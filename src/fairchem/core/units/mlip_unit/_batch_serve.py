"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any

import ray
import torch
from ray import serve

from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit import MLIPPredictUnit


@dataclass
class AutobatchConfig:
    """Configuration for probing-based autobatching.

    Attributes:
        min_batch_size: Minimum batch size (in atoms) to start probing from.
        max_batch_size_cap: Maximum batch size cap to avoid excessive probing.
        probe_steps: Number of probe steps to run at each batch size.
        backoff_factor: Factor to reduce batch size by after OOM (e.g., 0.8 = 80%).
        timeout_floor_s: Minimum batch wait timeout in seconds.
        timeout_ceil_s: Maximum batch wait timeout in seconds.
        timeout_latency_multiplier: Multiplier applied to median latency to compute timeout.
        warmup_steps: Number of warmup inference steps before probing.
    """

    min_batch_size: int = 128
    max_batch_size_cap: int = 16384
    probe_steps: int = 3
    backoff_factor: float = 0.8
    timeout_floor_s: float = 0.01
    timeout_ceil_s: float = 1.0
    timeout_latency_multiplier: float = 2.0
    warmup_steps: int = 2


@dataclass
class AutobatchResult:
    """Result from autobatch probing.

    Attributes:
        max_batch_size: Optimal maximum batch size in atoms.
        batch_wait_timeout_s: Optimal batch wait timeout in seconds.
        median_latency_s: Median inference latency observed during probing.
        probe_timestamp: Unix timestamp when probing was performed.
    """

    max_batch_size: int
    batch_wait_timeout_s: float
    median_latency_s: float
    probe_timestamp: float = field(default_factory=time.time)


def _expand_probe_data(
    data_list: list[AtomicData], target_num_atoms: int
) -> list[AtomicData]:
    """Expand probe data by repeating items to reach target atom count.

    Args:
        data_list: List of AtomicData objects to use as base data.
        target_num_atoms: Target total number of atoms for the batch.

    Returns:
        List of AtomicData objects with total atoms >= target_num_atoms.
    """
    if not data_list:
        raise ValueError("data_list cannot be empty")

    base_num_atoms = sum(data.natoms.sum().item() for data in data_list)

    if base_num_atoms >= target_num_atoms:
        return data_list

    num_repeats = (target_num_atoms + base_num_atoms - 1) // base_num_atoms

    expanded_list = []
    for _ in range(num_repeats):
        expanded_list.extend(data_list)

    return expanded_list


def probe_optimal_batch_size(
    model_key: str,
    probe_data: list[AtomicData],
    config: AutobatchConfig | None = None,
    device: str = "cuda",
) -> AutobatchResult:
    """Probe for optimal batch size and timeout using runtime GPU memory behavior.

    This function performs a binary search-like probing to find the maximum
    batch size that doesn't cause OOM errors, then derives an appropriate
    batch wait timeout from observed latencies.

    Args:
        model_key: Model key in format "{checkpoint_path_or_name}:{inference_settings}".
        probe_data: List of AtomicData objects to use for probing. If the total
            number of atoms is less than the target batch size being probed,
            the data will be repeated to reach the target size.
        config: Autobatch configuration. Uses defaults if None.
        device: Device to load the model on for probing.

    Returns:
        AutobatchResult with optimal parameters.
    """
    from fairchem.core.calculate import pretrained_mlip
    from fairchem.core.units.mlip_unit import load_predict_unit

    if config is None:
        config = AutobatchConfig()

    if not probe_data:
        raise ValueError("probe_data cannot be empty")

    # For CPU, use conservative defaults
    if "cuda" not in str(device):
        logging.info("Autobatch probing skipped for CPU device, using defaults")
        return AutobatchResult(
            max_batch_size=config.min_batch_size,
            batch_wait_timeout_s=config.timeout_ceil_s,
            median_latency_s=0.1,
        )

    # Parse model_key and load model for probing
    parts = model_key.split(":")
    checkpoint_name = parts[0]
    inference_settings = parts[1] if len(parts) > 1 else "default"

    is_local_path = (
        checkpoint_name.endswith(".pt")
        or "/" in checkpoint_name
        or "\\" in checkpoint_name
    )

    if is_local_path:
        predict_unit = load_predict_unit(
            checkpoint_name,
            inference_settings=inference_settings,
            device=device,
        )
    else:
        predict_unit = pretrained_mlip.get_predict_unit(
            checkpoint_name,
            inference_settings=inference_settings,
            device=device,
        )

    logging.info(f"Starting autobatch probing for model_key={model_key}...")

    free_mem, total_mem = (
        (0, 0) if not torch.cuda.is_available() else torch.cuda.mem_get_info()
    )
    logging.info(
        f"GPU memory: {free_mem / 1e9:.2f}GB free / {total_mem / 1e9:.2f}GB total"
    )

    # Warmup the model
    logging.info(f"Running {config.warmup_steps} warmup steps...")
    warmup_batch = atomicdata_list_to_batch(probe_data)
    for _ in range(config.warmup_steps):
        try:
            predict_unit.predict(warmup_batch, undo_element_references=False)
        except Exception as e:
            logging.warning(f"Warmup step failed: {e}")
    torch.cuda.empty_cache()

    # Binary search for optimal batch size
    low = config.min_batch_size
    high = config.max_batch_size_cap
    best_batch_size = low
    latencies: list[float] = []

    logging.info(f"Probing batch sizes in range [{low}, {high}]...")

    while low <= high:
        mid = (low + high) // 2
        success = True
        step_latencies = []

        logging.debug(f"Testing batch size: {mid} atoms")

        for step in range(config.probe_steps):
            try:
                expanded_data = _expand_probe_data(probe_data, mid)
                batch = atomicdata_list_to_batch(expanded_data)

                torch.cuda.synchronize()
                start = time.perf_counter()
                predict_unit.predict(batch, undo_element_references=False)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                step_latencies.append(elapsed)
                logging.debug(f"  Step {step + 1}: {elapsed:.4f}s")

            except torch.OutOfMemoryError:
                logging.debug(f"  OOM at batch size {mid}")
                success = False
                torch.cuda.empty_cache()
                break
            except Exception as e:
                logging.warning(f"  Probe failed at batch size {mid}: {e}")
                success = False
                break

        if success:
            best_batch_size = mid
            latencies.extend(step_latencies)
            low = mid + 1
            logging.debug(f"  Success at {mid}, trying larger...")
        else:
            high = mid - 1
            logging.debug(f"  Failed at {mid}, trying smaller...")

    final_batch_size = int(best_batch_size * config.backoff_factor)
    final_batch_size = max(final_batch_size, config.min_batch_size)

    if latencies:
        sorted_latencies = sorted(latencies)
        median_latency = sorted_latencies[len(sorted_latencies) // 2]
        timeout = median_latency * config.timeout_latency_multiplier
        timeout = max(config.timeout_floor_s, min(timeout, config.timeout_ceil_s))
    else:
        median_latency = 0.1
        timeout = config.timeout_ceil_s

    result = AutobatchResult(
        max_batch_size=final_batch_size,
        batch_wait_timeout_s=timeout,
        median_latency_s=median_latency,
    )

    logging.info(
        f"Autobatch probing complete: max_batch_size={result.max_batch_size}, "
        f"timeout={result.batch_wait_timeout_s:.4f}s, "
        f"median_latency={result.median_latency_s:.4f}s"
    )

    # Clean up the probe model
    del predict_unit
    torch.cuda.empty_cache()

    return result


def _batch_size_fn(requests: list[dict]) -> int:
    """Compute batch size as sum of atoms across all requests."""
    return sum(int(r["atomic_data"].natoms.sum()) for r in requests)


@serve.deployment(
    logging_config=serve.schema.LoggingConfig(log_level="WARNING"),
    max_ongoing_requests=300,
)
class BatchPredictServer:
    """
    Ray Serve deployment that batches incoming inference requests.

    This server always operates in multiplexed mode, loading models on-demand
    with LRU eviction. Single-model use is just multiplexed with one model_key.

    Request format:
        {
            "model_key": str,  # "{checkpoint_path_or_name}:{inference_settings}"
            "atomic_data": AtomicData,
            "undo_element_references": bool,
        }
    """

    def __init__(
        self,
        max_batch_size: int | None,
        batch_wait_timeout_s: float | None,
        split_oom_batch: bool = True,
        max_num_models_per_replica: int = 3,
    ):
        """
        Initialize the multiplexed batch prediction server.

        Args:
            max_batch_size: Maximum number of atoms in a batch. If None, batching
                must be configured via configure_batching() before running predictions.
            batch_wait_timeout_s: Timeout in seconds to wait for a batch.
                If None, batching must be configured before running predictions.
            split_oom_batch: If true will split batch if an OOM error is raised.
            max_num_models_per_replica: Maximum number of models to keep in memory
                per replica. Older models are evicted via LRU when exceeded.
        """
        self.split_oom_batch = split_oom_batch
        self._max_num_models_per_replica = max_num_models_per_replica
        self._batching_configured = False
        self._model_metadata_cache: dict[str, dict] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        if max_batch_size is not None and batch_wait_timeout_s is not None:
            self.configure_batching(max_batch_size, batch_wait_timeout_s)
        elif max_batch_size is not None or batch_wait_timeout_s is not None:
            raise ValueError(
                "Both max_batch_size and batch_wait_timeout_s must be provided together, "
                "or both must be None for later configuration."
            )
        else:
            logging.info(
                "BatchPredictServer initialized without batching configuration. "
                "Call configure_batching() before running predictions."
            )

        logging.info(
            f"BatchPredictServer initialized in multiplexed mode "
            f"(max_models={max_num_models_per_replica}, device={self._device})"
        )

    def configure_batching(
        self,
        max_batch_size: int,
        batch_wait_timeout_s: float,
    ):
        """Configure batching parameters.

        Args:
            max_batch_size: Maximum number of atoms in a batch.
            batch_wait_timeout_s: Maximum wait time before processing partial batch.
        """
        if max_batch_size is None or max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be a positive integer, got {max_batch_size}"
            )
        if batch_wait_timeout_s is None or batch_wait_timeout_s <= 0:
            raise ValueError(
                f"batch_wait_timeout_s must be a positive float, got {batch_wait_timeout_s}"
            )

        self.predict.set_max_batch_size(max_batch_size)
        self.predict.set_batch_wait_timeout_s(batch_wait_timeout_s)
        self._batching_configured = True
        logging.info(
            f"Batching configured: max_batch_size={max_batch_size}, "
            f"batch_wait_timeout_s={batch_wait_timeout_s}"
        )

    @serve.multiplexed(max_num_models_per_replica=3)
    async def get_model(self, model_key: str) -> MLIPPredictUnit:
        """
        Load model on demand with LRU eviction.

        model_key format: "{checkpoint_name_or_path}:{inference_settings}"
        e.g., "uma-s-1p1:default" or "/path/to/model.pt:turbo"

        If checkpoint_name looks like a file path (ends with .pt or contains /),
        it will be loaded directly from that path.
        Otherwise, it will be loaded from the pretrained model registry.
        """
        from fairchem.core.calculate import pretrained_mlip
        from fairchem.core.units.mlip_unit import load_predict_unit

        parts = model_key.split(":")
        checkpoint_name = parts[0]
        inference_settings = parts[1] if len(parts) > 1 else "default"

        start_time = (
            torch.cuda.Event(enable_timing=True) if self._device == "cuda" else None
        )
        end_time = (
            torch.cuda.Event(enable_timing=True) if self._device == "cuda" else None
        )

        if start_time:
            start_time.record()

        logging.info(f"Loading model '{model_key}'")

        is_local_path = (
            checkpoint_name.endswith(".pt")
            or "/" in checkpoint_name
            or "\\" in checkpoint_name
        )

        if is_local_path:
            logging.info(f"Loading local checkpoint from path: {checkpoint_name}")
            predict_unit = load_predict_unit(
                checkpoint_name,
                inference_settings=inference_settings,
                device=self._device,
            )
        else:
            predict_unit = pretrained_mlip.get_predict_unit(
                checkpoint_name,
                inference_settings=inference_settings,
                device=self._device,
            )

        self._cache_model_metadata(model_key, predict_unit)

        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            load_time = start_time.elapsed_time(end_time)
            logging.info(
                f"Successfully loaded model '{model_key}' in {load_time:.1f}ms"
            )
        else:
            logging.info(f"Successfully loaded model '{model_key}'")

        return predict_unit

    def _cache_model_metadata(self, model_key: str, predict_unit: MLIPPredictUnit):
        """Cache model metadata for client queries."""
        self._model_metadata_cache[model_key] = {
            "form_elem_refs": getattr(predict_unit, "form_elem_refs", {}),
            "atom_refs": getattr(predict_unit, "atom_refs", {}),
            "dataset_to_tasks": {
                name: [
                    {"name": t.name, "property": t.property, "level": t.level}
                    for t in tasks
                ]
                for name, tasks in predict_unit.dataset_to_tasks.items()
            },
        }

    async def fetch_model_metadata(self, model_key: str) -> dict[str, Any]:
        """
        Fetch metadata for a model, loading it if necessary.

        Returns cached metadata including dataset_to_tasks, form_elem_refs, atom_refs.
        """
        if model_key not in self._model_metadata_cache:
            await self.get_model(model_key)
        return self._model_metadata_cache.get(model_key, {})

    @serve.batch(batch_size_fn=_batch_size_fn)
    async def predict(self, requests: list[dict]) -> list[dict]:
        """
        Process a batch of inference requests.

        Args:
            requests: List of request dicts, each containing:
                - model_key: str
                - atomic_data: AtomicData
                - undo_element_references: bool

        Returns:
            List of prediction dictionaries, one per input request.
        """
        if not self._batching_configured:
            raise RuntimeError(
                "Batching has not been configured. Call configure_batching() "
                "before running predictions."
            )

        if not requests:
            return []

        # Group requests by model_key to prevent cross-model batching
        requests_by_model: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for idx, req in enumerate(requests):
            model_key = req["model_key"]
            requests_by_model[model_key].append((idx, req))

        # Pre-allocate results array to preserve original ordering
        results: list[dict | None] = [None] * len(requests)

        # Process each model group separately
        for model_key, indexed_requests in requests_by_model.items():
            predict_unit = await self.get_model(model_key)

            indices = [idx for idx, _ in indexed_requests]
            model_requests = [req for _, req in indexed_requests]

            data_list = [req["atomic_data"] for req in model_requests]
            undo_refs = model_requests[0].get("undo_element_references", True)

            # Process with OOM recovery
            split_preds = await self._predict_with_oom_recovery(
                predict_unit, data_list, undo_refs
            )

            # Place results back in original positions
            for idx, pred in zip(indices, split_preds, strict=False):
                results[idx] = pred

        return results

    async def _predict_with_oom_recovery(
        self,
        predict_unit: MLIPPredictUnit,
        data_list: list[AtomicData],
        undo_element_references: bool,
    ) -> list[dict]:
        """Run inference with OOM recovery by splitting batches."""
        data_deque = deque([data_list])
        prediction_list = []

        while len(data_deque) > 0:
            oom = False
            current_data_list = data_deque.popleft()
            batch = atomicdata_list_to_batch(current_data_list)

            try:
                predictions = predict_unit.predict(
                    batch, undo_element_references=undo_element_references
                )
                prediction_list.extend(self._split_predictions(predictions, batch))
            except torch.OutOfMemoryError as err:
                logging.warning(f"OutOfMemoryError during inference: {err}")
                if not self.split_oom_batch:
                    raise torch.OutOfMemoryError(
                        "Reduce max_batch_size or set split_oom_batch=True."
                    ) from err

                if len(current_data_list) == 1:
                    raise torch.OutOfMemoryError(
                        "Out of memory for a single system."
                    ) from err

                logging.warning("Splitting batch and retrying.")
                oom = True
                torch.cuda.empty_cache()

            if oom:
                mid = len(current_data_list) // 2
                data_deque.appendleft(current_data_list[mid:])
                data_deque.appendleft(current_data_list[:mid])

        return prediction_list

    def _split_predictions(
        self,
        predictions: dict,
        batch: AtomicData,
    ) -> list[dict]:
        """Split batched predictions back into individual system predictions."""
        split_preds = []
        for i in range(len(batch)):
            system_predictions = {}

            for key, pred in predictions.items():
                if pred.shape[0] == len(batch):
                    system_predictions[key] = pred[i : i + 1]
                elif pred.shape[0] == len(batch.batch):
                    mask = batch.batch == i
                    system_predictions[key] = pred[mask]
                else:
                    raise ValueError(
                        f"Cannot split prediction for key '{key}': "
                        f"unexpected shape {pred.shape} for batch size {len(batch)} "
                        f"and num_atoms {batch.num_atoms}"
                    )

            split_preds.append(system_predictions)

        return split_preds

    async def __call__(self, request: dict) -> dict:
        """
        Main entry point for inference requests.

        Handles both prediction and metadata requests.

        Args:
            request: Dict containing either:
                - For predictions: model_key, atomic_data, undo_element_references
                - For metadata: request_type="metadata", model_key

        Returns:
            Prediction dictionary or metadata dictionary.
        """
        if request.get("request_type") == "metadata":
            return await self.fetch_model_metadata(request["model_key"])

        return await self.predict(request)


def setup_batch_predict_server(
    max_batch_size: int | None = None,
    batch_wait_timeout_s: float | None = None,
    split_oom_batch: bool = True,
    num_replicas: int = 1,
    ray_actor_options: dict | None = None,
    deployment_name: str = "fairchem-inference",
    route_prefix: str = "/inference",
    autoscaling_config: dict | None = None,
    deployment_config: dict | None = None,
    max_num_models_per_replica: int = 3,
) -> serve.handle.DeploymentHandle:
    """
    Set up and deploy a BatchPredictServer for batched inference.

    The server operates in multiplexed mode, loading models on-demand with LRU eviction.

    Args:
        max_batch_size: Maximum number of atoms in a batch. If None, batching must
            be configured later via configure_batching() before running predictions.
        batch_wait_timeout_s: Maximum wait time before processing partial batch.
            If None, batching must be configured later.
        split_oom_batch: Whether to split batches that cause OOM errors.
        num_replicas: Number of deployment replicas (ignored if autoscaling_config provided).
        ray_actor_options: Ray actor options (e.g., {"num_gpus": 1, "num_cpus": 4}).
        deployment_name: Name for the Ray Serve deployment.
        route_prefix: HTTP route prefix for the deployment.
        autoscaling_config: Autoscaling configuration dict. If provided, overrides num_replicas.
            Example: {"min_replicas": 0, "max_replicas": 4, "downscale_delay_s": 60}
        deployment_config: Additional deployment configuration to merge in.
        max_num_models_per_replica: Maximum models to keep in memory per replica.

    Returns:
        Ray Serve deployment handle.
    """
    if ray_actor_options is None:
        ray_actor_options = {}

    cpus_per_actor = ray_actor_options.get("num_cpus", min(cpu_count(), 8))
    ray_actor_options["num_cpus"] = cpus_per_actor

    if torch.cuda.is_available() and "num_gpus" not in ray_actor_options:
        ray_actor_options["num_gpus"] = 1

    if not ray.is_initialized():
        ray.init(
            log_to_driver=False,
            logging_config=ray.LoggingConfig(log_level="WARNING"),
            num_cpus=cpus_per_actor * num_replicas,
        )
        logging.info("Ray initialized by setup_batch_predict_server")

    serve.start(
        logging_config=serve.schema.LoggingConfig(log_level="WARNING"),
    )
    logging.info("Ray Serve started by setup_batch_predict_server")

    # Build deployment options
    deploy_options = copy.deepcopy(deployment_config) if deployment_config else {}
    deploy_options["ray_actor_options"] = ray_actor_options

    if autoscaling_config:
        deploy_options["autoscaling_config"] = autoscaling_config
        logging.info(f"Autoscaling enabled: {autoscaling_config}")
    else:
        deploy_options["num_replicas"] = num_replicas

    deployment = BatchPredictServer.options(**deploy_options).bind(
        max_batch_size=max_batch_size,
        batch_wait_timeout_s=batch_wait_timeout_s,
        split_oom_batch=split_oom_batch,
        max_num_models_per_replica=max_num_models_per_replica,
    )

    handle = serve.run(deployment, name=deployment_name, route_prefix=route_prefix)

    logging.info(
        f"BatchPredictServer deployed: max_batch_size={max_batch_size}, "
        f"batch_wait_timeout_s={batch_wait_timeout_s}, "
        f"max_num_models_per_replica={max_num_models_per_replica}, "
        f"name={deployment_name}"
    )

    return handle


def wait_for_serve_ready(
    app_name: str = "fairchem-inference",
    poll_interval_seconds: float = 2.0,
) -> bool:
    """
    Wait for Ray Serve to be fully ready to accept requests.

    Blocks until:
    1. Ray Serve controller is running
    2. The specified application is deployed and RUNNING

    Args:
        app_name: Name of the Ray Serve application to wait for.
        poll_interval_seconds: How often to check status.

    Returns:
        True if server is ready.

    Raises:
        RuntimeError: If server fails to deploy.
    """
    from ray.serve.schema import ApplicationStatus

    logging.info("Waiting for Ray Serve controller to start...")
    while True:
        try:
            status = serve.status()
            logging.info("Ray Serve controller is running")
            break
        except Exception as e:
            error_msg = str(e)
            if (
                "SERVE_CONTROLLER_ACTOR" in error_msg
                or "Failed to look up actor" in error_msg
            ):
                logging.debug(f"Ray Serve controller not ready yet: {error_msg}")
                time.sleep(poll_interval_seconds)
            else:
                raise

    logging.info(f"Waiting for application '{app_name}' to be ready...")
    while True:
        try:
            status = serve.status()

            if app_name not in status.applications:
                logging.debug(f"Application '{app_name}' not found yet, waiting...")
                time.sleep(poll_interval_seconds)
                continue

            app_status = status.applications[app_name]

            if app_status.status == ApplicationStatus.RUNNING:
                logging.info(f"Application '{app_name}' is RUNNING and ready")
                return True
            elif app_status.status == ApplicationStatus.DEPLOYING:
                logging.debug(f"Application '{app_name}' is still deploying...")
                time.sleep(poll_interval_seconds)
            elif app_status.status in (
                ApplicationStatus.DEPLOY_FAILED,
                ApplicationStatus.UNHEALTHY,
            ):
                raise RuntimeError(
                    f"Application '{app_name}' failed to deploy. "
                    f"Status: {app_status.status}, Message: {app_status.message}"
                )
            else:
                logging.debug(f"Application '{app_name}' status: {app_status.status}")
                time.sleep(poll_interval_seconds)

        except RuntimeError:
            raise
        except Exception as e:
            logging.warning(f"Error checking serve status: {e}")
            time.sleep(poll_interval_seconds)
