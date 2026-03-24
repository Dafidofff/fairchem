"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Literal, Protocol

from fairchem.core.units.mlip_unit._batch_serve import (
    AutobatchConfig,
    AutobatchResult,
    probe_optimal_batch_size,
    setup_batch_predict_server,
    wait_for_serve_ready,
)
from fairchem.core.units.mlip_unit.predict import (
    BatchServerPredictUnit,
    MLIPPredictUnit,
    MLIPWorkerPredictUnit,
)

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData


__all__ = [
    "InferenceBatcher",
    "AutobatchConfig",
    "AutobatchResult",
    "probe_optimal_batch_size",
    "BatchServerPredictUnit",
    "MLIPPredictUnit",
    "MLIPWorkerPredictUnit",
    "wait_for_serve_ready",
]


class ExecutorProtocol(Protocol):
    def submit(self, fn, *args, **kwargs): ...
    def map(self, fn, *iterables, **kwargs): ...
    def shutdown(self, wait: bool = True): ...


class RayActorPoolExecutor:
    """
    Executor-like wrapper around Ray's ActorPool for concurrent task execution.

    This provides an interface compatible with ExecutorProtocol while using
    Ray actors for distributed execution.

    Note: Tasks submitted to this executor must be picklable and should not
    rely on shared state between workers.
    """

    def __init__(
        self,
        num_workers: int = 4,
        ray_actor_options: dict | None = None,
    ):
        """
        Args:
            num_workers: Number of Ray actor workers in the pool.
            ray_actor_options: Options to pass to Ray actor creation
                (e.g., {"num_cpus": 1, "num_gpus": 0}).
        """
        import ray
        from ray.util import ActorPool

        if ray_actor_options is None:
            ray_actor_options = {}

        if not ray.is_initialized():
            ray.init(
                log_to_driver=False,
                logging_config=ray.LoggingConfig(log_level="WARNING"),
            )

        @ray.remote
        class _TaskWorker:
            """Simple worker that executes arbitrary callables."""

            def execute(self, fn, *args, **kwargs):
                return fn(*args, **kwargs)

        # Create actors with configured options
        self._actors = [
            _TaskWorker.options(**ray_actor_options).remote()
            for _ in range(num_workers)
        ]
        self._pool = ActorPool(self._actors)
        self._current_actor_idx = 0

    def submit(self, fn, *args, **kwargs):
        """Submit a callable to be executed by a worker using round-robin selection.

        Args:
            fn: Callable to execute (must be picklable).
            *args: Positional arguments to pass to fn.
            **kwargs: Keyword arguments to pass to fn.

        Returns:
            A Ray ObjectRef. Use ray.get() to retrieve the result.
        """
        actor = self._actors[self._current_actor_idx]
        self._current_actor_idx = (self._current_actor_idx + 1) % len(self._actors)
        return actor.execute.remote(fn, *args, **kwargs)

    def map(self, fn, *iterables, **kwargs):
        """Map a callable over iterables using the actor pool.

        Args:
            fn: Callable to execute (must be picklable).
            *iterables: Iterables to map over.
            **kwargs: Additional keyword arguments (timeout is ignored).

        Returns:
            Iterator of results.
        """
        # Combine iterables into argument tuples
        args_list = list(zip(*iterables))

        # Use ActorPool's map for efficient distribution
        def _task(actor, args):
            return actor.execute.remote(fn, *args)

        return self._pool.map(_task, args_list)

    def shutdown(self, wait: bool = True):
        """Shutdown the actor pool.

        Args:
            wait: If True, wait for pending tasks (not fully supported with Ray actors).
        """
        import ray

        for actor in self._actors:
            ray.kill(actor)
        self._actors = []


def _get_concurrency_backend(
    backend: Literal["threads", "processes", "ray-actors"], options: dict
) -> ExecutorProtocol:
    """Get a backend to run ASE calculations concurrently.

    Args:
        backend: The concurrency backend type:
            - "threads": ThreadPoolExecutor for I/O-bound tasks (default).
            - "processes": ProcessPoolExecutor for CPU-bound tasks.
                Note: Tasks must be picklable; not suitable for GPU operations.
            - "ray-actors": Ray actor pool for distributed execution.
                Supports options like num_workers and ray_actor_options.
        options: Backend-specific options dictionary.

    Returns:
        An executor implementing ExecutorProtocol.

    Raises:
        ValueError: If an invalid backend is specified.
    """
    if backend == "threads":
        return ThreadPoolExecutor(**options)
    elif backend == "processes":
        return ProcessPoolExecutor(**options)
    elif backend == "ray-actors":
        return RayActorPoolExecutor(**options)
    raise ValueError(f"Invalid concurrency backend: {backend}")


class InferenceBatcher:
    """Batches incoming inference requests using Ray Serve.

    This class provides a high-level API for running concurrent simulations
    with batched inference calls to AI models. Models are loaded on-demand
    with LRU eviction, supporting both single-model and multi-model use cases.

    Example:
        >>> with InferenceBatcher(max_batch_size=1024) as batcher:
        ...     predict_unit = batcher.get_predict_unit("uma-s-1p1:default")
        ...     calc = FAIRChemCalculator(predict_unit, task_name="omat")
        ...     # Run concurrent simulations using batcher.executor
        ...     futures = [batcher.executor.submit(run_sim, atoms) for atoms in systems]

    Example with multiple models:
        >>> with InferenceBatcher(max_batch_size=1024, max_num_models_per_replica=3) as batcher:
        ...     unit_small = batcher.get_predict_unit("uma-s-1p1:default")
        ...     unit_large = batcher.get_predict_unit("uma-l-1p1:default")
        ...     calc_small = FAIRChemCalculator(unit_small, task_name="omat")
        ...     calc_large = FAIRChemCalculator(unit_large, task_name="omat")

    Example with autoscaling:
        >>> batcher = InferenceBatcher(
        ...     autoscaling_config={
        ...         "min_replicas": 0,
        ...         "max_replicas": 8,
        ...         "downscale_delay_s": 60,
        ...     }
        ... )
        >>> predict_unit = batcher.get_predict_unit("uma-s-1p1:default")
    """

    def __init__(
        self,
        max_batch_size: int | None = None,
        batch_wait_timeout_s: float | None = None,
        split_oom_batch: bool = True,
        num_replicas: int = 1,
        concurrency_backend: Literal["threads", "processes", "ray-actors"] = "threads",
        concurrency_backend_options: dict | None = None,
        ray_actor_options: dict | None = None,
        autoscaling_config: dict | None = None,
        deployment_config: dict | None = None,
        max_num_models_per_replica: int = 3,
        deployment_name: str = "fairchem-inference",
    ):
        """
        Args:
            max_batch_size: Maximum number of atoms in a batch. If None, must call
                auto_configure_batching() before running predictions to automatically
                determine optimal parameters.
            batch_wait_timeout_s: The maximum time to wait for a batch to be ready.
                If None, must call auto_configure_batching() before running predictions.
                Note: Both max_batch_size and batch_wait_timeout_s must be provided
                together, or both must be None for autobatch configuration.
            split_oom_batch: If True, split and retry on OOM errors.
            num_replicas: The number of replicas to use for inference (ignored if
                autoscaling_config is provided).
            concurrency_backend: The concurrency backend to use for running simulations:
                - "threads": ThreadPoolExecutor (default). Best for I/O-bound tasks.
                    Options: max_workers (int).
                - "processes": ProcessPoolExecutor. Best for CPU-bound tasks.
                    Note: Tasks must be picklable; not suitable for GPU operations.
                    Options: max_workers (int).
                - "ray-actors": Ray actor pool for distributed execution.
                    Options: num_workers (int), ray_actor_options (dict).
            concurrency_backend_options: Options to pass to the concurrency backend.
                See backend descriptions above for available options.
            ray_actor_options: Options to pass to the Ray actor running the batch server.
            autoscaling_config: Autoscaling configuration for the deployment.
                Example: {"min_replicas": 0, "max_replicas": 4, "downscale_delay_s": 60}
            deployment_config: Additional deployment configuration to merge in.
            max_num_models_per_replica: Maximum number of models to keep in memory
                per replica. Older models are evicted via LRU when exceeded.
            deployment_name: Name for the Ray Serve deployment.

        Raises:
            ValueError: If only one of max_batch_size or batch_wait_timeout_s is provided.
        """
        self.num_replicas = num_replicas
        self._concurrency_backend = concurrency_backend
        self._deployment_name = deployment_name

        self.predict_server_handle = setup_batch_predict_server(
            max_batch_size=max_batch_size,
            batch_wait_timeout_s=batch_wait_timeout_s,
            split_oom_batch=split_oom_batch,
            num_replicas=self.num_replicas,
            ray_actor_options=ray_actor_options or {},
            deployment_name=deployment_name,
            autoscaling_config=autoscaling_config,
            deployment_config=deployment_config,
            max_num_models_per_replica=max_num_models_per_replica,
        )

        if concurrency_backend_options is None:
            concurrency_backend_options = {}

        # Set default max_workers for thread and process backends
        if concurrency_backend in ("threads", "processes"):
            if "max_workers" not in concurrency_backend_options:
                concurrency_backend_options["max_workers"] = min(cpu_count(), 16)
        # Set default num_workers for ray-actors backend
        elif (
            concurrency_backend == "ray-actors"
            and "num_workers" not in concurrency_backend_options
        ):
            concurrency_backend_options["num_workers"] = min(cpu_count(), 16)

        self.executor: ExecutorProtocol = _get_concurrency_backend(
            concurrency_backend, concurrency_backend_options
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def get_predict_unit(self, model_key: str) -> BatchServerPredictUnit:
        """Get a predict unit for the specified model.

        Args:
            model_key: Model identifier in format "{checkpoint_path_or_name}:{inference_settings}"
                e.g., "uma-s-1p1:default" or "/path/to/model.pt:turbo"

        Returns:
            BatchServerPredictUnit configured for the specified model.
        """
        return BatchServerPredictUnit(
            server_handle=self.predict_server_handle,
            model_key=model_key,
        )

    def auto_configure_batching(
        self,
        model_key: str,
        probe_data: list[AtomicData],
        config: AutobatchConfig | None = None,
    ) -> AutobatchResult:
        """Probe for optimal batch size and timeout using representative data.

        This method runs inference with increasing batch sizes until OOM,
        then configures the server with optimal parameters.

        Args:
            model_key: Model to probe with in format "{checkpoint}:{settings}".
            probe_data: List of AtomicData objects to use for probing. The data
                will be repeated if needed to reach larger batch sizes during probing.
            config: Autobatch configuration. Uses defaults if None.

        Returns:
            AutobatchResult with the determined optimal parameters.
        """
        if config is None:
            config = AutobatchConfig()

        result = probe_optimal_batch_size(
            model_key=model_key,
            probe_data=probe_data,
            config=config,
        )

        # Update the server with the new batching parameters
        self.predict_server_handle.configure_batching.remote(
            result.max_batch_size, result.batch_wait_timeout_s
        )

        return result

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.

        Args:
            wait: If True, wait for pending tasks to complete before returning.
        """
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=wait)

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown(wait=False)
