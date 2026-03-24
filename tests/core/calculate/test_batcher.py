"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy.testing as npt
import pytest
import ray
import torch
from ase.build import bulk
from ray import serve

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.calculate._batch import AutobatchConfig, InferenceBatcher
from fairchem.core.datasets.atomic_data import AtomicData

# mark all tests in this module as serial (Ray needs serial execution due to large number of subprocesses)
pytestmark = pytest.mark.serial


@pytest.fixture(scope="module")
def uma_model_key():
    """Get a UMA model key for testing."""
    uma_models = [name for name in pretrained_mlip.available_models if "uma" in name]
    return f"{uma_models[0]}:default"


@pytest.fixture(scope="module")
def uma_predict_unit(uma_model_key):
    """Get a UMA predict unit for testing (for serial comparison)."""
    model_name = uma_model_key.split(":")[0]
    return pretrained_mlip.get_predict_unit(model_name)


def setup_ray():
    pytest.importorskip("ray.serve", reason="ray[serve] not installed")

    if ray.is_initialized():
        with contextlib.suppress(Exception):
            serve.shutdown()
        ray.shutdown()

    ray.init(
        ignore_reinit_error=True,
        num_cpus=4,
        num_gpus=1 if torch.cuda.is_available() else 0,
        logging_level="ERROR",  # Reduce noise in test output
    )


def cleanup_ray():
    try:
        serve.shutdown()
    except Exception as e:
        print(f"Warning: Error during serve shutdown: {e}")
    try:
        ray.shutdown()
    except Exception as e:
        print(f"Warning: Error during ray shutdown: {e}")


@pytest.fixture()
def inference_batcher(uma_model_key):
    batcher = InferenceBatcher(
        max_batch_size=8,
        batch_wait_timeout_s=0.05,
        num_replicas=1,
        concurrency_backend="threads",
        concurrency_backend_options={"max_workers": 4},
    )

    yield batcher, uma_model_key

    cleanup_ray()


@pytest.mark.gpu()
def test_initialization_with_custom_concurrency_options():
    try:
        max_workers = 8
        batcher = InferenceBatcher(
            max_batch_size=16,
            batch_wait_timeout_s=0.1,
            num_replicas=1,
            concurrency_backend="threads",
            concurrency_backend_options={"max_workers": max_workers},
        )

        assert isinstance(batcher.executor, ThreadPoolExecutor)
    finally:
        cleanup_ray()


@pytest.mark.gpu()
def test_initialization_with_ray_actor_options():
    try:
        batcher = InferenceBatcher(
            max_batch_size=16,
            batch_wait_timeout_s=0.1,
            num_replicas=1,
            ray_actor_options={"num_cpus": 2},
        )

        assert hasattr(batcher, "predict_server_handle")
    finally:
        cleanup_ray()


@pytest.mark.gpu()
def test_context_manager_enter_exit():
    try:
        with InferenceBatcher(
            max_batch_size=16,
            batch_wait_timeout_s=0.1,
            num_replicas=1,
        ) as batcher:
            assert hasattr(batcher, "executor")
            assert hasattr(batcher, "predict_server_handle")
            executor = batcher.executor

        assert executor is not None

        with pytest.raises(
            RuntimeError, match="cannot schedule new futures after shutdown"
        ):
            executor.submit(time.sleep, 1)
    finally:
        cleanup_ray()


@pytest.mark.gpu()
def test_batched_atomic_data_predictions(inference_batcher):
    """Test batched predictions using AtomicData directly."""
    batcher, model_key = inference_batcher
    predict_unit = batcher.get_predict_unit(model_key)

    atoms_list = [bulk("Cu"), bulk("Al"), bulk("Fe")]
    atomic_data_list = [
        AtomicData.from_ase(atoms, task_name="omat") for atoms in atoms_list
    ]

    with ThreadPoolExecutor(max_workers=len(atoms_list)) as executor:
        futures = [
            executor.submit(predict_unit.predict, data) for data in atomic_data_list
        ]
        results = [future.result() for future in futures]

    assert len(results) == len(atoms_list)
    for i, preds in enumerate(results):
        assert "energy" in preds
        assert "forces" in preds
        assert preds["energy"].shape == (1,)
        assert preds["forces"].shape == (len(atoms_list[i]), 3)


@pytest.mark.gpu()
def test_batch_vs_serial_consistency(inference_batcher, uma_predict_unit):
    """Test that batched and serial calculations produce consistent results."""
    batcher, model_key = inference_batcher
    batch_predict_unit = batcher.get_predict_unit(model_key)

    atoms_list = [
        bulk("Cu"),
        bulk("Al"),
        bulk("Fe"),
        bulk("Ni"),
    ]

    def calculate_properties(atoms, predict_unit):
        atoms.calc = FAIRChemCalculator(predict_unit, task_name="omat")
        return {
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
        }

    results_batched = list(
        batcher.executor.map(
            partial(
                calculate_properties,
                predict_unit=batch_predict_unit,
            ),
            atoms_list,
        )
    )

    results_serial = [
        calculate_properties(atoms, uma_predict_unit) for atoms in atoms_list
    ]

    assert len(results_batched) == len(results_serial)
    for r_batch, r_serial in zip(results_batched, results_serial):
        npt.assert_allclose(r_batch["energy"], r_serial["energy"], atol=1e-4)
        npt.assert_allclose(r_batch["forces"], r_serial["forces"], atol=1e-4)


@pytest.mark.gpu()
def test_initialization_with_processes_backend():
    """Test initialization with ProcessPoolExecutor backend."""
    try:
        from concurrent.futures import ProcessPoolExecutor

        batcher = InferenceBatcher(
            max_batch_size=16,
            batch_wait_timeout_s=0.1,
            num_replicas=1,
            concurrency_backend="processes",
            concurrency_backend_options={"max_workers": 2},
        )

        assert isinstance(batcher.executor, ProcessPoolExecutor)
    finally:
        cleanup_ray()


@pytest.mark.gpu()
def test_initialization_with_ray_actors_backend():
    """Test initialization with Ray actor pool backend."""
    try:
        from fairchem.core.calculate._batch import RayActorPoolExecutor

        batcher = InferenceBatcher(
            max_batch_size=16,
            batch_wait_timeout_s=0.1,
            num_replicas=1,
            concurrency_backend="ray-actors",
            concurrency_backend_options={"num_workers": 2},
        )

        assert isinstance(batcher.executor, RayActorPoolExecutor)
    finally:
        cleanup_ray()


@pytest.mark.gpu()
def test_autobatch_config_initialization(uma_model_key):
    """Test initialization and auto_configure_batching method."""
    try:
        autobatch_config = AutobatchConfig(
            min_batch_size=64,
            max_batch_size_cap=1024,
            probe_steps=2,
            warmup_steps=1,
        )

        batcher = InferenceBatcher(
            split_oom_batch=True,
            num_replicas=1,
        )

        # Create probe data for autobatch configuration
        probe_data = [AtomicData.from_ase(bulk("Cu"), task_name="omat")]

        # Configure autobatch with probe data
        result = batcher.auto_configure_batching(
            model_key=uma_model_key,
            probe_data=probe_data,
            config=autobatch_config,
        )

        # Autobatch should return a result with max_batch_size and timeout
        assert result.max_batch_size >= autobatch_config.min_batch_size
        assert result.batch_wait_timeout_s > 0
    finally:
        cleanup_ray()


@pytest.mark.gpu()
def test_batcher_with_explicit_values(uma_model_key):
    """Test that explicit batch size and timeout values are used."""
    try:
        batcher = InferenceBatcher(
            max_batch_size=256,
            batch_wait_timeout_s=0.2,
            num_replicas=1,
        )

        # Batcher should be created successfully with explicit values
        assert hasattr(batcher, "predict_server_handle")

        # Should be able to get a predict unit
        predict_unit = batcher.get_predict_unit(uma_model_key)
        assert predict_unit is not None
    finally:
        cleanup_ray()


def test_probe_optimal_batch_size_cpu(uma_model_key):
    """Test probing on CPU returns defaults."""
    from fairchem.core.units.mlip_unit._batch_serve import (
        AutobatchConfig,
        probe_optimal_batch_size,
    )

    config = AutobatchConfig()
    # Create probe data for the test
    probe_data = [AtomicData.from_ase(bulk("Cu"), task_name="omat")]
    result = probe_optimal_batch_size(
        model_key=uma_model_key,
        probe_data=probe_data,
        config=config,
        device="cpu",
    )

    # CPU should return defaults
    assert result.max_batch_size == config.min_batch_size
    assert result.batch_wait_timeout_s == config.timeout_ceil_s


@pytest.mark.gpu()
def test_get_predict_unit_returns_batch_server_predict_unit(uma_model_key):
    """Test that get_predict_unit returns a properly configured BatchServerPredictUnit."""
    try:
        from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit

        batcher = InferenceBatcher(
            max_batch_size=16,
            batch_wait_timeout_s=0.1,
            num_replicas=1,
        )

        predict_unit = batcher.get_predict_unit(uma_model_key)

        assert isinstance(predict_unit, BatchServerPredictUnit)
        assert predict_unit._model_key == uma_model_key
    finally:
        cleanup_ray()


@pytest.mark.gpu()
def test_multiple_models_same_batcher():
    """Test that multiple models can be used with the same batcher."""
    try:
        uma_models = [
            name for name in pretrained_mlip.available_models if "uma" in name
        ]
        if len(uma_models) < 2:
            pytest.skip("Need at least 2 UMA models for this test")

        model_key_1 = f"{uma_models[0]}:default"
        model_key_2 = f"{uma_models[1]}:default" if len(uma_models) > 1 else model_key_1

        batcher = InferenceBatcher(
            max_batch_size=16,
            batch_wait_timeout_s=0.1,
            num_replicas=1,
            max_num_models_per_replica=3,
        )

        unit_1 = batcher.get_predict_unit(model_key_1)
        unit_2 = batcher.get_predict_unit(model_key_2)

        # Both should be usable
        assert unit_1._model_key == model_key_1
        assert unit_2._model_key == model_key_2

        # Should share the same server handle
        assert unit_1._handle is unit_2._handle
    finally:
        cleanup_ray()


@pytest.mark.gpu()
def test_autoscaling_config():
    """Test that autoscaling configuration is accepted."""
    try:
        batcher = InferenceBatcher(
            max_batch_size=16,
            batch_wait_timeout_s=0.1,
            autoscaling_config={
                "min_replicas": 1,
                "max_replicas": 2,
                "target_ongoing_requests": 2,
            },
        )

        assert hasattr(batcher, "predict_server_handle")
    finally:
        cleanup_ray()
