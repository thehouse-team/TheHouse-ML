"""ML-06 T04 (ML-06-R04): medición de rendimiento con un modelo sintético (spec ML-06 §6.5).

No necesita `models/` ni `data/`: es la prueba de humo informativa que corre en CI, sin umbral
estricto de latencia (spec §6.5: «sin umbral estricto, para no depender del runner»). La medición
oficial (mediana < 20 ms con `logreg-1.0.0`) está en
`tests/integration/test_onnx_benchmark_official.py` y en la evidencia de T04.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from thehouse_ml.export import ExportError
from thehouse_ml.export.benchmark import (
    BENCHMARK_ROOMS,
    MEASURED_RUNS,
    MEDIAN_BUDGET_MS,
    WARMUP_RUNS,
    BenchmarkResult,
    benchmark_batch,
    cpu_name,
    main,
    run_benchmark,
)
from thehouse_ml.export.onnx_graph import build_onnx_model, graph_constants, model_metadata, serialize_model
from thehouse_ml.features import default_schema
from thehouse_ml.models.train import LOGREG_FORMAT, LogRegModel
from thehouse_ml.models.train_config import PROFILE_ENV_VAR
from thehouse_ml.models.trained_breakdown import build_breakdown_model

LOWER = [0.0] * 10
UPPER = [30.0, 30.0, 1.0, 1.0, 1.0, 1.0, 1.0, 60.0, 1.0, 1.0]
# Cota de cordura de la prueba de humo, no el criterio de la spec: 50 veces el presupuesto, para
# que un runner lento o compartido no vuelva inestable el CI.
SMOKE_SANITY_CEILING_MS = 50 * MEDIAN_BUDGET_MS


def logreg_document() -> dict[str, Any]:
    """`logreg.json` sintético con la forma de `thehouse-ml/logreg-json/1`."""
    return {
        "synthetic": True,
        "format": LOGREG_FORMAT,
        "feature_order": list(default_schema().feature_order),
        "config": {"transform": "standard", "C": 0.01},
        "clip": {"lower": LOWER, "upper": UPPER},
        "transform": {
            "type": "standard",
            "mean": [5.2, 2.5, 0.55, 0.73, 0.3, 0.7, 1.0, 1.33, 0.14, 0.48],
            "scale": [6.3, 3.7, 0.23, 0.18, 0.46, 0.46, 1.0, 3.42, 0.22, 0.21],
        },
        "coef": [-3.35, -1.87, -0.996, 0.566, 1.006, 0.36, 0.0, -0.1, -0.0123, 0.439],
        "intercept": -4.13,
    }


@pytest.fixture(scope="module")
def model_bytes() -> bytes:
    breakdown = build_breakdown_model(LogRegModel.from_dict(logreg_document()))
    constants = graph_constants(breakdown)
    return serialize_model(build_onnx_model(constants, model_metadata(breakdown, constants, "0" * 64, "abc", "ci")))


def synthetic_features(rows: int, seed: int = 20261125) -> np.ndarray:
    """Vectores deterministas dentro de la caja del recorte (las binarias en {0, 1})."""
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(np.asarray(LOWER), np.asarray(UPPER), size=(rows, 10))
    matrix[:, 4:7] = np.round(matrix[:, 4:7])
    return matrix


def result_with(timings_ms: list[float], budget_ms: float = MEDIAN_BUDGET_MS) -> BenchmarkResult:
    return BenchmarkResult(
        rooms=BENCHMARK_ROOMS,
        warmup_runs=WARMUP_RUNS,
        timings_ns=tuple(round(value * 1_000_000) for value in timings_ms),
        intra_op_threads=0,
        inter_op_threads=0,
        budget_ms=budget_ms,
    )


def test_measurement_parameters_are_those_of_spec_6_5() -> None:
    assert (BENCHMARK_ROOMS, WARMUP_RUNS, MEASURED_RUNS, MEDIAN_BUDGET_MS) == (200, 10, 100, 20.0)


def test_batch_takes_the_first_200_rows_as_contiguous_double() -> None:
    features = synthetic_features(250).astype(np.float32)
    batch = benchmark_batch(features)
    assert batch.shape == (200, 10)
    assert batch.dtype == np.float64
    assert batch.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(batch, features[:200].astype(np.float64))


@pytest.mark.parametrize("shape", [(199, 10), (200, 9), (200,)])
def test_batch_rejects_too_few_rooms_or_a_wrong_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ExportError):
        benchmark_batch(np.zeros(shape))


def test_statistics_use_the_median_and_the_linear_95th_percentile() -> None:
    timings = [float(value) for value in range(1, 101)]  # 1..100 ms
    result = result_with(timings)
    assert result.median_ms == pytest.approx(50.5)
    assert result.p95_ms == pytest.approx(float(np.percentile(timings, 95)))
    assert result.p95_ms == pytest.approx(95.05)
    assert result.min_ms == pytest.approx(1.0)
    assert result.max_ms == pytest.approx(100.0)
    assert result.mean_ms == pytest.approx(50.5)


def test_criterion_is_median_strictly_below_the_budget_regardless_of_the_tail() -> None:
    assert result_with([1.0] * 99 + [500.0]).passed  # un pico aislado no cambia la mediana
    assert not result_with([20.0] * 100).passed  # 20 ms no es < 20 ms
    assert not result_with([25.0] * 51 + [1.0] * 49).passed


def test_an_empty_measurement_does_not_pass() -> None:
    result = result_with([])
    assert not result.passed
    with pytest.raises(ExportError):
        _ = result.median_ms


def test_report_has_the_fields_required_by_the_spec() -> None:
    report = result_with([1.0] * 100).as_dict()
    for name in (
        "rooms",
        "warmup_runs",
        "measured_runs",
        "timer",
        "session_created_once",
        "intra_op_num_threads",
        "inter_op_num_threads",
        "median_ms",
        "p95_ms",
        "max_ms",
        "passed",
        "environment",
    ):
        assert name in report
    assert report["timer"] == "time.perf_counter_ns"
    assert "PASS" in result_with([1.0] * 100).summary()
    assert "FALLA" in result_with([30.0] * 100).summary()


def test_smoke_run_on_the_synthetic_model(model_bytes: bytes) -> None:
    """Humo informativo (CI): corre la medición completa de la spec sin exigir los 20 ms."""
    result = run_benchmark(model_bytes, benchmark_batch(synthetic_features(BENCHMARK_ROOMS)))
    assert result.rooms == BENCHMARK_ROOMS
    assert result.warmup_runs == WARMUP_RUNS
    assert len(result.timings_ns) == MEASURED_RUNS
    assert all(value > 0 for value in result.timings_ns)
    assert result.min_ms <= result.median_ms <= result.p95_ms <= result.max_ms
    assert math.isfinite(result.median_ms)
    assert result.median_ms < SMOKE_SANITY_CEILING_MS
    env = result.environment
    assert env["providers"] == "CPUExecutionProvider"
    assert env["onnxruntime"]
    assert env["python"]
    assert env["cpu"]


def test_thread_configuration_is_applied_and_reported(model_bytes: bytes) -> None:
    result = run_benchmark(
        model_bytes,
        synthetic_features(8),
        warmup_runs=0,
        measured_runs=3,
        intra_op_threads=1,
        inter_op_threads=1,
    )
    assert (result.intra_op_threads, result.inter_op_threads) == (1, 1)
    assert result.as_dict()["intra_op_num_threads"] == 1
    assert len(result.timings_ns) == 3


@pytest.mark.parametrize(
    ("kwargs", "features"),
    [
        ({"measured_runs": 0}, synthetic_features(4)),
        ({"warmup_runs": -1}, synthetic_features(4)),
        ({"intra_op_threads": -1}, synthetic_features(4)),
        ({}, np.zeros((4, 9))),
        ({}, np.zeros((0, 10))),
    ],
)
def test_run_benchmark_rejects_invalid_arguments(
    model_bytes: bytes, kwargs: dict[str, int], features: np.ndarray
) -> None:
    with pytest.raises(ExportError):
        run_benchmark(model_bytes, features, **kwargs)


def test_cpu_name_is_not_empty() -> None:
    assert cpu_name().strip()


def test_main_reports_a_missing_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)
    assert main([], root=tmp_path) == 1
    assert "ejecute `export`" in capsys.readouterr().out


def test_main_reports_missing_test_pairs(
    tmp_path: Path, model_bytes: bytes, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PROFILE_ENV_VAR, "ci")
    (tmp_path / "dist" / "ci").mkdir(parents=True)
    (tmp_path / "dist" / "ci" / "model.onnx").write_bytes(model_bytes)
    assert main([], root=tmp_path) == 1
    assert "ejecute `prepare`" in capsys.readouterr().out


def test_main_measures_the_exported_model_on_the_test_pairs(
    tmp_path: Path, model_bytes: bytes, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import thehouse_ml.export.package as package

    monkeypatch.setenv(PROFILE_ENV_VAR, "ci")
    monkeypatch.setattr(package, "load_test_pair_features", lambda _paths: synthetic_features(300))
    (tmp_path / "dist" / "ci").mkdir(parents=True)
    (tmp_path / "dist" / "ci" / "model.onnx").write_bytes(model_bytes)
    assert main(["1"], root=tmp_path) == 0
    out = capsys.readouterr().out
    assert "dist/ci/model.onnx (perfil ci)" in out
    assert '"intra_op_num_threads": 1' in out
    assert '"measured_runs": 100' in out
