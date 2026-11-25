"""ML-06 T04 (ML-06-R04): rendimiento de `logreg-1.0.0` en onnxruntime CPU (spec ML-06 §6.5).

Medición de la spec: sesión creada una vez, lote `[200, 10]` con los primeros 200 pares de test de
ML-05, 10 corridas de calentamiento y 100 medidas con `time.perf_counter_ns`; criterio
**mediana < 20 ms** (constitución §13.5).

Necesita el modelo oficial (`models/`, fuera de git) y los pares de test (`data/processed/`, fuera
de git): en CI se omite con motivo explícito (ML-06-RK05) y corre solo la prueba de humo de
`tests/unit/test_onnx_benchmark.py`. Un control no ejecutado no se reporta como PASS.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from thehouse_ml.export.benchmark import (
    BENCHMARK_ROOMS,
    MEASURED_RUNS,
    MEDIAN_BUDGET_MS,
    WARMUP_RUNS,
    benchmark_batch,
    run_benchmark,
)
from thehouse_ml.export.onnx_graph import EXPECTED_LOGREG_SHA256
from thehouse_ml.export.package import export_model, load_test_pair_features
from thehouse_ml.models.train import sha256_file
from thehouse_ml.paths import ProjectPaths

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
OFFICIAL_MODELS = REPO / "models"


@pytest.fixture(scope="module")
def model_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    model = OFFICIAL_MODELS / "logreg.json"
    if not model.is_file() or not (OFFICIAL_MODELS / "train_manifest.json").is_file():
        pytest.skip("models/logreg.json oficial ausente (fuera de git; en CI corre solo el humo, ML-06-RK05)")
    if sha256_file(model) != EXPECTED_LOGREG_SHA256:
        pytest.skip("models/logreg.json local no es logreg-1.0.0 (SHA-256 distinto del de la spec ML-06 §1)")
    root = tmp_path_factory.mktemp("official_benchmark")
    (root / "models").mkdir()
    for name in ("logreg.json", "train_manifest.json"):
        shutil.copyfile(OFFICIAL_MODELS / name, root / "models" / name)
    result = export_model(ProjectPaths(root=root), profile="full", commit="test-commit")
    assert result is not None
    return result.model_path.read_bytes()


@pytest.fixture(scope="module")
def batch() -> np.ndarray:
    features = load_test_pair_features(ProjectPaths(root=REPO))
    if features is None:
        pytest.skip("faltan los pares de test de data/processed/ (fuera de git; ejecute `prepare`)")
    return benchmark_batch(features)


def test_200_rooms_score_with_a_median_below_20_ms(model_bytes: bytes, batch: np.ndarray) -> None:
    result = run_benchmark(model_bytes, batch)
    assert (result.rooms, result.warmup_runs, len(result.timings_ns)) == (BENCHMARK_ROOMS, WARMUP_RUNS, MEASURED_RUNS)
    assert result.environment["providers"] == "CPUExecutionProvider"
    assert result.median_ms < MEDIAN_BUDGET_MS, result.summary()
    assert result.passed
