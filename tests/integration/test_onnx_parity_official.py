"""ML-06 T02 (ML-06-R02): paridad Python ↔ onnxruntime con el modelo oficial `logreg-1.0.0`.

Conjuntos de la spec ML-06 §6.3:

1. los **pares de test** de ML-05 (30,755; solo se comparan puntajes, no se recalculan métricas);
2. la **rejilla** determinista fuera del `training_range` (`out_of_range_grid`);
3. los **casos dorados**: hasta que T03 entregue `tests/fixtures/logreg_golden_cases.json`, los
   vectores `expected.features` de los casos puntuados de `baseline_golden_cases.json`, calculados
   por el recuento independiente de ML-04 (`tests/golden_recount.py`, sin importar `thehouse_ml`).

Tolerancia: |dif| ≤ 1e-4 sin redondear y 0 diferencias en puntos redondeados y `matchScore`.
Sin el modelo oficial (fuera de git; p. ej. en CI, ML-06-RK05) o sin `data/processed/`, las pruebas
se omiten con motivo explícito: un control no ejecutado no se reporta como PASS.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from thehouse_ml.export.onnx_graph import EXPECTED_LOGREG_SHA256
from thehouse_ml.export.package import export_model, load_test_pair_features, run_export
from thehouse_ml.export.parity import check_parity, clip_matches_np_clip, out_of_range_grid
from thehouse_ml.features import default_schema
from thehouse_ml.models.train import LogRegModel, sha256_file
from thehouse_ml.models.trained_breakdown import BreakdownModel, build_breakdown_model
from thehouse_ml.paths import ProjectPaths

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
OFFICIAL_MODELS = REPO / "models"
BASELINE_GOLDEN = REPO / "tests" / "fixtures" / "baseline_golden_cases.json"
TEST_PAIRS = 30_755  # partición test de ML-05 (M-ML-043)


@pytest.fixture(scope="module")
def official_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    model = OFFICIAL_MODELS / "logreg.json"
    if not model.is_file() or not (OFFICIAL_MODELS / "train_manifest.json").is_file():
        pytest.skip("models/logreg.json oficial ausente (fuera de git; en CI se usa el perfil ci, ML-06-RK05)")
    if sha256_file(model) != EXPECTED_LOGREG_SHA256:
        pytest.skip("models/logreg.json local no es logreg-1.0.0 (SHA-256 distinto del de la spec ML-06 §1)")
    root = tmp_path_factory.mktemp("official_parity")
    (root / "models").mkdir()
    for name in ("logreg.json", "train_manifest.json"):
        shutil.copyfile(OFFICIAL_MODELS / name, root / "models" / name)
    return root


@pytest.fixture(scope="module")
def logreg_document(official_root: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((official_root / "models" / "logreg.json").read_text(encoding="utf-8"))
    return document


@pytest.fixture(scope="module")
def breakdown(logreg_document: dict[str, Any]) -> BreakdownModel:
    return build_breakdown_model(LogRegModel.from_dict(logreg_document))


@pytest.fixture(scope="module")
def model_bytes(official_root: Path) -> bytes:
    result = export_model(ProjectPaths(root=official_root), profile="full", commit="test-commit")
    assert result is not None
    return result.model_path.read_bytes()


def golden_vectors() -> tuple[list[str], np.ndarray]:
    """Vectores del recuento independiente de los 70 casos puntuados del baseline (sin excluidos ni incompletos)."""
    data = json.loads(BASELINE_GOLDEN.read_text(encoding="utf-8"))
    order = default_schema().feature_order
    assert data["metadata"]["feature_order"] == list(order)
    scored = [case for case in data["cases"] if case["expected"]["features"] is not None]
    scored = [case for case in scored if not case["expected"]["result"].get("excluded")]
    vectors = np.asarray([[case["expected"]["features"][name] for name in order] for case in scored], dtype=np.float64)
    return [case["id"] for case in scored], vectors


def test_all_inputs_constants_intermediates_and_outputs_are_double(model_bytes: bytes) -> None:
    model = onnx.load_from_string(model_bytes)
    graph = model.graph

    for value in (*graph.input, *graph.output):
        assert value.type.tensor_type.elem_type == TensorProto.DOUBLE, value.name
    for tensor in graph.initializer:
        # Solo los índices escalares de `Gather` son int64; ninguna constante es float32.
        expected = TensorProto.INT64 if tensor.name.startswith("factor_column_") else TensorProto.DOUBLE
        assert tensor.data_type == expected, tensor.name
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph.value_info
    assert len(inferred) == len(graph.node) - 2  # todos los tensores intermedios (sin las 2 salidas)
    assert all(value.type.tensor_type.elem_type == TensorProto.DOUBLE for value in inferred)
    assert TensorProto.FLOAT not in {tensor.data_type for tensor in graph.initializer}


def test_parity_on_the_test_pairs(model_bytes: bytes, breakdown: BreakdownModel) -> None:
    features = load_test_pair_features(ProjectPaths(root=REPO))
    if features is None:
        pytest.skip("data/processed/ ausente (fuera de git; ejecute `prepare`): paridad en pares de test no ejecutada")

    result = check_parity("pares de test", model_bytes, breakdown, features)

    assert result.rows == result.rows_score_features == TEST_PAIRS
    assert result.passed, result.as_dict()
    assert result.rounded_points_mismatches == 0 and result.match_score_mismatches == 0
    assert result.max_abs_diff_factor_points < 1e-9 and result.max_abs_diff_raw_total < 1e-9


def test_parity_on_the_out_of_range_grid(
    model_bytes: bytes, breakdown: BreakdownModel, logreg_document: dict[str, Any]
) -> None:
    grid = out_of_range_grid(logreg_document["clip"]["lower"], logreg_document["clip"]["upper"])

    result = check_parity("rejilla fuera del training_range", model_bytes, breakdown, grid)

    assert result.rows == len(grid) == 380
    assert result.rows_score_matrix > 0  # vectores fuera del `range` del esquema
    assert result.passed, result.as_dict()


def test_parity_on_the_golden_case_vectors(model_bytes: bytes, breakdown: BreakdownModel) -> None:
    ids, vectors = golden_vectors()

    result = check_parity("casos dorados (vectores del baseline)", model_bytes, breakdown, vectors)

    # 81 casos = 70 puntuados + 6 excluidos + 5 incompletos; el ONNX solo puntúa los 70 (spec §6.1).
    assert len(ids) == result.rows == result.rows_score_features == 70
    assert result.passed, [(ids[item.row], item.as_dict()) for item in result.mismatches]


def test_onnx_clip_is_np_clip_with_the_limits_of_logreg_json(
    model_bytes: bytes, logreg_document: dict[str, Any]
) -> None:
    lower, upper = logreg_document["clip"]["lower"], logreg_document["clip"]["upper"]
    grid = out_of_range_grid(lower, upper)
    _, golden = golden_vectors()

    assert clip_matches_np_clip(model_bytes, np.vstack([grid, golden]), lower, upper)


def test_export_stage_reports_formal_parity(official_root: Path) -> None:
    message = run_export(ProjectPaths(root=official_root), profile="full", commit="test-commit")

    assert "paridad Python <-> onnxruntime CPU" in message
    assert "pares de test: omitido; faltan los datos de data/processed/" in message  # raíz temporal sin datos
    assert "rejilla fuera del training_range: 380 vectores" in message
    assert message.rstrip().endswith("-> PASS")
    message.encode("cp1252")  # consola de Windows (M-ML-045)
