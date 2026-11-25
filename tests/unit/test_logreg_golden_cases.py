"""ML-06 T03 (ML-06-R03): casos dorados de `logreg-1.0.0`.

`tests/fixtures/logreg_golden_cases.json` se genera con `tests/build_logreg_golden_cases.py`,
un recuento independiente (fracciones exactas, sin importar `thehouse_ml`). Aquí:

1. formato, encabezado, cobertura de la spec ML-06 §6.4 y coherencia interna del archivo
   (sin el modelo: corren siempre);
2. con el modelo oficial (`models/logreg.json`, SHA-256 de la spec §1; fuera de git): el archivo
   versionado es idéntico a la salida del generador, el paquete de Python (`score_pair_logreg`,
   `score_features_logreg`) y onnxruntime coinciden con todos los casos (|dif| ≤ 1e-4 sin redondear;
   0 diferencias en puntos redondeados y `matchScore`). Sin el modelo (p. ej. en CI, ML-06-RK05)
   se omiten con motivo explícito: un control no ejecutado no se reporta como PASS.
"""

from __future__ import annotations

import ast
import json
import math
import shutil
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import onnxruntime as ort
import pytest

import golden_recount
from thehouse_ml.export.onnx_graph import EXPECTED_LOGREG_SHA256, FACTOR_OUTPUT, INPUT_NAME, TOTAL_OUTPUT
from thehouse_ml.export.package import export_model
from thehouse_ml.features import (
    Campus,
    IncompleteProfileError,
    compute_pair_features,
    default_schema,
    load_campuses,
    room_from_record,
    student_from_record,
)
from thehouse_ml.models import ExcludedPair, ScoreInputError, factor_points, match_score
from thehouse_ml.models.factor_groups import FACTOR_GROUPS
from thehouse_ml.models.train import LogRegModel, sha256_file
from thehouse_ml.models.trained_breakdown import (
    LOGREG_MODEL_VERSION,
    BreakdownModel,
    LogRegScoreResult,
    build_breakdown_model,
    score_features_logreg,
    score_matrix,
    score_pair_logreg,
)
from thehouse_ml.paths import ProjectPaths

REPO = Path(__file__).resolve().parents[2]
CASES_PATH = REPO / "tests" / "fixtures" / "logreg_golden_cases.json"
BASELINE_PATH = REPO / "tests" / "fixtures" / "baseline_golden_cases.json"
UNIVERSITIES_CSV = REPO / "data" / "universities.csv"
OFFICIAL_MODELS = REPO / "models"
DATA: dict[str, Any] = json.loads(CASES_PATH.read_text(encoding="utf-8"))
META: dict[str, Any] = DATA["metadata"]
CASES: list[dict[str, Any]] = DATA["cases"]
CASE_IDS = [case["id"] for case in CASES]
SCORED = [case for case in CASES if "matchScore" in case["expected"]["result"]]
VECTOR_CASES = [case for case in CASES if "features" in case["input"]]
FEATURE_TOLERANCE = float(META["tolerance"]["features"])
ONNX_TOLERANCE = float(META["tolerance"]["onnx"])
# Vectores fuera del `range` del esquema: el backend los rechaza antes del ONNX (spec §6.1 paso 4) y
# `score_features_logreg` también; su referencia de Python es `score_matrix` (M-ML-046, T02-H4).
OUT_OF_SCHEMA_RANGE = {"clip_vector_below_zero", "clip_vector_bounded_features_out", "clip_vector_epsilon_outside"}
MARGIN_REPLACEMENTS = {"null_all_optional_partial_room": "null_all_optional_partial_room_margin"}
HALF = Fraction(1, 2)


def _case(case_id: str) -> dict[str, Any]:
    return next(case for case in CASES if case["id"] == case_id)


def _round_points(value: Fraction) -> Fraction:
    """round6 y 1 decimal mitad hacia arriba (valor exacto ≥ 0), escrito con golden_recount."""
    return golden_recount.round_half_up(golden_recount.round_half_up(value, 6), 1)


def _backend_round(value: float, decimals: int) -> float:
    """.NET `Math.Round(Math.Round(x, 6, AwayFromZero), d, AwayFromZero)` con `-0` → `0` (M-ML-046)."""
    exact = Fraction(value)
    magnitude = golden_recount.round_half_up(golden_recount.round_half_up(abs(exact), 6), decimals)
    return float(-magnitude if exact < 0 else magnitude) + 0.0


# --- Encabezado, formato y cobertura (sin el modelo) -----------------------------------------------------


def test_header_identifies_logreg_1_0_0_and_its_sha256() -> None:
    schema = default_schema()
    assert META["modelVersion"] == LOGREG_MODEL_VERSION == "logreg-1.0.0"
    assert META["derived_from"] == {
        "model_id": "logreg-1.0.0",
        "file": "models/logreg.json",
        "format": "thehouse-ml/logreg-json/1",
        "sha256": EXPECTED_LOGREG_SHA256,
    }
    assert META["featureSchemaVersion"] == schema.version == "2.0.0"
    assert META["featureOrder"] == list(schema.feature_order)
    assert META["factorOrder"] == [group.id for group in FACTOR_GROUPS]
    assert [item["features"] for item in META["factors"]] == [list(group.features) for group in FACTOR_GROUPS]
    assert META["fixedTermFeatures"] == ["pets_compatible"]
    assert [item["maxPoints"] for item in META["factors"]] == [9.6, 72.6, 6.7, 4.9, 1.7, 4.5]
    assert math.isclose(sum(META["maxPointsRaw"]), 100.0, rel_tol=0.0, abs_tol=1e-12)
    assert META["synthetic"] is True
    assert "SINTÉTICOS" in META["description"]
    assert "golden_cases.json" in META["description"]
    assert META["spec"] == "ML-06 1.0.0 (§6.4)"
    assert META["generated_by"] == "tests/build_logreg_golden_cases.py"
    assert "no importa thehouse_ml" in META["how_computed"] and "Fraction" in META["how_computed"]
    assert META["tolerance"] == {
        "features": 1e-6,
        "value": 1e-6,
        "onnx": 1e-4,
        "points": 0,
        "matchScore": 0,
        "note": META["tolerance"]["note"],
    }
    for text in ("round6", "MidpointRounding.AwayFromZero", "ToEven", "-0"):
        assert text in META["dotnet"], text
    assert META["case_count"] == len(CASES) == len(set(CASE_IDS))
    assert META["category_counts"] == dict(Counter(case["category"] for case in CASES))


def test_header_campuses_match_universities_csv() -> None:
    campuses = load_campuses(UNIVERSITIES_CSV)
    assert set(META["campuses"]) == set(campuses)
    for code, campus in campuses.items():
        assert (META["campuses"][code]["lat"], META["campuses"][code]["lon"]) == (campus.lat, campus.lon)


def test_coverage_required_by_spec_6_4() -> None:
    counts = Counter(case["category"] for case in CASES)
    minimum = {
        "baseline_inputs": 81,
        "clipping": 4,
        "extremes": 2,
        "t05_h1_zone": 3,
        "fixed_term": 1,
        "factor_minimum": 1,
    }
    for category, at_least in minimum.items():
        assert counts[category] >= at_least, (category, counts[category])
    assert counts["factor_rounding"] + counts["total_rounding"] >= 6
    assert counts["factor_rounding"] >= 1 and counts["total_rounding"] >= 3


def test_same_inputs_as_the_81_baseline_cases() -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["cases"]
    by_id = {case["id"]: case for case in CASES}
    assert len(baseline) == 81
    for case in baseline:
        if case["id"] in MARGIN_REPLACEMENTS:
            replacement = by_id[MARGIN_REPLACEMENTS[case["id"]]]
            assert case["id"] not in by_id
            assert "regla de margen" in replacement["description"]
            assert replacement["input"]["student"] == case["input"]["student"]
            room = {key: value for key, value in replacement["input"]["room"].items() if key != "lat"}
            assert room == {key: value for key, value in case["input"]["room"].items() if key != "lat"}
            continue
        assert by_id[case["id"]]["input"] == case["input"], case["id"]
        assert by_id[case["id"]]["category"] == "baseline_inputs"
    results = [by_id[MARGIN_REPLACEMENTS.get(case["id"], case["id"])]["expected"]["result"] for case in baseline]
    assert sum(1 for item in results if item.get("excluded")) == 6
    assert sum(1 for item in results if item.get("incomplete")) == 5
    assert sum(1 for item in results if "matchScore" in item) == 70


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_every_case_has_the_format_of_spec_6_4(case: dict[str, Any]) -> None:
    assert set(case) == {"id", "category", "description", "input", "tolerance", "expected"}
    assert case["description"]
    assert case["tolerance"] == {"features": 1e-6, "value": 1e-6, "onnx": 1e-4, "points": 0}
    assert set(case["input"]) in ({"student", "room"}, {"features"})
    expected = case["expected"]
    result = expected["result"]
    assert result["modelVersion"] == "logreg-1.0.0"
    assert result["featureSchemaVersion"] == "2.0.0"
    if result.get("incomplete"):
        assert set(expected) == {"features", "result"} and expected["features"] is None
        assert set(result) == {"modelVersion", "featureSchemaVersion", "incomplete", "missingFields"}
        return
    assert list(expected["features"]) == META["featureOrder"]
    if result.get("excluded"):
        assert set(expected) == {"features", "result"}
        assert set(result) == {"modelVersion", "featureSchemaVersion", "excluded", "violatedConstraints"}
        return
    assert set(expected) == {"features", "clippedFeatures", "onnx", "result"}
    assert list(expected["clippedFeatures"]) == META["featureOrder"]
    assert set(expected["onnx"]) == {"factor_points", "raw_total"} and len(expected["onnx"]["factor_points"]) == 6
    assert set(result) == {"modelVersion", "featureSchemaVersion", "matchScore", "breakdown"}
    assert [item["factor"] for item in result["breakdown"]] == META["factorOrder"]
    for item, factor in zip(result["breakdown"], META["factors"], strict=True):
        assert item["maxPoints"] == factor["maxPoints"]
        if len(factor["features"]) == 1:
            assert set(item) == {"factor", "points", "maxPoints", "feature", "value"}
            assert item["feature"] == factor["features"][0]
            assert item["value"] == expected["features"][item["feature"]]
        else:
            assert set(item) == {"factor", "points", "maxPoints", "features"}
            assert [entry["feature"] for entry in item["features"]] == factor["features"]
            for entry in item["features"]:
                assert entry["value"] == expected["features"][entry["feature"]]  # sin recortar


@pytest.mark.parametrize("case", SCORED, ids=[case["id"] for case in SCORED])
def test_rounded_breakdown_is_consistent_with_the_raw_outputs(case: dict[str, Any]) -> None:
    """Puntos y `matchScore` = regla de redondeo aplicada a `expected.onnx`; el desglose suma el total."""
    onnx = case["expected"]["onnx"]
    result = case["expected"]["result"]
    raw = [Fraction(value) for value in onnx["factor_points"]]
    points = [Fraction(item["points"]) for item in result["breakdown"]]
    assert [float(_round_points(value)) for value in raw] == [item["points"] for item in result["breakdown"]]
    total = Fraction(onnx["raw_total"])
    assert math.floor(golden_recount.round_half_up(total, 6) + HALF) == result["matchScore"]
    assert abs(sum(raw) - total) <= Fraction(1, 10**9)
    assert 0 <= result["matchScore"] <= 100
    assert abs(total - result["matchScore"]) <= HALF + Fraction(1, 2 * 10**6)
    assert abs(sum(points) - total) <= Fraction(3, 10) + Fraction(6, 2 * 10**6)  # constitución §12, M-ML-041
    for value, item in zip(raw, result["breakdown"], strict=True):
        assert (
            0
            <= value
            <= Fraction(META["factors"][META["factorOrder"].index(item["factor"])]["maxPointsRaw"])
            + (Fraction(1, 10**12))
        )
        assert item["points"] <= item["maxPoints"]


def test_special_cases_of_the_spec() -> None:
    def points(case_id: str, factor: str) -> float:
        breakdown = _case(case_id)["expected"]["result"]["breakdown"]
        return float(next(item["points"] for item in breakdown if item["factor"] == factor))

    def score(case_id: str) -> int:
        return int(_case(case_id)["expected"]["result"]["matchScore"])

    # Extremos.
    assert score("extreme_100_raw") == score("extreme_vector_best_corner") == 100
    assert score("extreme_vector_worst_corner") == 0
    # Recorte: por encima del training_range = en el límite.
    assert (
        _case("clip_vector_above_upper_bounds")["expected"]["onnx"]
        == (_case("clip_vector_at_upper_bounds")["expected"]["onnx"])
    )
    raw_35 = _case("clip_raw_35km_walk")["expected"]
    assert raw_35["features"]["distance_km"] > 30 and raw_35["features"]["distance_ratio"] > 60
    assert raw_35["clippedFeatures"]["distance_km"] == 30 and raw_35["clippedFeatures"]["distance_ratio"] == 60
    assert points("clip_raw_35km_walk", "distance") == 0.0
    # Factores en su mínimo (-0 de T02-H1).
    assert points("factor_minimum_place_type", "place_type") == 0.0
    assert _case("factor_minimum_place_type")["expected"]["onnx"]["factor_points"][3] == 0.0
    # Bordes de redondeo (AwayFromZero sobre round6).
    assert points("factor_round_budget_4_65", "budget") == 4.7
    assert points("factor_round_budget_4_25", "budget") == 4.3
    assert points("factor_round_distance_40_25_below", "distance") == 40.3
    assert points("factor_round_distance_40_25_above", "distance") == 40.3
    assert points("factor_round_distance_40_25_down", "distance") == 40.2
    assert (score("logreg_total_x49999955_up"), score("logreg_total_x4999994_down")) == (81, 80)
    assert score("logreg_total_x5000004_up") == 81
    # Zona T05-H1: el factor de presupuesto sube con la renta mientras renta < budget.min.
    budget = [_case(f"t05_h1_rent_{rent}")["expected"]["onnx"]["factor_points"][0] for rent in (0, 100, 199, 200)]
    assert budget == sorted(budget) and budget[0] < budget[-1]
    for rent in (0, 100, 199, 200):
        student = _case(f"t05_h1_rent_{rent}")["input"]["student"]
        assert student["budget.max"] / student["budget.min"] > 76.2
    # pets_compatible va al término fijo: no cambia los puntos.
    assert (
        _case("fixed_term_pets_vector_0")["expected"]["onnx"] == _case("fixed_term_pets_vector_1")["expected"]["onnx"]
    )
    assert _case("fixed_term_pets_allowed_raw")["expected"]["features"]["pets_compatible"] == 1


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_generator_does_not_import_the_code_under_test() -> None:
    imported = _imported_modules(REPO / "tests" / "build_logreg_golden_cases.py")
    assert not any(name == "thehouse_ml" or name.startswith("thehouse_ml.") for name in imported), imported
    assert not {"onnx", "onnxruntime", "numpy"} & imported, imported
    source = (REPO / "tests" / "build_logreg_golden_cases.py").read_text(encoding="utf-8")
    for name in ("score_features_logreg(", "score_matrix(", "build_breakdown_model(", "check_parity("):
        assert name not in source, name


# --- Con el modelo oficial (se omiten sin él) -----------------------------------------------------------


@pytest.fixture(scope="module")
def official_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    model = OFFICIAL_MODELS / "logreg.json"
    if not model.is_file() or not (OFFICIAL_MODELS / "train_manifest.json").is_file():
        pytest.skip("models/logreg.json oficial ausente (fuera de git; en CI se usa el perfil ci, ML-06-RK05)")
    if sha256_file(model) != EXPECTED_LOGREG_SHA256:
        pytest.skip("models/logreg.json local no es logreg-1.0.0 (SHA-256 distinto del de la spec ML-06 §1)")
    root = tmp_path_factory.mktemp("logreg_golden")
    (root / "models").mkdir()
    for name in ("logreg.json", "train_manifest.json"):
        shutil.copyfile(OFFICIAL_MODELS / name, root / "models" / name)
    return root


@pytest.fixture(scope="module")
def builder(official_root: Path) -> ModuleType:
    import build_logreg_golden_cases

    return build_logreg_golden_cases


@pytest.fixture(scope="module")
def breakdown(official_root: Path) -> BreakdownModel:
    document = json.loads((official_root / "models" / "logreg.json").read_text(encoding="utf-8"))
    return build_breakdown_model(LogRegModel.from_dict(document))


@pytest.fixture(scope="module")
def onnx_outputs(official_root: Path) -> dict[str, tuple[np.ndarray, float]]:
    """Salidas de onnxruntime CPU para los casos puntuados, por id."""
    result = export_model(ProjectPaths(root=official_root), profile="full", commit="test-commit")
    assert result is not None
    session = ort.InferenceSession(result.model_path.read_bytes(), providers=["CPUExecutionProvider"])
    order = META["featureOrder"]
    batch = np.asarray([[case["expected"]["features"][name] for name in order] for case in SCORED], dtype=np.float64)
    factor, total = session.run([FACTOR_OUTPUT, TOTAL_OUTPUT], {INPUT_NAME: batch})
    return {case["id"]: (factor[row], float(total[row])) for row, case in enumerate(SCORED)}


@pytest.fixture(scope="module")
def campuses() -> dict[str, Campus]:
    return load_campuses(UNIVERSITIES_CSV)


def test_fixture_is_exactly_what_the_independent_recount_produces(builder: ModuleType) -> None:
    assert CASES_PATH.read_text(encoding="utf-8") == builder.render()


def _python_result(case: dict[str, Any], breakdown: BreakdownModel, campuses: dict[str, Campus]) -> Any:
    if "features" in case["input"]:
        vector = [case["input"]["features"][name] for name in META["featureOrder"]]
        return score_features_logreg(vector, breakdown)
    student = student_from_record(case["input"]["student"])
    room = room_from_record(case["input"]["room"])
    return score_pair_logreg(student, room, campuses, breakdown)


IN_SCHEMA_RANGE = [case for case in CASES if case["id"] not in OUT_OF_SCHEMA_RANGE]


@pytest.mark.parametrize("case", IN_SCHEMA_RANGE, ids=[case["id"] for case in IN_SCHEMA_RANGE])
def test_python_package_matches(case: dict[str, Any], breakdown: BreakdownModel, campuses: dict[str, Campus]) -> None:
    expected = case["expected"]
    if expected["result"].get("incomplete"):
        with pytest.raises(IncompleteProfileError) as error:
            _python_result(case, breakdown, campuses)
        assert list(error.value.missing) == expected["result"]["missingFields"]
        return
    if "student" in case["input"]:
        features = compute_pair_features(
            student_from_record(case["input"]["student"]), room_from_record(case["input"]["room"]), campuses
        )
        for name, value in expected["features"].items():
            assert math.isclose(features[name], value, rel_tol=0.0, abs_tol=FEATURE_TOLERANCE), name
    result = _python_result(case, breakdown, campuses)
    if expected["result"].get("excluded"):
        assert isinstance(result, ExcludedPair)
        assert result.to_dict() == expected["result"]
        return
    assert isinstance(result, LogRegScoreResult)
    produced = result.to_dict(include_raw=True)
    assert produced["matchScore"] == expected["result"]["matchScore"]
    assert produced["modelVersion"] == expected["result"]["modelVersion"]
    assert produced["featureSchemaVersion"] == expected["result"]["featureSchemaVersion"]
    assert abs(produced["rawTotal"] - expected["onnx"]["raw_total"]) <= ONNX_TOLERANCE
    for got, want, raw in zip(
        produced["breakdown"], expected["result"]["breakdown"], expected["onnx"]["factor_points"], strict=True
    ):
        assert got["factor"] == want["factor"]
        assert got["points"] == want["points"], (got["factor"], got["points"], want["points"])
        assert got["maxPoints"] == want["maxPoints"]
        assert abs(got["rawPoints"] - raw) <= ONNX_TOLERANCE
        if "feature" in want:
            assert got["feature"] == want["feature"]
            assert math.isclose(got["value"], want["value"], rel_tol=0.0, abs_tol=FEATURE_TOLERANCE)
        else:
            assert [item["feature"] for item in got["features"]] == [item["feature"] for item in want["features"]]
            for item_got, item_want in zip(got["features"], want["features"], strict=True):
                assert math.isclose(item_got["value"], item_want["value"], rel_tol=0.0, abs_tol=FEATURE_TOLERANCE)


@pytest.mark.parametrize("case_id", sorted(OUT_OF_SCHEMA_RANGE))
def test_python_out_of_schema_vectors_match_through_score_matrix(case_id: str, breakdown: BreakdownModel) -> None:
    """Vectores que el backend rechaza antes del ONNX: `score_features_logreg` los rechaza y `score_matrix` puntúa."""
    case = _case(case_id)
    vector = [case["input"]["features"][name] for name in META["featureOrder"]]
    with pytest.raises(ScoreInputError):
        score_features_logreg(vector, breakdown)
    points, total, _ = score_matrix(breakdown, np.asarray([vector], dtype=np.float64))
    expected = case["expected"]
    assert np.max(np.abs(points[0] - np.asarray(expected["onnx"]["factor_points"]))) <= ONNX_TOLERANCE
    assert abs(float(total[0]) - expected["onnx"]["raw_total"]) <= ONNX_TOLERANCE
    rounded = [factor_points(float(value), breakdown.rounding) for value in points[0]]
    assert rounded == [item["points"] for item in expected["result"]["breakdown"]]
    assert match_score(float(total[0]), breakdown.rounding) == expected["result"]["matchScore"]


@pytest.mark.parametrize("case", SCORED, ids=[case["id"] for case in SCORED])
def test_onnxruntime_matches(case: dict[str, Any], onnx_outputs: dict[str, tuple[np.ndarray, float]]) -> None:
    factor, total = onnx_outputs[case["id"]]
    expected = case["expected"]
    assert np.max(np.abs(factor - np.asarray(expected["onnx"]["factor_points"]))) <= ONNX_TOLERANCE
    assert abs(total - expected["onnx"]["raw_total"]) <= ONNX_TOLERANCE
    points = [_backend_round(float(value), 1) for value in factor]
    assert points == [item["points"] for item in expected["result"]["breakdown"]]
    assert all(math.copysign(1.0, value) == 1.0 for value in points)  # sin -0 tras la normalización
    assert int(_backend_round(total, 0)) == expected["result"]["matchScore"]


def test_onnxruntime_max_abs_diff_is_double_precision(onnx_outputs: dict[str, tuple[np.ndarray, float]]) -> None:
    diffs = [
        max(
            float(np.max(np.abs(onnx_outputs[case["id"]][0] - np.asarray(case["expected"]["onnx"]["factor_points"])))),
            abs(onnx_outputs[case["id"]][1] - case["expected"]["onnx"]["raw_total"]),
        )
        for case in SCORED
    ]
    assert len(diffs) == len(SCORED) and max(diffs) < 1e-9
