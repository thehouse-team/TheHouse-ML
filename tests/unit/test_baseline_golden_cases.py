"""ML-04 T05 (R05): casos dorados del modelo base y prueba de paridad.

`tests/fixtures/baseline_golden_cases.json` se genera con
`tests/build_baseline_golden_cases.py` a partir del recuento independiente de
`tests/golden_recount.py` (fracciones exactas, sin importar `thehouse_ml`). Aquí:

1. el archivo versionado es idéntico a lo que produce el recuento (si el
   recuento cambia o alguien edita un esperado a mano, falla);
2. el paquete (`score_pair`, `compute_pair_features`) devuelve lo esperado:
   features y `value` con 1e-6, `points` y `matchScore` exactos;
3. el encabezado coincide con `baseline_params.json` y `feature_schema.json`
   (si cambia una versión, falla) y el archivo cubre los casos límite pedidos.
"""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

import build_baseline_golden_cases as builder
import golden_recount
from thehouse_ml.features import (
    Campus,
    IncompleteProfileError,
    compute_pair_features,
    default_schema,
    load_campuses,
    room_from_record,
    student_from_record,
)
from thehouse_ml.models import ExcludedPair, ScoreResult, default_config, score_pair

REPO = Path(__file__).resolve().parents[2]
CASES_PATH = REPO / "tests" / "fixtures" / "baseline_golden_cases.json"
UNIVERSITIES_CSV = REPO / "data" / "universities.csv"
DATA: dict[str, Any] = json.loads(CASES_PATH.read_text(encoding="utf-8"))
META: dict[str, Any] = DATA["metadata"]
CASES: list[dict[str, Any]] = DATA["cases"]
CASE_IDS = [case["id"] for case in CASES]
INCOMPLETE = [case for case in CASES if case["expected"]["result"].get("incomplete")]
EXCLUDED = [case for case in CASES if case["expected"]["result"].get("excluded")]
SCORED = [case for case in CASES if case not in INCOMPLETE and case not in EXCLUDED]
TOLERANCE = float(META["tolerance"]["features"])

STUDENT_KEYS = {
    "university_code",
    "budget.min",
    "budget.max",
    "maxDistanceKm",
    "essentialServices",
    "livingWith",
    "placeType",
    "pets",
    "transport",
    "propertyTypes",
}
ROOM_KEYS = {
    "price_mxn",
    "lat",
    "lon",
    "services",
    "placeType",
    "propertyType",
    "pets_allowed",
    "livingWith",
    "bedrooms",
    "bathrooms",
    "area_m2",
    "university_code",
    "synthetic",
}
BREAKDOWN_KEYS = {"factor", "points", "maxPoints", "feature", "value"}


@pytest.fixture(scope="module")
def campuses() -> dict[str, Campus]:
    return load_campuses(UNIVERSITIES_CSV)


def _case(case_id: str) -> dict[str, Any]:
    return next(case for case in CASES if case["id"] == case_id)


def _feature(case: dict[str, Any], name: str) -> float:
    return float(case["expected"]["features"][name])


# --- Encabezado y formato ----------------------------------------------------------------------------


def test_file_has_at_least_50_cases_with_unique_ids() -> None:
    assert len(CASES) >= 50
    assert META["case_count"] == len(CASES)
    assert len(set(CASE_IDS)) == len(CASE_IDS)


def test_header_versions_match_the_config_and_the_schema() -> None:
    config = default_config()
    schema = default_schema()
    assert META["model_version"] == config.model_version == "baseline-1.0.0"
    assert META["feature_schema_version"] == schema.version == "2.0.0"
    assert META["feature_order"] == list(schema.feature_order)
    assert META["factors"] == [
        {"factor": factor.id, "feature": factor.feature, "maxPoints": factor.weight} for factor in config.factors
    ]
    assert META["hard_constraints"] == [item.id for item in schema.hard_constraints]
    rejected = [field.name for field in schema.student if field.missing and field.missing.action == "reject"]
    assert META["required_fields"] == rejected == list(golden_recount.REQUIRED_FIELDS)
    for case in CASES:
        result = case["expected"]["result"]
        assert result["modelVersion"] == config.model_version, case["id"]
        assert result["featureSchemaVersion"] == schema.version, case["id"]


def test_header_documents_generation_decision_and_dotnet_rounding() -> None:
    assert META["generated_at"] == "2026-11-23"
    assert META["spec"] == "ML-04 1.0.0"
    assert META["decision"].startswith("M-ML-033 (APROBADO)")
    assert "PROPUESTO" not in json.dumps(META, ensure_ascii=False)
    assert "incomplete = true" in META["format"]["expected.result"]
    assert META["synthetic"] is True
    assert "Fraction" in META["how_computed"]
    assert "no importa thehouse_ml" in META["how_computed"]
    dotnet = META["dotnet"]
    assert "round6" in dotnet
    assert "MidpointRounding.AwayFromZero" in dotnet
    assert "double o con decimal" in dotnet
    assert "ToEven" in dotnet
    assert META["tolerance"]["features"] == 1e-6
    assert META["tolerance"]["points"] == 0
    assert META["tolerance"]["matchScore"] == 0
    assert "golden_cases.json" in META["description"]


def test_header_campuses_match_universities_csv(campuses: dict[str, Campus]) -> None:
    assert set(META["campuses"]) == set(campuses)
    for code, campus in campuses.items():
        assert META["campuses"][code]["lat"] == campus.lat, code
        assert META["campuses"][code]["lon"] == campus.lon, code


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_every_case_has_all_fields(case: dict[str, Any]) -> None:
    assert set(case) == {"id", "category", "description", "input", "tolerance", "expected"}
    assert case["description"]
    assert set(case["input"]) == {"student", "room"}
    assert set(case["input"]["student"]) == STUDENT_KEYS
    assert set(case["input"]["room"]) == ROOM_KEYS
    assert case["tolerance"] == {"features": 1e-6, "value": 1e-6, "points": 0, "matchScore": 0}
    expected = case["expected"]
    result = expected["result"]
    if result.get("incomplete"):
        assert set(expected) == {"features", "result"}
        assert expected["features"] is None
        assert set(result) == {"modelVersion", "featureSchemaVersion", "incomplete", "missingFields"}
        assert result["incomplete"] is True
        assert result["missingFields"]
        assert set(result["missingFields"]) <= set(META["required_fields"])
        for name in result["missingFields"]:
            assert case["input"]["student"][name] is None
        return
    assert list(expected["features"]) == META["feature_order"]
    if result.get("excluded"):
        assert set(expected) == {"features", "result"}
        assert set(result) == {"modelVersion", "featureSchemaVersion", "excluded", "violatedConstraints"}
        assert result["excluded"] is True
        assert result["violatedConstraints"]
    else:
        assert set(expected) == {"features", "result", "rawTotal"}
        assert set(result) == {"modelVersion", "featureSchemaVersion", "matchScore", "breakdown"}
        assert isinstance(result["matchScore"], int)
        assert 0 <= result["matchScore"] <= 100
        assert [item["factor"] for item in result["breakdown"]] == [f["factor"] for f in META["factors"]]
        for item in result["breakdown"]:
            assert set(item) == BREAKDOWN_KEYS
            assert item["points"] == round(item["points"], 1)


# --- El esperado sale del recuento independiente -----------------------------------------------------


def test_fixture_is_exactly_what_the_independent_recount_produces() -> None:
    assert CASES_PATH.read_text(encoding="utf-8") == builder.render()


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("module", ["golden_recount.py", "build_baseline_golden_cases.py"])
def test_recount_does_not_import_the_package(module: str) -> None:
    imported = _imported_modules(REPO / "tests" / module)
    assert not any(name == "thehouse_ml" or name.startswith("thehouse_ml.") for name in imported), imported


def test_recount_rules_written_apart_agree_with_the_schema() -> None:
    schema = json.loads(golden_recount.FEATURE_SCHEMA.read_text(encoding="utf-8"))
    fields = {field["name"]: field for field in schema["student"]["fields"]}
    assert fields["budget.min"]["missing"]["value"] == golden_recount.DEFAULT_BUDGET_MIN
    assert fields["maxDistanceKm"]["missing"]["value"] == golden_recount.DEFAULT_MAX_DISTANCE_KM
    assert fields["transport"]["missing"]["value"] == golden_recount.DEFAULT_TRANSPORT
    assert fields["essentialServices"]["missing"]["value"] == []
    assert fields["pets"]["missing"]["value"] is False
    assert fields["propertyTypes"]["missing"]["action"] == "accept_all"
    factors = next(
        feature["parameters"]["transport_factors"]
        for feature in schema["pair_features"]
        if "transport_factors" in feature.get("parameters", {})
    )
    assert {key: Fraction(str(value)) for key, value in factors.items()} == golden_recount.TRANSPORT_FACTORS


def test_recount_rounding_matches_hand_values() -> None:
    rh = golden_recount.round_half_up
    assert rh(rh(Fraction(1335, 100), 6), 1) == Fraction(134, 10)
    assert rh(rh(Fraction(125, 100), 6), 1) == Fraction(13, 10)
    assert math.floor(rh(Fraction(8249999955, 10**8), 6) + Fraction(1, 2)) == 83
    assert math.floor(rh(Fraction(824999994, 10**7), 6) + Fraction(1, 2)) == 82
    assert math.floor(rh(Fraction(173, 2), 6) + Fraction(1, 2)) == 87


# --- Paridad con el paquete --------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_package_features_match(case: dict[str, Any], campuses: dict[str, Campus]) -> None:
    student = student_from_record(case["input"]["student"])
    room = room_from_record(case["input"]["room"])
    expected = case["expected"]["features"]
    if expected is None:
        with pytest.raises(IncompleteProfileError) as error:
            compute_pair_features(student, room, campuses)
        assert list(error.value.missing) == case["expected"]["result"]["missingFields"]
        return
    features = compute_pair_features(student, room, campuses)
    assert list(features) == list(expected)
    for name, value in expected.items():
        assert math.isclose(features[name], value, rel_tol=0.0, abs_tol=TOLERANCE), (name, features[name], value)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_package_score_matches(case: dict[str, Any], campuses: dict[str, Campus]) -> None:
    student = student_from_record(case["input"]["student"])
    room = room_from_record(case["input"]["room"])
    expected = case["expected"]["result"]
    if expected.get("incomplete"):
        with pytest.raises(IncompleteProfileError) as error:
            score_pair(student, room, campuses)
        assert list(error.value.missing) == expected["missingFields"]
        return
    result = score_pair(student, room, campuses)
    if expected.get("excluded"):
        assert isinstance(result, ExcludedPair)
        assert result.to_dict() == expected
        return
    assert isinstance(result, ScoreResult)
    produced = result.to_dict()
    assert set(produced) == set(expected)
    assert produced["modelVersion"] == expected["modelVersion"]
    assert produced["featureSchemaVersion"] == expected["featureSchemaVersion"]
    assert produced["matchScore"] == expected["matchScore"]
    assert len(produced["breakdown"]) == len(expected["breakdown"])
    for got, want in zip(produced["breakdown"], expected["breakdown"], strict=True):
        assert set(got) == set(want)
        assert got["factor"] == want["factor"]
        assert got["feature"] == want["feature"]
        assert got["maxPoints"] == want["maxPoints"]
        assert got["points"] == want["points"], (got["factor"], got["points"], want["points"])
        assert math.isclose(got["value"], want["value"], rel_tol=0.0, abs_tol=TOLERANCE), got["factor"]
    assert math.isclose(result.raw_total, case["expected"]["rawTotal"], rel_tol=0.0, abs_tol=TOLERANCE)


# --- Cobertura de casos límite -----------------------------------------------------------------------


def test_coverage_by_category() -> None:
    counts = Counter(case["category"] for case in CASES)
    minimum = {
        "perfect": 1,
        "worst": 1,
        "budget": 3,
        "distance": 4,
        "services": 3,
        "place_property": 2,
        "extras": 3,
        "factor_rounding": 3,
        "total_rounding": 3,
        "spec_example": 1,
        "transport": 4,
        "nulls": 1,
        "exclusion": 4,
        "incomplete": 4,
        "campus": 10,
    }
    for category, at_least in minimum.items():
        assert counts[category] >= at_least, (category, counts[category])
    assert set(counts) == set(minimum)


def test_perfect_and_worst_scores() -> None:
    scores = {case["expected"]["result"]["matchScore"] for case in SCORED}
    assert {0, 100} <= scores
    assert _case("perfect_100")["expected"]["result"]["matchScore"] == 100
    assert _case("worst_0")["expected"]["result"]["matchScore"] == 0
    assert all(item["points"] == 0 for item in _case("worst_0")["expected"]["result"]["breakdown"])


def _close_to(values: list[float], target: float) -> bool:
    return any(abs(value - target) <= TOLERANCE for value in values)


def test_factor_borders_are_covered() -> None:
    def values(name: str) -> list[float]:
        return [_feature(case, name) for case in SCORED]

    assert all(_close_to(values("price_below_min"), target) for target in (0.0, 0.5, 1.0))
    assert all(_close_to(values("distance_ratio"), target) for target in (0.5, 0.75, 1.0))
    assert any(value > 1.0 + TOLERANCE for value in values("distance_ratio"))
    coverage = values("services_coverage")
    assert _close_to(coverage, 0.0) and _close_to(coverage, 1.0)
    assert any(TOLERANCE < value < 1 - TOLERANCE for value in coverage)
    for name in ("place_type_match", "property_type_match"):
        assert set(values(name)) == {0.0, 1.0}
    extras = values("extra_services_ratio")
    assert _close_to(extras, 0.0) and _close_to(extras, 1.0)
    assert any(TOLERANCE < value < 1 - TOLERANCE for value in extras)


def test_rounding_borders_are_covered() -> None:
    def points(case_id: str, factor: str) -> float:
        breakdown = _case(case_id)["expected"]["result"]["breakdown"]
        return float(next(item["points"] for item in breakdown if item["factor"] == factor))

    assert points("budget_13_35", "budget") == 13.4
    assert points("distance_24_35", "distance") == 24.4
    assert points("budget_2_35", "budget") == 2.4
    assert points("distance_2_35", "distance") == 2.4
    assert points("extras_1_25", "extras") == 1.3
    totals = {
        "total_x5_exact": (86.5, 87),
        "total_x5_binary64_noise": (68.5, 69),
        "total_x4999995_up": (82.49999955, 83),
        "total_x4999994_down": (82.4999994, 82),
        "total_x5000004_up": (82.5000004, 83),
        "spec_6_4_example": (77.708333, 78),
    }
    for case_id, (raw_total, score) in totals.items():
        expected = _case(case_id)["expected"]
        assert abs(expected["rawTotal"] - raw_total) <= TOLERANCE, case_id
        assert expected["result"]["matchScore"] == score, case_id


def test_inputs_cover_transports_nulls_campuses_and_exclusions() -> None:
    students = [case["input"]["student"] for case in CASES]
    assert {student["transport"] for student in students} >= {"walk", "bike", "public", "car", None}
    for field in ("budget.min", "maxDistanceKm", "essentialServices", "pets", "transport", "propertyTypes"):
        assert any(student[field] is None for student in students), field
    assert {student["university_code"] for student in students} == set(META["campuses"])
    violated = [case["expected"]["result"]["violatedConstraints"] for case in EXCLUDED]
    assert {item for group in violated for item in group} == set(META["hard_constraints"])
    assert any(len(group) >= 2 for group in violated)
    assert any(len(group) == 3 for group in violated)


def test_incomplete_profiles_cover_each_required_field_and_all_three() -> None:
    groups = [case["expected"]["result"]["missingFields"] for case in INCOMPLETE]
    for name in META["required_fields"]:
        assert [name] in groups, name
    assert META["required_fields"] in groups
    wins = _case("incomplete_wins_over_exclusion")
    assert wins["input"]["student"]["pets"] is True
    assert wins["input"]["room"]["pets_allowed"] is False
    assert wins["expected"]["result"]["missingFields"] == ["livingWith"]
