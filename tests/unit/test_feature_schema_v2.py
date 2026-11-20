"""ML-04 T02 (R02, M-ML-029): esquema de features 2.0.0 y sus 3 features nuevas.

`distance_ratio`, `price_below_min` y `extra_services_ratio` se agregan al final
de `feature_order` (versión mayor, constitución §18); las 7 features de 1.1.0
conservan nombre, definición, rango, posición y valor.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from thehouse_ml.features import (
    Campus,
    FeatureSchema,
    Room,
    StudentProfile,
    compute_pair_features,
    default_schema,
    load_campuses,
    room_from_record,
    student_from_record,
)

REPO = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO / "src" / "thehouse_ml" / "features" / "feature_schema.json"
CASES_PATH = REPO / "tests" / "fixtures" / "pair_feature_cases.json"
UNIVERSITIES_CSV = REPO / "data" / "universities.csv"
CASES: dict[str, Any] = json.loads(CASES_PATH.read_text(encoding="utf-8"))

V1_FEATURES = (
    "distance_km",
    "effective_distance_km",
    "price_to_budget",
    "services_coverage",
    "place_type_match",
    "property_type_match",
    "pets_compatible",
)
NEW_FEATURES = ("distance_ratio", "price_below_min", "extra_services_ratio")
# SHA-256 de json.dumps(pair_features[:7], sort_keys=True, ensure_ascii=False) en el esquema 1.1.0 (commit fac23bc).
V1_DEFINITIONS_SHA256 = "938939bbca96c423b9f1e27798a07b61a83e5d9d9c35281b96e48fafb8360318"


@pytest.fixture(scope="module")
def schema() -> FeatureSchema:
    return default_schema()


@pytest.fixture(scope="module")
def raw_schema() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def campuses() -> dict[str, Campus]:
    return load_campuses(UNIVERSITIES_CSV)


def _base() -> tuple[StudentProfile, Room]:
    case = next(case for case in CASES["cases"] if case["id"] == "budget_exact")
    return student_from_record(case["input"]["student"]), room_from_record(case["input"]["room"])


def _feature(schema: FeatureSchema, name: str) -> Any:
    return next(feature for feature in schema.pair_features if feature.name == name)


# --- Versión, changelog y orden ---------------------------------------------


def test_the_schema_loads_as_major_version_2_0_0(schema: FeatureSchema, raw_schema: dict[str, Any]) -> None:
    assert schema.version == "2.0.0"
    assert schema.metadata.spec == "ML-04 1.0.0"
    changelog = raw_schema["metadata"]["changelog"]
    assert [entry["version"] for entry in changelog] == ["1.0.0", "1.1.0", "2.0.0"]
    last = changelog[-1]
    assert last["decision"] == "M-ML-029"
    for word in ("ONNX", "NV-23", *NEW_FEATURES):
        assert word in last["change"]
    assert "M-ML-029" in raw_schema["metadata"]["decision"]


def test_versioning_says_that_adding_removing_or_reordering_features_is_major(raw_schema: dict[str, Any]) -> None:
    versioning = raw_schema["metadata"]["versioning"]
    for word in ("versión mayor", "agregar, quitar o reordenar", "feature_order", "NV-11", "NV-23"):
        assert word in versioning


def test_the_new_features_are_appended_at_the_end(schema: FeatureSchema) -> None:
    assert schema.feature_order[:7] == V1_FEATURES
    assert schema.feature_order[7:] == NEW_FEATURES
    assert tuple(feature.name for feature in schema.pair_features) == schema.feature_order


def test_the_seven_previous_definitions_did_not_change(raw_schema: dict[str, Any]) -> None:
    previous = raw_schema["pair_features"][:7]
    assert tuple(item["name"] for item in previous) == V1_FEATURES
    digest = hashlib.sha256(json.dumps(previous, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert digest == V1_DEFINITIONS_SHA256


def test_the_table_has_the_35_previous_cases_and_the_new_ones() -> None:
    ids = [case["id"] for case in CASES["cases"]]
    assert len(ids) == len(set(ids)) == 48
    assert CASES["metadata"]["schema_version"] == "2.0.0"
    assert tuple(CASES["metadata"]["feature_order"]) == (*V1_FEATURES, *NEW_FEATURES)


# --- Rangos ------------------------------------------------------------------


def test_ranges_of_the_new_features(schema: FeatureSchema) -> None:
    ratio = _feature(schema, "distance_ratio")
    assert (ratio.type, ratio.range.min, ratio.range.max) == ("number", 0.0, None)  # sin límite superior
    assert ratio.training_range is not None
    assert (ratio.training_range.min, ratio.training_range.max) == (0.0, 60.0)  # 30 km / 0.5 km
    assert set(ratio.depends_on) == {"distance_km", "student.maxDistanceKm"}  # distancia real, no la efectiva
    for name in ("price_below_min", "extra_services_ratio"):
        feature = _feature(schema, name)
        assert (feature.type, feature.range.min, feature.range.max) == ("number", 0.0, 1.0)
        assert feature.training_range is None


def test_the_training_range_of_distance_ratio_is_the_bound_of_the_synthetic_catalog(schema: FeatureSchema) -> None:
    # Cuartos a ≤ 30 km (M-ML-021) y maxDistanceKm ≥ 0.5 (NV-11): distance_ratio ≤ 60 en los datos sintéticos.
    max_distance = schema.student_field("maxDistanceKm")
    distance = _feature(schema, "distance_km")
    assert max_distance.range is not None
    assert distance.training_range is not None
    assert distance.training_range.max is not None
    ratio = _feature(schema, "distance_ratio")
    assert ratio.training_range.max == distance.training_range.max / max_distance.range.min


# --- Las 7 features anteriores no cambian de valor ---------------------------


@pytest.mark.parametrize(
    "case", [case for case in CASES["cases"] if "expected" in case], ids=lambda case: str(case["id"])
)
def test_the_first_seven_values_equal_those_of_schema_1_1_0(
    case: dict[str, Any], schema: FeatureSchema, campuses: dict[str, Campus]
) -> None:
    legacy = dataclasses.replace(schema, pair_features=schema.pair_features[:7], feature_order=V1_FEATURES)
    student = student_from_record(case["input"]["student"])
    room = room_from_record(case["input"]["room"])
    current = compute_pair_features(student, room, campuses, schema)
    previous = compute_pair_features(student, room, campuses, legacy)
    assert tuple(previous) == V1_FEATURES
    assert {name: current[name] for name in V1_FEATURES} == previous


# --- Fórmulas ----------------------------------------------------------------


@pytest.mark.parametrize("transport", ["walk", "bike", "public", "car", None])
@pytest.mark.parametrize("max_distance", [0.5, 10, 30, None])
def test_distance_ratio_uses_the_real_distance_and_the_resolved_max_distance(
    transport: str | None, max_distance: float | None, campuses: dict[str, Campus]
) -> None:
    student, room = _base()
    student = dataclasses.replace(student, transport=transport, max_distance_km=max_distance)
    room = dataclasses.replace(room, lat=room.lat + 0.09)
    result = compute_pair_features(student, room, campuses)
    resolved_max = 30.0 if max_distance is None else max_distance  # M-ML-022
    assert result["distance_ratio"] == result["distance_km"] / resolved_max
    assert result["distance_ratio"] >= 0.0


@pytest.mark.parametrize("budget_min", [None, 0, 1, 1000, 2000, 4000])
def test_price_below_min_is_in_range_zero_at_or_above_the_minimum_and_non_increasing(
    budget_min: int | None, campuses: dict[str, Campus]
) -> None:
    student, room = _base()
    student = dataclasses.replace(student, budget_min=budget_min, budget_max=4000)
    previous = None
    for price in range(0, 6001, 250):
        value = compute_pair_features(student, dataclasses.replace(room, price_mxn=price), campuses)["price_below_min"]
        assert 0.0 <= value <= 1.0
        if not budget_min or price >= budget_min:
            assert value == 0.0
        else:
            assert value == (budget_min - price) / budget_min
        if previous is not None:
            assert value <= previous
        previous = value


def test_extra_services_ratio_over_subsets_of_the_catalog(schema: FeatureSchema, campuses: dict[str, Campus]) -> None:
    catalog = schema.catalog("essentialServices")
    assert len(catalog) == 11
    student, room = _base()
    checked = 0
    for size in range(len(catalog) + 1):
        for requested in itertools.islice(itertools.combinations(catalog, size), 3):
            for offered in (frozenset(), frozenset(catalog[:4]), frozenset(catalog[3:]), frozenset(catalog)):
                result = compute_pair_features(
                    dataclasses.replace(student, essential_services=frozenset(requested)),
                    dataclasses.replace(room, services=offered),
                    campuses,
                )
                value = result["extra_services_ratio"]
                assert 0.0 <= value <= 1.0
                not_requested = len(catalog) - len(requested)
                expected = 0.0 if not_requested == 0 else len(offered - set(requested)) / not_requested
                assert value == expected
                checked += 1
    assert checked > 100


def test_a_null_essential_services_counts_as_nothing_requested(campuses: dict[str, Campus]) -> None:
    student, room = _base()
    room = dataclasses.replace(room, services=frozenset({"wifi", "gas"}))
    null = compute_pair_features(dataclasses.replace(student, essential_services=None), room, campuses)
    empty = compute_pair_features(dataclasses.replace(student, essential_services=frozenset()), room, campuses)
    assert null["extra_services_ratio"] == empty["extra_services_ratio"] == 2 / 11
