"""ML-03 T02 (R02): features del par con la implementación de referencia.

Los casos por tabla viven en `tests/fixtures/pair_feature_cases.json` para
reutilizarse como casos dorados del backend (ML-06). Sus salidas se calcularon
a mano, sin este código: los cuartos con distancia > 0 están sobre el meridiano
del campus, donde haversine = 6371 km por delta_phi.

ML-04 T02 (R02, M-ML-029): esquema 2.0.0 con `distance_ratio`,
`price_below_min` y `extra_services_ratio` al final de `feature_order`. Los
valores nuevos de la tabla salieron de un recuento independiente (aritmética
exacta con fracciones, sin este código).
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import pytest

from thehouse_ml import universities
from thehouse_ml.features import (
    Campus,
    FeatureSchema,
    IncompleteProfileError,
    PairInputError,
    Room,
    StudentProfile,
    compute_pair_features,
    default_schema,
    feature_vector,
    load_campuses,
    pairs,
    resolve_student,
    room_from_record,
    student_from_record,
    validate_room,
)

REPO = Path(__file__).resolve().parents[2]
CASES_PATH = REPO / "tests" / "fixtures" / "pair_feature_cases.json"
UNIVERSITIES_CSV = REPO / "data" / "universities.csv"
CASES: dict[str, Any] = json.loads(CASES_PATH.read_text(encoding="utf-8"))
TOLERANCE = float(CASES["metadata"]["tolerance"])


@pytest.fixture(scope="module")
def schema() -> FeatureSchema:
    return default_schema()


@pytest.fixture(scope="module")
def campuses() -> dict[str, Campus]:
    return load_campuses(UNIVERSITIES_CSV)


def _case(case_id: str) -> dict[str, Any]:
    for case in CASES["cases"]:
        if case["id"] == case_id:
            return dict(case)
    raise KeyError(case_id)


def _base_student(**changes: Any) -> StudentProfile:
    student = student_from_record(_case("budget_exact")["input"]["student"])
    return dataclasses.replace(student, **changes)


def _base_room(**changes: Any) -> Room:
    room = room_from_record(_case("budget_exact")["input"]["room"])
    return dataclasses.replace(room, **changes)


# --- Casos por tabla -------------------------------------------------------


@pytest.mark.parametrize("case", CASES["cases"], ids=[case["id"] for case in CASES["cases"]])
def test_table_case(case: dict[str, Any], campuses: dict[str, Campus], schema: FeatureSchema) -> None:
    student = student_from_record(case["input"]["student"])
    room = room_from_record(case["input"]["room"])
    if "expected_error" in case:
        assert case["expected_error"]["type"] == "IncompleteProfileError"
        with pytest.raises(IncompleteProfileError) as raised:
            compute_pair_features(student, room, campuses, schema)
        assert list(raised.value.missing) == case["expected_error"]["missing"]
        assert "quick setup" in str(raised.value)
        return
    result = compute_pair_features(student, room, campuses, schema)
    assert list(result) == list(case["expected"])
    for name, expected in case["expected"].items():
        assert math.isclose(result[name], expected, rel_tol=0.0, abs_tol=TOLERANCE), (name, result[name], expected)


def test_the_table_covers_every_requested_scenario() -> None:
    ids = {case["id"] for case in CASES["cases"]}
    required = {
        "budget_exact",
        "distance_zero",
        "no_services_requested",
        "room_without_services",
        "pets_student_false_room_false",
        "pets_student_false_room_true",
        "pets_student_true_room_true",
        "pets_student_true_room_false",
        "property_types_empty",
        "transport_walk",
        "transport_bike",
        "transport_public",
        "transport_car",
        "null_transport",
        "null_budget_min_and_max_distance",
        "null_essential_services",
        "null_pets",
        "null_property_types",
        "distance_over_30_km",
        "incomplete_budget_max",
        "incomplete_livingWith",
        "incomplete_placeType",
        "budget_max_zero_price_zero",
        "budget_max_zero_price_positive",
        "room_assumed_ranges_exceeded",
        # Esquema 2.0.0 (ML-04 T02)
        "distance_ratio_null_max_distance",
        "distance_ratio_equal_max_distance",
        "distance_ratio_half",
        "price_below_min_null_budget_min",
        "price_equal_budget_min",
        "price_below_budget_min",
        "price_far_below_budget_min",
        "price_zero_with_budget_min",
        "extras_no_services_requested",
        "extras_all_eleven_requested",
        "extras_room_without_extra_services",
        "extras_room_with_every_service",
    }
    assert required <= ids
    assert len(ids) == len(CASES["cases"]), "ids duplicados"
    assert TOLERANCE == 1e-6


def test_the_table_uses_the_campus_coordinates_of_universities_csv(campuses: dict[str, Campus]) -> None:
    for code, coords in CASES["metadata"]["campuses"].items():
        assert campuses[code].lat == coords["lat"]
        assert campuses[code].lon == coords["lon"]


def test_the_table_metadata_matches_the_schema(schema: FeatureSchema) -> None:
    assert CASES["metadata"]["schema_version"] == schema.version
    assert tuple(CASES["metadata"]["feature_order"]) == schema.feature_order
    assert CASES["metadata"]["synthetic"] is True
    for case in CASES["cases"]:
        if "expected" in case:
            assert tuple(case["expected"]) == schema.feature_order, case["id"]


def test_distances_beyond_30_km_or_max_distance_are_returned_unclipped(campuses: dict[str, Campus]) -> None:
    case = _case("distance_over_30_km")
    student = student_from_record(case["input"]["student"])
    result = compute_pair_features(student, room_from_record(case["input"]["room"]), campuses)
    assert student.max_distance_km is not None
    assert result["distance_km"] > 30.0
    assert result["distance_km"] > student.max_distance_km
    assert result["effective_distance_km"] == result["distance_km"]


@pytest.mark.parametrize("case_id", ["distance_over_30_km", "distance_over_30_km_car"])
def test_a_distance_beyond_30_km_is_valid_against_the_schema(
    case_id: str, schema: FeatureSchema, campuses: dict[str, Campus]
) -> None:
    case = _case(case_id)
    result = compute_pair_features(
        student_from_record(case["input"]["student"]), room_from_record(case["input"]["room"]), campuses, schema
    )
    assert result["distance_km"] > 30.0
    for name in ("distance_km", "effective_distance_km"):
        feature = next(feature for feature in schema.pair_features if feature.name == name)
        assert feature.range.min == 0.0
        assert feature.range.max is None  # sin límite superior (M-ML-023)
        assert feature.range.contains(result[name])
        assert not feature.range.contains(-0.001)
        assert feature.training_range is not None
        assert (feature.training_range.min, feature.training_range.max) == (0.0, 30.0)
    distance = next(feature for feature in schema.pair_features if feature.name == "distance_km")
    assert distance.training_range is not None
    assert not distance.training_range.contains(result["distance_km"])  # fuera del dominio de entrenamiento


def test_price_to_budget_is_one_formula_with_max_budget_one(campuses: dict[str, Campus]) -> None:
    for budget_max, price in ((0, 0), (0, 3000), (1, 3000), (4000, 4000), (4000, 5000)):
        result = compute_pair_features(
            _base_student(budget_min=0, budget_max=budget_max), _base_room(price_mxn=price), campuses
        )
        assert result["price_to_budget"] == price / max(budget_max, 1)


def test_a_price_to_budget_of_3000_is_valid_against_the_schema(
    schema: FeatureSchema, campuses: dict[str, Campus]
) -> None:
    case = _case("budget_max_zero_price_positive")
    result = compute_pair_features(
        student_from_record(case["input"]["student"]), room_from_record(case["input"]["room"]), campuses, schema
    )
    assert result["price_to_budget"] == 3000.0
    feature = next(feature for feature in schema.pair_features if feature.name == "price_to_budget")
    assert feature.range.min == 0.0
    assert feature.range.max is None  # sin límite superior (M-ML-023, hallazgo 9)
    assert feature.range.contains(result["price_to_budget"])
    assert not feature.range.contains(-0.001)
    assert feature.training_range is not None
    assert (feature.training_range.min, feature.training_range.max) == (0.0, 1.0)  # pares que pasan HC01
    assert not feature.training_range.contains(result["price_to_budget"])
    assert feature.training_range.contains(1.0)  # presupuesto exacto


@pytest.mark.parametrize(
    "case", [case for case in CASES["cases"] if "expected" in case], ids=lambda case: str(case["id"])
)
def test_every_computed_feature_is_valid_against_its_schema_range(
    case: dict[str, Any], schema: FeatureSchema, campuses: dict[str, Campus]
) -> None:
    result = compute_pair_features(
        student_from_record(case["input"]["student"]), room_from_record(case["input"]["room"]), campuses, schema
    )
    for feature in schema.pair_features:
        assert feature.range.contains(result[feature.name]), (feature.name, result[feature.name])


def test_only_unbounded_features_can_leave_their_training_range(schema: FeatureSchema) -> None:
    # Las features con rango cerrado no pueden rebasarlo por construcción: cobertura (0 a 1) y coincidencias (0/1).
    closed = {feature.name for feature in schema.pair_features if feature.range.max is not None}
    unbounded = {feature.name for feature in schema.pair_features if feature.range.max is None}
    assert closed == {
        "services_coverage",
        "place_type_match",
        "property_type_match",
        "pets_compatible",
        "price_below_min",
        "extra_services_ratio",
    }
    assert unbounded == {"distance_km", "effective_distance_km", "price_to_budget", "distance_ratio"}
    for feature in schema.pair_features:
        assert (feature.training_range is not None) == (feature.name in unbounded)


def test_assumed_room_ranges_do_not_reject_the_room(schema: FeatureSchema) -> None:
    for name in ("bedrooms", "bathrooms", "area_m2"):
        spec = schema.room_field(name)
        assert spec.range_is_assumption
        assert spec.range is not None  # el rango sigue documentado en el esquema
    assert not schema.room_field("price_mxn").range_is_assumption
    validate_room(_base_room(bedrooms=0, bathrooms=0, area_m2=0.0), schema)
    validate_room(_base_room(bedrooms=50, bathrooms=20, area_m2=5000.0), schema)


# --- Orden de las features -------------------------------------------------


def test_features_come_back_in_the_exact_order_of_the_schema(
    schema: FeatureSchema, campuses: dict[str, Campus]
) -> None:
    result = compute_pair_features(_base_student(), _base_room(), campuses, schema)
    assert tuple(result) == schema.feature_order
    assert schema.feature_order == (
        "distance_km",
        "effective_distance_km",
        "price_to_budget",
        "services_coverage",
        "place_type_match",
        "property_type_match",
        "pets_compatible",
        "distance_ratio",
        "price_below_min",
        "extra_services_ratio",
    )
    assert feature_vector(result, schema) == tuple(float(result[name]) for name in schema.feature_order)
    assert feature_vector(result) == feature_vector(result, schema)


def test_the_order_is_taken_from_the_schema_not_from_the_code(
    schema: FeatureSchema, campuses: dict[str, Campus]
) -> None:
    reordered = dataclasses.replace(schema, feature_order=tuple(reversed(schema.feature_order)))
    result = compute_pair_features(_base_student(), _base_room(), campuses, reordered)
    assert tuple(result) == tuple(reversed(schema.feature_order))


def test_a_schema_feature_without_reference_implementation_fails(
    schema: FeatureSchema, campuses: dict[str, Campus]
) -> None:
    broken = dataclasses.replace(schema, feature_order=(*schema.feature_order, "noise_level"))
    with pytest.raises(PairInputError, match="noise_level"):
        compute_pair_features(_base_student(), _base_room(), campuses, broken)


def test_integer_features_are_zero_or_one(schema: FeatureSchema, campuses: dict[str, Campus]) -> None:
    result = compute_pair_features(_base_student(), _base_room(), campuses, schema)
    for feature in schema.pair_features:
        if feature.type == "integer":
            assert isinstance(result[feature.name], int)
            assert result[feature.name] in (0, 1)


# --- Reglas para faltantes del esquema --------------------------------------


def test_missing_rules_come_from_the_schema(schema: FeatureSchema) -> None:
    resolved = resolve_student(
        StudentProfile(university_code="UTTT", budget_max=4000, living_with="any", place_type="room")
    )
    assert resolved.budget_min == schema.student_field("budget.min").missing.value  # type: ignore[union-attr]
    assert resolved.max_distance_km == float(schema.student_field("maxDistanceKm").missing.value)  # type: ignore[union-attr]
    assert resolved.essential_services == frozenset()
    assert resolved.pets is False
    assert resolved.transport == "public"
    assert resolved.property_types == frozenset()


def test_dataclasses_cover_every_field_of_the_schema(schema: FeatureSchema) -> None:
    assert set(pairs.STUDENT_ATTRIBUTES) == {spec.name for spec in schema.student}
    assert set(pairs.ROOM_ATTRIBUTES) == {spec.name for spec in schema.room}
    assert set(pairs.STUDENT_ATTRIBUTES.values()) == {f.name for f in dataclasses.fields(StudentProfile)}
    assert set(pairs.ROOM_ATTRIBUTES.values()) == {f.name for f in dataclasses.fields(Room)}


def test_a_non_nullable_student_field_without_rule_is_rejected(schema: FeatureSchema) -> None:
    student = dataclasses.replace(_base_student(), university_code=None)  # type: ignore[arg-type]
    with pytest.raises(PairInputError, match="university_code"):
        resolve_student(student, schema)


# --- Validación de entradas -----------------------------------------------


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"place_type": "castle"}, "placeType"),
        ({"living_with": "everyone"}, "livingWith"),
        ({"transport": "teleport"}, "transport"),
        ({"essential_services": frozenset({"pool"})}, "essentialServices"),
        ({"essential_services": frozenset({1})}, "essentialServices"),
        ({"property_types": frozenset({"castle"})}, "propertyTypes"),
        ({"budget_max": 60000}, "budget.max"),
        ({"budget_max": 4000.5}, "budget.max"),
        ({"budget_max": True}, "budget.max"),
        ({"budget_min": -1}, "budget.min"),
        ({"max_distance_km": 0.1}, "maxDistanceKm"),
        ({"max_distance_km": math.nan}, "maxDistanceKm"),
        ({"max_distance_km": "10"}, "maxDistanceKm"),
        ({"pets": 1}, "pets"),
        ({"university_code": "MIT"}, "university_code"),
        ({"budget_min": 5000}, "budget.min"),
    ],
)
def test_invalid_student_values_are_rejected(
    changes: dict[str, Any], message: str, campuses: dict[str, Campus]
) -> None:
    with pytest.raises(PairInputError, match=message.replace(".", r"\.")):
        compute_pair_features(_base_student(**changes), _base_room(), campuses)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"price_mxn": -1}, "price_mxn"),
        ({"price_mxn": 3000.5}, "price_mxn"),
        ({"lat": 91.0}, "lat"),
        ({"lon": math.inf}, "lon"),
        ({"services": frozenset({"pool"})}, "services"),
        ({"services": ["wifi"]}, "services"),
        ({"place_type": "castle"}, "placeType"),
        ({"property_type": "castle"}, "propertyType"),
        ({"pets_allowed": "yes"}, "pets_allowed"),
        ({"bedrooms": -1}, "bedrooms"),
        ({"bedrooms": 2.5}, "bedrooms"),
        ({"bathrooms": -2}, "bathrooms"),
        ({"area_m2": -1.0}, "area_m2"),
        ({"area_m2": "12"}, "area_m2"),
        ({"synthetic": None}, "synthetic"),
    ],
)
def test_invalid_room_values_are_rejected(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(PairInputError, match=message):
        validate_room(_base_room(**changes))


def test_a_student_university_without_campus_coordinates_is_rejected(schema: FeatureSchema) -> None:
    with pytest.raises(PairInputError, match="sin coordenadas de campus"):
        compute_pair_features(_base_student(), _base_room(), {}, schema)


def test_a_schema_without_transport_factors_is_rejected(schema: FeatureSchema, campuses: dict[str, Campus]) -> None:
    features = tuple(
        dataclasses.replace(feature, parameters={}) if feature.name == "effective_distance_km" else feature
        for feature in schema.pair_features
    )
    broken = dataclasses.replace(schema, pair_features=features)
    with pytest.raises(PairInputError, match="transport_factors"):
        compute_pair_features(_base_student(), _base_room(), campuses, broken)


def test_records_with_unknown_or_missing_keys_are_rejected() -> None:
    student = dict(_case("budget_exact")["input"]["student"])
    room = dict(_case("budget_exact")["input"]["room"])
    with pytest.raises(PairInputError, match="desconocidos"):
        student_from_record({**student, "gender": "x"})
    with pytest.raises(PairInputError, match="university_code"):
        student_from_record({key: value for key, value in student.items() if key != "university_code"})
    with pytest.raises(PairInputError, match="essentialServices"):
        student_from_record({**student, "essentialServices": "wifi"})
    with pytest.raises(PairInputError, match="faltantes"):
        room_from_record({key: value for key, value in room.items() if key != "lat"})
    with pytest.raises(PairInputError, match="desconocidos"):
        room_from_record({**room, "host_name": "x"})


# --- Pureza y una sola implementación de haversine -------------------------


def test_the_function_is_deterministic_and_does_not_mutate_inputs(campuses: dict[str, Campus]) -> None:
    case = _case("transport_bike")
    student = student_from_record(case["input"]["student"])
    room = room_from_record(case["input"]["room"])
    before = (dataclasses.asdict(student), dataclasses.asdict(room), dict(campuses))
    first = compute_pair_features(student, room, campuses)
    second = compute_pair_features(student, room, campuses)
    assert first == second
    assert (dataclasses.asdict(student), dataclasses.asdict(room), dict(campuses)) == before


def test_distance_reuses_the_single_haversine_of_ml02() -> None:
    assert pairs.haversine_m is universities.haversine_m
    tree = ast.parse(Path(pairs.__file__).read_text(encoding="utf-8"))
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not any("haversine" in name for name in names)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
        if node.func.value.id == "math"
    }
    assert not calls & {"asin", "atan2", "sin", "cos", "radians"}


def test_the_pairs_module_has_no_io_or_randomness() -> None:
    tree = ast.parse(Path(pairs.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    imported |= {
        node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not imported & {"random", "numpy", "os", "pathlib", "io", "csv", "json", "urllib", "time", "datetime"}
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "open" not in called
