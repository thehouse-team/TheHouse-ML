"""ML-03 T01 (R01): `feature_schema.json` coincide con NV-11 y con las universidades.

Un cambio en el snapshot de NV-11 (`data/seed/nv11_catalogs.json`) hace fallar
estas pruebas: según la regla 1 de la spec, obliga a una versión mayor del
esquema.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from thehouse_ml.features import FeatureSchema, default_schema, load_schema, schema_version

REPO = Path(__file__).resolve().parents[2]
NV11_SNAPSHOT = REPO / "data" / "seed" / "nv11_catalogs.json"
UNIVERSITIES_CSV = REPO / "data" / "universities.csv"


@pytest.fixture(scope="module")
def nv11() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(NV11_SNAPSHOT.read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def schema() -> FeatureSchema:
    return load_schema()


def test_the_packaged_schema_loads_validates_and_exposes_its_version(schema: FeatureSchema) -> None:
    assert schema.version == "2.0.0"
    assert schema_version() == "2.0.0"
    assert default_schema() is default_schema()
    assert schema.metadata.spec == "ML-04 1.0.0"
    assert schema.metadata.synthetic_training_data is True


def test_the_schema_cites_the_snapshot_commit(schema: FeatureSchema, nv11: dict[str, Any]) -> None:
    assert schema.metadata.nv11_source["commit"] == nv11["source"]["commit"] == "b7d7595"
    assert schema.metadata.nv11_source["verified_commit"] == nv11["source"]["verified_commit"] == "ea10a4c"
    assert schema.currency == nv11["currency"] == "MXN"


def test_catalogs_are_exactly_those_of_nv11(schema: FeatureSchema, nv11: dict[str, Any]) -> None:
    expected = {name: tuple(values) for name, values in nv11["catalogs"].items()}

    assert dict(schema.nv11_catalogs) == expected


@pytest.mark.parametrize(
    ("field", "catalog"),
    [
        ("student.essentialServices", "essentialServices"),
        ("student.livingWith", "livingWith"),
        ("student.placeType", "placeType"),
        ("student.transport", "transport"),
        ("student.propertyTypes", "propertyTypes"),
        ("room.services", "essentialServices"),
        ("room.placeType", "placeType"),
        ("room.propertyType", "propertyTypes"),
        ("room.livingWith", "livingWith"),
    ],
)
def test_student_and_room_fields_use_the_nv11_catalogs(schema: FeatureSchema, field: str, catalog: str) -> None:
    group, name = field.split(".", 1)
    spec = schema.student_field(name) if group == "student" else schema.room_field(name)

    assert spec.catalog == catalog
    assert catalog in schema.nv11_catalogs


def test_budget_and_distance_ranges_are_those_of_nv11(schema: FeatureSchema, nv11: dict[str, Any]) -> None:
    budget = nv11["ranges"]["budget"]
    distance = nv11["ranges"]["maxDistanceKm"]

    for name in ("budget.min", "budget.max"):
        spec = schema.student_field(name)
        assert spec.type == budget["type"]
        assert spec.range is not None
        assert (spec.range.min, spec.range.max) == (budget["min"], budget["max"])
    spec = schema.student_field("maxDistanceKm")
    assert spec.type == distance["type"]
    assert spec.range is not None
    assert (spec.range.min, spec.range.max) == (distance["min"], distance["max"])
    assert budget["constraint"] in schema.student_constraints
    price = schema.room_field("price_mxn")
    assert price.range is not None
    assert (price.range.min, price.range.max) == (budget["min"], budget["max"])


def test_booleans_of_nv11_are_booleans(schema: FeatureSchema, nv11: dict[str, Any]) -> None:
    for name in nv11["booleans"]:
        assert schema.student_field(name).type == "boolean"


def test_nullable_fields_are_those_of_nv11(schema: FeatureSchema, nv11: dict[str, Any]) -> None:
    nullable = {spec.name for spec in schema.student if spec.nullable}

    assert nullable == set(nv11["nullable"])


def test_every_nullable_field_has_a_missing_rule(schema: FeatureSchema) -> None:
    for spec in schema.student + schema.room:
        if spec.nullable:
            assert spec.missing is not None, spec.name
            assert spec.missing.description


def test_quick_setup_fields_cannot_be_null_for_scoring(schema: FeatureSchema, nv11: dict[str, Any]) -> None:
    required = {spec.name for spec in schema.student if spec.nullable and spec.required_for_scoring}

    assert required == set(nv11["quick_setup_required"])
    for name in required:
        rule = schema.student_field(name).missing
        assert rule is not None
        assert rule.action == "reject"


@pytest.mark.parametrize(
    ("name", "action", "value"),
    [
        ("budget.min", "default", 0),
        ("maxDistanceKm", "default", 30),
        ("essentialServices", "default", []),
        ("pets", "default", False),
        ("transport", "default", "public"),
        ("propertyTypes", "accept_all", None),
    ],
)
def test_missing_rules_of_m_ml_022(schema: FeatureSchema, name: str, action: str, value: object) -> None:
    rule = schema.student_field(name).missing

    assert rule is not None
    assert (rule.action, rule.value) == (action, value)


def test_feature_order_has_no_duplicates_and_covers_every_pair_feature(schema: FeatureSchema) -> None:
    order = schema.feature_order

    assert len(order) == len(set(order))
    assert set(order) == {feature.name for feature in schema.pair_features}
    assert order == (
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


def test_transport_factors_cover_the_transport_catalog(schema: FeatureSchema) -> None:
    feature = next(feature for feature in schema.pair_features if feature.name == "effective_distance_km")

    assert feature.parameters["transport_factors"] == {"walk": 1.0, "bike": 0.5, "public": 0.4, "car": 0.25}
    assert set(feature.parameters["transport_factors"]) == set(schema.nv11_catalogs["transport"])


def test_hard_constraints_have_ids_for_budget_living_with_and_pets(schema: FeatureSchema) -> None:
    constraints = {constraint.id: constraint for constraint in schema.hard_constraints}

    assert set(constraints) == {"HC01_budget", "HC02_living_with", "HC03_pets"}
    assert constraints["HC01_budget"].excludes_when == "room.price_mxn > student.budget.max"
    assert set(constraints["HC02_living_with"].fields) == {"student.livingWith", "room.livingWith"}
    assert set(constraints["HC03_pets"].fields) == {"student.pets", "room.pets_allowed"}


def test_room_backend_status_matches_property(schema: FeatureSchema) -> None:
    status = {spec.name: spec.backend_status for spec in schema.room}

    missing = {"lat", "lon", "placeType", "propertyType", "pets_allowed", "livingWith", "services"}
    present = {"price_mxn": "Price", "bedrooms": "Bedrooms", "bathrooms": "Bathrooms", "area_m2": "AreaM2"}
    present["university_code"] = "UniversityId"
    assert {name for name, value in status.items() if value == "missing_in_property"} == missing
    assert {name for name, value in status.items() if value == "present"} == set(present)
    for name, backend_field in present.items():
        assert schema.room_field(name).backend_field == backend_field
    assert schema.room_field("synthetic").type == "boolean"


def test_university_codes_match_the_universities_csv(schema: FeatureSchema) -> None:
    with UNIVERSITIES_CSV.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert schema.ml_catalogs["university_code"] == tuple(row["code"] for row in rows)
    assert set(schema.unverified_university_codes) == {
        row["code"] for row in rows if row["final_status"] == "seed_sin_verificar"
    }
    for group in (schema.student_field("university_code"), schema.room_field("university_code")):
        assert group.catalog == "university_code"


BACKEND_TO_SCHEMA = {"int?": "integer", "decimal?": "number", "bool?": "boolean"}


def test_student_types_match_the_backend_types_of_the_snapshot(schema: FeatureSchema, nv11: dict[str, Any]) -> None:
    for name, backend_type in nv11["backend_types"].items():
        assert schema.student_field(name).type == BACKEND_TO_SCHEMA[backend_type.split(" ")[0]], name


def test_the_snapshot_records_the_human_verification(nv11: dict[str, Any], schema: FeatureSchema) -> None:
    assert "César Yael López Trujillo" in nv11["source"]["verified_by"]
    assert "ea10a4c" in nv11["source"]["verified_by"]
    assert "ea10a4c" in schema.metadata.nv11_source["verified_by"]


def test_room_types_from_property_and_documented_assumptions() -> None:
    raw = json.loads((REPO / "src" / "thehouse_ml" / "features" / "feature_schema.json").read_text(encoding="utf-8"))
    fields = {field["name"]: field for field in raw["room"]["fields"]}

    assert fields["price_mxn"]["backend_type"] == "decimal"
    assert fields["price_mxn"]["type"] == "integer"
    assert "redondea a entero" in fields["price_mxn"]["description"]
    for name, backend_type in (("bedrooms", "int"), ("bathrooms", "int"), ("area_m2", "double")):
        assert fields[name]["backend_type"] == backend_type
        assert fields[name]["range_basis"].startswith("SUPUESTO")
    assert raw["pair_features"][1]["parameters_status"].startswith("APROBADO")
