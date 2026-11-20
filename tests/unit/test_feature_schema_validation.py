"""ML-03 T01 (R01): la validación estructural rechaza esquemas rotos."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from thehouse_ml.features import SchemaError, load_schema, parse_schema, schema_transport_factors

Mutation = Callable[[dict[str, Any]], None]


@pytest.fixture(scope="module")
def raw_schema() -> dict[str, Any]:
    text = resources.files("thehouse_ml.features").joinpath("feature_schema.json").read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


def student(data: dict[str, Any], name: str) -> dict[str, Any]:
    field: dict[str, Any] = next(item for item in data["student"]["fields"] if item["name"] == name)
    return field


def room(data: dict[str, Any], name: str) -> dict[str, Any]:
    field: dict[str, Any] = next(item for item in data["room"]["fields"] if item["name"] == name)
    return field


def problems_for(raw_schema: dict[str, Any], mutate: Mutation) -> str:
    data = copy.deepcopy(raw_schema)
    mutate(data)
    with pytest.raises(SchemaError) as error:
        parse_schema(data)
    return "\n".join(error.value.problems)


def test_an_unchanged_copy_parses(raw_schema: dict[str, Any]) -> None:
    schema = parse_schema(copy.deepcopy(raw_schema))

    assert schema.version == "2.0.0"
    assert schema.catalog("university_code")[0] == "UTTT"
    assert schema.catalog("placeType") == ("room", "room_private_bath", "studio", "shared")
    with pytest.raises(KeyError):
        schema.student_field("no_existe")


def test_load_schema_from_a_path(tmp_path: Path, raw_schema: dict[str, Any]) -> None:
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(raw_schema), encoding="utf-8")

    assert load_schema(path).version == "2.0.0"


def test_invalid_json_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "schema.json"
    path.write_text("{no es json", encoding="utf-8")

    with pytest.raises(SchemaError, match="JSON inválido"):
        load_schema(path)


def test_a_non_object_root_is_rejected() -> None:
    with pytest.raises(SchemaError, match="raíz: se esperaba un objeto"):
        parse_schema([])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["metadata"].update(schema_version="1.0"), "no es semver"),
        (lambda d: d["metadata"].update(synthetic_training_data=False), "son sintéticos"),
        (lambda d: d["metadata"].update(synthetic_training_data="sí"), "se esperaba un booleano"),
        (lambda d: d["metadata"]["nv11_source"].pop("commit"), "nv11_source.commit"),
        (lambda d: d.update(currency=""), "currency"),
        (lambda d: d["catalogs"]["nv11"].update(livingWith=["women", "women"]), "duplicados"),
        (lambda d: d["catalogs"]["nv11"].update(transport=[]), "catálogo vacío"),
        (lambda d: d["catalogs"]["nv11"].update(pets=[1, 2]), "textos no vacíos"),
        (lambda d: d["catalogs"]["ml"].update(placeType=["x"]), "repetidos entre nv11 y ml"),
        (lambda d: d["universities"].update(unverified_codes=["XYZ"]), "no está en catalogs.ml"),
        (lambda d: d["student"].update(fields={}), "student.fields: se esperaba una lista"),
        (lambda d: d["student"]["fields"].append(copy.deepcopy(d["student"]["fields"][0])), "campo duplicado"),
        (lambda d: student(d, "pets").update(type="flag"), "tipo desconocido"),
        (lambda d: student(d, "budget.min").pop("range"), "necesita range"),
        (lambda d: student(d, "budget.min").update(range={"min": 10, "max": 1}), "mayor que max"),
        (lambda d: student(d, "budget.min").update(range={"min": "0", "max": 1}), "deben ser numéricos"),
        (lambda d: student(d, "livingWith").pop("catalog"), "necesita catalog"),
        (lambda d: student(d, "livingWith").update(catalog="gender"), "catálogo desconocido"),
        (lambda d: student(d, "pets").pop("missing"), "nullable sin regla para faltantes"),
        (lambda d: student(d, "university_code").update(missing=student(d, "pets")["missing"]), "no nullable"),
        (lambda d: student(d, "transport")["missing"].update(action="guess"), "acción desconocida"),
        (lambda d: student(d, "budget.max")["missing"].update(action="default", value=0), "solo admite"),
        (lambda d: student(d, "pets")["missing"].update(action="reject"), "solo aplica a campos obligatorios"),
        (lambda d: student(d, "transport")["missing"].update(action="accept_all"), "solo aplica a category_set"),
        (lambda d: student(d, "transport")["missing"].update(value="plane"), "fuera de tipo"),
        (lambda d: student(d, "pets")["missing"].update(value=0), "fuera de tipo"),
        (lambda d: student(d, "maxDistanceKm")["missing"].update(value=31), "fuera de tipo"),
        (lambda d: student(d, "maxDistanceKm")["missing"].update(value=True), "fuera de tipo"),
        (lambda d: student(d, "budget.min")["missing"].update(value=0.5), "fuera de tipo"),
        (lambda d: student(d, "essentialServices")["missing"].update(value=["pool"]), "fuera de tipo"),
        (lambda d: student(d, "essentialServices")["missing"].update(value="wifi"), "fuera de tipo"),
        (lambda d: student(d, "pets")["missing"].pop("description"), "missing.description"),
        (lambda d: room(d, "lat").update(backend_status="maybe"), "valor desconocido"),
        (lambda d: room(d, "lat").pop("backend_status"), "backend_status"),
        (lambda d: d["pair_features"].append(copy.deepcopy(d["pair_features"][0])), "feature duplicada"),
        (lambda d: d["pair_features"][0].update(type="category"), "debe ser numérica"),
        (lambda d: d["pair_features"][0].update(depends_on=["room.floor"]), "dependencia desconocida"),
        (lambda d: d["pair_features"][0].update(formula=""), "formula"),
        (
            lambda d: d["pair_features"][1]["parameters"]["transport_factors"].pop("car"),
            "no cubre exactamente el catálogo",
        ),
        (lambda d: d.update(hard_constraints=[]), "al menos una restricción"),
        (lambda d: d["hard_constraints"].append(copy.deepcopy(d["hard_constraints"][0])), "id duplicado"),
        (lambda d: d["hard_constraints"][0].update(fields=["student.age"]), "campo desconocido"),
        (lambda d: d.update(feature_order=[]), "lista vacía"),
        (lambda d: d["feature_order"].append("distance_km"), "duplicados"),
        (lambda d: d["feature_order"].pop(), "exactamente las features"),
        # Esquema 1.1.0 (M-ML-023): `max: null` solo en features del par; training_range [min, max] dentro de range.
        (lambda d: student(d, "maxDistanceKm").update(range={"min": 0.5, "max": None}), "deben ser numéricos"),
        (lambda d: d["pair_features"][0].update(range={"min": "0", "max": None}), "min debe ser numérico"),
        (lambda d: d["pair_features"][0].update(training_range=[0]), "se esperaba [min, max]"),
        (lambda d: d["pair_features"][0].update(training_range=[0, "30"]), "se esperaba [min, max]"),
        (lambda d: d["pair_features"][0].update(training_range=[30, 0]), "mayor que max"),
        (lambda d: d["pair_features"][0].update(training_range=[-1, 30]), "fuera de range [0, ∞)"),
        (lambda d: d["pair_features"][3].update(training_range=[0, 2]), "fuera de range [0, 1]"),
    ],
)
def test_broken_schemas_are_rejected(raw_schema: dict[str, Any], mutate: Mutation, message: str) -> None:
    assert message in problems_for(raw_schema, mutate)


def test_every_problem_is_reported_at_once(raw_schema: dict[str, Any]) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["metadata"]["schema_version"] = "v1"
        student(data, "pets").pop("missing")
        data["feature_order"] = []

    problems = problems_for(raw_schema, mutate)

    assert "no es semver" in problems
    assert "nullable sin regla" in problems
    assert "lista vacía" in problems


def test_transport_factors_are_optional(raw_schema: dict[str, Any]) -> None:
    data = copy.deepcopy(raw_schema)
    data["pair_features"][1].pop("parameters")

    assert schema_transport_factors(parse_schema(data)) is None
