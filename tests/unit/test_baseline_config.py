"""ML-04 T01 (R01): configuración versionada `baseline-1.0.0` y su validación.

La configuración empaquetada (`thehouse_ml/models/baseline_params.json`) debe
reproducir la spec ML-04 §6.2 a 6.3 (M-ML-029); cada copia rota debe fallar con
un `BaselineConfigError` claro.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from thehouse_ml.features import FeatureSchema, default_schema
from thehouse_ml.models import (
    CONFIG_RESOURCE,
    BaselineConfig,
    BaselineConfigError,
    Curve,
    default_config,
    load_config,
    parse_config,
    schema_is_compatible,
)


@pytest.fixture(scope="module")
def raw_config() -> dict[str, Any]:
    text = resources.files("thehouse_ml.models").joinpath(CONFIG_RESOURCE).read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


@pytest.fixture(scope="module")
def config() -> BaselineConfig:
    return default_config()


@pytest.fixture(scope="module")
def schema() -> FeatureSchema:
    return default_schema()


def _problems(raw: dict[str, Any], change: Callable[[dict[str, Any]], object], schema: FeatureSchema) -> str:
    data = copy.deepcopy(raw)
    change(data)
    with pytest.raises(BaselineConfigError) as error:
        parse_config(data, schema)
    assert str(error.value).startswith("baseline_params.json inválido:")
    return "\n".join(error.value.problems)


def _factor(data: dict[str, Any], factor_id: str) -> dict[str, Any]:
    return next(item for item in data["factors"] if item["id"] == factor_id)  # type: ignore[no-any-return]


# --- La configuración empaquetada = spec §6.2 a 6.3 ------------------------------


def test_the_packaged_config_is_baseline_1_0_0_for_schema_2_0_0(config: BaselineConfig) -> None:
    assert config.model_version == "baseline-1.0.0"
    assert config.feature_schema_version == "2.0.0"
    assert config.spec == "ML-04 1.0.0"
    assert "M-ML-029" in config.decision
    assert config.total_points == 100
    assert default_config() is default_config()


def test_factors_weights_and_features_follow_the_spec_table(config: BaselineConfig) -> None:
    assert [(f.id, f.weight, f.feature) for f in config.factors] == [
        ("budget", 30, "price_below_min"),
        ("distance", 25, "distance_ratio"),
        ("essential_services", 20, "services_coverage"),
        ("place_type", 10, "place_type_match"),
        ("property_type", 10, "property_type_match"),
        ("extras", 5, "extra_services_ratio"),
    ]
    assert math.fsum(config.weights.values()) == 100
    assert config.factor("distance").label == "Distancia"
    with pytest.raises(KeyError):
        config.factor("gender")


def test_curves_are_the_declared_piecewise_linear_points(config: BaselineConfig) -> None:
    identity = ((0.0, 0.0), (1.0, 1.0))
    assert config.factor("budget").curve.points == ((0.0, 1.0), (1.0, 0.0))
    assert config.factor("distance").curve.points == ((0.0, 1.0), (0.5, 1.0), (1.0, 0.0))
    for factor_id in ("essential_services", "place_type", "property_type", "extras"):
        assert config.factor(factor_id).curve.points == identity


@pytest.mark.parametrize(
    ("factor_id", "x", "expected"),
    [
        ("budget", 0.0, 1.0),
        ("budget", 0.15, 0.85),
        ("budget", 0.9, 0.1),
        ("budget", 1.0, 0.0),
        ("distance", 0.0, 1.0),
        ("distance", 0.5, 1.0),
        ("distance", 0.66, 0.68),
        ("distance", 0.75, 0.5),
        ("distance", 1.0, 0.0),
        ("distance", 1.2, 0.0),
        ("distance", 59.8, 0.0),
        ("essential_services", 2 / 3, 2 / 3),
        ("place_type", 0.0, 0.0),
        ("place_type", 1.0, 1.0),
        ("extras", 3 / 7, 3 / 7),
    ],
)
def test_curve_values_match_the_formulas_of_the_spec(
    config: BaselineConfig, factor_id: str, x: float, expected: float
) -> None:
    assert math.isclose(config.factor(factor_id).curve.evaluate(x), expected, rel_tol=0.0, abs_tol=1e-12)


def test_the_curve_keeps_the_end_values_outside_its_points() -> None:
    curve = Curve(points=((1.0, 0.2), (2.0, 0.6)))
    assert curve.evaluate(0.0) == 0.2
    assert curve.evaluate(1.5) == pytest.approx(0.4)
    assert curve.evaluate(9.0) == 0.6


def test_the_example_of_spec_6_4_adds_up_to_77_976190(config: BaselineConfig) -> None:
    # Solo comprueba pesos y curvas; el redondeo y la salida son T03.
    values = {
        "price_below_min": 0.15,
        "distance_ratio": 0.66,
        "services_coverage": 2 / 3,
        "place_type_match": 1,
        "property_type_match": 1,
        "extra_services_ratio": 3 / 7,
    }
    total = math.fsum(f.weight * f.curve.evaluate(values[f.feature]) for f in config.factors)
    assert round(total, 6) == 77.97619


def test_every_schema_feature_scores_or_has_a_reason(config: BaselineConfig, schema: FeatureSchema) -> None:
    scoring = [factor.feature for factor in config.factors]
    excluded = {item.feature: item.reason for item in config.non_scoring_features}
    assert set(excluded) == {"pets_compatible", "price_to_budget", "effective_distance_km", "distance_km"}
    assert all(reason.strip() for reason in excluded.values())
    assert sorted([*scoring, *excluded]) == sorted(schema.feature_order)
    assert "HC03" in excluded["pets_compatible"]
    assert "HC01" in excluded["price_to_budget"]


def test_rounding_rules_of_m_ml_029(config: BaselineConfig, raw_config: dict[str, Any]) -> None:
    rules = config.rounding
    assert (rules.factor_decimals, rules.factor_mode) == (1, "half_up")
    assert (rules.total_pre_round_decimals, rules.total_mode) == (6, "half_up")
    assert rules.total_formula == "floor(round6(Σ puntos sin redondear) + 0.5)"
    assert (rules.ranking_order_by, rules.ranking_direction) == ("unrounded_sum", "desc")
    assert (rules.tie_breaker, rules.tie_breaker_direction) == ("room_id", "asc")
    assert (rules.total_tolerance, rules.breakdown_tolerance) == (0.5, 0.3)
    assert "MidpointRounding.AwayFromZero" in rules.dotnet
    assert "ToEven" in rules.dotnet
    assert raw_config["rounding"]["total"]["range"] == [0, 100]


def test_no_factor_uses_a_sensitive_attribute(
    config: BaselineConfig, raw_config: dict[str, Any], schema: FeatureSchema
) -> None:
    sensitive = {"gender", "genero", "género", "age", "edad", "origin", "origen"}
    dependencies = {
        dependency.split(".", 1)[-1].lower()
        for feature in schema.pair_features
        if feature.name in {factor.feature for factor in config.factors}
        for dependency in feature.depends_on
    }
    assert not dependencies & sensitive
    assert not {factor.feature for factor in config.factors} & sensitive
    assert "constitution §11.2" in raw_config["metadata"]["sensitive_attributes"]


def test_load_config_from_a_path(tmp_path: Path, raw_config: dict[str, Any], schema: FeatureSchema) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(raw_config), encoding="utf-8")
    assert load_config(path, schema) == default_config()
    path.write_text("{no es json", encoding="utf-8")
    with pytest.raises(BaselineConfigError, match="JSON inválido"):
        load_config(path, schema)


# --- Versión del esquema -----------------------------------------------------


@pytest.mark.parametrize(
    ("required", "actual", "compatible"),
    [
        ("2.0.0", "2.0.0", True),
        ("2.0.0", "2.1.0", True),
        ("2.0.0", "2.0.3", True),
        ("2.1.0", "2.0.0", False),
        ("2.0.0", "1.1.0", False),
        ("2.0.0", "3.0.0", False),
        ("2.0", "2.0.0", False),
        ("2.0.0", "v2", False),
    ],
)
def test_schema_compatibility_is_same_major_and_not_older(required: str, actual: str, compatible: bool) -> None:
    assert schema_is_compatible(required, actual) is compatible


@pytest.mark.parametrize("version", ["1.1.0", "3.0.0"])
def test_a_schema_with_another_major_version_is_rejected(
    raw_config: dict[str, Any], schema: FeatureSchema, version: str
) -> None:
    other = dataclasses.replace(schema, metadata=dataclasses.replace(schema.metadata, schema_version=version))
    with pytest.raises(BaselineConfigError, match=rf"requiere el esquema 2\.0\.0.*esquema cargado es {version}"):
        parse_config(copy.deepcopy(raw_config), other)


def test_a_later_minor_schema_is_accepted(raw_config: dict[str, Any], schema: FeatureSchema) -> None:
    newer = dataclasses.replace(schema, metadata=dataclasses.replace(schema.metadata, schema_version="2.1.0"))
    assert parse_config(copy.deepcopy(raw_config), newer).feature_schema_version == "2.0.0"


# --- Copias rotas ------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        # Pesos
        (lambda d: _factor(d, "budget").update(weight=31), "los pesos suman 101, no 100"),
        (lambda d: _factor(d, "extras").update(weight=0), "debe ser positivo"),
        (lambda d: _factor(d, "extras").update(weight=-5), "debe ser positivo"),
        (lambda d: _factor(d, "extras").update(weight="5"), "se esperaba un número finito"),
        (lambda d: _factor(d, "extras").update(weight=True), "se esperaba un número finito"),
        (lambda d: d["factors"].pop(), "los pesos suman 95"),
        (lambda d: d.update(total_points=99), "total_points: se esperaba 100"),
        (lambda d: d.update(factors=[]), "al menos un factor"),
        # Features
        (lambda d: _factor(d, "budget").update(feature="noise_level"), "'noise_level' no existe en feature_schema"),
        (lambda d: _factor(d, "extras").update(feature="distance_ratio"), "puntúa en más de un factor"),
        (lambda d: d["non_scoring_features"].pop(), "sin factor ni motivo"),
        (
            lambda d: d["non_scoring_features"].append({"feature": "distance_ratio", "reason": "x"}),
            "también puntúa en un factor",
        ),
        (
            lambda d: d["non_scoring_features"].append({"feature": "gender", "reason": "x"}),
            "'gender' no existe en feature_schema.json",
        ),
        (lambda d: d["non_scoring_features"].append(dict(d["non_scoring_features"][0])), "feature duplicada"),
        (lambda d: d["non_scoring_features"][0].update(reason=""), "reason: se esperaba un texto no vacío"),
        (lambda d: _factor(d, "extras").update(id="budget"), "id duplicado 'budget'"),
        # Curvas
        (lambda d: _factor(d, "distance")["curve"].update(points=[[0, 1], [1, 0], [0.5, 1]]), "creciente"),
        (lambda d: _factor(d, "distance")["curve"].update(points=[[0, 1], [0, 0]]), "estrictamente creciente"),
        (lambda d: _factor(d, "budget")["curve"].update(points=[[0, 1.2], [1, 0]]), "y = 1.2 fuera de [0, 1]"),
        (lambda d: _factor(d, "budget")["curve"].update(points=[[0, 1], [1, -0.1]]), "y = -0.1 fuera de [0, 1]"),
        (lambda d: _factor(d, "budget")["curve"].update(points=[[0, 1]]), "al menos 2 puntos"),
        (lambda d: _factor(d, "budget")["curve"].update(points=[[0, 1], [1.5, 0]]), "x = 1.5 fuera del rango [0, 1]"),
        (lambda d: _factor(d, "distance")["curve"].update(points=[[-1, 1], [1, 0]]), "x = -1 fuera del rango [0, ∞)"),
        (lambda d: _factor(d, "budget")["curve"].update(points=[[0, 1], ["1", 0]]), "se esperaba [x, y] numérico"),
        (lambda d: _factor(d, "budget")["curve"].update(points=[[0, 1], [1, 0, 2]]), "se esperaba [x, y] numérico"),
        (lambda d: _factor(d, "budget")["curve"].update(type="spline"), "'spline' no soportado"),
        (lambda d: d["curve_semantics"].update(extrapolation="linear"), "'linear' no soportado"),
        # Versión y metadatos
        (lambda d: d["metadata"].update(feature_schema_version="1.1.0"), "requiere el esquema 1.1.0"),
        (lambda d: d["metadata"].update(feature_schema_version="2.1.0"), "requiere el esquema 2.1.0"),
        (lambda d: d["metadata"].update(feature_schema_version="2"), "no es semver X.Y.Z"),
        (lambda d: d["metadata"].update(model_version="v1"), "no es baseline-X.Y.Z"),
        (lambda d: d["metadata"].pop("spec"), "metadata.spec"),
        # Redondeo
        (lambda d: d["rounding"]["factor_points"].update(mode="half_even"), "'half_even' no soportado"),
        (lambda d: d["rounding"]["factor_points"].update(decimals=-1), "entero no negativo"),
        (lambda d: d["rounding"]["total"].update(range=[0, 10]), "rounding.total.range"),
        (lambda d: d["rounding"]["ranking"].update(tie_breaker="price_mxn"), "'price_mxn' no soportado"),
        (lambda d: d["rounding"]["tolerances"].update(total_vs_unrounded_sum=-0.5), "no puede ser negativa"),
        (lambda d: d["rounding"].update(dotnet="Math.Round(x)"), "AwayFromZero"),
        (lambda d: d.pop("rounding"), "rounding: se esperaba un objeto"),
    ],
)
def test_broken_configs_are_rejected_with_a_clear_message(
    raw_config: dict[str, Any], schema: FeatureSchema, change: Callable[[dict[str, Any]], object], message: str
) -> None:
    assert message in _problems(raw_config, change, schema)


def test_every_problem_is_reported_at_once(raw_config: dict[str, Any], schema: FeatureSchema) -> None:
    def break_many(data: dict[str, Any]) -> None:
        _factor(data, "budget").update(weight=40, feature="noise_level")
        _factor(data, "distance")["curve"].update(points=[[1, 0], [0, 1]])

    problems = _problems(raw_config, break_many, schema)
    assert "los pesos suman 110" in problems
    assert "'noise_level' no existe" in problems
    assert "estrictamente creciente" in problems


def test_a_non_object_root_is_rejected(schema: FeatureSchema) -> None:
    with pytest.raises(BaselineConfigError, match="raíz: se esperaba un objeto"):
        parse_config([], schema)
