"""Configuración versionada del modelo base por reglas (ML-04 T01, ML-04-R01).

`baseline_params.json` vive junto a este módulo y se empaqueta con él, igual
que `feature_schema.json`. Declara factores, pesos, la feature de cada factor,
curvas lineales por tramos, las features que no puntúan y las reglas de
redondeo (spec ML-04 §6.2 y §6.3, M-ML-029). Se valida con la biblioteca
estándar contra el esquema de features:

- los pesos son positivos y suman 100 (`total_points`);
- cada feature existe en `feature_schema.json` y cada feature del esquema
  puntúa en un factor o está en `non_scoring_features` con su motivo;
- cada curva tiene al menos 2 puntos, `x` estrictamente creciente dentro del
  rango de su feature e `y` en [0, 1];
- la versión del esquema es compatible: misma versión mayor y versión del
  esquema ≥ `feature_schema_version` (M-ML-030).

Si algo no cuadra, `BaselineConfigError` lista todos los problemas.
"""

from __future__ import annotations

import json
import math
import re
from bisect import bisect_right
from dataclasses import dataclass
from functools import cache
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Any, TypeGuard

from thehouse_ml.features import FeatureSchema, default_schema

CONFIG_PACKAGE = "thehouse_ml.models"
CONFIG_RESOURCE = "baseline_params.json"
TOTAL_POINTS = 100
CURVE_TYPE = "piecewise_linear"
EXTRAPOLATION = "constant"
ROUNDING_MODES = frozenset({"half_up"})
_MODEL_VERSION = re.compile(r"^baseline-(\d+)\.(\d+)\.(\d+)$")
_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class BaselineConfigError(ValueError):
    """La configuración del modelo base no cuadra; lista todos los problemas."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("baseline_params.json inválido:\n- " + "\n- ".join(problems))


@dataclass(frozen=True)
class Curve:
    """Función lineal por tramos con extremos constantes."""

    points: tuple[tuple[float, float], ...]

    def evaluate(self, x: float) -> float:
        """Valor de la curva en `x` (interpolación lineal; fuera de los extremos, el valor del extremo)."""
        xs = [point[0] for point in self.points]
        if x <= xs[0]:
            return self.points[0][1]
        if x >= xs[-1]:
            return self.points[-1][1]
        index = bisect_right(xs, x)
        (x0, y0), (x1, y1) = self.points[index - 1], self.points[index]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass(frozen=True)
class Factor:
    """Factor del puntaje: `weight * curve(feature)`."""

    id: str
    label: str
    weight: float
    feature: str
    curve: Curve
    description: str


@dataclass(frozen=True)
class NonScoringFeature:
    """Feature del esquema que no puntúa en este modelo, con su motivo."""

    feature: str
    reason: str


@dataclass(frozen=True)
class RoundingRules:
    """Reglas de redondeo y de orden (spec ML-04 §6.3)."""

    factor_pre_round_decimals: int
    factor_decimals: int
    factor_mode: str
    factor_formula: str
    total_pre_round_decimals: int
    total_mode: str
    total_formula: str
    ranking_order_by: str
    ranking_direction: str
    tie_breaker: str
    tie_breaker_direction: str
    total_tolerance: float
    breakdown_tolerance: float
    dotnet: str


@dataclass(frozen=True)
class BaselineConfig:
    """Configuración `baseline-x.y.z` validada contra el esquema de features."""

    model_version: str
    spec: str
    decision: str
    feature_schema_version: str
    total_points: int
    factors: tuple[Factor, ...]
    non_scoring_features: tuple[NonScoringFeature, ...]
    rounding: RoundingRules

    @property
    def weights(self) -> dict[str, float]:
        return {factor.id: factor.weight for factor in self.factors}

    def factor(self, factor_id: str) -> Factor:
        for factor in self.factors:
            if factor.id == factor_id:
                return factor
        raise KeyError(factor_id)


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


class _Reader:
    """Lee el JSON acumulando problemas en lugar de fallar en el primero."""

    def __init__(self) -> None:
        self.problems: list[str] = []

    def obj(self, value: object, where: str) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        self.problems.append(f"{where}: se esperaba un objeto")
        return {}

    def lst(self, value: object, where: str) -> list[Any]:
        if isinstance(value, list):
            return value
        self.problems.append(f"{where}: se esperaba una lista")
        return []

    def text(self, data: dict[str, Any], key: str, where: str) -> str:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
        self.problems.append(f"{where}.{key}: se esperaba un texto no vacío")
        return ""

    def number(self, data: dict[str, Any], key: str, where: str) -> float:
        value = data.get(key)
        if _is_number(value):
            return float(value)
        self.problems.append(f"{where}.{key}: se esperaba un número finito")
        return math.nan

    def integer(self, data: dict[str, Any], key: str, where: str) -> int:
        value = data.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        self.problems.append(f"{where}.{key}: se esperaba un entero no negativo")
        return -1

    def choice(self, data: dict[str, Any], key: str, where: str, allowed: frozenset[str]) -> str:
        value = self.text(data, key, where)
        if value and value not in allowed:
            self.problems.append(f"{where}.{key}: {value!r} no soportado (permitidos: {sorted(allowed)})")
        return value


def _semver(version: str, pattern: re.Pattern[str]) -> tuple[int, int, int] | None:
    match = pattern.match(version)
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def schema_is_compatible(required: str, actual: str) -> bool:
    """Misma versión mayor y `actual` ≥ `required` (semver X.Y.Z)."""
    wanted, have = _semver(required, _SEMVER), _semver(actual, _SEMVER)
    if wanted is None or have is None:
        return False
    return have[0] == wanted[0] and have >= wanted


def _read_curve(reader: _Reader, raw: object, where: str, schema: FeatureSchema, feature: str) -> Curve:
    data = reader.obj(raw, where)
    reader.choice(data, "type", where, frozenset({CURVE_TYPE}))
    points: list[tuple[float, float]] = []
    for index, item in enumerate(reader.lst(data.get("points"), f"{where}.points")):
        if not (isinstance(item, list) and len(item) == 2 and all(_is_number(value) for value in item)):
            reader.problems.append(f"{where}.points[{index}]: se esperaba [x, y] numérico y finito")
            continue
        points.append((float(item[0]), float(item[1])))
    if len(points) < 2:
        reader.problems.append(f"{where}.points: se esperaban al menos 2 puntos")
    for (x0, _), (x1, _) in pairwise(points):
        if x1 <= x0:
            reader.problems.append(f"{where}.points: x debe ser estrictamente creciente ({x0:g} → {x1:g})")
    for _x, y in points:
        if not 0.0 <= y <= 1.0:
            reader.problems.append(f"{where}.points: y = {y:g} fuera de [0, 1]")
    spec = next((item for item in schema.pair_features if item.name == feature), None)
    if spec is not None:
        for x, _ in points:
            if not spec.range.contains(x):
                reader.problems.append(f"{where}.points: x = {x:g} fuera del rango {spec.range.render()} de {feature}")
    return Curve(points=tuple(points))


def _read_factors(reader: _Reader, raw: object, schema: FeatureSchema) -> tuple[Factor, ...]:
    known = set(schema.feature_order)
    factors: list[Factor] = []
    for index, item in enumerate(reader.lst(raw, "factors")):
        data = reader.obj(item, f"factors[{index}]")
        factor_id = reader.text(data, "id", f"factors[{index}]")
        where = f"factors[{factor_id or index}]"
        feature = reader.text(data, "feature", where)
        if feature and feature not in known:
            reader.problems.append(f"{where}.feature: {feature!r} no existe en feature_schema.json {schema.version}")
        weight = reader.number(data, "weight", where)
        if not math.isnan(weight) and weight <= 0:
            reader.problems.append(f"{where}.weight: debe ser positivo (llegó {data.get('weight')!r})")
        factors.append(
            Factor(
                id=factor_id,
                label=reader.text(data, "label", where),
                weight=weight,
                feature=feature,
                curve=_read_curve(reader, data.get("curve"), f"{where}.curve", schema, feature),
                description=reader.text(data, "description", where),
            )
        )
    if not factors:
        reader.problems.append("factors: se esperaba al menos un factor")
    return tuple(factors)


def _read_non_scoring(reader: _Reader, raw: object, schema: FeatureSchema) -> tuple[NonScoringFeature, ...]:
    result: list[NonScoringFeature] = []
    for index, item in enumerate(reader.lst(raw, "non_scoring_features")):
        data = reader.obj(item, f"non_scoring_features[{index}]")
        feature = reader.text(data, "feature", f"non_scoring_features[{index}]")
        if feature and feature not in schema.feature_order:
            reader.problems.append(
                f"non_scoring_features[{index}].feature: {feature!r} no existe en feature_schema.json {schema.version}"
            )
        result.append(
            NonScoringFeature(feature=feature, reason=reader.text(data, "reason", f"non_scoring_features[{index}]"))
        )
    return tuple(result)


def _read_rounding(reader: _Reader, raw: object, total_points: int) -> RoundingRules:
    data = reader.obj(raw, "rounding")
    factor = reader.obj(data.get("factor_points"), "rounding.factor_points")
    total = reader.obj(data.get("total"), "rounding.total")
    ranking = reader.obj(data.get("ranking"), "rounding.ranking")
    tolerances = reader.obj(data.get("tolerances"), "rounding.tolerances")
    score_range = total.get("range")
    if score_range != [0, total_points]:
        reader.problems.append(f"rounding.total.range: se esperaba [0, {total_points}], llegó {score_range!r}")
    rules = RoundingRules(
        factor_pre_round_decimals=reader.integer(factor, "pre_round_decimals", "rounding.factor_points"),
        factor_decimals=reader.integer(factor, "decimals", "rounding.factor_points"),
        factor_mode=reader.choice(factor, "mode", "rounding.factor_points", ROUNDING_MODES),
        factor_formula=reader.text(factor, "formula", "rounding.factor_points"),
        total_pre_round_decimals=reader.integer(total, "pre_round_decimals", "rounding.total"),
        total_mode=reader.choice(total, "mode", "rounding.total", ROUNDING_MODES),
        total_formula=reader.text(total, "formula", "rounding.total"),
        ranking_order_by=reader.choice(ranking, "order_by", "rounding.ranking", frozenset({"unrounded_sum"})),
        ranking_direction=reader.choice(ranking, "direction", "rounding.ranking", frozenset({"desc"})),
        tie_breaker=reader.choice(ranking, "tie_breaker", "rounding.ranking", frozenset({"room_id"})),
        tie_breaker_direction=reader.choice(ranking, "tie_breaker_direction", "rounding.ranking", frozenset({"asc"})),
        total_tolerance=reader.number(tolerances, "total_vs_unrounded_sum", "rounding.tolerances"),
        breakdown_tolerance=reader.number(tolerances, "reported_sum_vs_unrounded_sum", "rounding.tolerances"),
        dotnet=reader.text(data, "dotnet", "rounding"),
    )
    if rules.dotnet and "AwayFromZero" not in rules.dotnet:
        reader.problems.append("rounding.dotnet: debe indicar MidpointRounding.AwayFromZero para .NET")
    for name, value in (
        ("total_vs_unrounded_sum", rules.total_tolerance),
        ("reported_sum_vs_unrounded_sum", rules.breakdown_tolerance),
    ):
        if value < 0:
            reader.problems.append(f"rounding.tolerances.{name}: no puede ser negativa")
    return rules


def _check_coverage(
    problems: list[str], factors: tuple[Factor, ...], non_scoring: tuple[NonScoringFeature, ...], schema: FeatureSchema
) -> None:
    ids = [factor.id for factor in factors]
    for factor_id in sorted({item for item in ids if item and ids.count(item) > 1}):
        problems.append(f"factors: id duplicado {factor_id!r}")
    scoring = [factor.feature for factor in factors if factor.feature]
    for feature in sorted({item for item in scoring if scoring.count(item) > 1}):
        problems.append(f"factors: la feature {feature!r} puntúa en más de un factor")
    excluded = [item.feature for item in non_scoring if item.feature]
    for feature in sorted({item for item in excluded if excluded.count(item) > 1}):
        problems.append(f"non_scoring_features: feature duplicada {feature!r}")
    for feature in sorted(set(scoring) & set(excluded)):
        problems.append(f"non_scoring_features: {feature!r} también puntúa en un factor")
    missing = [name for name in schema.feature_order if name not in set(scoring) | set(excluded)]
    if missing:
        problems.append(f"features del esquema sin factor ni motivo en non_scoring_features: {missing}")


def parse_config(data: object, schema: FeatureSchema | None = None) -> BaselineConfig:
    """Convierte el JSON ya decodificado en `BaselineConfig` y lo valida contra el esquema."""
    schema = schema or default_schema()
    reader = _Reader()
    root = reader.obj(data, "raíz")
    meta = reader.obj(root.get("metadata"), "metadata")
    model_version = reader.text(meta, "model_version", "metadata")
    spec = reader.text(meta, "spec", "metadata")
    decision = reader.text(meta, "decision", "metadata")
    if model_version and _semver(model_version, _MODEL_VERSION) is None:
        reader.problems.append(f"metadata.model_version: {model_version!r} no es baseline-X.Y.Z")
    required = reader.text(meta, "feature_schema_version", "metadata")
    if required and _semver(required, _SEMVER) is None:
        reader.problems.append(f"metadata.feature_schema_version: {required!r} no es semver X.Y.Z")
    elif required and not schema_is_compatible(required, schema.version):
        reader.problems.append(
            f"metadata.feature_schema_version: la configuración requiere el esquema {required} "
            f"(misma versión mayor, igual o posterior) y el esquema cargado es {schema.version}"
        )
    semantics = reader.obj(root.get("curve_semantics"), "curve_semantics")
    reader.choice(semantics, "type", "curve_semantics", frozenset({CURVE_TYPE}))
    reader.choice(semantics, "extrapolation", "curve_semantics", frozenset({EXTRAPOLATION}))
    total_points = root.get("total_points")
    if total_points != TOTAL_POINTS or isinstance(total_points, bool):
        reader.problems.append(f"total_points: se esperaba {TOTAL_POINTS}, llegó {total_points!r}")
    factors = _read_factors(reader, root.get("factors"), schema)
    total = math.fsum(factor.weight for factor in factors)
    if math.isfinite(total) and not math.isclose(total, TOTAL_POINTS, rel_tol=0.0, abs_tol=1e-9):
        weights = ", ".join(f"{factor.id} {factor.weight:g}" for factor in factors)
        reader.problems.append(f"factors: los pesos suman {total:g}, no {TOTAL_POINTS} ({weights})")
    non_scoring = _read_non_scoring(reader, root.get("non_scoring_features"), schema)
    rounding = _read_rounding(reader, root.get("rounding"), TOTAL_POINTS)
    _check_coverage(reader.problems, factors, non_scoring, schema)
    if reader.problems:
        raise BaselineConfigError(reader.problems)
    return BaselineConfig(
        model_version=model_version,
        spec=spec,
        decision=decision,
        feature_schema_version=required,
        total_points=TOTAL_POINTS,
        factors=factors,
        non_scoring_features=non_scoring,
        rounding=rounding,
    )


def load_config(path: Path | None = None, schema: FeatureSchema | None = None) -> BaselineConfig:
    """Carga la configuración empaquetada (o `path`) y la valida contra el esquema."""
    if path is None:
        text = resources.files(CONFIG_PACKAGE).joinpath(CONFIG_RESOURCE).read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise BaselineConfigError([f"JSON inválido: {error}"]) from error
    return parse_config(data, schema)


@cache
def default_config() -> BaselineConfig:
    """Configuración empaquetada, cargada y validada una sola vez."""
    return load_config()


__all__ = [
    "CONFIG_RESOURCE",
    "BaselineConfig",
    "BaselineConfigError",
    "Curve",
    "Factor",
    "NonScoringFeature",
    "RoundingRules",
    "default_config",
    "load_config",
    "parse_config",
    "schema_is_compatible",
]
