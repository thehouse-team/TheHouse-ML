"""Puntaje del modelo base por reglas con desglose por factor (ML-04 T03, ML-04-R03).

Implementación de referencia, pura (sin I/O, red ni aleatoriedad), de la spec
ML-04 §6.2 a §6.4 (M-ML-029). Pesos, curvas, decimales y orden del ranking se
leen de `baseline_params.json` (`baseline-1.0.0`); el orden de las features,
de `feature_schema.json` (2.0.0). Nada de eso se escribe en este módulo.

Pasos (una sola implementación de cada uno):

1. `score_pair` aplica las restricciones duras (`check_hard_constraints`,
   ML-03 T03). Un par excluido **no** recibe puntaje: el resultado lista las
   restricciones violadas (constitution §11.1).
2. Si el par pasa, `compute_pair_features` (ML-03 T02 / ML-04 T02) calcula las
   10 features y `feature_vector` las ordena como el esquema.
3. `score_features` recibe ese vector (lo mismo que recibirá el ONNX y que
   calculará NV-23) y devuelve el desglose y `matchScore`.
4. `rank` ordena por la suma sin redondear (desc) y desempata por `room_id` (asc).

Redondeo «mitad hacia arriba» (M-ML-031, APROBADO): se replica el algoritmo de
.NET `Math.Round(x, d, MidpointRounding.AwayFromZero)`: se escala `x` por
`10^d` en binary64 y se redondea el valor escalado, con la mitad hacia arriba.
El total y cada factor se redondean primero a 6 decimales (round6) para
absorber el error de punto flotante: 30 * (1 - 0.555) = 13.349999999999998 →
13.35 → 13.4, igual que a mano y que .NET con `decimal`. Todos los valores
redondeados aquí son ≥ 0.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from thehouse_ml.features import (
    Campus,
    FeatureSchema,
    Room,
    StudentProfile,
    check_hard_constraints,
    compute_pair_features,
    default_schema,
    feature_vector,
)
from thehouse_ml.models.baseline_config import BaselineConfig, default_config, schema_is_compatible


class ScoreInputError(ValueError):
    """El vector de features no cuadra con el esquema o la configuración no es compatible."""


@dataclass(frozen=True)
class FactorScore:
    """Puntos de un factor: `max_points * curva(value)`."""

    factor: str
    feature: str
    value: float
    raw_points: float
    points: float
    max_points: float


@dataclass(frozen=True)
class ScoreResult:
    """Puntaje de un par que pasó las restricciones duras."""

    model_version: str
    feature_schema_version: str
    match_score: int
    raw_total: float
    breakdown: tuple[FactorScore, ...]

    @property
    def reported_sum(self) -> float:
        """Suma de los puntos reportados (1 decimal); puede diferir de `match_score` hasta 0.8 (§6.3)."""
        return sum(item.points for item in self.breakdown)

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        """Salida JSON de la spec ML-04 §6.4; con `include_raw`, también los valores sin redondear."""
        breakdown: list[dict[str, Any]] = []
        for item in self.breakdown:
            entry: dict[str, Any] = {
                "factor": item.factor,
                "points": item.points,
                "maxPoints": _json_number(item.max_points),
                "feature": item.feature,
                "value": _json_number(item.value),
            }
            if include_raw:
                entry["rawPoints"] = item.raw_points
            breakdown.append(entry)
        result: dict[str, Any] = {
            "modelVersion": self.model_version,
            "featureSchemaVersion": self.feature_schema_version,
            "matchScore": self.match_score,
        }
        if include_raw:
            result["rawTotal"] = self.raw_total
        result["breakdown"] = breakdown
        return result


@dataclass(frozen=True)
class ExcludedPair:
    """Par excluido por restricciones duras: sin `matchScore` (constitution §11.1)."""

    model_version: str
    feature_schema_version: str
    violated: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "modelVersion": self.model_version,
            "featureSchemaVersion": self.feature_schema_version,
            "excluded": True,
            "violatedConstraints": list(self.violated),
        }


@dataclass(frozen=True)
class ScoredRoom:
    """Candidato para el ranking: el cuarto y su puntaje."""

    room_id: str
    score: ScoreResult


def round_half_up(value: float, decimals: int) -> float:
    """Redondeo mitad hacia arriba como .NET `Math.Round(value, decimals, AwayFromZero)` para `value` ≥ 0."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"round_half_up: se esperaba un número finito ≥ 0, llegó {value!r}")
    if decimals < 0:
        raise ValueError(f"round_half_up: decimals debe ser ≥ 0, llegó {decimals}")
    scale = 10.0**decimals
    scaled = value * scale
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    return whole / scale


def match_score(raw_total: float, config: BaselineConfig | None = None) -> int:
    """`matchScore` = floor(round6(raw_total) + 0.5), recortado a [0, total_points]."""
    config = config or default_config()
    rules = config.rounding
    total = round_half_up(round_half_up(raw_total, rules.total_pre_round_decimals), 0)
    return min(max(int(total), 0), config.total_points)


def factor_points(raw_points: float, config: BaselineConfig | None = None) -> float:
    """Puntos de un factor en el desglose = floor(round6(puntos) * 10 + 0.5) / 10 (M-ML-031)."""
    config = config or default_config()
    rules = config.rounding
    return round_half_up(round_half_up(raw_points, rules.factor_pre_round_decimals), rules.factor_decimals)


def _check_vector(features: Sequence[float], schema: FeatureSchema) -> dict[str, float]:
    if isinstance(features, str | bytes) or not isinstance(features, Sequence):
        raise ScoreInputError("features: se esperaba una secuencia en el orden de feature_order")
    order = schema.feature_order
    if len(features) != len(order):
        raise ScoreInputError(
            f"features: se esperaban {len(order)} valores en el orden de feature_order "
            f"(esquema {schema.version}), llegaron {len(features)}"
        )
    specs = {spec.name: spec for spec in schema.pair_features}
    values: dict[str, float] = {}
    problems: list[str] = []
    for name, raw in zip(order, features, strict=True):
        if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
            problems.append(f"{name}: se esperaba un número finito, llegó {raw!r}")
            continue
        value = float(raw)
        spec = specs[name]
        if not spec.range.contains(value):
            problems.append(f"{name}: {value!r} fuera del rango {spec.range.render()}")
        if spec.type == "integer" and not value.is_integer():
            problems.append(f"{name}: {value!r} no es entero")
        values[name] = value
    if problems:
        raise ScoreInputError("features inválidas:\n- " + "\n- ".join(problems))
    return values


def _check_compatible(config: BaselineConfig, schema: FeatureSchema) -> None:
    if not schema_is_compatible(config.feature_schema_version, schema.version):
        raise ScoreInputError(
            f"{config.model_version} requiere el esquema {config.feature_schema_version} "
            f"(misma versión mayor, igual o posterior) y llegó {schema.version}"
        )
    missing = [factor.feature for factor in config.factors if factor.feature not in schema.feature_order]
    if missing:
        raise ScoreInputError(f"features de {config.model_version} sin posición en feature_order: {missing}")


def score_features(
    features: Sequence[float],
    config: BaselineConfig | None = None,
    schema: FeatureSchema | None = None,
) -> ScoreResult:
    """Puntaje y desglose a partir de las features del par en el orden de `feature_order`.

    Cada factor vale `peso * curva(feature)` (sin redondear); `raw_total` es su
    suma en el orden de la configuración, de izquierda a derecha. Los puntos del
    desglose son `floor(round6(puntos) * 10 + 0.5) / 10` y `matchScore` es
    `floor(round6(raw_total) + 0.5)`. Lanza `ScoreInputError` si el vector no
    tiene la longitud del esquema, trae un valor no finito, fuera de rango o no
    entero en una feature `integer`, o si la configuración no es compatible.
    """
    config = config or default_config()
    schema = schema or default_schema()
    _check_compatible(config, schema)
    values = _check_vector(features, schema)
    breakdown: list[FactorScore] = []
    raw_total = 0.0
    for factor in config.factors:
        value = values[factor.feature]
        raw_points = factor.weight * factor.curve.evaluate(value)
        raw_total += raw_points
        breakdown.append(
            FactorScore(
                factor=factor.id,
                feature=factor.feature,
                value=value,
                raw_points=raw_points,
                points=factor_points(raw_points, config),
                max_points=factor.weight,
            )
        )
    return ScoreResult(
        model_version=config.model_version,
        feature_schema_version=schema.version,
        match_score=match_score(raw_total, config),
        raw_total=raw_total,
        breakdown=tuple(breakdown),
    )


def score_pair(
    student: StudentProfile,
    room: Room,
    campuses: Mapping[str, Campus],
    config: BaselineConfig | None = None,
    schema: FeatureSchema | None = None,
) -> ScoreResult | ExcludedPair:
    """Puntaje de un par estudiante-cuarto, o las restricciones duras que viola.

    Reutiliza `check_hard_constraints`, `compute_pair_features` y
    `feature_vector`; no repite ninguna regla. Propaga `IncompleteProfileError`
    y `PairInputError` de esas funciones.
    """
    config = config or default_config()
    schema = schema or default_schema()
    _check_compatible(config, schema)
    constraints = check_hard_constraints(student, room, schema)
    if not constraints.allowed:
        return ExcludedPair(
            model_version=config.model_version,
            feature_schema_version=schema.version,
            violated=constraints.violated,
        )
    features = compute_pair_features(student, room, campuses, schema)
    return score_features(feature_vector(features, schema), config, schema)


def rank(candidates: Iterable[ScoredRoom]) -> list[ScoredRoom]:
    """Candidatos por `raw_total` descendente; empates por `room_id` ascendente (M-ML-029)."""
    return sorted(candidates, key=lambda item: (-item.score.raw_total, item.room_id))


def _json_number(value: float) -> int | float:
    """Entero si el valor es entero (30, 1); si no, el float sin redondear."""
    return int(value) if value.is_integer() else value


__all__ = [
    "ExcludedPair",
    "FactorScore",
    "ScoreInputError",
    "ScoreResult",
    "ScoredRoom",
    "factor_points",
    "match_score",
    "rank",
    "round_half_up",
    "score_features",
    "score_pair",
]
