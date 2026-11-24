"""Puntaje 0-100 y desglose por factor de la regresión logística (ML-05 T05, ML-05-R05).

Implementación de referencia, pura (sin I/O, red ni aleatoriedad), de la spec ML-05 §6.6
(M-ML-036 decisión 5). Parte de `models/logreg.json` de T02 (no reentrena nada):

1. **Recorte:** cada feature se recorta a su `training_range` (o `range`) con los límites
   guardados en el modelo (`LogRegModel.clip_lower`/`clip_upper`, los mismos de T02).
2. **Contribución lineal:** con la transformación `standard`, el logit es
   `b0 + Σ_j coef_j · (x_j - mean_j) / scale_j`; cada término depende de una sola feature.
3. **Factores:** los términos se agrupan en los 6 factores de `factor_groups.FACTOR_GROUPS`
   (mismos `id` que `baseline-1.0.0`). Las features fuera de los grupos
   (`pets_compatible`, constante 1 en los candidatos) van al **término fijo** junto con `b0`.
4. **Calibración afín:** para cada factor, su contribución mínima y máxima dentro del
   recorte definen [0, `maxPoints`]; con `S = Σ_k (max_k - min_k)`,
   `puntos_k = 100 · (C_k - min_k) / S` y `maxPoints_k = 100 · (max_k - min_k) / S`, así
   que Σ `maxPoints` = 100 y el total es una transformación afín creciente del logit:
   `logit = fijo + Σ_k min_k + S · raw_total / 100`. Cada término se calcula como
   `|coef_j / scale_j| · distancia de x_j al extremo que minimiza el término` (≥ 0 y ≤ su
   rango en binary64), de modo que cada factor queda en [0, `maxPoints_k`] sin recortes.
5. **Redondeo:** el de M-ML-029/031 con las funciones de `baseline_score` (`factor_points`
   y `match_score`): round6 y 1 decimal mitad hacia arriba por factor; total entero
   `floor(round6(Σ) + 0.5)`; `raw_total` es la suma de los factores de izquierda a derecha.

La salida tiene el formato del baseline (`modelVersion`, `featureSchemaVersion`,
`matchScore`, `breakdown`) con `maxPoints` por factor de la calibración y, en los factores
con varias features, una lista `features` en lugar de `feature`/`value` (spec §6.6).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
from thehouse_ml.models.baseline_config import BaselineConfig, default_config
from thehouse_ml.models.baseline_score import ExcludedPair, _check_vector, factor_points, match_score
from thehouse_ml.models.factor_groups import FACTOR_GROUPS, OUTSIDE_GROUPS, FactorGroup, check_groups
from thehouse_ml.models.train import LogRegModel

LOGREG_MODEL_VERSION = "logreg-1.0.0"
FULL_SCALE = 100.0
SUPPORTED_TRANSFORM = "standard"
MONOTONE_TOLERANCE = 1e-9  # puntos; diferencias menores son ruido de binary64
AFFINE_TOLERANCE = 1e-9  # logit; |logit - afín(raw_total)| máximo admitido

# Dirección esperada de cada feature respecto a su variable natural (spec §6.6, «hallazgo de monotonía»):
# +1 = más valor debe sumar (o no restar) puntos; -1 = más valor no debe sumar puntos.
EXPECTED_DIRECTION: dict[str, int] = {
    "price_to_budget": -1,  # más renta → no más puntos
    "price_below_min": +1,  # más alto = renta más baja: con más renta baja, así que no debe sumar al subir la renta
    "distance_km": -1,
    "effective_distance_km": -1,
    "distance_ratio": -1,
    "services_coverage": +1,
    "place_type_match": +1,
    "property_type_match": +1,
    "extra_services_ratio": +1,
}
DIRECTION_NOTES: dict[str, str] = {
    "price_to_budget": "renta / budget.max: sube con la renta",
    "price_below_min": "(budget.min - renta) / budget.min: baja al subir la renta",
    "distance_km": "distancia real al campus",
    "effective_distance_km": "distancia x factor de transporte",
    "distance_ratio": "distancia / maxDistanceKm",
    "services_coverage": "servicios esenciales cubiertos",
    "place_type_match": "coincide el tipo de lugar",
    "property_type_match": "coincide el tipo de inmueble",
    "extra_services_ratio": "servicios no pedidos que ofrece el cuarto",
}


# Desviaciones conocidas aceptadas por el responsable (M-ML-041, APROBADO 2026-11-24, hallazgo T05-H1): una
# comprobación de monotonía que falla y está aquí no bloquea la condición 3; cualquier otra sí. Se decidió después
# de ver el resultado de esa verificación adicional (no forma parte del criterio de la spec §6.5, que exige el
# desglose aditivo que suma el total). Corrección prevista: `logreg-1.1.0`, evaluada con datos nuevos en ML-07.
ACCEPTED_DEVIATIONS: dict[str, str] = {
    "monotonicity_budget": (
        "M-ML-041 (T05-H1): el coeficiente de price_below_min es negativo; por debajo de budget.min el factor "
        "presupuesto sube con la renta solo si budget.max / budget.min > 76.2 (máximo +0.124 puntos sobre 100; "
        "ningún par observado en esa zona). Aceptada por el responsable después de ver la verificación; "
        "corrección en logreg-1.1.0 (ML-07)"
    ),
}


class BreakdownError(ValueError):
    """El modelo no admite el desglose aditivo calibrado (transformación, orden o grupos)."""


# ---------------------------------------------------------------------------
# Calibración
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureTerm:
    """Término lineal de una feature: `coef · (clip(x) - mean) / scale` dentro de [lower, upper]."""

    name: str
    index: int
    coef: float
    mean: float
    scale: float
    lower: float
    upper: float

    @property
    def slope(self) -> float:
        """Cambio del logit por unidad de la feature (`coef / scale`)."""
        return self.coef / self.scale

    def contribution(self, value: float) -> float:
        """Término del logit en `value` (recortado)."""
        clipped = min(max(value, self.lower), self.upper)
        return self.coef * (clipped - self.mean) / self.scale

    @property
    def min_contribution(self) -> float:
        return min(self.contribution(self.lower), self.contribution(self.upper))

    @property
    def max_contribution(self) -> float:
        return max(self.contribution(self.lower), self.contribution(self.upper))

    @property
    def span(self) -> float:
        """Rango del término en logit: `|slope| · (upper - lower)` (misma fórmula que `gain`)."""
        return abs(self.slope) * (self.upper - self.lower)

    def gain(self, clipped: np.ndarray) -> np.ndarray:
        """Término menos su mínimo, ≥ 0 y ≤ `span`: `|slope|` por la distancia al extremo que lo minimiza."""
        distance = clipped - self.lower if self.slope >= 0 else self.upper - clipped
        result: np.ndarray = abs(self.slope) * distance
        return result


@dataclass(frozen=True)
class FactorCalibration:
    """Un factor del desglose: sus términos y su rango de contribución dentro del recorte."""

    id: str
    label: str
    terms: tuple[FeatureTerm, ...]
    min_logit: float
    max_logit: float
    span: float  # Σ span de los términos (logit)
    raw_max_points: float  # 100 · span / S
    max_points: float  # raw_max_points con el redondeo por factor (1 decimal)

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(term.name for term in self.terms)


@dataclass(frozen=True)
class BreakdownModel:
    """Regresión logística calibrada a 0-100 con desglose por los 6 factores."""

    model_version: str
    feature_schema_version: str
    logreg: LogRegModel
    factors: tuple[FactorCalibration, ...]
    fixed_terms: tuple[FeatureTerm, ...]
    intercept: float
    total_span: float  # S = Σ_k span_k (logit)
    rounding: BaselineConfig

    @property
    def min_logit_sum(self) -> float:
        """Σ_k min_k: logit de los factores cuando todos valen 0 puntos."""
        return sum(factor.min_logit for factor in self.factors)

    def clip(self, features: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(np.asarray(features, dtype=np.float64))
        return np.clip(values, self.logreg.clip_lower, self.logreg.clip_upper)

    def raw_points(self, features: np.ndarray) -> np.ndarray:
        """Puntos sin redondear (filas x 6 factores) de vectores en el orden de `feature_order`."""
        clipped = self.clip(features)
        columns: list[np.ndarray] = []
        for factor in self.factors:
            gain = np.zeros(len(clipped), dtype=np.float64)
            for term in factor.terms:
                gain = gain + term.gain(clipped[:, term.index])
            columns.append(FULL_SCALE * gain / self.total_span)
        return np.column_stack(columns)

    def raw_total(self, points: np.ndarray) -> np.ndarray:
        """Suma de los factores de izquierda a derecha en binary64 (M-ML-031 decisión 3)."""
        total = np.zeros(len(points), dtype=np.float64)
        for column in range(points.shape[1]):
            total = total + points[:, column]
        return total

    def fixed_term(self, features: np.ndarray) -> np.ndarray:
        """`b0` + términos de las features fuera de los factores (`pets_compatible`)."""
        clipped = self.clip(features)
        fixed = np.full(len(clipped), self.intercept, dtype=np.float64)
        for term in self.fixed_terms:
            fixed = fixed + term.coef * (clipped[:, term.index] - term.mean) / term.scale
        return fixed

    def affine_logit(self, raw_total: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        """Logit reconstruido desde el puntaje: `fijo + Σ min_k + S · raw_total / 100`."""
        result: np.ndarray = fixed + self.min_logit_sum + self.total_span * raw_total / FULL_SCALE
        return result

    def as_dict(self) -> dict[str, Any]:
        """Parámetros de la calibración (para reportes, ML-06 y NV-23)."""
        return {
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "transform": self.logreg.config.transform,
            "method": "afín por factor sobre el recorte: puntos_k = 100 · (C_k - min_k) / S",
            "total_span_logit": self.total_span,
            "intercept": self.intercept,
            "min_logit_sum": self.min_logit_sum,
            "factors": [
                {
                    "factor": factor.id,
                    "label": factor.label,
                    "features": list(factor.features),
                    "min_logit": factor.min_logit,
                    "max_logit": factor.max_logit,
                    "span_logit": factor.span,
                    "raw_max_points": factor.raw_max_points,
                    "maxPoints": factor.max_points,
                }
                for factor in self.factors
            ],
            "fixed_terms": [
                {
                    "feature": term.name,
                    "coef": term.coef,
                    "constant_in_candidates": True,
                    "reason": OUTSIDE_GROUPS.get(term.name, ""),
                }
                for term in self.fixed_terms
            ],
            "raw_max_points_sum": sum(factor.raw_max_points for factor in self.factors),
            "max_points_sum": sum(factor.max_points for factor in self.factors),
        }


def _terms(logreg: LogRegModel) -> dict[str, FeatureTerm]:
    if logreg.config.transform != SUPPORTED_TRANSFORM or logreg.mean is None or logreg.scale is None:
        raise BreakdownError(
            f"la calibración de T05 está definida para la transformación {SUPPORTED_TRANSFORM!r} "
            f"(la elegida en T02); el modelo usa {logreg.config.transform!r}"
        )
    return {
        name: FeatureTerm(
            name=name,
            index=index,
            coef=float(logreg.coef[index]),
            mean=float(logreg.mean[index]),
            scale=float(logreg.scale[index]),
            lower=float(logreg.clip_lower[index]),
            upper=float(logreg.clip_upper[index]),
        )
        for index, name in enumerate(logreg.feature_order)
    }


def build_breakdown_model(
    logreg: LogRegModel,
    schema: FeatureSchema | None = None,
    rounding: BaselineConfig | None = None,
    groups: Sequence[FactorGroup] = FACTOR_GROUPS,
    model_version: str = LOGREG_MODEL_VERSION,
) -> BreakdownModel:
    """Calibra la regresión logística de T02 a 0-100 por factor (spec §6.6); no reentrena nada."""
    schema = schema or default_schema()
    rounding = rounding or default_config()
    if logreg.feature_order != schema.feature_order:
        raise BreakdownError("feature_order del modelo distinto del esquema")
    check_groups(schema.feature_order, groups)
    terms = _terms(logreg)
    spans = {group.id: sum(terms[name].span for name in group.features) for group in groups}
    total_span = sum(spans.values())
    if not math.isfinite(total_span) or total_span <= 0:
        raise BreakdownError("todos los factores tienen contribución constante: no hay escala 0-100")
    factors: list[FactorCalibration] = []
    for group in groups:
        group_terms = tuple(terms[name] for name in group.features)
        raw_max = FULL_SCALE * spans[group.id] / total_span
        factors.append(
            FactorCalibration(
                id=group.id,
                label=group.label,
                terms=group_terms,
                min_logit=sum(term.min_contribution for term in group_terms),
                max_logit=sum(term.max_contribution for term in group_terms),
                span=spans[group.id],
                raw_max_points=raw_max,
                max_points=factor_points(raw_max, rounding),
            )
        )
    grouped = {name for group in groups for name in group.features}
    fixed = tuple(terms[name] for name in schema.feature_order if name not in grouped)
    return BreakdownModel(
        model_version=model_version,
        feature_schema_version=schema.version,
        logreg=logreg,
        factors=tuple(factors),
        fixed_terms=fixed,
        intercept=logreg.intercept,
        total_span=total_span,
        rounding=rounding,
    )


# ---------------------------------------------------------------------------
# Puntaje de un par
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogRegFactorScore:
    """Puntos de un factor: `100 · (C_k - min_k) / S`, con sus features y valores de entrada."""

    factor: str
    features: tuple[tuple[str, float], ...]
    raw_points: float
    points: float
    raw_max_points: float
    max_points: float


@dataclass(frozen=True)
class LogRegScoreResult:
    """Puntaje 0-100 y desglose de un par que pasó las restricciones duras."""

    model_version: str
    feature_schema_version: str
    match_score: int
    raw_total: float
    logit: float
    breakdown: tuple[LogRegFactorScore, ...]

    @property
    def reported_sum(self) -> float:
        return sum(item.points for item in self.breakdown)

    @property
    def probability(self) -> float:
        """Sigmoide del logit: dato informativo; no es el puntaje porque no es aditiva (spec §6.6)."""
        return 1.0 / (1.0 + math.exp(-self.logit))

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        """Salida con el formato del baseline (ML-04 §6.4) y `features` en los factores de varias features."""
        breakdown: list[dict[str, Any]] = []
        for item in self.breakdown:
            entry: dict[str, Any] = {
                "factor": item.factor,
                "points": item.points,
                "maxPoints": _json_number(item.max_points),
            }
            if len(item.features) == 1:
                name, value = item.features[0]
                entry["feature"] = name
                entry["value"] = _json_number(value)
            else:
                entry["features"] = [{"feature": name, "value": _json_number(value)} for name, value in item.features]
            if include_raw:
                entry["rawPoints"] = item.raw_points
                entry["rawMaxPoints"] = item.raw_max_points
            breakdown.append(entry)
        result: dict[str, Any] = {
            "modelVersion": self.model_version,
            "featureSchemaVersion": self.feature_schema_version,
            "matchScore": self.match_score,
        }
        if include_raw:
            result["rawTotal"] = self.raw_total
            result["logit"] = self.logit
            result["probability"] = self.probability
        result["breakdown"] = breakdown
        return result


def _json_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def score_matrix(model: BreakdownModel, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Puntos sin redondear (n x 6), `raw_total` y logit afín para muchos pares (verificación en test)."""
    points = model.raw_points(features)
    total = model.raw_total(points)
    return points, total, model.affine_logit(total, model.fixed_term(features))


def score_features_logreg(
    features: Sequence[float], model: BreakdownModel, schema: FeatureSchema | None = None
) -> LogRegScoreResult:
    """Puntaje 0-100 y desglose de un vector en el orden de `feature_order` (valida como el baseline)."""
    schema = schema or default_schema()
    if schema.version != model.feature_schema_version:
        raise BreakdownError(
            f"{model.model_version} usa el esquema {model.feature_schema_version}, llegó {schema.version}"
        )
    values = _check_vector(features, schema)
    row = np.asarray([[values[name] for name in schema.feature_order]], dtype=np.float64)
    points, total, logit = score_matrix(model, row)
    raw_total = float(total[0])
    breakdown = tuple(
        LogRegFactorScore(
            factor=factor.id,
            features=tuple((name, values[name]) for name in factor.features),
            raw_points=float(points[0, index]),
            points=factor_points(float(points[0, index]), model.rounding),
            raw_max_points=factor.raw_max_points,
            max_points=factor.max_points,
        )
        for index, factor in enumerate(model.factors)
    )
    return LogRegScoreResult(
        model_version=model.model_version,
        feature_schema_version=model.feature_schema_version,
        match_score=match_score(raw_total, model.rounding),
        raw_total=raw_total,
        logit=float(logit[0]),
        breakdown=breakdown,
    )


def score_pair_logreg(
    student: StudentProfile,
    room: Room,
    campuses: Mapping[str, Campus],
    model: BreakdownModel,
    schema: FeatureSchema | None = None,
) -> LogRegScoreResult | ExcludedPair:
    """Como `baseline_score.score_pair`: perfil incompleto → `IncompleteProfileError`; HC01-HC03 → excluido."""
    schema = schema or default_schema()
    constraints = check_hard_constraints(student, room, schema)
    if not constraints.allowed:
        return ExcludedPair(
            model_version=model.model_version,
            feature_schema_version=schema.version,
            violated=constraints.violated,
        )
    features = compute_pair_features(student, room, campuses, schema)
    return score_features_logreg(feature_vector(features, schema), model, schema)


@dataclass(frozen=True)
class RankedRoom:
    room_id: str
    score: LogRegScoreResult


def rank_logreg(candidates: Iterable[RankedRoom]) -> list[RankedRoom]:
    """Por `raw_total` descendente y `room_id` ascendente (misma regla que el baseline)."""
    return sorted(candidates, key=lambda item: (-item.score.raw_total, item.room_id))


# ---------------------------------------------------------------------------
# Verificación sobre todos los pares de test (condición 3 de la spec §6.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SumCheck:
    """El desglose suma el total, con las tolerancias y rangos de M-ML-029/031."""

    pairs: int
    max_total_gap: float  # max |raw_total - matchScore|
    max_reported_gap: float  # max |Σ puntos reportados - raw_total|
    total_violations: int
    reported_violations: int
    factor_out_of_range: int
    score_out_of_range: int
    max_affine_gap: float  # max |logit - afín(raw_total)|
    total_tolerance: float
    breakdown_tolerance: float

    @property
    def passed(self) -> bool:
        return (
            self.pairs > 0
            and self.total_violations == 0
            and self.reported_violations == 0
            and self.factor_out_of_range == 0
            and self.score_out_of_range == 0
            and self.max_affine_gap <= AFFINE_TOLERANCE
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "pairs": self.pairs,
            "max_abs_raw_total_minus_matchScore": self.max_total_gap,
            "max_abs_reported_sum_minus_raw_total": self.max_reported_gap,
            "total_tolerance": self.total_tolerance,
            "breakdown_tolerance": self.breakdown_tolerance,
            "total_tolerance_violations": self.total_violations,
            "breakdown_tolerance_violations": self.reported_violations,
            "factors_outside_0_maxPoints": self.factor_out_of_range,
            "matchScore_outside_0_100": self.score_out_of_range,
            "max_abs_logit_minus_affine": self.max_affine_gap,
            "affine_tolerance": AFFINE_TOLERANCE,
        }


@dataclass(frozen=True)
class OrderCheck:
    """El orden por `raw_total` (desempate `room_id`) es el del logit en cada estudiante."""

    students: int
    identical: int
    differing: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.students > 0 and self.identical == self.students

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "students": self.students,
            "identical_order": self.identical,
            "differing_students": list(self.differing[:10]),
        }


@dataclass(frozen=True)
class ScoredPairs:
    """Puntaje de cada par evaluado (alineado con las features de entrada)."""

    raw_points: np.ndarray
    points: np.ndarray
    raw_total: np.ndarray
    match_score: np.ndarray


def score_pairs(model: BreakdownModel, features: np.ndarray) -> ScoredPairs:
    """Puntos, totales y `matchScore` de todos los pares con el redondeo de `baseline_score`."""
    raw_points, raw_total, _ = score_matrix(model, features)
    points = np.vectorize(lambda value: factor_points(float(value), model.rounding), otypes=[np.float64])(raw_points)
    scores = np.asarray([match_score(float(value), model.rounding) for value in raw_total], dtype=np.int64)
    return ScoredPairs(raw_points=raw_points, points=points, raw_total=raw_total, match_score=scores)


def check_sums(model: BreakdownModel, features: np.ndarray, logit: np.ndarray, scored: ScoredPairs) -> SumCheck:
    rules = model.rounding.rounding
    total_gap = np.abs(scored.raw_total - scored.match_score)
    reported = np.zeros(len(scored.points), dtype=np.float64)
    for column in range(scored.points.shape[1]):
        reported = reported + scored.points[:, column]
    reported_gap = np.abs(reported - scored.raw_total)
    raw_max = np.asarray([factor.raw_max_points for factor in model.factors])
    rounded_max = np.asarray([factor.max_points for factor in model.factors])
    outside = (
        (scored.raw_points < 0) | (scored.raw_points > raw_max) | (scored.points < 0) | (scored.points > rounded_max)
    )
    affine = model.affine_logit(scored.raw_total, model.fixed_term(features))
    total_points = model.rounding.total_points
    return SumCheck(
        pairs=len(scored.raw_total),
        max_total_gap=float(total_gap.max(initial=0.0)),
        max_reported_gap=float(reported_gap.max(initial=0.0)),
        total_violations=int((total_gap > rules.total_tolerance).sum()),
        reported_violations=int((reported_gap > rules.breakdown_tolerance + 1e-12).sum()),
        factor_out_of_range=int(outside.any(axis=1).sum()),
        score_out_of_range=int(((scored.match_score < 0) | (scored.match_score > total_points)).sum()),
        max_affine_gap=float(np.abs(np.asarray(logit, dtype=np.float64) - affine).max(initial=0.0)),
        total_tolerance=rules.total_tolerance,
        breakdown_tolerance=rules.breakdown_tolerance,
    )


def check_order(raw_total: np.ndarray, logit: np.ndarray, group: np.ndarray, room_id: np.ndarray) -> OrderCheck:
    """Compara, por estudiante, el orden (-raw_total, room_id) con el orden (-logit, room_id)."""
    rows: dict[str, list[int]] = {}
    for position, student in enumerate(group.tolist()):
        rows.setdefault(str(student), []).append(position)
    differing: list[str] = []
    for student in sorted(rows):
        positions = rows[student]
        by_score = sorted(positions, key=lambda i: (-float(raw_total[i]), str(room_id[i])))
        by_logit = sorted(positions, key=lambda i: (-float(logit[i]), str(room_id[i])))
        if by_score != by_logit:
            differing.append(student)
    return OrderCheck(students=len(rows), identical=len(rows) - len(differing), differing=tuple(differing))


# ---------------------------------------------------------------------------
# Monotonía (hallazgo obligatorio de T05)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoefficientRow:
    feature: str
    factor: str | None
    coef: float
    scale: float
    slope: float
    expected: int | None

    @property
    def consistent(self) -> bool | None:
        """Signo del término coherente con la dirección esperada (`None` si no hay expectativa o es 0)."""
        if self.expected is None:
            return None
        return bool(self.slope == 0 or (self.slope > 0) == (self.expected > 0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "factor": self.factor,
            "coef_standardized": self.coef,
            "scale": self.scale,
            "logit_per_unit": self.slope,
            "expected_direction": self.expected,
            "sign_consistent": self.consistent,
            "note": DIRECTION_NOTES.get(self.feature, OUTSIDE_GROUPS.get(self.feature, "")),
        }


def coefficient_table(model: BreakdownModel) -> tuple[CoefficientRow, ...]:
    factor_of = {term.name: factor.id for factor in model.factors for term in factor.terms}
    terms = _terms(model.logreg)
    return tuple(
        CoefficientRow(
            feature=name,
            factor=factor_of.get(name),
            coef=terms[name].coef,
            scale=terms[name].scale,
            slope=terms[name].slope,
            expected=EXPECTED_DIRECTION.get(name),
        )
        for name in model.logreg.feature_order
    )


@dataclass(frozen=True)
class MonotonicityCheck:
    """Rejilla de un factor a lo largo de su variable natural (renta o distancia)."""

    factor: str
    variable: str
    domain: str
    curves: int
    steps: int
    violations: int
    worst_increase: float
    worst_case: Mapping[str, Any] | None

    @property
    def passed(self) -> bool:
        return self.curves > 0 and self.violations == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "factor": self.factor,
            "variable": self.variable,
            "domain": self.domain,
            "curves": self.curves,
            "steps": self.steps,
            "violations": self.violations,
            "worst_increase_points": self.worst_increase,
            "worst_case": dict(self.worst_case) if self.worst_case else None,
            "tolerance_points": MONOTONE_TOLERANCE,
        }


GRID_CAMPUS = Campus(code="UTTT", lat=20.0, lon=-99.0)
KM_PER_DEGREE_LAT = 6371.0 * math.pi / 180.0
BUDGET_MINS: tuple[int | None, ...] = (
    None,
    1,
    10,
    50,
    100,
    300,
    500,
    1000,
    1500,
    2000,
    3000,
    5000,
    10000,
    25000,
    50000,
)
BUDGET_MAXES: tuple[int, ...] = (500, 1000, 2000, 2700, 3500, 5000, 10000, 25000, 50000)
RENT_STEPS = 101
MAX_DISTANCES: tuple[float | None, ...] = (None, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)
DISTANCE_STEP_KM = 0.25
DISTANCE_LIMIT_KM = 36.0  # más allá del training_range (30 km): comprueba también el recorte


def _grid_student(
    budget_min: int | None, budget_max: int, max_distance: float | None, transport: str
) -> StudentProfile:
    return StudentProfile(
        university_code=GRID_CAMPUS.code,
        budget_min=budget_min,
        budget_max=budget_max,
        max_distance_km=max_distance,
        essential_services=frozenset({"wifi"}),
        living_with="any",
        place_type="room",
        pets=False,
        transport=transport,
        property_types=None,
    )


def _grid_room(price: int, distance_km: float) -> Room:
    return Room(
        price_mxn=price,
        lat=GRID_CAMPUS.lat + distance_km / KM_PER_DEGREE_LAT,
        lon=GRID_CAMPUS.lon,
        services=frozenset({"wifi", "water"}),
        place_type="room",
        property_type="house",
        pets_allowed=True,
        living_with="any",
        bedrooms=1,
        bathrooms=1,
        area_m2=12.0,
        university_code=GRID_CAMPUS.code,
        synthetic=True,
    )


def _sweep(
    model: BreakdownModel,
    schema: FeatureSchema,
    factor_id: str,
    curves: Iterable[tuple[dict[str, Any], list[tuple[float, StudentProfile, Room]]]],
    variable: str,
    domain: str,
) -> MonotonicityCheck:
    """Factor a lo largo de cada curva (variable creciente): no debe subir más de `MONOTONE_TOLERANCE`."""
    column = next(index for index, factor in enumerate(model.factors) if factor.id == factor_id)
    campuses = {GRID_CAMPUS.code: GRID_CAMPUS}
    n_curves = steps = violations = 0
    worst = 0.0
    worst_case: dict[str, Any] | None = None
    for label, points in curves:
        vectors = [feature_vector(compute_pair_features(s, r, campuses, schema), schema) for _x, s, r in points]
        factor = model.raw_points(np.asarray(vectors, dtype=np.float64))[:, column]
        n_curves += 1
        steps += len(points) - 1
        increases = np.diff(factor)
        bad = np.flatnonzero(increases > MONOTONE_TOLERANCE)
        violations += len(bad)
        if len(bad):
            index = int(bad[np.argmax(increases[bad])])
            if increases[index] > worst:
                worst = float(increases[index])
                worst_case = {
                    **label,
                    f"{variable}_from": points[index][0],
                    f"{variable}_to": points[index + 1][0],
                    "points_from": float(factor[index]),
                    "points_to": float(factor[index + 1]),
                }
    return MonotonicityCheck(factor_id, variable, domain, n_curves, steps, violations, worst, worst_case)


def budget_monotonicity(model: BreakdownModel, schema: FeatureSchema | None = None) -> MonotonicityCheck:
    """Factor presupuesto al subir la renta de 0 a `budget.max` (pares que pasan HC01; p2b en [0, 1])."""
    schema = schema or default_schema()
    curves: list[tuple[dict[str, Any], list[tuple[float, StudentProfile, Room]]]] = []
    for budget_max in BUDGET_MAXES:
        for budget_min in BUDGET_MINS:
            if budget_min is not None and budget_min > budget_max:
                continue
            rents = {round(float(value)) for value in np.linspace(0, budget_max, RENT_STEPS)}
            if budget_min is not None:
                rents |= {rent for rent in (budget_min - 1, budget_min, budget_min + 1) if 0 <= rent <= budget_max}
            student = _grid_student(budget_min, budget_max, None, "public")
            points = [(float(rent), student, _grid_room(rent, 1.0)) for rent in sorted(rents)]
            curves.append(({"budget.min": budget_min, "budget.max": budget_max}, points))
    domain = (
        f"NV-11: budget.min ∈ {{nulo, {', '.join(str(v) for v in BUDGET_MINS[1:])}}} ≤ budget.max ∈ "
        f"{{{', '.join(str(v) for v in BUDGET_MAXES)}}}; renta entera de 0 a budget.max ({RENT_STEPS} pasos más "
        "budget.min ± 1): price_to_budget recorre su training_range [0, 1] y price_below_min su range [0, 1]"
    )
    return _sweep(model, schema, "budget", curves, "rent_mxn", domain)


def distance_monotonicity(model: BreakdownModel, schema: FeatureSchema | None = None) -> MonotonicityCheck:
    """Factor distancia al alejar el cuarto de 0 a 36 km, por transporte y `maxDistanceKm`."""
    schema = schema or default_schema()
    transports = schema.catalog("transport")
    distances = np.arange(0.0, DISTANCE_LIMIT_KM + DISTANCE_STEP_KM / 2, DISTANCE_STEP_KM)
    curves: list[tuple[dict[str, Any], list[tuple[float, StudentProfile, Room]]]] = []
    for transport in transports:
        for max_distance in MAX_DISTANCES:
            student = _grid_student(None, 3000, max_distance, transport)
            points = [(float(d), student, _grid_room(1500, float(d))) for d in distances]
            curves.append(({"transport": transport, "maxDistanceKm": max_distance}, points))
    domain = (
        f"transporte ∈ {{{', '.join(transports)}}} x maxDistanceKm ∈ {{nulo→30, "
        f"{', '.join(f'{v:g}' for v in MAX_DISTANCES[1:])}}}; distancia de 0 a {DISTANCE_LIMIT_KM:g} km cada "
        f"{DISTANCE_STEP_KM:g} km (incluye > 30 km para comprobar el recorte)"
    )
    return _sweep(model, schema, "distance", curves, "distance_km", domain)


def budget_ratio_threshold(model: BreakdownModel) -> float | None:
    """`budget.max / budget.min` a partir del cual el factor presupuesto sube con la renta (renta < budget.min).

    Con renta < budget.min, d(factor)/d(renta) ∝ slope(p2b) / budget.max - slope(pbm) / budget.min;
    es > 0 si `slope(pbm) < 0` y `budget.max / budget.min > slope(p2b) / slope(pbm)`. `None` si no existe.
    """
    terms = _terms(model.logreg)
    p2b, pbm = terms["price_to_budget"].slope, terms["price_below_min"].slope
    if pbm >= 0:
        return None
    if p2b >= 0:
        return 0.0
    return p2b / pbm


def observed_budget_ratio(features: np.ndarray, feature_order: Sequence[str]) -> tuple[float | None, int]:
    """Mayor `budget.max / budget.min` implícito en los pares con renta < budget.min: (1 - pbm) / p2b."""
    p2b = np.asarray(features, dtype=np.float64)[:, list(feature_order).index("price_to_budget")]
    pbm = np.asarray(features, dtype=np.float64)[:, list(feature_order).index("price_below_min")]
    mask = (pbm > 0) & (p2b > 0)
    if not mask.any():
        return None, 0
    return float(((1.0 - pbm[mask]) / p2b[mask]).max()), int(mask.sum())


@dataclass(frozen=True)
class MonotonicityResult:
    coefficients: tuple[CoefficientRow, ...]
    budget: MonotonicityCheck
    distance: MonotonicityCheck
    budget_ratio_threshold: float | None
    observed_ratio: float | None
    observed_pairs_below_min: int

    @property
    def passed(self) -> bool:
        return self.budget.passed and self.distance.passed

    @property
    def observed_violations(self) -> bool | None:
        """¿Algún par observado cae en la zona donde el presupuesto sube con la renta?"""
        if self.budget_ratio_threshold is None or self.observed_ratio is None:
            return False if self.budget_ratio_threshold is None else None
        return self.observed_ratio > self.budget_ratio_threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "coefficients": [row.as_dict() for row in self.coefficients],
            "sign_inconsistent_features": [row.feature for row in self.coefficients if row.consistent is False],
            "budget": self.budget.as_dict(),
            "distance": self.distance.as_dict(),
            "budget_ratio_threshold": self.budget_ratio_threshold,
            "observed_max_budget_ratio": self.observed_ratio,
            "observed_pairs_below_budget_min": self.observed_pairs_below_min,
            "observed_pairs_in_violation_zone": self.observed_violations,
        }


def check_monotonicity(
    model: BreakdownModel, observed_features: np.ndarray, schema: FeatureSchema | None = None
) -> MonotonicityResult:
    schema = schema or default_schema()
    ratio, below = observed_budget_ratio(observed_features, schema.feature_order)
    return MonotonicityResult(
        coefficients=coefficient_table(model),
        budget=budget_monotonicity(model, schema),
        distance=distance_monotonicity(model, schema),
        budget_ratio_threshold=budget_ratio_threshold(model),
        observed_ratio=ratio,
        observed_pairs_below_min=below,
    )


# ---------------------------------------------------------------------------
# Resultado completo (condición 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BreakdownEvaluation:
    """Desglose verificado sobre todos los pares de la partición evaluada y monotonía por factor."""

    model: BreakdownModel
    scored: ScoredPairs
    sums: SumCheck
    order: OrderCheck
    metrics_unchanged: bool
    monotonicity: MonotonicityResult

    accepted_deviations: Mapping[str, str] = field(default_factory=lambda: dict(ACCEPTED_DEVIATIONS))

    @property
    def sums_total(self) -> bool:
        """Criterio de la spec §6.5 (condición 3): el desglose suma el total, con rangos, orden y métricas."""
        return self.sums.passed and self.order.passed and self.metrics_unchanged

    def _checks(self) -> tuple[tuple[str, bool], ...]:
        return (
            ("sums", self.sums.passed),
            ("order", self.order.passed),
            ("metrics_unchanged", self.metrics_unchanged),
            ("monotonicity_budget", self.monotonicity.budget.passed),
            ("monotonicity_distance", self.monotonicity.distance.passed),
        )

    @property
    def failed_checks(self) -> list[str]:
        """Comprobaciones que fallan y no son una desviación aceptada (bloquean la condición 3)."""
        return [name for name, ok in self._checks() if not ok and name not in self.accepted_deviations]

    @property
    def deviations_applied(self) -> list[str]:
        """Comprobaciones que fallan y son desviaciones aceptadas (se reportan, no bloquean)."""
        return [name for name, ok in self._checks() if not ok and name in self.accepted_deviations]

    @property
    def passed(self) -> bool:
        """Condición 3: el desglose suma el total y ninguna comprobación falla fuera de las desviaciones aceptadas."""
        return self.sums_total and not self.failed_checks

    def condition_detail(self) -> dict[str, Any]:
        failed = self.failed_checks
        return {
            "status": "verified_t05",
            "model_version": self.model.model_version,
            "breakdown_sums_total": self.sums.passed,
            "order_identical_to_logit": self.order.passed,
            "metrics_unchanged_by_calibration": self.metrics_unchanged,
            "factor_monotonicity": self.monotonicity.passed,
            "failed_checks": failed,
            "accepted_deviations": {name: self.accepted_deviations[name] for name in self.deviations_applied},
            "pairs_verified": self.sums.pairs,
        }


def evaluate_breakdown(
    model: BreakdownModel,
    features: np.ndarray,
    logit: np.ndarray,
    group: np.ndarray,
    room_id: np.ndarray,
    observed_features: np.ndarray,
    metrics_unchanged: bool,
    schema: FeatureSchema | None = None,
) -> BreakdownEvaluation:
    """Desglose de todos los pares evaluados, sus comprobaciones y la rejilla de monotonía (sin I/O)."""
    scored = score_pairs(model, features)
    return BreakdownEvaluation(
        model=model,
        scored=scored,
        sums=check_sums(model, features, logit, scored),
        order=check_order(scored.raw_total, logit, group, room_id),
        metrics_unchanged=metrics_unchanged,
        monotonicity=check_monotonicity(model, observed_features, schema),
    )


__all__ = [
    "ACCEPTED_DEVIATIONS",
    "LOGREG_MODEL_VERSION",
    "BreakdownError",
    "BreakdownEvaluation",
    "BreakdownModel",
    "CoefficientRow",
    "FactorCalibration",
    "FeatureTerm",
    "LogRegFactorScore",
    "LogRegScoreResult",
    "MonotonicityCheck",
    "MonotonicityResult",
    "OrderCheck",
    "RankedRoom",
    "ScoredPairs",
    "SumCheck",
    "budget_monotonicity",
    "budget_ratio_threshold",
    "build_breakdown_model",
    "check_monotonicity",
    "check_order",
    "check_sums",
    "coefficient_table",
    "distance_monotonicity",
    "evaluate_breakdown",
    "observed_budget_ratio",
    "rank_logreg",
    "score_features_logreg",
    "score_matrix",
    "score_pair_logreg",
    "score_pairs",
]
