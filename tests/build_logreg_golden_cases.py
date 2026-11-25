"""Genera `tests/fixtures/logreg_golden_cases.json` (ML-06 T03, ML-06-R03).

Casos dorados del modelo promovido `logreg-1.0.0` con **recuento independiente**
(spec ML-06 §6.4, como M-ML-033): este script **no importa `thehouse_ml`** (ni
`score_features_logreg`, `score_matrix`, `parity.py` ni `onnx_graph.py`). Reutiliza
el recuento de features, restricciones duras y perfil incompleto de
`tests/golden_recount.py` y las entradas de los 81 casos de
`tests/build_baseline_golden_cases.py`; lee **como datos** `models/logreg.json`
(recorte, media, escala y coeficientes; SHA-256 verificado) y `feature_schema.json`;
y escribe aparte, paso a paso y con `fractions.Fraction` sobre los binary64 del
JSON, la fórmula de M-ML-041:

1. recorte de cada feature a `[clip.lower, clip.upper]`;
2. escalado `z = (x - mean) / scale`;
3. contribución lineal `t = coef · z`;
4. suma por factor `C_k` (los 6 factores de M-ML-040; `pets_compatible` al término fijo);
5. `min_k`/`max_k` (suma de los extremos de cada término en la caja del recorte),
   `S = Σ_k (max_k - min_k)` y calibración `puntos_k = 100 · (C_k - min_k) / S`;
6. redondeo: round6 y 1 decimal mitad hacia arriba por factor; total
   `floor(round6(Σ) + 1/2)` (M-ML-029/031).

Uso: `uv run python tests/build_logreg_golden_cases.py` (reescribe el archivo). Sin
`models/logreg.json` oficial (fuera de git) el script falla: el archivo solo se
genera con `logreg-1.0.0`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import build_baseline_golden_cases as baseline
from golden_recount import (
    FEATURE_SCHEMA,
    HALF,
    INTEGER_FEATURES,
    REPO_ROOT,
    REQUIRED_FIELDS,
    TRANSPORT_FACTORS,
    load_contract,
    recount,
    round6_margin,
    round_half_up,
)

OUTPUT = REPO_ROOT / "tests" / "fixtures" / "logreg_golden_cases.json"
LOGREG_JSON = REPO_ROOT / "models" / "logreg.json"
LOGREG_SHA256 = "341ea3ec705c2f877b36d1b931eafe2dea4630523560fe8764ce59e0a70b27fa"  # spec ML-06 §1
LOGREG_FORMAT = "thehouse-ml/logreg-json/1"
MODEL_VERSION = "logreg-1.0.0"
GENERATED_AT = "2026-11-25"

FEATURE_TOLERANCE = 1e-6
ONNX_TOLERANCE = 1e-4
CASE_TOLERANCE = {"features": FEATURE_TOLERANCE, "value": FEATURE_TOLERANCE, "onnx": ONNX_TOLERANCE, "points": 0}
# Regla de margen (M-ML-033 decisión 3): todo valor no exacto a ≥ 1e-9 de un punto medio de round6
# (en unidades de 10^-6: 1/1000).
MIN_ROUND6_MARGIN = Fraction(1, 1000)
FULL_SCALE = 100
FACTOR_DECIMALS = 1
PRE_ROUND_DECIMALS = 6
# Coherencia del desglose (constitución §12, comprobación de M-ML-041): |Σ sin redondear - matchScore| ≤
# 1/2 + 1/2 x 10^-6 (round6 antes del entero: 80.49999955 → 81) y |Σ puntos redondeados - Σ sin redondear| ≤
# 6 x (1/20 + 1/2 x 10^-6).
MAX_SCORE_GAP = HALF + Fraction(1, 2 * 10**PRE_ROUND_DECIMALS)
MAX_REPORTED_GAP = 6 * (Fraction(1, 20) + Fraction(1, 2 * 10**PRE_ROUND_DECIMALS))

# Los 6 factores del baseline con sus features (M-ML-040 decisión 1), escritos aparte de
# `thehouse_ml.models.factor_groups`. `pets_compatible` no pertenece a ningún factor (M-ML-041 decisión 3).
FACTORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("budget", ("price_to_budget", "price_below_min")),
    ("distance", ("distance_km", "effective_distance_km", "distance_ratio")),
    ("essential_services", ("services_coverage",)),
    ("place_type", ("place_type_match",)),
    ("property_type", ("property_type_match",)),
    ("extras", ("extra_services_ratio",)),
)
FIXED_TERM_FEATURES = ("pets_compatible",)

DOTNET_NOTE = (
    "Paridad en .NET (NV-23): el ONNX devuelve factor_points [N, 6] y raw_total [N] sin redondear; el backend "
    "redondea con round6 ANTES del redondeo final: points = Math.Round(Math.Round(p, 6, "
    "MidpointRounding.AwayFromZero), 1, MidpointRounding.AwayFromZero) y matchScore = "
    "Math.Round(Math.Round(raw_total, 6, MidpointRounding.AwayFromZero), MidpointRounding.AwayFromZero); "
    "maxPoints = maxPointsRaw con la regla de points. Usar siempre MidpointRounding.AwayFromZero: el Math.Round "
    "por omisión (ToEven) NO es compatible (factor_round_budget_4_65: 4.65 → 4.6 en lugar de 4.7; "
    "logreg_total_x5000004_up: 80.5 → 80 en lugar de 81). Después de redondear, normalizar -0 a 0 en points y "
    "matchScore (M-ML-046): el ONNX puede devolver negativos de ~1e-16 en un factor que está en su mínimo "
    "(factor_minimum_place_type) y Math.Round(-2.45e-16, 6, AwayFromZero) da -0.0, que System.Text.Json "
    "escribe como -0."
)


class GoldenError(ValueError):
    """El modelo no es `logreg-1.0.0` o un caso incumple una regla del generador."""


# ---------------------------------------------------------------------------
# Parámetros de logreg-1.0.0 (leídos como datos) y recuento exacto
# ---------------------------------------------------------------------------


def sha256_of(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class LogRegParams:
    """Parámetros de `models/logreg.json` como `Fraction` exactas de sus binary64."""

    sha256: str
    feature_order: tuple[str, ...]
    feature_schema_version: str
    lower: dict[str, Fraction]
    upper: dict[str, Fraction]
    mean: dict[str, Fraction]
    scale: dict[str, Fraction]
    coef: dict[str, Fraction]
    intercept: Fraction

    def term(self, name: str, value: Fraction) -> Fraction:
        """Pasos 2-3: escalado y contribución lineal de una feature ya recortada."""
        z = (value - self.mean[name]) / self.scale[name]
        return self.coef[name] * z

    def term_extremes(self, name: str) -> tuple[Fraction, Fraction]:
        """Mínimo y máximo de la contribución de una feature dentro del recorte."""
        at_lower = self.term(name, self.lower[name])
        at_upper = self.term(name, self.upper[name])
        return min(at_lower, at_upper), max(at_lower, at_upper)

    def factor_min(self, features: tuple[str, ...]) -> Fraction:
        return sum((self.term_extremes(name)[0] for name in features), Fraction(0))

    def factor_max(self, features: tuple[str, ...]) -> Fraction:
        return sum((self.term_extremes(name)[1] for name in features), Fraction(0))

    @property
    def total_span(self) -> Fraction:
        """S = Σ_k (max_k - min_k)."""
        return sum((self.factor_max(features) - self.factor_min(features) for _, features in FACTORS), Fraction(0))

    @property
    def max_points_raw(self) -> tuple[Fraction, ...]:
        span = self.total_span
        return tuple(
            FULL_SCALE * (self.factor_max(features) - self.factor_min(features)) / span for _, features in FACTORS
        )

    @property
    def max_points(self) -> tuple[Fraction, ...]:
        return tuple(round_points(value) for value in self.max_points_raw)

    def points_per_unit(self, name: str) -> Fraction:
        """|∂ puntos / ∂ feature| dentro del recorte: 100 · |coef / scale| / S."""
        return FULL_SCALE * abs(self.coef[name] / self.scale[name]) / self.total_span


def load_logreg() -> LogRegParams:
    """Lee `models/logreg.json` como datos tras verificar su SHA-256 (spec ML-06 §1)."""
    if not LOGREG_JSON.is_file():
        raise GoldenError(f"{LOGREG_JSON} ausente: los casos dorados solo se generan con logreg-1.0.0")
    digest = sha256_of(LOGREG_JSON)
    if digest != LOGREG_SHA256:
        raise GoldenError(f"models/logreg.json tiene SHA-256 {digest}; se esperaba {LOGREG_SHA256} (logreg-1.0.0)")
    document = json.loads(LOGREG_JSON.read_text(encoding="utf-8"))
    schema = json.loads(FEATURE_SCHEMA.read_text(encoding="utf-8"))
    order = tuple(document["feature_order"])
    checks = {
        "format": document["format"] == LOGREG_FORMAT,
        "synthetic": document["synthetic"] is True,
        "transform": document["transform"]["type"] == "standard",
        "feature_order": order == tuple(schema["feature_order"]),
        "feature_schema_version": document["feature_schema_version"] == schema["metadata"]["schema_version"],
        "factors": sorted([name for _, names in FACTORS for name in names] + list(FIXED_TERM_FEATURES))
        == sorted(order),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise GoldenError(f"models/logreg.json no cumple: {failed}")

    def column(values: list[float]) -> dict[str, Fraction]:
        if len(values) != len(order):
            raise GoldenError("longitud de un parámetro distinta de feature_order")
        return {name: Fraction(float(value)) for name, value in zip(order, values, strict=True)}

    return LogRegParams(
        sha256=digest,
        feature_order=order,
        feature_schema_version=schema["metadata"]["schema_version"],
        lower=column(document["clip"]["lower"]),
        upper=column(document["clip"]["upper"]),
        mean=column(document["transform"]["mean"]),
        scale=column(document["transform"]["scale"]),
        coef=column(document["coef"]),
        intercept=Fraction(float(document["intercept"])),
    )


def round_points(value: Fraction) -> Fraction:
    """Puntos de un factor: round6 y 1 decimal mitad hacia arriba (valores ≥ 0)."""
    return round_half_up(round_half_up(value, PRE_ROUND_DECIMALS), FACTOR_DECIMALS)


def match_score(raw_total: Fraction) -> int:
    """`floor(round6(Σ) + 1/2)`."""
    return math.floor(round_half_up(raw_total, PRE_ROUND_DECIMALS) + HALF)


def backend_round(value: float, decimals: int) -> Fraction:
    """.NET `Math.Round(Math.Round(x, 6, AwayFromZero), d, AwayFromZero)` sobre un binary64 de cualquier signo.

    Para redondear las salidas del ONNX en la prueba: la magnitud exacta del binary64 se redondea
    mitad lejos de cero y conserva el signo; `-0` queda como `0` (`Fraction` no tiene cero negativo,
    la regla de M-ML-046).
    """
    if not math.isfinite(value):
        raise GoldenError(f"valor no finito {value!r}")
    exact = Fraction(value)
    magnitude = round_half_up(round_half_up(abs(exact), PRE_ROUND_DECIMALS), decimals)
    return -magnitude if exact < 0 else magnitude


@dataclass(frozen=True)
class LogRegRecount:
    clipped: dict[str, Fraction]
    contributions: dict[str, Fraction]
    factor_logits: tuple[Fraction, ...]
    raw_points: tuple[Fraction, ...]
    points: tuple[Fraction, ...]
    raw_total: Fraction
    match_score: int
    fixed_term: Fraction


def score_vector(features: dict[str, Fraction], params: LogRegParams) -> LogRegRecount:
    """Pasos 1-6 de M-ML-041 sobre un vector de features (valores exactos)."""
    clipped = {name: min(max(features[name], params.lower[name]), params.upper[name]) for name in params.feature_order}
    contributions = {name: params.term(name, clipped[name]) for name in params.feature_order}
    span = params.total_span
    factor_logits: list[Fraction] = []
    raw_points: list[Fraction] = []
    for _, names in FACTORS:
        logit = sum((contributions[name] for name in names), Fraction(0))
        factor_logits.append(logit)
        raw_points.append(FULL_SCALE * (logit - params.factor_min(names)) / span)
    for value in raw_points:
        if value < 0:
            raise GoldenError(f"puntos negativos {value}: el recorte no se aplicó")  # pragma: no cover
    raw_total = sum(raw_points, Fraction(0))
    fixed = params.intercept + sum((contributions[name] for name in FIXED_TERM_FEATURES), Fraction(0))
    return LogRegRecount(
        clipped=clipped,
        contributions=contributions,
        factor_logits=tuple(factor_logits),
        raw_points=tuple(raw_points),
        points=tuple(round_points(value) for value in raw_points),
        raw_total=raw_total,
        match_score=match_score(raw_total),
        fixed_term=fixed,
    )


def check_margin(case_id: str, recount_result: LogRegRecount) -> None:
    """Regla de margen: ningún valor queda a menos de 1e-9 de un punto medio de round6."""
    for value in (*recount_result.raw_points, recount_result.raw_total):
        margin = round6_margin(value, PRE_ROUND_DECIMALS)
        if margin != HALF and margin < MIN_ROUND6_MARGIN:
            raise GoldenError(f"{case_id}: {float(value)!r} a {float(margin) * 1e-6:.3g} de un punto medio de round6")


def check_consistency(case_id: str, recount_result: LogRegRecount) -> None:
    """El desglose redondeado suma el total (constitución §12) con la tolerancia del redondeo."""
    reported = sum(recount_result.points, Fraction(0))
    if abs(recount_result.raw_total - recount_result.match_score) > MAX_SCORE_GAP:
        raise GoldenError(f"{case_id}: |Σ - matchScore| > 1/2")  # pragma: no cover
    if abs(reported - recount_result.raw_total) > MAX_REPORTED_GAP:
        raise GoldenError(f"{case_id}: |Σ puntos - Σ| > 0.3")  # pragma: no cover
    if not 0 <= recount_result.match_score <= FULL_SCALE:
        raise GoldenError(f"{case_id}: matchScore fuera de [0, 100]")  # pragma: no cover


# ---------------------------------------------------------------------------
# Casos
# ---------------------------------------------------------------------------

PARAMS = load_logreg()
CONTRACT = load_contract()
CAMPUSES = baseline.CAMPUSES
SERVICES = baseline.services
CASES: list[dict[str, Any]] = []


def _raw_input(
    student: dict[str, Any] | None, room: dict[str, Any] | None, distance_km: Fraction | float, direction: int
) -> dict[str, Any]:
    student_record = {**baseline.STUDENT_BASE, **(student or {})}
    code = student_record["university_code"]
    campus = CAMPUSES[code]
    lat = baseline.meridian_lat(code, distance_km, direction) if distance_km else campus["lat"]
    room_record = {
        **baseline.ROOM_BASE,
        "lat": lat,
        "lon": campus["lon"],
        "university_code": code,
        **(room or {}),
    }
    return {
        "student": {key: student_record[key] for key in baseline.STUDENT_KEYS},
        "room": {key: room_record[key] for key in baseline.ROOM_KEYS},
    }


def add(
    case_id: str,
    category: str,
    description: str,
    *,
    student: dict[str, Any] | None = None,
    room: dict[str, Any] | None = None,
    distance_km: Fraction | float = 0,
    direction: int = 1,
) -> None:
    """Caso con entradas crudas (cuarto sobre el meridiano del campus a `distance_km`)."""
    CASES.append(
        {
            "id": case_id,
            "category": category,
            "description": description,
            "input": _raw_input(student, room, distance_km, direction),
        }
    )


def add_vector(case_id: str, category: str, description: str, features: dict[str, float | int]) -> None:
    """Caso solo vector (`input.features`): prueba la inferencia, no el cálculo de features."""
    if list(features) != list(PARAMS.feature_order):
        raise GoldenError(f"{case_id}: el vector debe seguir feature_order")
    CASES.append({"id": case_id, "category": category, "description": description, "input": {"features": features}})


def features_of(case_input: dict[str, Any]) -> dict[str, Fraction]:
    """Features exactas de una entrada cruda (recuento de golden_recount) o del vector dado."""
    if "features" in case_input:
        return {name: Fraction(value) for name, value in case_input["features"].items()}
    result = recount(case_input["student"], case_input["room"], CAMPUSES, CONTRACT)
    if result.features is None or result.excluded:
        raise GoldenError("se esperaba un par que pasa las reglas previas")
    return result.features


def distance_slope(transport: str, max_distance_km: Fraction) -> Fraction:
    """Puntos de distancia que se pierden por km (dentro del recorte): ∂/∂d de las 3 features."""
    return (
        PARAMS.points_per_unit("distance_km")
        + PARAMS.points_per_unit("effective_distance_km") * TRANSPORT_FACTORS[transport]
        + PARAMS.points_per_unit("distance_ratio") / max_distance_km
    )


def distance_for(
    target: Fraction, *, factor: str | None, student: dict[str, Any], room: dict[str, Any] | None = None
) -> Fraction:
    """km sobre el meridiano para que el factor `factor` (o el total, si es None) valga `target`."""
    at_campus = score_vector(features_of(_raw_input(student, room, 0, 1)), PARAMS)
    factor_ids = [name for name, _ in FACTORS]
    start = at_campus.raw_total if factor is None else at_campus.raw_points[factor_ids.index(factor)]
    record = {**baseline.STUDENT_BASE, **student}
    slope = distance_slope(record["transport"], Fraction(record["maxDistanceKm"]))
    return (start - target) / slope


# --- (1) Las mismas entradas que los 81 casos del baseline ---------------------------------------------
# Regla de margen (spec §6.4): si una entrada del baseline deja un valor de logreg a < 1e-9 de un punto
# medio de round6, se reemplaza la entrada (no se ajusta el esperado). id del baseline → (id nuevo, cambio).
MARGIN_REPLACEMENTS: dict[str, tuple[str, str, Fraction]] = {
    "null_all_optional_partial_room": (
        "null_all_optional_partial_room_margin",
        "12 km → 12.01 km: con 12 km el factor de distancia de logreg (53.13840450060224) queda a 6.0e-10 del "
        "punto medio de round6 53.1384045",
        Fraction(1201, 100),
    ),
}
for item in baseline.CASES:
    if item["id"] in MARGIN_REPLACEMENTS:
        new_id, change, km = MARGIN_REPLACEMENTS[item["id"]]
        student = dict(item["input"]["student"])
        room = {key: value for key, value in item["input"]["room"].items() if key not in ("lat", "lon")}
        CASES.append(
            {
                "id": new_id,
                "category": "baseline_inputs",
                "description": (
                    f"Entradas del caso {item['id']} de baseline_golden_cases.json (categoría {item['category']}) "
                    f"con un cambio por la regla de margen de la spec ML-06 §6.4: {change}. Resultado esperado "
                    f"con {MODEL_VERSION}."
                ),
                "input": _raw_input(student, room, km, 1),
            }
        )
        continue
    CASES.append(
        {
            "id": item["id"],
            "category": "baseline_inputs",
            "description": (
                f"Mismas entradas que el caso {item['id']} de baseline_golden_cases.json (categoría "
                f"{item['category']}); resultado esperado con {MODEL_VERSION}."
            ),
            "input": item["input"],
        }
    )

# --- (2) Recorte al training_range ------------------------------------------------------------------
add(
    "clip_raw_35km_walk",
    "clipping",
    "Cuarto a 35 km a pie con maxDistanceKm 0.5 (mínimo de NV-11): distance_km 35 y effective_distance_km 35 > 30 "
    "y distance_ratio 70 > 60; las tres se recortan a su training_range y el factor de distancia vale 0 (su mínimo). "
    "breakdown.value conserva los valores sin recortar.",
    student={"maxDistanceKm": 0.5},
    distance_km=35,
)
add(
    "clip_raw_31km_car_ratio_only",
    "clipping",
    "Cuarto a 31 km en auto (0.25) con maxDistanceKm 0.5: distance_km 31 > 30 y distance_ratio 62 > 60 se recortan; "
    "effective_distance_km 7.75 queda dentro del rango y sigue sumando.",
    student={"maxDistanceKm": 0.5, "transport": "car"},
    distance_km=31,
    direction=-1,
)
add(
    "clip_raw_29km_control",
    "clipping",
    "Control del borde: cuarto a 29 km a pie con maxDistanceKm 0.5: distance_ratio 58 < 60 y distance_km y "
    "effective_distance_km 29 < 30 quedan dentro del training_range (sin recorte; el factor de distancia es > 0).",
    student={"maxDistanceKm": 0.5},
    distance_km=29,
)
add_vector(
    "clip_vector_at_upper_bounds",
    "clipping",
    "Vector justo en los límites superiores del training_range de las features sin límite en el esquema "
    "(distance_km 30, effective_distance_km 30, price_to_budget 1, distance_ratio 60): el recorte no cambia nada "
    "y los factores de presupuesto y distancia quedan en su mínimo salvo price_below_min.",
    {
        "distance_km": 30,
        "effective_distance_km": 30,
        "price_to_budget": 1,
        "services_coverage": 0.5,
        "place_type_match": 1,
        "property_type_match": 1,
        "pets_compatible": 1,
        "distance_ratio": 60,
        "price_below_min": 0,
        "extra_services_ratio": 0.5,
    },
)
add_vector(
    "clip_vector_above_upper_bounds",
    "clipping",
    "Vector por encima de los límites (distance_km 45, effective_distance_km 40.5, price_to_budget 1.8, "
    "distance_ratio 90), todo dentro del range del esquema (sin máximo): mismos puntos que "
    "clip_vector_at_upper_bounds porque se recorta a 30, 30, 1 y 60.",
    {
        "distance_km": 45,
        "effective_distance_km": 40.5,
        "price_to_budget": 1.8,
        "services_coverage": 0.5,
        "place_type_match": 1,
        "property_type_match": 1,
        "pets_compatible": 1,
        "distance_ratio": 90,
        "price_below_min": 0,
        "extra_services_ratio": 0.5,
    },
)
add_vector(
    "clip_vector_below_zero",
    "clipping",
    "Vector con features negativas (fuera del range del esquema; el backend lo rechaza en la validación del "
    "paso 4 y el ONNX nunca lo recibe): distance_km -2, effective_distance_km -0.5, price_to_budget -0.3, "
    "distance_ratio -1 y price_below_min -0.25 se recortan a 0. Solo prueba el recorte del ONNX.",
    {
        "distance_km": -2,
        "effective_distance_km": -0.5,
        "price_to_budget": -0.3,
        "services_coverage": 1,
        "place_type_match": 1,
        "property_type_match": 1,
        "pets_compatible": 1,
        "distance_ratio": -1,
        "price_below_min": -0.25,
        "extra_services_ratio": 1,
    },
)
add_vector(
    "clip_vector_bounded_features_out",
    "clipping",
    "Vector con las features acotadas fuera de [0, 1] (fuera del range del esquema; solo prueba el recorte): "
    "services_coverage 1.5 → 1, place_type_match 2 → 1, property_type_match -1 → 0, price_below_min 1.3 → 1 y "
    "extra_services_ratio -0.2 → 0.",
    {
        "distance_km": 3,
        "effective_distance_km": 1.2,
        "price_to_budget": 0.75,
        "services_coverage": 1.5,
        "place_type_match": 2,
        "property_type_match": -1,
        "pets_compatible": 1,
        "distance_ratio": 0.1,
        "price_below_min": 1.3,
        "extra_services_ratio": -0.2,
    },
)
add_vector(
    "clip_vector_epsilon_outside",
    "clipping",
    "Vector a 1e-9 fuera de cada límite (por debajo de 0 en las acotadas abajo y por encima del límite superior en "
    "distance_km, effective_distance_km, price_to_budget y distance_ratio): el recorte deja exactamente el borde.",
    {
        "distance_km": 30.000000001,
        "effective_distance_km": 30.000000001,
        "price_to_budget": 1.000000001,
        "services_coverage": -1e-9,
        "place_type_match": 0,
        "property_type_match": 0,
        "pets_compatible": 1,
        "distance_ratio": 60.000000001,
        "price_below_min": -1e-9,
        "extra_services_ratio": -1e-9,
    },
)

# --- (3) Extremos ---------------------------------------------------------------------------------------
add(
    "extreme_100_raw",
    "extremes",
    "Entradas de la app que dan 100: renta 0 sin budget.min (price_to_budget 0 y price_below_min 0), cuarto en el "
    "campus, servicios pedidos cubiertos, tipo de lugar e inmueble iguales y el cuarto con todos los servicios no "
    "pedidos. Cada factor vale su maxPoints.",
    student={"budget.min": None},
    room={"price_mxn": 0},
)
add_vector(
    "extreme_vector_best_corner",
    "extremes",
    "Esquina de la caja que maximiza todos los factores: distancias 0, price_to_budget 0, price_below_min 0, "
    "cobertura 1, tipo de lugar e inmueble 1 y extras 1. Total 100.",
    {
        "distance_km": 0,
        "effective_distance_km": 0,
        "price_to_budget": 0,
        "services_coverage": 1,
        "place_type_match": 1,
        "property_type_match": 1,
        "pets_compatible": 1,
        "distance_ratio": 0,
        "price_below_min": 0,
        "extra_services_ratio": 1,
    },
)
add_vector(
    "extreme_vector_worst_corner",
    "extremes",
    "Esquina de la caja que minimiza todos los factores: distancias en 30/30/60, price_to_budget 1, "
    "price_below_min 1, cobertura 0, tipo de lugar e inmueble 0 y extras 0. Total 0 (no alcanzable con entradas "
    "de la app, T05-H4: price_to_budget 1 exige renta = budget.max y price_below_min 1 exige renta 0).",
    {
        "distance_km": 30,
        "effective_distance_km": 30,
        "price_to_budget": 1,
        "services_coverage": 0,
        "place_type_match": 0,
        "property_type_match": 0,
        "pets_compatible": 1,
        "distance_ratio": 60,
        "price_below_min": 1,
        "extra_services_ratio": 0,
    },
)

# --- Factores en su mínimo (T02-H1: el ONNX da ~-1e-16 donde el recuento da 0 exacto) ------------------
add(
    "factor_minimum_place_type",
    "factor_minimum",
    "placeType room frente a studio: place_type en su mínimo, 0 puntos exactos. En onnxruntime sale ~-2.45e-16 "
    "(T02-H1); el backend redondea con AwayFromZero y normaliza -0 a 0 (M-ML-046).",
    room={"placeType": "studio"},
)
add(
    "factor_minimum_five_factors",
    "factor_minimum",
    "Distancia, servicios esenciales, tipo de lugar, tipo de inmueble y extras en su mínimo a la vez (35 km a pie "
    "con maxDistanceKm 0.5, sin los servicios pedidos, tipos distintos y sin servicios no pedidos); solo el "
    "presupuesto suma. Los 5 factores dan 0 puntos (sin -0 en el JSON).",
    student={"maxDistanceKm": 0.5},
    room={"services": [], "placeType": "shared", "propertyType": "apartment"},
    distance_km=35,
    direction=-1,
)

# --- (4) Bordes de redondeo ------------------------------------------------------------------------------
add(
    "factor_round_budget_4_65",
    "factor_rounding",
    "budget.max 2039, renta 1062, sin budget.min: presupuesto 4.64999994 → round6 4.65 → 4.7 (mitad hacia "
    "arriba). Sin round6 daría 4.6 y con ToEven 4.6.",
    student={"budget.min": None, "budget.max": 2039},
    room={"price_mxn": 1062},
)
add(
    "factor_round_budget_4_25",
    "factor_rounding",
    "budget.max 2445, renta 1377, sin budget.min: presupuesto 4.24999983 → round6 4.25 → 4.3. Sin round6 daría "
    "4.2 y con ToEven 4.2.",
    student={"budget.min": None, "budget.max": 2445},
    room={"price_mxn": 1377},
)

ROUNDING_STUDENT: dict[str, Any] = {"maxDistanceKm": 30, "transport": "walk"}
FACTOR_TARGETS = (
    (
        "factor_round_distance_40_25_below",
        Fraction(402499996, 10**7),
        "40.2499996 → round6 40.25 → 40.3 (sin round6 o con ToEven, 40.2)",
    ),
    (
        "factor_round_distance_40_25_above",
        Fraction(402500004, 10**7),
        "40.2500004 → round6 40.25 → 40.3 (ToEven sobre round6 daría 40.2)",
    ),
    ("factor_round_distance_40_25_down", Fraction(402499994, 10**7), "40.2499994 → round6 40.249999 → 40.2"),
)
for case_id, target, text in FACTOR_TARGETS:
    add(
        case_id,
        "factor_rounding",
        f"Distancia (a pie, maxDistanceKm 30) ajustada para que el factor de distancia valga {text}. El valor "
        "queda a ≥ 1e-9 de un punto medio de round6 (regla de margen).",
        student=ROUNDING_STUDENT,
        distance_km=distance_for(target, factor="distance", student=ROUNDING_STUDENT),
    )

TOTAL_TARGETS = (
    ("logreg_total_x49999955_up", Fraction(8049999955, 10**8), "80.49999955 → round6 80.5 → 81"),
    ("logreg_total_x4999994_down", Fraction(804999994, 10**7), "80.4999994 → round6 80.499999 → 80"),
    ("logreg_total_x5000004_up", Fraction(805000004, 10**7), "80.5000004 → round6 80.5 → 81 (ToEven daría 80)"),
)
for case_id, target, text in TOTAL_TARGETS:
    add(
        case_id,
        "total_rounding",
        f"Distancia (a pie, maxDistanceKm 30) ajustada para un total sin redondear de {text}. El total no puede "
        "ser un punto medio exacto con estos parámetros; x.49999955 queda a 5e-8 del punto medio de round6.",
        student=ROUNDING_STUDENT,
        distance_km=distance_for(target, factor=None, student=ROUNDING_STUDENT),
    )

# --- (5) Zona de la desviación T05-H1 -------------------------------------------------------------------
T05_H1_STUDENT: dict[str, Any] = {"budget.min": 200, "budget.max": 50000}
for rent in (0, 100, 199, 200):
    add(
        f"t05_h1_rent_{rent}",
        "t05_h1_zone",
        f"Zona T05-H1: budget.max / budget.min = 250 > 76.2 y renta {rent}"
        + (" < budget.min" if rent < 200 else " = budget.min (referencia)")
        + ". El coeficiente negativo de price_below_min hace que el factor de presupuesto SUBA con la renta "
        "mientras renta < budget.min (t05_h1_rent_0 < _100 < _199 < _200 sin redondear; máximo +0.124 puntos). "
        "Desviación aceptada que se conserva en logreg-1.0.0 (M-ML-041).",
        student=T05_H1_STUDENT,
        room={"price_mxn": rent},
    )

# --- (6) pets_compatible en el término fijo -------------------------------------------------------------
add(
    "fixed_term_pets_allowed_raw",
    "fixed_term",
    "Estudiante con mascota y cuarto que la admite: pasa HC03 y pets_compatible = 1. pets_compatible no pertenece a "
    "ningún factor (su término va al término fijo con el intercepto) y no aparece en el breakdown.",
    student={"pets": True},
    room={"pets_allowed": True, "price_mxn": 2500},
    distance_km=4,
)
PETS_VECTOR: dict[str, float | int] = {
    "distance_km": 4,
    "effective_distance_km": 4,
    "price_to_budget": 0.625,
    "services_coverage": 1,
    "place_type_match": 1,
    "property_type_match": 1,
    "pets_compatible": 1,
    "distance_ratio": 0.4,
    "price_below_min": 0,
    "extra_services_ratio": 1,
}
add_vector(
    "fixed_term_pets_vector_1",
    "fixed_term",
    "Vector con pets_compatible = 1 (el único valor posible en un par que pasa HC03).",
    PETS_VECTOR,
)
add_vector(
    "fixed_term_pets_vector_0",
    "fixed_term",
    "El mismo vector con pets_compatible = 0 (solo vector; HC03 lo excluye antes del ONNX): factor_points y "
    "raw_total idénticos a fixed_term_pets_vector_1, porque pets_compatible solo entra al término fijo "
    "(coeficiente 0).",
    {**PETS_VECTOR, "pets_compatible": 0},
)


# ---------------------------------------------------------------------------
# Esperados
# ---------------------------------------------------------------------------


def json_number(value: Fraction) -> int | float:
    """Entero si el valor es entero; si no, el binary64 más cercano."""
    return int(value) if value.denominator == 1 else float(value)


def feature_json(name: str, value: Fraction) -> int | float:
    return int(value) if name in INTEGER_FEATURES and value.denominator == 1 else float(value)


def breakdown_json(features: dict[str, Fraction], points: tuple[Fraction, ...]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for (factor, names), value, max_points in zip(FACTORS, points, PARAMS.max_points, strict=True):
        entry: dict[str, Any] = {"factor": factor, "points": float(value), "maxPoints": json_number(max_points)}
        if len(names) == 1:
            entry["feature"] = names[0]
            entry["value"] = json_number(features[names[0]])
        else:
            entry["features"] = [{"feature": name, "value": json_number(features[name])} for name in names]
        entries.append(entry)
    return entries


def expected_for(case: dict[str, Any]) -> dict[str, Any]:
    head = {"modelVersion": MODEL_VERSION, "featureSchemaVersion": PARAMS.feature_schema_version}
    case_input = case["input"]
    if "features" in case_input:
        features = {name: Fraction(value) for name, value in case_input["features"].items()}
    else:
        base = recount(case_input["student"], case_input["room"], CAMPUSES, CONTRACT)
        if base.features is None:
            return {"features": None, "result": {**head, "incomplete": True, "missingFields": list(base.missing)}}
        features = base.features
        if base.excluded:
            return {
                "features": {name: feature_json(name, value) for name, value in features.items()},
                "result": {**head, "excluded": True, "violatedConstraints": list(base.violated)},
            }
    result = score_vector(features, PARAMS)
    check_margin(case["id"], result)
    check_consistency(case["id"], result)
    return {
        "features": {name: feature_json(name, value) for name, value in features.items()},
        "clippedFeatures": {name: feature_json(name, value) for name, value in result.clipped.items()},
        "onnx": {
            "factor_points": [float(value) for value in result.raw_points],
            "raw_total": float(result.raw_total),
        },
        "result": {**head, "matchScore": result.match_score, "breakdown": breakdown_json(features, result.points)},
    }


def _check_targets(cases: list[dict[str, Any]]) -> None:
    """Los casos ajustados con la distancia caen donde dice su descripción (a < 1e-9)."""
    by_id = {case["id"]: case for case in cases}
    for case_id, target, _ in FACTOR_TARGETS:
        got = Fraction(by_id[case_id]["expected"]["onnx"]["factor_points"][1])
        if abs(got - target) > Fraction(1, 10**9):
            raise GoldenError(f"{case_id}: distancia {float(got)!r} lejos del objetivo {float(target)}")
    for case_id, target, _ in TOTAL_TARGETS:
        got = Fraction(by_id[case_id]["expected"]["onnx"]["raw_total"])
        if abs(got - target) > Fraction(1, 10**9):
            raise GoldenError(f"{case_id}: total {float(got)!r} lejos del objetivo {float(target)}")


def build() -> dict[str, Any]:
    """Contenido completo del archivo de casos dorados de logreg-1.0.0."""
    cases = [{**case, "tolerance": dict(CASE_TOLERANCE), "expected": expected_for(case)} for case in CASES]
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise GoldenError("ids duplicados")
    _check_targets(cases)
    span = PARAMS.total_span
    categories: dict[str, int] = {}
    for case in cases:
        categories[case["category"]] = categories.get(case["category"], 0) + 1
    return {
        "metadata": {
            "description": (
                "Casos dorados del modelo promovido logreg-1.0.0 (ML-06 T03, ML-06-R03) para la paridad Python ↔ "
                "onnxruntime ↔ backend (.NET, NV-23). Se empaquetan como golden_cases.json junto a model.onnx "
                "(spec ML-06 §6.6). DATOS SINTÉTICOS: el modelo se entrenó con datos sintéticos."
            ),
            "modelVersion": MODEL_VERSION,
            "featureSchemaVersion": PARAMS.feature_schema_version,
            "derived_from": {
                "model_id": MODEL_VERSION,
                "file": "models/logreg.json",
                "format": LOGREG_FORMAT,
                "sha256": PARAMS.sha256,
            },
            "spec": "ML-06 1.0.0 (§6.4)",
            "decision": "M-ML-041, M-ML-044 y M-ML-046; T03: M-ML-047",
            "generated_at": GENERATED_AT,
            "generated_by": "tests/build_logreg_golden_cases.py",
            "synthetic": True,
            "how_computed": (
                "Recuento independiente: no importa thehouse_ml (ni score_features_logreg, score_matrix, parity.py "
                "ni onnx_graph.py). Features, restricciones duras y perfil incompleto con tests/golden_recount.py "
                "(el recuento de ML-04); parámetros de models/logreg.json leídos como datos (SHA-256 verificado) "
                "y convertidos a fractions.Fraction exactas de sus binary64; recorte → escalado → contribución "
                "lineal → suma por factor → min_k/max_k en la caja del recorte, S = Σ (max_k - min_k) → puntos_k = "
                "100 · (C_k - min_k) / S (M-ML-041) en aritmética exacta. Redondeo sobre el valor exacto: round6 y "
                "1 decimal mitad hacia arriba por factor; total floor(round6(Σ) + 1/2). La distancia es la única "
                "parte en binary64 (haversine con atan2) y se convierte a Fraction. Regla de margen: todo valor no "
                "exacto queda a ≥ 1e-9 de un punto medio de round6; el generador falla si un caso la incumple."
            ),
            "rounding": {
                "factor_points": "round_half_up(round_half_up(p, 6), 1)",
                "matchScore": "floor(round_half_up(raw_total, 6) + 0.5)",
                "maxPoints": "maxPointsRaw con la regla de factor_points",
                "mode": "half_up (MidpointRounding.AwayFromZero), nunca ToEven",
                "negative_zero": "normalizar -0 a 0 después de redondear (M-ML-046)",
            },
            "dotnet": DOTNET_NOTE,
            "tolerance": {
                "features": FEATURE_TOLERANCE,
                "value": FEATURE_TOLERANCE,
                "onnx": ONNX_TOLERANCE,
                "points": 0,
                "matchScore": 0,
                "note": "expected.features y breakdown[].value con tolerancia absoluta 1e-6; expected.onnx "
                "(factor_points y raw_total sin redondear) con 1e-4; points, maxPoints, matchScore, factor, "
                "feature, excluded, violatedConstraints, incomplete y missingFields exactos.",
            },
            "format": {
                "input.student / input.room": "Entradas crudas de NV-11 con los nombres de feature_schema.json, "
                "como en baseline_golden_cases.json (null = campo nulo; se aplican las reglas para faltantes).",
                "input.features": "Casos solo vector: las 10 features en el orden de feature_order; solo prueban "
                "la inferencia (recorte incluido), no el cálculo de features. Algunos están fuera del range del "
                "esquema y el backend los rechazaría antes del ONNX (se indica en la descripción).",
                "expected.features": "Las 10 features sin recortar (también en los excluidos, como referencia); "
                "null si el perfil está incompleto.",
                "expected.clippedFeatures": "Informativo: las features tras el recorte al training_range.",
                "expected.onnx": "Salidas esperadas del ONNX sin redondear: factor_points [6] en el orden de "
                "factorOrder y raw_total. Solo en los pares que pasan (el ONNX no puntúa excluidos ni incompletos).",
                "expected.result": "JSON del backend (spec ML-06 §6.1 paso 7): modelVersion, featureSchemaVersion, "
                "matchScore y breakdown (factor, points, maxPoints y feature/value, o features [{feature, value}] "
                "en budget y distance; value sin recortar); o excluded = true con violatedConstraints; o "
                "incomplete = true con missingFields.",
                "category": "Grupo del caso para el conteo de cobertura (spec ML-06 §6.4).",
            },
            "featureOrder": list(PARAMS.feature_order),
            "factorOrder": [factor for factor, _ in FACTORS],
            "factors": [
                {
                    "factor": factor,
                    "features": list(names),
                    "maxPointsRaw": float(raw),
                    "maxPoints": json_number(rounded),
                    "minLogit": float(PARAMS.factor_min(names)),
                    "maxLogit": float(PARAMS.factor_max(names)),
                }
                for (factor, names), raw, rounded in zip(FACTORS, PARAMS.max_points_raw, PARAMS.max_points, strict=True)
            ],
            "maxPointsRaw": [float(value) for value in PARAMS.max_points_raw],
            "totalSpanLogit": float(span),
            "fixedTermFeatures": list(FIXED_TERM_FEATURES),
            "campuses": {
                code: {"lat": value["lat"], "lon": value["lon"], "final_status": value["final_status"]}
                for code, value in CAMPUSES.items()
            },
            "hard_constraints": list(CONTRACT.hard_constraints),
            "required_fields": list(REQUIRED_FIELDS),
            "category_counts": categories,
            "case_count": len(cases),
        },
        "cases": cases,
    }


def render() -> str:
    """Texto JSON estable (UTF-8, 2 espacios, salto de línea final)."""
    return json.dumps(build(), ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    OUTPUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"{OUTPUT.relative_to(REPO_ROOT)}: {len(CASES)} casos")


if __name__ == "__main__":
    main()
