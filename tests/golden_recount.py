"""Recuento independiente de los casos dorados del modelo base (ML-04 T05, ML-04-R05).

Calcula features, restricciones duras, puntos por factor y `matchScore` de un par
estudiante-cuarto **sin** el código bajo prueba: no importa `thehouse_ml` (ni
`baseline_score` ni `features.pairs`), escribe aparte las fórmulas de la spec
ML-04 §6.1-6.3 y de ML-03 (M-ML-022 a M-ML-024) y usa aritmética exacta con
`fractions.Fraction`. Solo lee, como datos, los pesos, curvas y decimales de
`baseline_params.json` y el orden de las features, el catálogo de servicios y
los ids de las restricciones de `feature_schema.json`.

La única parte en punto flotante es la distancia (haversine con `atan2`, otra
formulación que la de `thehouse_ml.universities.haversine_m`, que usa `asin`):
su valor binary64 se convierte a `Fraction` y todo lo demás es exacto. El
redondeo es la regla de la spec §6.3 sobre el valor exacto:
`round6(x) = floor(x · 10^6 + 1/2) / 10^6`, factor `floor(round6 · 10 + 1/2) / 10`
y total `floor(round6(Σ) + 1/2)`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PARAMS = REPO_ROOT / "src" / "thehouse_ml" / "models" / "baseline_params.json"
FEATURE_SCHEMA = REPO_ROOT / "src" / "thehouse_ml" / "features" / "feature_schema.json"

EARTH_RADIUS_KM = 6371
HALF = Fraction(1, 2)

# Reglas para faltantes de NV-11 (M-ML-022), escritas aparte del esquema.
DEFAULT_BUDGET_MIN = 0
DEFAULT_MAX_DISTANCE_KM = 30
DEFAULT_TRANSPORT = "public"
# Campos del quick setup que no pueden ser nulos (M-ML-022, `missing.action = "reject"`), en el orden de NV-11.
REQUIRED_FIELDS = ("budget.max", "livingWith", "placeType")
# Factores de distancia efectiva por transporte (M-ML-022), escritos aparte del esquema.
TRANSPORT_FACTORS: dict[str, Fraction] = {
    "walk": Fraction(1),
    "bike": Fraction(1, 2),
    "public": Fraction(2, 5),
    "car": Fraction(1, 4),
}
INTEGER_FEATURES = frozenset({"place_type_match", "property_type_match", "pets_compatible"})


class RecountError(ValueError):
    """El caso no se puede recontar (perfil incompleto o entrada inválida)."""


@dataclass(frozen=True)
class FactorRule:
    id: str
    weight: int
    feature: str
    points: tuple[tuple[Fraction, Fraction], ...]

    def curve(self, x: Fraction) -> Fraction:
        """Lineal por tramos con extremos constantes (curve_semantics de baseline_params.json)."""
        if x <= self.points[0][0]:
            return self.points[0][1]
        if x >= self.points[-1][0]:
            return self.points[-1][1]
        for (x0, y0), (x1, y1) in zip(self.points, self.points[1:], strict=False):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        raise RecountError(f"{self.id}: {x} fuera de la curva")  # pragma: no cover


@dataclass(frozen=True)
class Contract:
    model_version: str
    feature_schema_version: str
    total_points: int
    factors: tuple[FactorRule, ...]
    factor_pre_round: int
    factor_decimals: int
    total_pre_round: int
    feature_order: tuple[str, ...]
    services_catalog: frozenset[str]
    hard_constraints: tuple[str, ...]


def load_contract() -> Contract:
    """Lee pesos, curvas y decimales (baseline_params.json) y orden y catálogos (feature_schema.json)."""
    params = json.loads(BASELINE_PARAMS.read_text(encoding="utf-8"))
    schema = json.loads(FEATURE_SCHEMA.read_text(encoding="utf-8"))
    rounding = params["rounding"]
    if rounding["factor_points"]["mode"] != "half_up" or rounding["total"]["mode"] != "half_up":
        raise RecountError("el recuento solo implementa el redondeo half_up")
    if params["curve_semantics"]["extrapolation"] != "constant":
        raise RecountError("el recuento solo implementa extremos constantes")
    factors = tuple(
        FactorRule(
            id=item["id"],
            weight=int(item["weight"]),
            feature=item["feature"],
            points=tuple((Fraction(str(x)), Fraction(str(y))) for x, y in item["curve"]["points"]),
        )
        for item in params["factors"]
    )
    return Contract(
        model_version=params["metadata"]["model_version"],
        feature_schema_version=schema["metadata"]["schema_version"],
        total_points=int(params["total_points"]),
        factors=factors,
        factor_pre_round=int(rounding["factor_points"]["pre_round_decimals"]),
        factor_decimals=int(rounding["factor_points"]["decimals"]),
        total_pre_round=int(rounding["total"]["pre_round_decimals"]),
        feature_order=tuple(schema["feature_order"]),
        services_catalog=frozenset(schema["catalogs"]["nv11"]["essentialServices"]),
        hard_constraints=tuple(item["id"] for item in schema["hard_constraints"]),
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia de gran círculo con radio 6371 km (formulación con atan2)."""
    phi1 = lat1 * math.pi / 180.0
    phi2 = lat2 * math.pi / 180.0
    d_phi = (lat2 - lat1) * math.pi / 180.0
    d_lambda = (lon2 - lon1) * math.pi / 180.0
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def round_half_up(value: Fraction, decimals: int) -> Fraction:
    """Mitad hacia arriba exacta (valores ≥ 0)."""
    scale = 10**decimals
    return Fraction(math.floor(value * scale + HALF), scale)


def round6_margin(value: Fraction, decimals: int = 6) -> Fraction:
    """Distancia de `value · 10^decimals` al punto medio más cercano (0 = punto medio exacto)."""
    scaled = value * 10**decimals
    frac = scaled - math.floor(scaled)
    return abs(frac - HALF)


@dataclass(frozen=True)
class Recount:
    features: dict[str, Fraction] | None
    violated: tuple[str, ...]
    raw_points: tuple[Fraction, ...]
    points: tuple[Fraction, ...]
    raw_total: Fraction
    match_score: int | None
    missing: tuple[str, ...] = ()

    @property
    def incomplete(self) -> bool:
        return bool(self.missing)

    @property
    def excluded(self) -> bool:
        return bool(self.violated)


def _as_set(value: Any) -> frozenset[str]:
    return frozenset(value) if value else frozenset()


def recount(student: dict[str, Any], room: dict[str, Any], campuses: dict[str, Any], contract: Contract) -> Recount:
    """Features, restricciones, puntos y `matchScore` del par, con las fórmulas escritas aquí.

    Un perfil con un campo del quick setup nulo no se evalúa (ni restricciones ni
    features): el resultado lista **todos** los campos que faltan, en el orden de
    `REQUIRED_FIELDS`.
    """
    missing = tuple(name for name in REQUIRED_FIELDS if student.get(name) is None)
    if missing:
        return Recount(None, (), (), (), Fraction(0), None, missing)
    budget_min = DEFAULT_BUDGET_MIN if student.get("budget.min") is None else int(student["budget.min"])
    budget_max = int(student["budget.max"])
    max_km = student.get("maxDistanceKm")
    max_distance = Fraction(DEFAULT_MAX_DISTANCE_KM) if max_km is None else Fraction(float(max_km))
    requested = _as_set(student.get("essentialServices"))
    pets = student.get("pets") is True
    transport = student.get("transport") or DEFAULT_TRANSPORT
    property_types = _as_set(student.get("propertyTypes"))
    living_with = student["livingWith"]

    price = int(room["price_mxn"])
    services = _as_set(room["services"])

    violated: list[str] = []
    checks = {
        "HC01_budget": price > budget_max,
        "HC02_living_with": living_with != "any" and room["livingWith"] != living_with,
        "HC03_pets": pets and not room["pets_allowed"],
    }
    for constraint in contract.hard_constraints:
        if checks[constraint]:
            violated.append(constraint)

    campus = campuses[student["university_code"]]
    distance = Fraction(
        haversine_km(float(campus["lat"]), float(campus["lon"]), float(room["lat"]), float(room["lon"]))
    )
    not_requested = len(contract.services_catalog) - len(requested)
    features: dict[str, Fraction] = {
        "distance_km": distance,
        "effective_distance_km": distance * TRANSPORT_FACTORS[transport],
        "price_to_budget": Fraction(price, max(budget_max, 1)),
        "services_coverage": Fraction(len(requested & services), len(requested)) if requested else Fraction(1),
        "place_type_match": Fraction(int(student["placeType"] == room["placeType"])),
        "property_type_match": Fraction(int(not property_types or room["propertyType"] in property_types)),
        "pets_compatible": Fraction(int(not (pets and not room["pets_allowed"]))),
        "distance_ratio": distance / max_distance,
        "price_below_min": (
            Fraction(budget_min - price, budget_min) if budget_min > 0 and price < budget_min else Fraction(0)
        ),
        "extra_services_ratio": (Fraction(len(services - requested), not_requested) if not_requested else Fraction(0)),
    }
    ordered = {name: features[name] for name in contract.feature_order}
    if violated:
        return Recount(ordered, tuple(violated), (), (), Fraction(0), None)

    raw_points = tuple(factor.weight * factor.curve(ordered[factor.feature]) for factor in contract.factors)
    points = tuple(
        round_half_up(round_half_up(raw, contract.factor_pre_round), contract.factor_decimals) for raw in raw_points
    )
    raw_total = sum(raw_points, Fraction(0))
    match_score = math.floor(round_half_up(raw_total, contract.total_pre_round) + HALF)
    return Recount(ordered, (), raw_points, points, raw_total, match_score)


def json_number(value: Fraction) -> int | float:
    """Entero si el valor es entero; si no, el binary64 más cercano."""
    return int(value) if value.denominator == 1 else float(value)


def expected_json(result: Recount, contract: Contract) -> dict[str, Any]:
    """Bloque `expected` del caso dorado a partir del recuento."""
    head = {"modelVersion": contract.model_version, "featureSchemaVersion": contract.feature_schema_version}
    if result.features is None:
        return {
            "features": None,
            "result": {**head, "incomplete": True, "missingFields": list(result.missing)},
        }
    features = {
        name: (int(value) if name in INTEGER_FEATURES else float(value)) for name, value in result.features.items()
    }
    if result.excluded:
        return {
            "features": features,
            "result": {**head, "excluded": True, "violatedConstraints": list(result.violated)},
        }
    breakdown = [
        {
            "factor": factor.id,
            "points": float(points),
            "maxPoints": factor.weight,
            "feature": factor.feature,
            "value": json_number(result.features[factor.feature]),
        }
        for factor, points in zip(contract.factors, result.points, strict=True)
    ]
    return {
        "features": features,
        "result": {**head, "matchScore": result.match_score, "breakdown": breakdown},
        "rawTotal": float(result.raw_total),
    }
