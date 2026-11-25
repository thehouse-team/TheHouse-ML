"""Genera `tests/fixtures/baseline_golden_cases.json` (ML-04 T05, ML-04-R05).

Los valores esperados salen **solo** del recuento independiente
(`tests/golden_recount.py`: fracciones exactas y fórmulas escritas aparte); este
script no importa `thehouse_ml`. La prueba `tests/unit/test_baseline_golden_cases.py`
comprueba que el archivo versionado es idéntico a lo que genera este script y
compara cada caso con el paquete (`score_pair`).

Uso: `uv run python tests/build_baseline_golden_cases.py` (reescribe el archivo).
"""

from __future__ import annotations

import csv
import json
import math
from fractions import Fraction
from typing import Any

from golden_recount import (
    REPO_ROOT,
    REQUIRED_FIELDS,
    Contract,
    expected_json,
    load_contract,
    recount,
    round6_margin,
)

OUTPUT = REPO_ROOT / "tests" / "fixtures" / "baseline_golden_cases.json"
UNIVERSITIES_CSV = REPO_ROOT / "data" / "universities.csv"
GENERATED_AT = "2026-11-23"
FEATURE_TOLERANCE = 1e-6
# Margen mínimo (en unidades de 10^-6) entre un valor que depende de la distancia y
# un punto medio de round6: evita casos cuyo resultado dependa del último ulp.
MIN_ROUND6_MARGIN = Fraction(1, 1000)

CATALOG = (
    "wifi",
    "water",
    "electricity",
    "gas",
    "hot_water",
    "laundry",
    "kitchen",
    "furnished",
    "parking",
    "security",
    "cleaning",
)
ALL_SERVICES = list(CATALOG)

STUDENT_BASE: dict[str, Any] = {
    "university_code": "UTTT",
    "budget.min": 2000,
    "budget.max": 4000,
    "maxDistanceKm": 10,
    "essentialServices": ["wifi", "water"],
    "livingWith": "any",
    "placeType": "room",
    "pets": False,
    "transport": "walk",
    "propertyTypes": ["house"],
}
ROOM_BASE: dict[str, Any] = {
    "price_mxn": 3000,
    "services": ALL_SERVICES,
    "placeType": "room",
    "propertyType": "house",
    "pets_allowed": False,
    "livingWith": "any",
    "bedrooms": 3,
    "bathrooms": 1,
    "area_m2": 12.0,
    "synthetic": True,
}
STUDENT_KEYS = tuple(STUDENT_BASE)
ROOM_KEYS = (
    "price_mxn",
    "lat",
    "lon",
    "services",
    "placeType",
    "propertyType",
    "pets_allowed",
    "livingWith",
    "bedrooms",
    "bathrooms",
    "area_m2",
    "university_code",
    "synthetic",
)

DOTNET_NOTE = (
    "Paridad en .NET (NV-23): evaluar las curvas y sumar los puntos en el orden de breakdown, de izquierda a "
    "derecha; aplicar round6 ANTES de redondear, con double o con decimal: matchScore = "
    "Math.Round(Math.Round(sum, 6, MidpointRounding.AwayFromZero), MidpointRounding.AwayFromZero) y points = "
    "Math.Round(Math.Round(points, 6, MidpointRounding.AwayFromZero), 1, MidpointRounding.AwayFromZero). "
    "Usar siempre MidpointRounding.AwayFromZero: el Math.Round por omisión (ToEven, mitad al par) NO es "
    "compatible (86.5 → 86 en lugar de 87; 1.25 → 1.2 en lugar de 1.3). Sin round6, double da 68.49999999999999 → "
    "68 en el caso total_x5_binary64_noise (esperado 69) y 13.349999999999998 → 13.3 en budget_13_35 (esperado 13.4)."
)


def load_campuses() -> dict[str, dict[str, Any]]:
    """Coordenadas finales de los 10 campus (data/universities.csv), leídas con csv."""
    with UNIVERSITIES_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        row["code"]: {
            "lat": float(row["final_lat"]),
            "lon": float(row["final_lon"]),
            "final_status": row["final_status"],
            "source": "data/universities.csv (final_lat, final_lon)",
        }
        for row in rows
    }


CAMPUSES = load_campuses()


def services(*names: str) -> list[str]:
    """Servicios en el orden del catálogo."""
    unknown = set(names) - set(CATALOG)
    if unknown:
        raise ValueError(f"servicios desconocidos {sorted(unknown)}")
    return [name for name in CATALOG if name in names]


def meridian_lat(code: str, distance_km: Fraction | float, direction: int = 1) -> float:
    """Latitud a `distance_km` sobre el meridiano del campus (haversine = 6371 km x Δφ)."""
    lat: float = CAMPUSES[code]["lat"]
    return lat + direction * math.degrees(float(distance_km) / 6371)


CASES: list[dict[str, Any]] = []


def add(
    case_id: str,
    category: str,
    description: str,
    *,
    student: dict[str, Any] | None = None,
    room: dict[str, Any] | None = None,
    distance_km: Fraction | float = 0,
    direction: int = 1,
    offset: tuple[float, float] | None = None,
) -> None:
    """Registra un caso: el cuarto va sobre el meridiano a `distance_km` o desplazado por `offset` (Δlat, Δlon)."""
    student_record = {**STUDENT_BASE, **(student or {})}
    code = student_record["university_code"]
    campus = CAMPUSES[code]
    if offset is not None:
        lat, lon = campus["lat"] + offset[0], campus["lon"] + offset[1]
    elif distance_km:
        lat, lon = meridian_lat(code, distance_km, direction), campus["lon"]
    else:
        lat, lon = campus["lat"], campus["lon"]
    room_record = {**ROOM_BASE, "lat": lat, "lon": lon, "university_code": code, **(room or {})}
    CASES.append(
        {
            "id": case_id,
            "category": category,
            "description": description,
            "input": {
                "student": {key: student_record[key] for key in STUDENT_KEYS},
                "room": {key: room_record[key] for key in ROOM_KEYS},
            },
        }
    )


def ratio_for_distance_points(points: Fraction) -> Fraction:
    """distance_ratio en (0.5, 1) que da `points` en el factor de distancia (25 · 2 · (1 - r))."""
    return 1 - points / 50


# --- Puntaje perfecto y peor puntaje ---------------------------------------------------------------
add(
    "perfect_100",
    "perfect",
    "Todas las curvas en 1: renta dentro de [budget.min, budget.max], cuarto en el campus, servicios pedidos "
    "cubiertos, tipo de lugar e inmueble iguales y el cuarto con los 9 servicios no pedidos. 30+25+20+10+10+5 = 100.",
)
add(
    "perfect_100_all_optional_null",
    "perfect",
    "Todos los campos opcionales de NV-11 nulos (budget.min → 0, maxDistanceKm → 30, essentialServices → vacío, "
    "pets → false, transport → public, propertyTypes → acepta todos) y cuarto con los 11 servicios: "
    "cobertura 1, extras 11/11 = 1. Total 100.",
    student={
        "budget.min": None,
        "maxDistanceKm": None,
        "essentialServices": None,
        "pets": None,
        "transport": None,
        "propertyTypes": None,
    },
    distance_km=3,
)
add(
    "perfect_100_price_at_max_ratio_half",
    "perfect",
    "Renta = budget.max (HC01 solo excluye >) y distance_ratio 0.5 (5 km de 10): los dos siguen en 1. Total 100.",
    room={"price_mxn": 4000},
    distance_km=5,
)
add(
    "worst_0",
    "worst",
    "Peores valores en todas las curvas: renta 0 con budget.min 2000 (price_below_min 1 → 0), distance_ratio 1.5 "
    "(15 km de 10 → 0), cobertura 0, tipo de lugar y de inmueble distintos y cuarto sin servicios (extras 0). Total 0.",
    room={"price_mxn": 0, "services": [], "placeType": "studio", "propertyType": "apartment"},
    distance_km=15,
)
add(
    "worst_0_ratio_exactly_one",
    "worst",
    "Igual que worst_0 pero con distance_ratio = 1 (10 km de 10): la distancia vale 0 en el borde. Total 0.",
    room={"price_mxn": 0, "services": [], "placeType": "shared", "propertyType": "student_residence"},
    distance_km=10,
    direction=-1,
)

# --- Presupuesto (price_below_min) -----------------------------------------------------------------
add(
    "budget_price_equals_min",
    "budget",
    "Renta = budget.min (2000): price_below_min 0 → 30 puntos (borde inferior del tramo plano). Total 100.",
    room={"price_mxn": 2000},
)
add(
    "budget_price_below_min_half",
    "budget",
    "budget.min 4000, renta 2000: price_below_min 0.5 → 15 puntos. Total 85.",
    student={"budget.min": 4000, "budget.max": 6000},
    room={"price_mxn": 2000},
)
add(
    "budget_price_zero",
    "budget",
    "Renta 0 con budget.min 2000: price_below_min 1 → 0 puntos. Total 70.",
    room={"price_mxn": 0},
)
add(
    "budget_price_below_min_fifth",
    "budget",
    "budget.min 2500, renta 2000: price_below_min 0.2 → 24 puntos. Total 94.",
    student={"budget.min": 2500},
    room={"price_mxn": 2000},
)
add(
    "budget_min_null_low_rent",
    "budget",
    "budget.min nulo → 0: no hay piso, price_below_min 0 aunque la renta sea 500. Total 100.",
    student={"budget.min": None},
    room={"price_mxn": 500},
)
add(
    "budget_min_zero_rent_zero",
    "budget",
    "budget.min 0 explícito y renta 0: price_below_min 0 (sin piso), price_to_budget 0. Total 100.",
    student={"budget.min": 0},
    room={"price_mxn": 0},
)
add(
    "budget_max_zero_rent_zero",
    "budget",
    "budget.min y budget.max 0, renta 0: pasa HC01 (0 > 0 es falso), price_to_budget = 0 / max(0, 1) = 0. Total 100.",
    student={"budget.min": 0, "budget.max": 0},
    room={"price_mxn": 0},
)

# --- Distancia (distance_ratio) --------------------------------------------------------------------
add(
    "distance_ratio_half_south",
    "distance",
    "distance_ratio 0.5 (2 km al sur con maxDistanceKm 4): fin de la meseta, 25 puntos. Total 100.",
    student={"maxDistanceKm": 4},
    distance_km=2,
    direction=-1,
)
add(
    "distance_ratio_three_quarters",
    "distance",
    "distance_ratio 0.75 (7.5 km de 10): 2 x (1 - 0.75) x 25 = 12.5 puntos. Total 87.5 → 88.",
    distance_km=Fraction(15, 2),
)
add(
    "distance_ratio_one",
    "distance",
    "distance_ratio 1 (10 km de 10): 0 puntos. Total 75.",
    distance_km=10,
)
add(
    "distance_ratio_above_one",
    "distance",
    "distance_ratio 1.2 (12 km de 10): la distancia no es restricción dura; 0 puntos (extremo constante). Total 75.",
    distance_km=12,
)
add(
    "distance_ratio_far_tiny_max",
    "distance",
    "29.9 km con maxDistanceKm 0.5 (mínimo de NV-11): distance_ratio 59.8, cerca del borde de training_range. "
    "0 puntos. Total 75.",
    student={"maxDistanceKm": 0.5},
    distance_km=Fraction(299, 10),
)
add(
    "distance_ratio_decimal_max",
    "distance",
    "maxDistanceKm decimal 2.5 y 1.5 km: distance_ratio 0.6 → 20 puntos. Total 95.",
    student={"maxDistanceKm": 2.5},
    distance_km=Fraction(3, 2),
)
add(
    "distance_max_null_two_thirds",
    "distance",
    "maxDistanceKm nulo → 30; 20 km: distance_ratio 2/3 → 16.666667 → 16.7. Total 91.666667 → 92.",
    student={"maxDistanceKm": None},
    distance_km=20,
)
add(
    "distance_max_null_half",
    "distance",
    "maxDistanceKm nulo → 30; 15 km: distance_ratio 0.5 → 25 puntos. Total 100.",
    student={"maxDistanceKm": None},
    distance_km=15,
    direction=-1,
)

# --- Servicios esenciales (services_coverage) ------------------------------------------------------
add(
    "services_coverage_zero",
    "services",
    "Pidió wifi y water; el cuarto tiene gas y kitchen: cobertura 0 → 0 puntos; extras 2/9 → 1.111111 → 1.1. "
    "Total 76.111111 → 76.",
    room={"services": services("gas", "kitchen")},
)
add(
    "services_coverage_one_third",
    "services",
    "Pidió wifi, water y gas; el cuarto tiene wifi y kitchen: cobertura 1/3 → 6.666667 → 6.7; extras 1/8 → 0.625 "
    "→ 0.6. Total 82.291667 → 82.",
    student={"essentialServices": services("wifi", "water", "gas")},
    room={"services": services("wifi", "kitchen")},
)
add(
    "services_coverage_half",
    "services",
    "Pidió wifi y water; el cuarto solo tiene wifi: cobertura 1/2 → 10 puntos; extras 0. Total 85.",
    room={"services": services("wifi")},
)
add(
    "services_coverage_one_no_extras",
    "services",
    "Pidió wifi y water; el cuarto tiene exactamente esos: cobertura 1 → 20; extras 0. Total 95.",
    room={"services": services("wifi", "water")},
)
add(
    "services_empty_list",
    "services",
    "essentialServices [] (vacío, igual que nulo): cobertura 1; extras 4/11 → 1.818182 → 1.8. Total 96.818182 → 97.",
    student={"essentialServices": []},
    room={"services": services("wifi", "water", "gas", "kitchen")},
)
add(
    "services_all_eleven_requested",
    "services",
    "Pidió los 11 servicios; el cuarto tiene 5: cobertura 5/11 → 9.090909 → 9.1; extras 0 (sin denominador). "
    "Total 84.090909 → 84.",
    student={"essentialServices": ALL_SERVICES},
    room={"services": services("wifi", "water", "electricity", "gas", "kitchen")},
)

# --- Tipo de lugar y tipo de inmueble --------------------------------------------------------------
add(
    "place_type_mismatch",
    "place_property",
    "placeType room frente a studio: place_type_match 0 → 0 puntos. Total 90.",
    room={"placeType": "studio"},
)
add(
    "property_type_mismatch",
    "place_property",
    "propertyTypes [house] frente a apartment: property_type_match 0 → 0 puntos. Total 90.",
    room={"propertyType": "apartment"},
)
add(
    "place_and_property_mismatch",
    "place_property",
    "placeType shared frente a room_private_bath y propertyTypes [apartment] frente a house. Total 80.",
    student={"placeType": "shared", "propertyTypes": ["apartment"]},
    room={"placeType": "room_private_bath"},
)
add(
    "property_types_null",
    "place_property",
    "propertyTypes nulo → acepta todos: student_residence vale 1. Total 100.",
    student={"propertyTypes": None},
    room={"propertyType": "student_residence"},
)
add(
    "property_types_empty_list",
    "place_property",
    "propertyTypes [] (vacío) → acepta todos: apartment vale 1. Total 100.",
    student={"propertyTypes": []},
    room={"propertyType": "apartment"},
)
add(
    "property_types_several",
    "place_property",
    "propertyTypes [apartment, student_residence] con un cuarto student_residence y placeType shared igual. Total 100.",
    student={"placeType": "shared", "propertyTypes": ["apartment", "student_residence"]},
    room={"placeType": "shared", "propertyType": "student_residence"},
)

# --- Extras (extra_services_ratio) -----------------------------------------------------------------
add(
    "extras_zero",
    "extras",
    "Pidió wifi, water y laundry; el cuarto solo tiene esos: extras 0/8 → 0 puntos. Total 95.",
    student={"essentialServices": services("wifi", "water", "laundry")},
    room={"services": services("wifi", "water", "laundry")},
)
add(
    "extras_two_ninths",
    "extras",
    "Pidió wifi y water; el cuarto agrega gas y parking: extras 2/9 → 1.111111 → 1.1. Total 96.111111 → 96.",
    room={"services": services("wifi", "water", "gas", "parking")},
)
add(
    "extras_three_eighths",
    "extras",
    "Pidió wifi, water y laundry; el cuarto agrega gas, kitchen y security: extras 3/8 → 1.875 → 1.9. "
    "Total 96.875 → 97.",
    student={"essentialServices": services("wifi", "water", "laundry")},
    room={"services": services("wifi", "water", "laundry", "gas", "kitchen", "security")},
)
add(
    "extras_one_ten_requested",
    "extras",
    "Pidió 10 servicios (todos menos cleaning); el cuarto tiene los 11: extras 1/1 = 1 → 5 puntos. Total 100.",
    student={"essentialServices": [name for name in CATALOG if name != "cleaning"]},
)

# --- Bordes de redondeo por factor (1 decimal, round6 antes) ---------------------------------------
add(
    "budget_13_35",
    "factor_rounding",
    "budget.min 2000, renta 890: price_below_min 0.555 → 30 x 0.445 = 13.35 (binary64 13.349999999999998) → "
    "round6 13.35 → 13.4. Total 83.35 → 83.",
    room={"price_mxn": 890},
)
add(
    "distance_24_35",
    "factor_rounding",
    "distance_ratio 0.513 (5.13 km de 10): 2 x 0.487 x 25 = 24.35 → 24.4. Total 99.35 → 99.",
    distance_km=Fraction(513, 100),
)
add(
    "budget_2_35",
    "factor_rounding",
    "budget.min 6000, renta 470: 30 x 470/6000 = 2.35 → 2.4. Total 72.35 → 72. (Extras no puede dar 2.35 con "
    "entradas de la app: su denominador es ≤ 11.)",
    student={"budget.min": 6000, "budget.max": 8000},
    room={"price_mxn": 470},
)
add(
    "distance_2_35",
    "factor_rounding",
    "distance_ratio 0.953 (9.53 km de 10): 2 x 0.047 x 25 = 2.35 → 2.4. Total 77.35 → 77.",
    distance_km=Fraction(953, 100),
)
add(
    "budget_28_65",
    "factor_rounding",
    "budget.min 2000, renta 1910: 30 x 0.955 = 28.65 → 28.7. Total 98.65 → 99.",
    room={"price_mxn": 1910},
)
add(
    "extras_1_25",
    "factor_rounding",
    "Pidió wifi, water y laundry; el cuarto agrega gas y kitchen: extras 2/8 → 1.25 → 1.3 (ToEven daría 1.2). "
    "Total 96.25 → 96.",
    student={"essentialServices": services("wifi", "water", "laundry")},
    room={"services": services("wifi", "water", "laundry", "gas", "kitchen")},
)
add(
    "extras_3_75",
    "factor_rounding",
    "Pidió wifi, water y laundry; el cuarto agrega 6 de los 8 no pedidos: extras 6/8 → 3.75 → 3.8. Total 98.75 → 99.",
    student={"essentialServices": services("wifi", "water", "laundry")},
    room={
        "services": services(
            "wifi", "water", "laundry", "electricity", "gas", "hot_water", "kitchen", "furnished", "parking"
        )
    },
)

# --- Bordes de redondeo del total ------------------------------------------------------------------
add(
    "total_x5_exact",
    "total_rounding",
    "budget.min 6000, renta 4300: 30 x 4300/6000 = 21.5; el cuarto solo tiene los servicios pedidos (extras 0); "
    "total exacto 86.5 → 87 (ToEven daría 86).",
    student={"budget.min": 6000, "budget.max": 8000},
    room={"price_mxn": 4300, "services": services("wifi", "water")},
)
add(
    "total_x5_binary64_noise",
    "total_rounding",
    "budget.min 3600, renta 20 (30 x 20/3600 = 1/6), cobertura 2/2 y extras 6/9 (10/3): total exacto 68.5; en "
    "binary64 de izquierda a derecha 68.49999999999999 → round6 68.5 → 69 (sin round6 daría 68).",
    student={"budget.min": 3600, "budget.max": 5000},
    room={
        "price_mxn": 20,
        "services": services("wifi", "water", "electricity", "gas", "hot_water", "laundry", "kitchen", "furnished"),
    },
)
add(
    "total_x4999995_up",
    "total_rounding",
    "Distancia ajustada para 12.49999955 puntos: total 82.49999955 → round6 82.5 → 83. Con entradas válidas de la "
    "app no existe un total con exactamente 7 decimales (los factores racionales tienen ≤ 4 decimales y la "
    "distancia es irracional); se usa x.49999955, a 5e-8 del punto medio de round6, para que el resultado no "
    "dependa del último ulp.",
    room={"services": services("wifi", "water")},
    distance_km=10 * ratio_for_distance_points(Fraction(1249999955, 10**8)),
)
add(
    "total_x4999994_down",
    "total_rounding",
    "Distancia ajustada para 12.4999994 puntos: total 82.4999994 → round6 82.499999 → 82.",
    room={"services": services("wifi", "water")},
    distance_km=10 * ratio_for_distance_points(Fraction(124999994, 10**7)),
)
add(
    "total_x5000004_up",
    "total_rounding",
    "Distancia ajustada para 12.5000004 puntos: total 82.5000004 → round6 82.5 → 83 (ToEven daría 82).",
    room={"services": services("wifi", "water")},
    distance_km=10 * ratio_for_distance_points(Fraction(125000004, 10**7)),
)

# --- Ejemplo de la spec §6.4 -----------------------------------------------------------------------
add(
    "spec_6_4_example",
    "spec_example",
    "Ejemplo de la spec §6.4 con entradas de la app: price_below_min 0.15 (25.5), distance_ratio 0.66 (17.0), "
    "cobertura 2/3 (13.3), tipo de lugar 1 (10.0), tipo de inmueble 1 (10.0) → matchScore 78. Extras 3/8 (1.9) en "
    "lugar de 3/7: con 3 servicios pedidos el denominador de extras es 8 (3/7 exige 4 pedidos y entonces la "
    "cobertura no puede ser 2/3). Total 77.708333 → 78.",
    student={"essentialServices": services("wifi", "water", "laundry")},
    room={"price_mxn": 1700, "services": services("wifi", "water", "gas", "kitchen", "security")},
    distance_km=Fraction(66, 10),
)

# --- Medios de transporte (effective_distance_km) --------------------------------------------------
add(
    "transport_walk",
    "transport",
    "walk (factor 1.0), 2 km con maxDistanceKm 3: effective_distance_km 2; distance_ratio 2/3 → 16.666667 → 16.7. "
    "Total 91.666667 → 92.",
    student={"transport": "walk", "maxDistanceKm": 3},
    distance_km=2,
)
add(
    "transport_bike",
    "transport",
    "bike (factor 0.5), 4 km con maxDistanceKm 5: effective_distance_km 2; distance_ratio 0.8 → 10. Total 85.",
    student={"transport": "bike", "maxDistanceKm": 5},
    distance_km=4,
)
add(
    "transport_public",
    "transport",
    "public (factor 0.4), 6 km con maxDistanceKm 10: effective_distance_km 2.4; distance_ratio 0.6 → 20. Total 95.",
    student={"transport": "public"},
    distance_km=6,
    direction=-1,
)
add(
    "transport_car",
    "transport",
    "car (factor 0.25), 9 km con maxDistanceKm 12: effective_distance_km 2.25; distance_ratio 0.75 → 12.5. "
    "Total 87.5 → 88.",
    student={"transport": "car", "maxDistanceKm": 12},
    distance_km=9,
)
add(
    "transport_null_public",
    "transport",
    "transport nulo → public (0.4), 1.2 km con maxDistanceKm nulo → 30: effective_distance_km 0.48; distance_ratio "
    "0.04 → 25. Total 100.",
    student={"transport": None, "maxDistanceKm": None},
    distance_km=Fraction(12, 10),
)

# --- Campos nulos ----------------------------------------------------------------------------------
add(
    "null_pets_room_without_pets",
    "nulls",
    "pets nulo → false: pasa HC03 aunque el cuarto no admita mascotas; pets_compatible 1. Total 100.",
    student={"pets": None},
)
add(
    "null_all_optional_partial_room",
    "nulls",
    "Todos los opcionales nulos; cuarto a 12 km con 3 servicios: distance_ratio 0.4 (12 de 30) → 25; extras 3/11 "
    "→ 1.363636 → 1.4. Total 96.363636 → 96.",
    student={
        "budget.min": None,
        "maxDistanceKm": None,
        "essentialServices": None,
        "pets": None,
        "transport": None,
        "propertyTypes": None,
    },
    room={"price_mxn": 1000, "services": services("wifi", "gas", "parking"), "propertyType": "apartment"},
    distance_km=12,
)
add(
    "null_services_room_without_services",
    "nulls",
    "essentialServices nulo y cuarto sin servicios: cobertura 1, extras 0/11 = 0. Total 95.",
    student={"essentialServices": None},
    room={"services": []},
)

# --- Perfil incompleto (quick setup nulo): ni puntaje ni restricciones -----------------------------
add(
    "incomplete_budget_max",
    "incomplete",
    "budget.max nulo (quick setup): el perfil está incompleto y no se evalúa; missingFields [budget.max].",
    student={"budget.max": None},
)
add(
    "incomplete_living_with",
    "incomplete",
    "livingWith nulo (quick setup): perfil incompleto; missingFields [livingWith].",
    student={"livingWith": None},
)
add(
    "incomplete_place_type",
    "incomplete",
    "placeType nulo (quick setup): perfil incompleto; missingFields [placeType].",
    student={"placeType": None},
)
add(
    "incomplete_all_three",
    "incomplete",
    "budget.max, livingWith y placeType nulos: missingFields lista los tres, en el orden de NV-11.",
    student={"budget.max": None, "livingWith": None, "placeType": None},
)
add(
    "incomplete_wins_over_exclusion",
    "incomplete",
    "livingWith nulo con mascota y un cuarto sin mascotas a renta 9000: el perfil incompleto se detecta antes que "
    "las restricciones duras; el resultado es incomplete (no excluded) con missingFields [livingWith].",
    student={"livingWith": None, "pets": True},
    room={"price_mxn": 9000},
)

# --- Restricciones duras (sin puntaje) y controles que pasan ---------------------------------------
add(
    "excluded_hc01_budget",
    "exclusion",
    "Renta 4001 > budget.max 4000: excluido por HC01_budget, sin matchScore.",
    room={"price_mxn": 4001},
)
add(
    "excluded_hc02_women_men",
    "exclusion",
    "livingWith women frente a un cuarto men: excluido por HC02_living_with.",
    student={"livingWith": "women"},
    room={"livingWith": "men"},
)
add(
    "excluded_hc02_men_any",
    "exclusion",
    "livingWith men frente a un cuarto any: excluido por HC02_living_with (men solo acepta cuartos men).",
    student={"livingWith": "men"},
    room={"livingWith": "any"},
)
add(
    "excluded_hc03_pets",
    "exclusion",
    "pets true y el cuarto no admite mascotas: excluido por HC03_pets.",
    student={"pets": True},
)
add(
    "excluded_hc01_hc03",
    "exclusion",
    "Renta 5000 > 4000 y mascota en un cuarto sin mascotas: HC01_budget y HC03_pets, en el orden del esquema.",
    student={"pets": True},
    room={"price_mxn": 5000},
)
add(
    "excluded_all_three",
    "exclusion",
    "Renta 4500 > 4000, livingWith women frente a any y mascota sin permiso: las tres restricciones en orden.",
    student={"livingWith": "women", "pets": True},
    room={"price_mxn": 4500, "livingWith": "any"},
)
add(
    "passes_any_student_women_room",
    "exclusion",
    "Control: livingWith any acepta un cuarto women (no hay exclusión). Total 100.",
    room={"livingWith": "women"},
)
add(
    "passes_pets_allowed",
    "exclusion",
    "Control: pets true con un cuarto que admite mascotas pasa HC03. Total 100.",
    student={"pets": True},
    room={"pets_allowed": True},
)
add(
    "passes_women_women_at_max",
    "exclusion",
    "Control: women con cuarto women y renta = budget.max pasa HC01 y HC02. Total 100.",
    student={"livingWith": "women"},
    room={"livingWith": "women", "price_mxn": 4000},
)

# --- Un caso por campus (posición genérica: cambian latitud y longitud) -----------------------------
CAMPUS_CASES: list[tuple[str, tuple[float, float], dict[str, Any], dict[str, Any]]] = [
    ("UTTT", (0.012, -0.018), {"maxDistanceKm": 4}, {}),
    ("UAEH-ESTE", (-0.021, 0.009), {"maxDistanceKm": 5, "transport": "bike"}, {"price_mxn": 1500}),
    ("UAEH-ESTL", (0.004, 0.027), {"maxDistanceKm": 6.5, "transport": "car"}, {"services": services("wifi")}),
    ("UAEH-ESAT", (-0.008, -0.006), {"transport": "public"}, {"placeType": "studio"}),
    ("ITSOEH", (0.031, 0.015), {"maxDistanceKm": 5, "transport": "walk"}, {}),
    ("TESJI", (-0.015, -0.022), {"maxDistanceKm": None, "transport": None}, {"propertyType": "apartment"}),
    ("UMB-JILO", (0.006, 0.004), {"maxDistanceKm": 1.5}, {"price_mxn": 1800}),
    ("UMB-VDC", (0.045, -0.035), {"maxDistanceKm": 20, "transport": "public"}, {}),
    ("UCAN", (-0.002, 0.013), {"maxDistanceKm": 2, "essentialServices": None}, {"services": services("wifi", "gas")}),
    ("UNE-TULA", (0.019, 0.0), {"maxDistanceKm": 3, "transport": "car"}, {}),
]
for code, offset, student_extra, room_extra in CAMPUS_CASES:
    status = CAMPUSES[code]["final_status"]
    note = " (coordenada seed_sin_verificar: si cambia en data/universities.csv, se regenera el caso)"
    add(
        f"campus_{code.lower().replace('-', '_')}",
        "campus",
        f"Campus {code} con el cuarto desplazado {offset[0]:+.3f}° de latitud y {offset[1]:+.3f}° de longitud "
        f"(haversine general, no meridiano){note if status == 'seed_sin_verificar' else ''}.",
        student={"university_code": code, **student_extra},
        room=room_extra,
        offset=offset,
    )


def _check_margins(case: dict[str, Any], contract: Contract) -> None:
    """Un valor que depende de la distancia no puede quedar cerca de un punto medio de round6."""
    result = recount(case["input"]["student"], case["input"]["room"], CAMPUSES, contract)
    if result.excluded or result.incomplete:
        return
    for value in (*result.raw_points, result.raw_total):
        margin = round6_margin(value)
        if margin != Fraction(1, 2) and margin < MIN_ROUND6_MARGIN:
            raise ValueError(f"{case['id']}: {float(value)!r} a {float(margin)} de un punto medio de round6")


def build() -> dict[str, Any]:
    """Contenido completo del archivo de casos dorados."""
    contract = load_contract()
    cases = []
    for case in CASES:
        _check_margins(case, contract)
        result = recount(case["input"]["student"], case["input"]["room"], CAMPUSES, contract)
        cases.append(
            {
                **case,
                "tolerance": {"features": FEATURE_TOLERANCE, "value": FEATURE_TOLERANCE, "points": 0, "matchScore": 0},
                "expected": expected_json(result, contract),
            }
        )
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("ids duplicados")
    return {
        "metadata": {
            "description": (
                "Casos dorados del modelo base por reglas (ML-04 T05, ML-04-R05) para la prueba de paridad Python "
                "↔ backend (.NET, NV-23) y, en ML-06, Python ↔ onnxruntime (constitution §13.4). En ML-06 se "
                "empaquetan como golden_cases.json junto al ONNX."
            ),
            "model_version": contract.model_version,
            "feature_schema_version": contract.feature_schema_version,
            "spec": "ML-04 1.0.0",
            "decision": "M-ML-033 (APROBADO); reglas: M-ML-029, M-ML-030 y M-ML-031 (APROBADO)",
            "generated_at": GENERATED_AT,
            "generated_by": "tests/build_baseline_golden_cases.py",
            "synthetic": True,
            "how_computed": (
                "Recuento independiente (tests/golden_recount.py): no importa thehouse_ml; fórmulas de las features, "
                "reglas para faltantes, factores de transporte y restricciones duras escritas aparte (M-ML-022 a "
                "M-ML-024, spec ML-04 §6.1); pesos, curvas y decimales leídos de baseline_params.json y orden de las "
                "features de feature_schema.json; aritmética exacta con fractions.Fraction. La distancia es la única "
                "parte en binary64 (haversine con atan2, radio 6371 km) y se convierte a Fraction. Redondeo sobre el "
                "valor exacto: round6 y después mitad hacia arriba (spec §6.3, M-ML-031). Un valor que depende de la "
                "distancia queda a ≥ 1e-9 de un punto medio de round6. La prueba compara este recuento y el paquete "
                "(score_pair) con lo esperado."
            ),
            "dotnet": DOTNET_NOTE,
            "tolerance": {
                "features": FEATURE_TOLERANCE,
                "value": FEATURE_TOLERANCE,
                "rawTotal": FEATURE_TOLERANCE,
                "points": 0,
                "matchScore": 0,
                "note": "features, breakdown[].value y rawTotal con tolerancia absoluta 1e-6; points, maxPoints, "
                "matchScore, factor, feature, excluded y violatedConstraints exactos.",
            },
            "format": {
                "input.student": "Perfil de NV-11 con los nombres de feature_schema.json; los 10 campos siempre "
                "presentes, null = campo nulo (se aplican las reglas para faltantes del esquema).",
                "input.room": "Cuarto con los nombres de feature_schema.json (ninguno nulo); lat/lon en WGS84.",
                "expected.features": "Las 10 features del par en el orden de feature_order (también en los pares "
                "excluidos, como referencia; el modelo no las puntúa); null si el perfil está incompleto.",
                "expected.result": "Salida de la spec §6.4: modelVersion, featureSchemaVersion, matchScore y "
                "breakdown (factor, points, maxPoints, feature, value) en el orden de los factores; o, si el par "
                "viola restricciones duras, excluded = true y violatedConstraints en el orden del esquema, sin "
                "matchScore ni breakdown; o, si falta un campo del quick setup (budget.max, livingWith o placeType "
                "nulos), incomplete = true y missingFields con todos los que faltan en ese orden, sin matchScore, "
                "breakdown ni restricciones (el perfil incompleto se detecta primero; en Python, "
                "IncompleteProfileError con .missing).",
                "expected.rawTotal": "Suma sin redondear de los puntos (solo pares que pasan); informativa, para "
                "depurar el redondeo. No es parte del contrato con TheHouse-GenAI.",
                "category": "Grupo del caso para el conteo de cobertura.",
            },
            "campuses": {
                code: {"lat": value["lat"], "lon": value["lon"], "final_status": value["final_status"]}
                for code, value in CAMPUSES.items()
            },
            "campuses_source": "data/universities.csv (final_lat, final_lon); TESJI, UMB-VDC y UCAN son "
            "seed_sin_verificar: si su coordenada cambia, se regeneran los casos con este script.",
            "feature_order": list(contract.feature_order),
            "factors": [
                {"factor": factor.id, "feature": factor.feature, "maxPoints": factor.weight}
                for factor in contract.factors
            ],
            "hard_constraints": list(contract.hard_constraints),
            "required_fields": list(REQUIRED_FIELDS),
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
