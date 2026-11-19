"""Etapa ``prepare`` (catálogo sintético de cuartos): Parquet y reporte (ML-03 T05, ML-03-R05).

Lee de `data/processed/` la renta p50 y las proporciones del Censo de
`regional_calibration.parquet` (ML-02 T04), la forma de la renta del estado de
`rent_histogram.parquet` (ML-02 T08) y los campus de `data/universities.csv`;
genera los cuartos con `thehouse_ml.synthetic.rooms` y escribe:

- `data/processed/rooms.parquet` (fuera de git) con la columna `synthetic = true`;
- `reports/rooms_catalog.md` (en git), titulado «DATOS SINTÉTICOS».

El reporte no lleva fechas ni rutas absolutas: correr `prepare` dos veces no
cambia ningún archivo versionado.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from thehouse_ml.features import default_schema
from thehouse_ml.features.campuses import load_campuses
from thehouse_ml.features.loader import FeatureSchema
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.synthetic.rooms import (
    PLACE_TYPE_RENT_ORDER,
    ROOMS_PARAMS_RESOURCE,
    CampusInputs,
    HistogramBin,
    RentShape,
    RoomParams,
    SyntheticRoom,
    effective_multipliers,
    generate_rooms,
    load_params,
    room_record,
)
from thehouse_ml.synthetic.students_stage import (
    REGIONAL_PARQUET,
    RentAnchor,
    _money,
    _pct,
    _row,
    quartiles,
    rent_anchors,
)

HISTOGRAM_PARQUET = "rent_histogram.parquet"
ROOMS_PARQUET = "rooms.parquet"
ROOMS_REPORT = "rooms_catalog.md"
ROOMS_SKIP_MESSAGE = (
    f"rooms: omitido; falta data/processed/{REGIONAL_PARQUET}, data/processed/{HISTOGRAM_PARQUET} "
    "o data/universities.csv (ejecute `download` y `prepare` de ML-02)"
)
COLUMNS = (
    "room_id",
    "university_code",
    "seed_verified",
    "campus_distance_km",
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
    "synthetic",
)
DTYPES: Mapping[str, str] = {
    "room_id": "string",
    "university_code": "string",
    "seed_verified": "bool",
    "campus_distance_km": "float64",
    "price_mxn": "int64",
    "lat": "float64",
    "lon": "float64",
    "placeType": "string",
    "propertyType": "string",
    "pets_allowed": "bool",
    "livingWith": "string",
    "bedrooms": "int64",
    "bathrooms": "int64",
    "area_m2": "float64",
    "synthetic": "bool",
}


class RoomsError(RuntimeError):
    """Falta un insumo de calibración (renta, histograma o Censo) o un campus para generar los cuartos."""


@dataclass(frozen=True)
class CensusShare:
    """Proporción del Censo de un servicio en el municipio de un campus y su nivel de respaldo."""

    indicator: str
    estimate: float
    level: str


@dataclass(frozen=True)
class CampusCalibration:
    """Todo lo que la etapa sabe de un campus: renta p50, estado, histograma y Censo."""

    anchor: RentAnchor
    entidad: str
    histogram: tuple[HistogramBin, ...]
    census: Mapping[str, CensusShare]

    def inputs(self) -> CampusInputs:
        return CampusInputs(
            rent_p50=self.anchor.p50,
            histogram=self.histogram,
            census={service: share.estimate for service, share in self.census.items()},
        )


def state_histograms(histogram: pd.DataFrame) -> dict[str, tuple[HistogramBin, ...]]:
    """Intervalos del histograma de renta por ámbito (`scope`: entidad o `nacional`)."""
    result: dict[str, tuple[HistogramBin, ...]] = {}
    for scope, rows in histogram.groupby("scope", sort=True):
        result[str(scope)] = tuple(
            HistogramBin(
                lower=float(row.lower),
                upper=None if pd.isna(row.upper) else float(row.upper),
                weight=float(row.frecuencia_expandida),
            )
            for row in rows.sort_values("lower").itertuples(index=False)
        )
    return result


def campus_calibration(
    regional: pd.DataFrame, histogram: pd.DataFrame, params: RoomParams
) -> dict[str, CampusCalibration]:
    """Renta p50, histograma del estado y proporciones del Censo de cada campus."""
    try:
        anchors = rent_anchors(regional, params.municipio_by_campus)
    except RuntimeError as error:
        raise RoomsError(str(error)) from error
    histograms = state_histograms(histogram)
    result: dict[str, CampusCalibration] = {}
    for code, anchor in anchors.items():
        rows = regional.loc[regional["municipio_clave"].astype(str) == anchor.municipio_clave]
        entidad = str(rows.iloc[0]["entidad"])
        if entidad not in histograms:
            raise RoomsError(f"{code}: sin histograma de renta para la entidad {entidad} en {HISTOGRAM_PARQUET}")
        census: dict[str, CensusShare] = {}
        for service, indicator in params.census_indicators.items():
            match = rows.loc[rows["indicador"] == indicator]
            if len(match) != 1 or pd.isna(match.iloc[0]["estimate"]):
                raise RoomsError(
                    f"{code}: sin la proporción del Censo «{indicator}» para el municipio {anchor.municipio_clave}"
                )
            census[service] = CensusShare(
                indicator=indicator, estimate=float(match.iloc[0]["estimate"]), level=str(match.iloc[0]["level"])
            )
        result[code] = CampusCalibration(anchor=anchor, entidad=entidad, histogram=histograms[entidad], census=census)
    return result


def rooms_frame(rooms: Sequence[SyntheticRoom], schema: FeatureSchema) -> pd.DataFrame:
    """Tabla de cuartos con los nombres del esquema y tipos fijos."""
    frame = pd.DataFrame([room_record(room, schema) for room in rooms], columns=list(COLUMNS))
    return frame.astype(dict(DTYPES))


def run_rooms(paths: ProjectPaths) -> str:
    """Genera `rooms.parquet` y `reports/rooms_catalog.md`."""
    regional_path = paths.data_processed / REGIONAL_PARQUET
    histogram_path = paths.data_processed / HISTOGRAM_PARQUET
    if not (regional_path.is_file() and histogram_path.is_file() and paths.universities_csv.is_file()):
        return ROOMS_SKIP_MESSAGE
    schema = default_schema()
    params = load_params(schema)
    calibration = campus_calibration(pd.read_parquet(regional_path), pd.read_parquet(histogram_path), params)
    campuses = load_campuses(paths.universities_csv)
    missing = [code for code in schema.catalog("university_code") if code not in campuses]
    if missing:
        raise RoomsError(f"data/universities.csv: faltan campus {missing}")
    rooms = generate_rooms(params, {code: item.inputs() for code, item in calibration.items()}, campuses, schema)

    paths.data_processed.mkdir(parents=True, exist_ok=True)
    parquet_path = paths.data_processed / ROOMS_PARQUET
    rooms_frame(rooms, schema).to_parquet(parquet_path, index=False)
    report_path = paths.reports / ROOMS_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(rooms, params, calibration, schema), encoding="utf-8", newline="\n")
    return f"rooms: {len(rooms)} cuartos sintéticos -> {parquet_path.as_posix()}; reporte {report_path.as_posix()}"


# ---------------------------------------------------------------------------
# Verificaciones que usan el reporte y las pruebas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RentRow:
    """Renta generada de un campus frente a la renta p50 regional."""

    code: str
    municipio: str
    level: str
    rent_p50: float
    p25: float
    median: float
    p75: float
    deviation: float
    within: bool


def rent_rows(rooms: Sequence[SyntheticRoom], params: RoomParams, anchors: Mapping[str, RentAnchor]) -> list[RentRow]:
    """Por campus: renta p50 regional, cuartiles de la renta generada y si la mediana cumple la tolerancia."""
    rows: list[RentRow] = []
    for code, anchor in anchors.items():
        prices = [float(item.room.price_mxn) for item in rooms if item.room.university_code == code]
        p25, median, p75 = quartiles(prices)
        deviation = median / anchor.p50 - 1.0
        rows.append(
            RentRow(
                code=code,
                municipio=anchor.municipio_nombre,
                level=anchor.level,
                rent_p50=anchor.p50,
                p25=p25,
                median=median,
                p75=p75,
                deviation=deviation,
                within=abs(deviation) <= params.calibration_tolerance,
            )
        )
    return rows


def census_tolerance(params: RoomParams, share: float) -> float:
    """Tolerancia absoluta frente al Censo: k errores estándar binomiales con n = cuartos por campus (mínimo 2 pts)."""
    k = float(params.value("services.census_tolerance_se"))
    return max(0.02, k * math.sqrt(share * (1.0 - share) / params.per_campus))


@dataclass(frozen=True)
class CensusRow:
    """Proporción generada de un servicio del Censo en un campus."""

    code: str
    service: str
    census: float
    level: str
    generated: float
    tolerance: float
    within: bool


def census_rows(
    rooms: Sequence[SyntheticRoom], params: RoomParams, census: Mapping[str, Mapping[str, CensusShare]]
) -> list[CensusRow]:
    """Por campus y servicio del Censo: proporción generada frente a la del Censo."""
    rows: list[CensusRow] = []
    for code, shares in census.items():
        group = [item for item in rooms if item.room.university_code == code]
        for service, share in shares.items():
            generated = sum(1 for item in group if service in item.room.services) / len(group)
            tolerance = census_tolerance(params, share.estimate)
            rows.append(
                CensusRow(
                    code=code,
                    service=service,
                    census=share.estimate,
                    level=share.level,
                    generated=generated,
                    tolerance=tolerance,
                    within=abs(generated - share.estimate) <= tolerance,
                )
            )
    return rows


def median_by_place_type(rooms: Sequence[SyntheticRoom], code: str | None = None) -> dict[str, float]:
    """Mediana de la renta por placeType de un campus (o de todo el catálogo si `code` es `None`)."""
    result: dict[str, float] = {}
    for place_type in PLACE_TYPE_RENT_ORDER:
        prices = [
            float(item.room.price_mxn)
            for item in rooms
            if item.room.place_type == place_type and (code is None or item.room.university_code == code)
        ]
        result[place_type] = float(statistics.median(prices)) if prices else float("nan")
    return result


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def _by_campus(rooms: Sequence[SyntheticRoom], schema: FeatureSchema) -> list[tuple[str, list[SyntheticRoom]]]:
    return [(code, [r for r in rooms if r.room.university_code == code]) for code in schema.catalog("university_code")]


def _share_table(
    rooms: Sequence[SyntheticRoom],
    schema: FeatureSchema,
    labels: Sequence[str],
    predicate: Callable[[SyntheticRoom, str], bool],
    *,
    counts: bool = False,
) -> list[str]:
    header = ["Campus", *(f"`{label}`" for label in labels)]
    lines = [_row(header), _row(["---", *(["---:"] * len(labels))])]
    for code, group in [*_by_campus(rooms, schema), ("**Total**", list(rooms))]:
        cells = []
        for label in labels:
            hits = sum(1 for item in group if predicate(item, label))
            cells.append(f"{hits} ({_pct(hits, len(group))})" if counts else _pct(hits, len(group)))
        lines.append(_row([code, *cells]))
    return lines


def _band_label(band: Mapping[str, float]) -> str:
    return f"{float(band['min']):g}-{float(band['max']):g} km"


def render_report(
    rooms: Sequence[SyntheticRoom],
    params: RoomParams,
    calibration: Mapping[str, CampusCalibration],
    schema: FeatureSchema,
) -> str:
    """Reporte Markdown determinista del catálogo sintético de cuartos."""
    total = len(rooms)
    anchors = {code: item.anchor for code, item in calibration.items()}
    tolerance = params.calibration_tolerance
    groups = _by_campus(rooms, schema)
    unverified = [f"`{code}`" for code, group in groups if group and not group[0].seed_verified]
    lines = [
        "# DATOS SINTÉTICOS — Catálogo sintético de cuartos (ML-03 T05)",
        "",
        "> **DATOS SINTÉTICOS.** Ningún cuarto de este reporte es un anuncio real. Los cuartos se generan con",
        "> distribuciones documentadas y semilla fija (constitución §8), calibrados con agregados públicos de INEGI;",
        "> las métricas que se calculen con ellos no son desempeño real.",
        "",
        "Generado por `uv run python -m thehouse_ml prepare` (`thehouse_ml.synthetic.rooms_stage`). Parámetros:",
        f"`src/thehouse_ml/synthetic/{ROOMS_PARAMS_RESOURCE}` versión {params.metadata.get('params_version', '—')}"
        f" (decisión {params.metadata.get('decision', '—')}).",
        f"Salida: `data/processed/{ROOMS_PARQUET}` (fuera de git, columna `synthetic = true`).",
        f"Esquema de features {schema.version}.",
        "",
        "## 1. Resumen",
        "",
        f"- Cuartos: **{total:,}** ({params.per_campus} por campus, {len(groups)} campus; semilla {params.seed}).",
        "- Ids sintéticos sin datos personales (`"
        + (f"{rooms[0].room_id}` a `{rooms[-1].room_id}`" if rooms else "—`")
        + "); sin títulos, descripciones, nombres ni datos de anfitriones.",
        "- Sin ningún campo de género (constitución §11): `livingWith` es la política que declara el cuarto.",
        f"- Todos a ≤ {params.max_distance_km:g} km de su campus (distancia haversine, radio 6,371 km).",
        "- Campus `seed_verified = false` (coordenadas sin verificar, M-ML-017; sus distancias son orientativas): "
        + (", ".join(unverified) or "ninguno")
        + ".",
        "- Rentas en pesos de julio de 2024 (M-ML-014), igual que la renta regional y el presupuesto de los",
        "  estudiantes.",
        "",
        "## 2. Calibración de la renta",
        "",
        "La forma de la renta es el histograma ENVI 2020 del estado del campus (`rent_histogram.parquet`), escalado",
        "para que su mediana sea la renta p50 regional (`regional_calibration.parquet`, indicador `renta`; el nivel",
        "indica el respaldo de M-ML-015). Los multiplicadores por `placeType` se normalizan para que la mediana",
        f"general de cada campus siga en esa p50; tolerancia ±{tolerance * 100:.0f} %.",
        "",
        _row(["Campus", "Municipio", "Nivel", "Renta p50 regional", "p25", "p50", "p75", "Desviación p50", "Dentro"]),
        _row(["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---"]),
    ]
    for row in rent_rows(rooms, params, anchors):
        lines.append(
            _row(
                [
                    row.code,
                    row.municipio,
                    row.level,
                    _money(row.rent_p50),
                    _money(row.p25),
                    _money(row.median),
                    _money(row.p75),
                    f"{row.deviation * 100:+.1f} %",
                    "sí" if row.within else "**no**",
                ]
            )
        )
    multipliers = params.value("rent.place_type_multipliers")
    entidades = sorted({item.entidad for item in calibration.values()})
    effective = {
        entidad: effective_multipliers(
            params,
            RentShape(
                next(item.histogram for item in calibration.values() if item.entidad == entidad),
                float(params.value("rent.open_bin_upper")),
            ),
        )
        for entidad in entidades
    }
    lines += [
        "",
        "## 3. Renta por tipo de lugar (`placeType`)",
        "",
        "La renta de la ENVI es de vivienda completa, no de cuarto; cada `placeType` aplica un multiplicador",
        "SUPUESTO (orden `shared` < `room` < `room_private_bath` < `studio`). El multiplicador efectivo ya incluye el",
        "factor común que mantiene la mediana general del campus en la renta p50 regional.",
        "",
        _row(["`placeType`", "Multiplicador", *(f"Efectivo (entidad {entidad})" for entidad in entidades)]),
        _row(["---", "---:", *(["---:"] * len(entidades))]),
    ]
    for place_type in PLACE_TYPE_RENT_ORDER:
        lines.append(
            _row(
                [
                    f"`{place_type}`",
                    f"{float(multipliers[place_type]):.2f}",
                    *(f"{effective[entidad][place_type]:.3f}" for entidad in entidades),
                ]
            )
        )
    lines += [
        "",
        "Mediana de la renta por campus y `placeType`:",
        "",
        _row(["Campus", *(f"`{place_type}`" for place_type in PLACE_TYPE_RENT_ORDER), "Orden"]),
        _row(["---", *(["---:"] * len(PLACE_TYPE_RENT_ORDER)), "---"]),
    ]
    for code, _group in [*groups, ("**Total**", list(rooms))]:
        medians = median_by_place_type(rooms, None if code == "**Total**" else code)
        ordered = all(medians[low] < medians[high] for low, high in itertools.pairwise(PLACE_TYPE_RENT_ORDER))
        lines.append(
            _row(
                [
                    code,
                    *(_money(medians[place_type]) for place_type in PLACE_TYPE_RENT_ORDER),
                    "sí" if ordered else "**no**",
                ]
            )
        )
    bands = params.value("location.distance_bands_km")
    lines += [
        "",
        "## 4. Distancia al campus",
        "",
        "Rumbo uniforme y distancia por bandas SUPUESTAS (uniforme dentro de cada banda). Probabilidad de cada",
        "banda: " + " · ".join(f"{_band_label(band)} {float(band['probability']) * 100:.0f} %" for band in bands) + ".",
        "",
        _row(["Campus", "Verificado", *(_band_label(band) for band in bands), "p50", "Máxima"]),
        _row(["---", "---", *(["---:"] * len(bands)), "---:", "---:"]),
    ]
    for code, group in [*groups, ("**Total**", list(rooms))]:
        distances = [item.campus_distance_km for item in group]
        cells = [
            _pct(
                sum(
                    1
                    for distance in distances
                    if float(band["min"]) <= distance < float(band["max"])
                    or (index == len(bands) - 1 and distance >= float(band["max"]))
                ),
                len(group),
            )
            for index, band in enumerate(bands)
        ]
        verified = "—" if code == "**Total**" else ("sí" if group[0].seed_verified else "**no**")
        lines.append(
            _row(
                [
                    code,
                    verified,
                    *cells,
                    f"{statistics.median(distances):.2f} km",
                    f"{max(distances):.2f} km",
                ]
            )
        )
    lines += [
        "",
        "## 5. Servicios frente al Censo 2020",
        "",
        "Agua, electricidad, gas e internet (→ `wifi`) siguen la proporción ponderada del Censo 2020 del municipio",
        "del campus (`regional_calibration.parquet`; drenaje no tiene equivalente en NV-11). Tolerancia: "
        f"±{float(params.value('services.census_tolerance_se')):g} errores estándar binomiales con n = "
        f"{params.per_campus} (mínimo ±2 puntos).",
        "",
        _row(["Campus", "Servicio", "Indicador del Censo", "Nivel", "Censo", "Generado", "Tolerancia", "Dentro"]),
        _row(["---", "---", "---", "---", "---:", "---:", "---:", "---"]),
    ]
    census = {code: item.census for code, item in calibration.items()}
    for share_row in census_rows(rooms, params, census):
        lines.append(
            _row(
                [
                    share_row.code,
                    f"`{share_row.service}`",
                    f"`{census[share_row.code][share_row.service].indicator}`",
                    share_row.level,
                    f"{share_row.census * 100:.1f} %",
                    f"{share_row.generated * 100:.1f} %",
                    f"±{share_row.tolerance * 100:.1f}",
                    "sí" if share_row.within else "**no**",
                ]
            )
        )
    priors = params.value("services.priors_by_place_type")
    lines += [
        "",
        "Servicios que el Censo no mide: probabilidad SUPUESTA por `placeType` y proporción generada en todo el",
        "catálogo.",
        "",
        _row(
            [
                "Servicio",
                *(f"`{place_type}` (prior / generado)" for place_type in PLACE_TYPE_RENT_ORDER),
                "Total generado",
            ]
        ),
        _row(["---", *(["---:"] * (len(PLACE_TYPE_RENT_ORDER) + 1))]),
    ]
    for service in [item for item in schema.catalog("essentialServices") if item in priors]:
        cells = []
        for place_type in PLACE_TYPE_RENT_ORDER:
            group = [item for item in rooms if item.room.place_type == place_type]
            hits = sum(1 for item in group if service in item.room.services)
            cells.append(f"{float(priors[service][place_type]) * 100:.0f} % / {_pct(hits, len(group))}")
        hits = sum(1 for item in rooms if service in item.room.services)
        lines.append(_row([f"`{service}`", *cells, _pct(hits, total)]))
    lines += ["", "## 6. Tipo de lugar (`placeType`)", ""]
    lines += _share_table(rooms, schema, schema.catalog("placeType"), lambda item, label: item.room.place_type == label)
    lines += ["", "## 7. Tipo de inmueble (`propertyType`)", ""]
    lines += _share_table(
        rooms, schema, schema.catalog("propertyTypes"), lambda item, label: item.room.property_type == label
    )
    lines += ["", "## 8. Mascotas (`pets_allowed`)", ""]
    lines += _share_table(
        rooms, schema, ("true", "false"), lambda item, label: str(item.room.pets_allowed).lower() == label
    )
    minimum = params.min_living_with_per_campus
    lines += [
        "",
        "## 9. Política de convivencia (`livingWith`)",
        "",
        "El cuarto declara quién vive ahí. Con HC02 (M-ML-024) los estudiantes `women` y `men` solo ven cuartos de",
        f"su mismo valor, así que cada campus necesita al menos {minimum} cuartos `women` y {minimum} `men`",
        "(nota para T05, M-ML-025).",
        "",
    ]
    lines += _share_table(
        rooms, schema, schema.catalog("livingWith"), lambda item, label: item.room.living_with == label, counts=True
    )
    lines += [
        "",
        "## 10. Recámaras, baños y superficie",
        "",
        "Recámaras y baños son del inmueble; la superficie es la del espacio que se ofrece. Rangos SUPUESTOS del",
        "esquema (M-ML-022): recámaras 1-10, baños 1-6, superficie 6-400 m².",
        "",
        _row(["`placeType`", "n", "Recámaras (mín-máx)", "Baños (mín-máx)", "Superficie p50", "Superficie (mín-máx)"]),
        _row(["---", "---:", "---", "---", "---:", "---"]),
    ]
    for place_type in PLACE_TYPE_RENT_ORDER:
        offered = [item.room for item in rooms if item.room.place_type == place_type]
        if not offered:
            continue
        areas = [room.area_m2 for room in offered]
        lines.append(
            _row(
                [
                    f"`{place_type}`",
                    len(offered),
                    f"{min(room.bedrooms for room in offered)}-{max(room.bedrooms for room in offered)}",
                    f"{min(room.bathrooms for room in offered)}-{max(room.bathrooms for room in offered)}",
                    f"{statistics.median(areas):.0f} m²",
                    f"{min(areas):.0f}-{max(areas):.0f} m²",
                ]
            )
        )
    lines += [
        "",
        "## 11. Privacidad y limitaciones",
        "",
        "- Sin nombres, títulos, descripciones, fotos ni identificadores reales; los ids son consecutivos",
        "  (constitución §7). Sin género, edad ni origen (constitución §11).",
        "- La renta regional es estatal (no municipal) y de vivienda completa (no de cuarto): los multiplicadores",
        "  por `placeType` son SUPUESTOS y las proporciones del Censo son de todas las viviendas del municipio,",
        "  no solo de las rentadas.",
        "- Ubicación, tipo de lugar, tipo de inmueble, mascotas, convivencia y los servicios que el Censo no mide",
        "  son supuestos documentados en `rooms_params.json`; validar con el equipo y reemplazar con señales",
        "  reales en ML-07 (riesgo RK02).",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "HISTOGRAM_PARQUET",
    "ROOMS_PARQUET",
    "ROOMS_REPORT",
    "ROOMS_SKIP_MESSAGE",
    "CampusCalibration",
    "CensusRow",
    "CensusShare",
    "RentRow",
    "RoomsError",
    "campus_calibration",
    "census_rows",
    "census_tolerance",
    "median_by_place_type",
    "render_report",
    "rent_rows",
    "rooms_frame",
    "run_rooms",
    "state_histograms",
]
