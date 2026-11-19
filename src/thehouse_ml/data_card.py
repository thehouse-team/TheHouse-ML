"""Data card de ML-02 (T08), generada automáticamente desde datos del proyecto.

Todas las cifras salen de ``data/sources.toml``, ``data/manifests/``,
``data/universities.csv`` y los Parquet de ``data/processed/``. No hay números
escritos a mano en el reporte: ``render_data_card`` es la única fuente de la
tabla y ``prepare`` la invoca al final de la etapa.

El reporte respeta la spec §8: no publica celdas con menos de 30 viviendas en
muestra (los intervalos de renta se agrupan antes de dibujarse), cita a INEGI
como autor del producto sin sugerir respaldo y no atribuye Inside Airbnb como
fuente de datos (M-ML-019).
"""

from __future__ import annotations

import csv
import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from thehouse_ml.denue import DENUE_COLUMNS, RADII_M, UNVERIFIED_STATUS
from thehouse_ml.inpc import ENIGH_SURVEY_MONTHS, ENVI_SURVEY_MONTHS, REFERENCE_MONTH
from thehouse_ml.manifests import Manifest, load_manifest
from thehouse_ml.national import ENIGH_SOURCE, ENTITY_NAMES, ENVI_SOURCE
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.regional import CENSO_SOURCE
from thehouse_ml.rent_histogram import BIN_WIDTH, RentBin, bins_from_frame, group_small_bins
from thehouse_ml.sources import SourceSpec, load_sources
from thehouse_ml.thresholds import MAX_CV, MIN_DWELLINGS

DATA_CARD_FILENAME = "data_card.md"
NATIONAL_PARQUET = "national_baselines.parquet"
REGIONAL_PARQUET = "regional_calibration.parquet"
CAMPUS_PARQUET = "campus_context.parquet"
RENT_HISTOGRAM_PARQUET = "rent_histogram.parquet"

BAR_WIDTH = 40
BAR_CHAR = "█"

# Ejemplo numérico de la nota de redondeo de la sección 7 (no una estimación).
ROUND_RENT_EXAMPLE = 2_500

# Nombres abreviados de mes para mostrar REFERENCE_MONTH como "jul-2024".
MONTH_LABELS: tuple[str, ...] = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")

# Decisión de memoria que respalda cada fuente (la de las descartadas se lee
# de `status_reference` en sources.toml, que es la autoridad).
DECISION_BY_SOURCE: dict[str, str] = {
    CENSO_SOURCE: "M-ML-015",
    ENVI_SOURCE: "M-ML-013, M-ML-014",
    ENIGH_SOURCE: "M-ML-013, M-ML-014",
    "inegi_denue_vigente": "M-ML-018",
    "inegi_inpc": "M-ML-014",
    "openstreetmap": "M-ML-016, M-ML-017",
    "thehouse_seed_nv07_nv08_nv11": "M-ML-016",
}

# Columnas que el lector del DENUE nunca abre (M-ML-018); se documentan aquí
# para que la data card no dependa de un reporte escrito a mano.
PRIVATE_DENUE_COLUMNS: tuple[str, ...] = (
    "nom_estab",
    "raz_social",
    "telefono",
    "correoelec",
    "www",
    "tipo_vial",
    "nom_vial",
    "numero_ext",
    "numero_int",
    "colonia",
    "codigo_postal",
    "nombre_act",
    "clee",
)

UNIVERSITY_COLUMNS: tuple[str, ...] = ("code", "name", "final_status", "final_source", "final_lat", "final_lon")

DATA_CARD_SKIP_MESSAGE = "data card: omitido (faltan salidas de prepare; ejecute `prepare`)"
REQUIRED_INPUTS: tuple[str, ...] = (
    NATIONAL_PARQUET,
    REGIONAL_PARQUET,
    CAMPUS_PARQUET,
    RENT_HISTOGRAM_PARQUET,
)


class DataCardError(RuntimeError):
    """Falta una entrada necesaria para generar la data card."""


@dataclass(frozen=True)
class University:
    """Universidad final de la T07 con su coordenada y procedencia."""

    code: str
    name: str
    final_status: str
    final_source: str
    final_lat: str
    final_lon: str


@dataclass(frozen=True)
class DataCardInputs:
    """Entradas leídas de los archivos del proyecto."""

    sources: tuple[SourceSpec, ...]
    source_meta: Mapping[str, Mapping[str, object]]
    manifests: Mapping[str, Manifest]
    baselines: pd.DataFrame
    regional: pd.DataFrame
    campus: pd.DataFrame
    rent_histogram: pd.DataFrame
    universities: tuple[University, ...]


def collect_inputs(paths: ProjectPaths) -> DataCardInputs:
    """Lee el registro, los manifiestos, los Parquet y el CSV de universidades."""
    sources = tuple(load_sources(paths.sources_toml))
    source_meta = _read_source_meta(paths.sources_toml)
    manifests: dict[str, Manifest] = {}
    for path in sorted(paths.data_manifests.glob("*.json")):
        manifest = load_manifest(path)
        if manifest is not None:
            manifests[manifest.source_id] = manifest
    return DataCardInputs(
        sources=sources,
        source_meta=source_meta,
        manifests=manifests,
        baselines=_read_parquet(paths.data_processed / NATIONAL_PARQUET),
        regional=_read_parquet(paths.data_processed / REGIONAL_PARQUET),
        campus=_read_parquet(paths.data_processed / CAMPUS_PARQUET),
        rent_histogram=_read_parquet(paths.data_processed / RENT_HISTOGRAM_PARQUET),
        universities=_read_universities(paths.universities_csv),
    )


def render_data_card(inputs: DataCardInputs) -> str:
    """Renderiza las 11 secciones de la data card a partir de las entradas."""
    sections = (
        _purpose_section(inputs),
        _sources_section(inputs),
        _attribution_section(inputs),
        _weighting_section(inputs),
        _levels_section(inputs),
        _samples_section(inputs),
        _histograms_section(inputs),
        _universities_section(inputs),
        _transformations_section(inputs),
        _limitations_section(inputs),
        _privacy_section(inputs),
    )
    header = [
        "# Data card — ML-02",
        "",
        "Generada automáticamente por `uv run python -m thehouse_ml prepare`",
        "(`thehouse_ml.data_card.render_data_card`). Todas las cifras proceden de",
        "`data/sources.toml`, `data/manifests/`, `data/universities.csv` y los Parquet de",
        "`data/processed/`; ninguna está escrita a mano.",
        "",
    ]
    return "\n".join([*header, *("\n".join(section) for section in sections), ""])


def run_data_card(paths: ProjectPaths) -> str:
    """Genera ``reports/data_card.md`` y devuelve el mensaje para la consola."""
    needed = (
        paths.sources_toml,
        paths.universities_csv,
        *(paths.data_processed / name for name in REQUIRED_INPUTS),
    )
    if any(not path.is_file() for path in needed):
        return DATA_CARD_SKIP_MESSAGE
    text = render_data_card(collect_inputs(paths))
    report = paths.reports / DATA_CARD_FILENAME
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8", newline="\n")
    return f"data card: {report.as_posix()}"


# ---------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------


def _purpose_section(inputs: DataCardInputs) -> list[str]:
    municipalities = inputs.regional["municipio_clave"].nunique() if not inputs.regional.empty else 0
    by_entity = (
        inputs.regional.groupby("entidad")["municipio_clave"].nunique().to_dict() if not inputs.regional.empty else {}
    )
    entity_breakdown = ", ".join(
        f"{count} en {ENTITY_NAMES.get(entity, entity)}" for entity, count in sorted(by_entity.items())
    )
    n_universities = len(inputs.universities)
    return [
        "## 1. Propósito y alcance",
        "",
        "Datos públicos para el **Smart Match** de TheHouse en la zona de Tula/Tepeji:",
        f"{municipalities} municipios ({entity_breakdown}) y {n_universities} universidades.",
        "Las fuentes describen viviendas y su entorno, no cuartos: el catálogo de cuartos será",
        "**sintético en ML-03**, calibrado con estas bases y etiquetado como tal (M-ML-019).",
        "",
    ]


def _sources_section(inputs: DataCardInputs) -> list[str]:
    lines = [
        "## 2. Fuentes",
        "",
        "Tamaños y SHA-256 se leen de los manifiestos de `data/manifests/`.",
        "",
        "| id | producto | edición | URL | licencia | archivos (tamaño y SHA-256) | estado | decisión |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for source in sorted(inputs.sources, key=lambda item: item.id):
        meta = inputs.source_meta.get(source.id, {})
        edition = str(meta.get("edition", "—"))
        url = str(meta.get("url", "—"))
        files = _source_files(inputs.manifests.get(source.id), source.variables_status)
        decision = source.status_reference or DECISION_BY_SOURCE.get(source.id, "—")
        state = f"DESCARTADA ({decision})" if source.variables_status == "discarded" else source.variables_status
        lines.append(
            f"| `{source.id}` | {source.name} | {edition} | {url} | {source.license} | {files} | {state} | {decision} |"
        )
    lines.append("")
    return lines


def _attribution_section(inputs: DataCardInputs) -> list[str]:
    lines = [
        "## 3. Atribuciones",
        "",
        "Se cita a cada productor como autor de su producto, sin sugerir que respalda el proyecto.",
    ]
    for source in sorted(inputs.sources, key=lambda item: item.id):
        if source.variables_status == "discarded" or "INEGI" not in source.attribution:
            continue
        lines.append(f"- {source.attribution}")
    for source in inputs.sources:
        if source.id == "openstreetmap":
            lines.append(f"- {source.attribution}")
    lines.extend(
        [
            "- INEGI no respalda este proyecto: solo se le cita como autor de los productos anteriores.",
            "",
        ]
    )
    return lines


def _weighting_section(inputs: DataCardInputs) -> list[str]:
    lines = [
        "## 4. Ponderación y deflactación",
        "",
        "Toda estimación de encuesta usa su factor de expansión; los conteos sin ponderar solo",
        "deciden si la muestra alcanza (spec §6, M-ML-003).",
        "",
        "| Encuesta | factor de expansión | diseño (estrato / UPM) |",
        "| --- | --- | --- |",
    ]
    for source_id in (CENSO_SOURCE, ENVI_SOURCE, ENIGH_SOURCE):
        source = next((item for item in inputs.sources if item.id == source_id), None)
        if source is None:
            continue
        factor, strata, psu = _design_variables(source)
        lines.append(f"| `{source.id}` | {factor} | {strata} / {psu} |")

    lines.extend(
        [
            "",
            f"Las rentas se expresan en **pesos de {_reference_label()}** (último mes del INPC de datos",
            "abiertos; M-ML-014). El deflactor es el INPC general y sale de `national_baselines.parquet`:",
            "",
            "| Fuente | periodo del levantamiento | deflactor |",
            "| --- | --- | ---: |",
        ]
    )
    for source_id, months in ((ENVI_SOURCE, ENVI_SURVEY_MONTHS), (ENIGH_SOURCE, ENIGH_SURVEY_MONTHS)):
        row = _first_baseline(inputs.baselines, source_id)
        if row is None:
            continue
        deflator = _as_float(row["deflactor"])
        lines.append(f"| `{source_id}` | {', '.join(months)} | {deflator:.6f} |")
    lines.append("")
    return lines


def _levels_section(inputs: DataCardInputs) -> list[str]:
    lines = [
        "## 5. Niveles de respaldo",
        "",
        f"Regla de suficiencia: al menos **{MIN_DWELLINGS} viviendas rentadas en muestra** y",
        f"**CV ≤ {MAX_CV * 100:.0f} %**. Si no se cumple, se usa el estado y, si tampoco, el país",
        "(M-ML-003, M-ML-015). Cada fila registra su nivel y su fuente.",
        "",
        "| Municipio | indicador | nivel | fuente | estimación | viviendas en muestra | CV |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for record in sorted(
        inputs.regional.to_dict("records"),
        key=lambda row: (str(row["municipio_clave"]), str(row["indicador"])),
    ):
        indicator = str(record["indicador"])
        rented = int(record["n_rentadas"])
        if rented < MIN_DWELLINGS:
            estimate = cv = "suprimido"
        else:
            estimate = _format_estimate(record["estimate"], percent=indicator != "renta")
            cv = "—" if _is_missing(record["cv"]) else f"{float(record['cv']) * 100:.1f} %"
        n = _suppress(rented, f"{rented:,}")
        lines.append(
            f"| {record['municipio_nombre']} | {indicator} | {record['level']} | {record['source']} | "
            f"{estimate} | {n} | {cv} |"
        )
    lines.extend(
        [
            "",
            "El CV se muestra solo para proporciones (linealización de Taylor con ESTRATO/UPM). "
            "La renta es una mediana ponderada; su CV no se calcula y se toma del nivel estatal "
            "porque ningún municipio alcanza la suficiencia.",
            "",
        ]
    )
    return lines


def _samples_section(inputs: DataCardInputs) -> list[str]:
    lines = [
        "## 6. Tamaños de muestra",
        "",
        "Viviendas en muestra (conteo sin ponderar) y viviendas expandidas (suma del factor).",
        f"Las celdas con menos de {MIN_DWELLINGS} viviendas en muestra se marcan `suprimido`",
        "y no publican ninguna de las dos cifras (spec §8).",
        "",
        "| Fuente | ámbito | viviendas en muestra | viviendas expandidas |",
        "| --- | --- | ---: | ---: |",
    ]
    for record in sorted(
        inputs.baselines.to_dict("records"),
        key=lambda row: (str(row["source"]), str(row["level"]), str(row["level_value"])),
    ):
        n = int(record["n_muestra"])
        sample = _suppress(n, f"{n:,}")
        expanded = _suppress(n, f"{float(record['suma_factores']):,.0f}")
        scope = _scope_label(str(record["level"]), str(record["level_value"]))
        lines.append(f"| `{record['source']}` | {scope} | {sample} | {expanded} |")
    lines.extend(
        [
            "",
            "Los tamaños por municipio están en la sección 5; el detalle completo vive en",
            f"`data/processed/{NATIONAL_PARQUET}` y `data/processed/{REGIONAL_PARQUET}` (fuera de git).",
            "",
        ]
    )
    return lines


def _histograms_section(inputs: DataCardInputs) -> list[str]:
    lines = [
        "## 7. Histogramas",
        "",
        "Barras en texto: la longitud es proporcional a la frecuencia del intervalo más frecuente.",
        f"Los intervalos con menos de {MIN_DWELLINGS} viviendas en muestra se agrupan con los",
        "adyacentes antes de dibujarse; la suma no cambia.",
        "",
        f"### Renta mensual ponderada ENVI (pesos de {_reference_label()})",
        "",
    ]
    envi_row = _first_baseline(inputs.baselines, ENVI_SOURCE)
    envi_deflator = _as_float(envi_row["deflactor"]) if envi_row is not None else 1.0
    for scope in ("nacional", "13", "15"):
        bins = bins_from_frame(inputs.rent_histogram.loc[inputs.rent_histogram["scope"] == scope])
        grouped = group_small_bins(bins)
        total = _baseline_expanded(inputs.baselines, scope)
        title = _scope_label("nacional" if scope == "nacional" else "entidad", scope)
        lines.extend(_render_rent_histogram(title, grouped, total))
        lines.append("")
        if scope == "nacional":
            lines.extend([_rounding_note(envi_deflator), ""])

    line = RADII_M[1]
    lines.extend(
        [
            f"### Negocios totales a {line:,} m por universidad (DENUE)",
            "",
        ]
    )
    totals = inputs.campus.loc[(inputs.campus["radius_m"] == line) & (inputs.campus["category"] == "total")].to_dict(
        "records"
    )
    lines.extend(_render_denue_histogram(totals))
    lines.append("")
    return lines


def _universities_section(inputs: DataCardInputs) -> list[str]:
    lines = [
        "## 8. Universidades",
        "",
        "Coordenada final y procedencia según la verificación de la T07 (M-ML-016, M-ML-017).",
        "",
        "| code | universidad | final_status | fuente de la coordenada | latitud | longitud |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for university in sorted(inputs.universities, key=lambda item: item.code):
        lines.append(
            f"| `{university.code}` | {university.name} | `{university.final_status}` | "
            f"{university.final_source} | {university.final_lat} | {university.final_lon} |"
        )
    unverified = [item.code for item in inputs.universities if item.final_status == UNVERIFIED_STATUS]
    lines.extend(["", "### Advertencia"])
    if unverified:
        lines.extend(
            [
                "",
                f"**{len(unverified)} universidades sin verificar:** "
                + ", ".join(f"`{code}`" for code in unverified)
                + ".",
                "Conservan la coordenada del seed y no deben tratarse como verificadas; el backend",
                "debe confirmar su nombre y ubicación (en particular `UCAN`). Sus conteos son orientativos.",
            ]
        )
    else:
        lines.extend(["", "No hay universidades con coordenada sin verificar."])
    lines.append("")
    return lines


def _transformations_section(inputs: DataCardInputs) -> list[str]:
    manifests = ", ".join(f"`data/manifests/{source_id}.json`" for source_id in sorted(inputs.manifests))
    return [
        "## 9. Transformaciones",
        "",
        "| etapa | entrada | salida |",
        "| --- | --- | --- |",
        "| `download` | `data/sources.toml` | "
        + (manifests or "— (sin manifiestos)")
        + ", `reports/variables_check.md` |",
        f"| `prepare` nacional | ENVI 2020, ENIGH 2022 e INPC (en `data/raw/`) | "
        f"`data/processed/{NATIONAL_PARQUET}`, `data/processed/{RENT_HISTOGRAM_PARQUET}`, "
        "`reports/national_baselines.md` |",
        f"| `prepare` regional | Censo 2020 y `data/processed/{NATIONAL_PARQUET}` | "
        f"`data/processed/{REGIONAL_PARQUET}`, `reports/regional_calibration.md` |",
        f"| `prepare` DENUE | DENUE 05/2026, `data/universities.csv` | "
        f"`data/processed/{CAMPUS_PARQUET}`, `reports/denue_campus.md` |",
        "| `prepare` universidades | seed NV-07/NV-11, respuestas OSM y revisión humana | "
        "`data/universities.csv`, `reports/universities_check.md` |",
        "| data card | registro, manifiestos, CSV y Parquet anteriores | `reports/data_card.md` |",
        "",
    ]


def _limitations_section(inputs: DataCardInputs) -> list[str]:
    years = sorted({int(value) for value in inputs.baselines["year"].unique()}) if not inputs.baselines.empty else []
    discarded = [item for item in inputs.sources if item.variables_status == "discarded"]
    unverified = [item for item in inputs.universities if item.final_status == UNVERIFIED_STATUS]
    rent_levels = (
        sorted({str(value) for value in inputs.regional.loc[inputs.regional["indicador"] == "renta", "level"]})
        if not inputs.regional.empty
        else []
    )
    lines = [
        "## 10. Limitaciones",
        "",
    ]
    if years:
        lines.append(
            f"- Las encuestas son de {', '.join(str(year) for year in years)}, anteriores a {_reference_label()};"
            " sus montos se deflactaron al INPC de ese mes y la serie no continúa después de esa fecha."
        )
    if rent_levels and set(rent_levels) <= {"estado", "nacional"}:
        lines.append(
            "- La renta no se estima a nivel municipal: no cumple la regla de suficiencia y se usa la"
            f" estimación {' estatal' if 'estado' in rent_levels else ' nacional'} de ENVI 2020."
        )
    if unverified:
        lines.append(f"- {len(unverified)} universidades conservan una coordenada sin verificar (M-ML-017).")
    lines.extend(
        [
            "- El DENUE cuenta establecimientos, no calidad, precios ni disponibilidad de cuartos.",
            "- No hay datos reales de cuartos: el catálogo se genera sintético en ML-03 (M-ML-019).",
        ]
    )
    for source in discarded:
        lines.append(f"- {source.name} queda descartada por {source.status_reference or 'decisión aprobada'}.")
    lines.append("")
    return lines


def _privacy_section(inputs: DataCardInputs) -> list[str]:
    allowed = ", ".join(f"`{name}`" for name in DENUE_COLUMNS)
    forbidden = ", ".join(f"`{name}`" for name in PRIVATE_DENUE_COLUMNS)
    return [
        "## 11. Privacidad",
        "",
        f"- Del DENUE **nunca se leen** nombres, razones sociales, teléfonos, correos, sitios web ni"
        f" direcciones: {forbidden}.",
        f"- Solo se leen las variables públicas registradas: {allowed}; se publican únicamente conteos agregados.",
        f"- Ninguna celda con menos de {MIN_DWELLINGS} viviendas en muestra se publica (spec §8).",
        "- A git solo entran agregados y metadatos: `sources.toml`, manifiestos, `data/universities.csv`",
        "  y los reportes. Los datos crudos y los Parquet quedan fuera del repositorio (constitution §6).",
        "",
    ]


# ---------------------------------------------------------------------------
# Ayudas
# ---------------------------------------------------------------------------


def _source_files(manifest: Manifest | None, status: str) -> str:
    if status == "discarded":
        return "— (descartada, sin descarga ni hash)"
    if manifest is None or not manifest.files:
        return "— (sin manifiesto)"
    return "<br>".join(f"`{item.filename}`: {item.size_bytes:,} B, `{item.sha256}`" for item in manifest.files)


def _design_variables(source: SourceSpec) -> tuple[str, str, str]:
    lower = [name.lower() for name in source.variables_used]
    factor = ", ".join(f"`{name}`" for name, item in zip(source.variables_used, lower, strict=True) if item == "factor")
    strata = ", ".join(
        f"`{name}`"
        for name, item in zip(source.variables_used, lower, strict=True)
        if "estrat" in item or item == "est_dis"
    )
    psu = ", ".join(
        f"`{name}`" for name, item in zip(source.variables_used, lower, strict=True) if item.startswith("upm")
    )
    return factor or "—", strata or "—", psu or "—"


def _first_baseline(frame: pd.DataFrame, source: str) -> Mapping[str, object] | None:
    selected = frame.loc[frame["source"] == source]
    return selected.iloc[0].to_dict() if not selected.empty else None


def _baseline_expanded(frame: pd.DataFrame, scope: str) -> float | None:
    if scope == "nacional":
        selected = frame.loc[(frame["source"] == ENVI_SOURCE) & (frame["level"] == "nacional")]
    else:
        selected = frame.loc[
            (frame["source"] == ENVI_SOURCE) & (frame["level"] == "entidad") & (frame["level_value"] == scope)
        ]
    return float(selected.iloc[0]["suma_factores"]) if not selected.empty else None


def _scope_label(level: str, value: str) -> str:
    if level == "nacional" or value == "nacional":
        return "Nacional"
    if level == "entidad":
        return ENTITY_NAMES.get(value, value)
    labels = {"tam_loc": "tamaño de localidad", "tipo_viv": "tipo de vivienda"}
    prefix = labels.get(level, level)
    if value == "&":  # código de "no especificado" en `tipo_viv` de ENIGH
        return f"{prefix} no especificado (&)"
    return f"{prefix} {value}"


def _rounding_note(deflator: float) -> str:
    """Nota sobre el desplazamiento de intervalos al deflactar rentas redondas."""
    deflated = round(ROUND_RENT_EXAMPLE * deflator)
    return (
        "Las rentas se reportan en cifras redondas (2,000, 2,500…); al deflactarlas a pesos de "
        f"{_reference_label()} (factor {deflator:.6f}) se desplazan de intervalo "
        f"({ROUND_RENT_EXAMPLE:,} → {deflated:,.0f}), por lo que algunos intervalos quedan casi vacíos. "
        "No es un error de procesamiento"
    )


def _reference_label() -> str:
    """``2024-07`` se muestra como ``jul-2024`` sin cifras escritas a mano."""
    year, month = REFERENCE_MONTH.split("-")
    return f"{MONTH_LABELS[int(month) - 1]}-{year}"


def _suppress(n: int, rendered: str) -> str:
    return "suprimido" if n < MIN_DWELLINGS else rendered


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _format_estimate(value: object, *, percent: bool) -> str:
    if _is_missing(value):
        return "—"
    number = _as_float(value)
    return f"{number * 100:.1f} %" if percent else f"${number:,.0f}"


def _as_float(value: object) -> float:
    return float(str(value))


def _as_int(value: object) -> int:
    return int(float(str(value)))


def _render_rent_histogram(title: str, bins: Sequence[RentBin], total: float | None) -> list[str]:
    lines = ["```text", f"# {title} — ENVI 2020, intervalos de {BIN_WIDTH:,.0f} pesos"]
    if not bins:
        lines.extend([f"suprimido: menos de {MIN_DWELLINGS} viviendas en muestra", "```"])
        return lines
    largest = max(bin_.frecuencia_expandida for bin_ in bins)
    width = {}
    for bin_ in bins:
        length = (
            0
            if bin_.frecuencia_expandida <= 0 or largest <= 0
            else max(1, round(bin_.frecuencia_expandida / largest * BAR_WIDTH))
        )
        width[bin_.label] = BAR_CHAR * length
        lines.append(
            f"{bin_.label:<12} | {width[bin_.label]:<{BAR_WIDTH}} "
            f"{bin_.frecuencia_expandida:>12,.0f} (n={bin_.n_muestra:,})"
        )
    if total is not None:
        lines.append(f"{'Total expandido':<12} | {'':<{BAR_WIDTH}} {total:>12,.0f}")
    lines.append("```")
    return lines


def _render_denue_histogram(rows: Sequence[Mapping[str, object]]) -> list[str]:
    lines = ["```text", f"# {' Universidad':<12} | {'':<{BAR_WIDTH}} negocios totales"]
    if not rows:
        lines.append("sin datos")
        lines.append("```")
        return lines
    largest = max(_as_int(row["count"]) for row in rows)
    for row in sorted(rows, key=lambda item: str(item["code"])):
        count = _as_int(row["count"])
        length = 0 if count <= 0 or largest <= 0 else max(1, round(count / largest * BAR_WIDTH))
        lines.append(f"{row['code']!s:<12} | {BAR_CHAR * length:<{BAR_WIDTH}} {count:>10,}")
    lines.append("```")
    return lines


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DataCardError(f"Falta {path.as_posix()}; ejecute primero `prepare`")
    return pd.read_parquet(path)


def _read_source_meta(path: Path) -> dict[str, Mapping[str, object]]:
    try:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataCardError(f"No se puede leer {path}: {exc}") from exc
    entries = document.get("sources", [])
    return {
        str(entry["id"]): entry for entry in entries if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }


def _read_universities(path: Path) -> tuple[University, ...]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = set(UNIVERSITY_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise DataCardError(f"{path.name}: faltan columnas {', '.join(sorted(missing))}")
            universities = tuple(
                University(
                    code=str(row.get("code") or "").strip(),
                    name=str(row.get("name") or "").strip(),
                    final_status=str(row.get("final_status") or "").strip(),
                    final_source=str(row.get("final_source") or "").strip(),
                    final_lat=str(row.get("final_lat") or "").strip(),
                    final_lon=str(row.get("final_lon") or "").strip(),
                )
                for row in reader
                if str(row.get("code") or "").strip()
            )
    except OSError as exc:
        raise DataCardError(f"No se puede leer {path}: {exc}") from exc
    if not universities:
        raise DataCardError(f"{path.name}: no hay universidades")
    return universities


__all__ = [
    "DATA_CARD_FILENAME",
    "DECISION_BY_SOURCE",
    "PRIVATE_DENUE_COLUMNS",
    "DataCardError",
    "DataCardInputs",
    "University",
    "collect_inputs",
    "render_data_card",
    "run_data_card",
]
