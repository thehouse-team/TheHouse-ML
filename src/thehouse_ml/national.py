"""Etapa ``prepare`` (parte nacional): bases ponderadas de renta (ML-02 T03).

Fuentes: ENVI 2020 (``TVIVIENDA``) y ENIGH 2022 (``viviendas``). Para cada
fuente y cada nivel (nacional, entidad, tamaño de localidad y clase/tipo de
vivienda) se calculan percentiles ponderados de la renta mensual en pesos del
año de la encuesta y en pesos de julio de 2024, la proporción ponderada de
viviendas rentadas y los tamaños de muestra sin ponderar.

Salidas: ``data/processed/national_baselines.parquet`` (fuera de git) y
``reports/national_baselines.md`` (en git).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from thehouse_ml.csvio import CsvReadError, read_zip_csv
from thehouse_ml.inpc import (
    ENIGH_SURVEY_MONTHS,
    ENVI_SURVEY_MONTHS,
    REFERENCE_MONTH,
    Deflators,
    build_deflators,
    general_index,
)
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.percentiles import weighted_percentiles
from thehouse_ml.sources import SourceSpec, load_sources

ENVI_SOURCE = "inegi_envi_2020"
ENIGH_SOURCE = "inegi_enigh_2022"
INPC_SOURCE = "inegi_inpc"

ENVI_COLUMNS = ("P5_1", "P5_2", "P4_3", "ENT", "TLOC", "FACTOR")
ENIGH_COLUMNS = ("tenencia", "renta", "factor", "ubica_geo", "tam_loc", "tipo_viv")
INPC_COLUMNS = ("FECHA", "CONCEPTO", "VALOR")

# Códigos especiales de P5_2 en ENVI: 0 = "Nada", 999999888 = "No responde",
# 999999999 = "No sabe" y blanco. El rango válido [100, 50000] los excluye;
# se listan además de forma explícita para hacer la regla verificable.
ENVI_RENT_SPECIAL_CODES = frozenset({"0", "000000000", "999999888", "999999999"})
ENVI_RENT_MIN = 100.0
ENVI_RENT_MAX = 50_000.0

BASELINE_COLUMNS: tuple[str, ...] = (
    "source",
    "year",
    "level",
    "level_value",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p10_jul2024",
    "p25_jul2024",
    "p50_jul2024",
    "p75_jul2024",
    "p90_jul2024",
    "rented_share",
    "n_muestra",
    "suma_factores",
    "n_total",
    "suma_factores_total",
    "deflactor",
)

ENTITY_NAMES = {"13": "Hidalgo", "15": "Estado de México"}
_SUMMARY_ORDER = ("nacional", "13", "15")


class PrepareError(RuntimeError):
    """Falta una fuente, un archivo o una variable necesaria para ``prepare``."""


@dataclass(frozen=True)
class BaselineRow:
    """Una estimación por fuente, nivel y valor del nivel."""

    source: str
    year: int
    level: str
    level_value: str
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p10_jul2024: float
    p25_jul2024: float
    p50_jul2024: float
    p75_jul2024: float
    p90_jul2024: float
    rented_share: float
    n_muestra: int
    suma_factores: float
    n_total: int
    suma_factores_total: float
    deflactor: float


@dataclass(frozen=True)
class PreparedData:
    """Salidas de la etapa: filas, deflactores y rutas escritas."""

    rows: tuple[BaselineRow, ...]
    deflators: Deflators
    parquet_path: Path
    report_path: Path


def run_prepare(paths: ProjectPaths) -> str:
    """Genera las bases nacionales con los archivos descargados y sus manifiestos."""
    sources = {source.id: source for source in load_sources(paths.sources_toml)}
    envi_frame = _load_frame(paths, sources, ENVI_SOURCE, ENVI_COLUMNS)
    enigh_frame = _load_frame(paths, sources, ENIGH_SOURCE, ENIGH_COLUMNS)
    inpc_frame = _load_frame(paths, sources, INPC_SOURCE, INPC_COLUMNS)

    index = general_index(inpc_frame.to_dict("records"))
    deflators = build_deflators(index)

    envi = _normalize_envi(envi_frame)
    enigh = _normalize_enigh(enigh_frame)
    rows = [
        *_levels(envi, ENVI_SOURCE, 2020, deflators.envi),
        *_levels(enigh, ENIGH_SOURCE, 2022, deflators.enigh),
    ]

    parquet_path = paths.data_processed / "national_baselines.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    baselines = rows_to_frame(rows)
    baselines.to_parquet(parquet_path, index=False)

    report_path = paths.reports / "national_baselines.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(rows, deflators, index), encoding="utf-8")
    return f"prepare: {parquet_path.as_posix()} ({len(rows)} filas) - {report_path.as_posix()}"


def rows_to_frame(rows: list[BaselineRow] | tuple[BaselineRow, ...]) -> pd.DataFrame:
    """Convierte las filas en un ``DataFrame`` con columnas estables."""
    frame = pd.DataFrame([asdict(row) for row in rows], columns=list(BASELINE_COLUMNS))
    return frame.sort_values(["source", "level", "level_value"], kind="stable").reset_index(drop=True)


def _load_frame(
    paths: ProjectPaths,
    sources: dict[str, SourceSpec],
    source_id: str,
    usecols: tuple[str, ...],
) -> pd.DataFrame:
    """Lee el CSV de una fuente registrada desde su ZIP en ``data/raw``."""
    source = sources.get(source_id)
    if source is None:
        raise PrepareError(f"Fuente no registrada: {source_id}")
    data_files = [file for file in source.download_url if file.kind == "data"]
    if len(data_files) != 1:
        raise PrepareError(f"{source_id}: se esperaba un archivo de datos, se encontraron {len(data_files)}")
    file = data_files[0]
    local = paths.data_raw / source_id / file.filename
    if not local.is_file():
        raise PrepareError(f"{source_id}: falta el archivo {local.as_posix()}; ejecute primero `download`")
    try:
        return read_zip_csv(local, file.table, file.encoding, usecols)
    except CsvReadError as exc:
        raise PrepareError(str(exc)) from exc


def _normalize_envi(frame: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas de viviendas rentadas y renta válida de ENVI 2020."""
    raw_rent = frame["P5_2"].astype(str).str.strip()
    rent = pd.to_numeric(raw_rent, errors="coerce")
    rented = frame["P5_1"].astype(str).str.strip().eq("1")
    valid = (
        rented & raw_rent.ne("") & ~raw_rent.isin(ENVI_RENT_SPECIAL_CODES) & rent.between(ENVI_RENT_MIN, ENVI_RENT_MAX)
    )
    return pd.DataFrame(
        {
            "entidad": frame["ENT"].astype(str).str.strip(),
            "tam_loc": frame["TLOC"].astype(str).str.strip(),
            "tipo_viv": frame["P4_3"].astype(str).str.strip(),
            "rented": rented,
            "valid_rent": valid,
            "rent": rent,
            "weight": _weights(frame["FACTOR"]),
        }
    )


def _normalize_enigh(frame: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas de viviendas rentadas y renta válida de ENIGH 2022."""
    rent = pd.to_numeric(frame["renta"].astype(str).str.strip(), errors="coerce")
    rented = frame["tenencia"].astype(str).str.strip().eq("1")
    return pd.DataFrame(
        {
            "entidad": frame["ubica_geo"].astype(str).str.strip().str[:2],
            "tam_loc": frame["tam_loc"].astype(str).str.strip(),
            "tipo_viv": frame["tipo_viv"].astype(str).str.strip(),
            "rented": rented,
            "valid_rent": rented & rent.gt(0.0),
            "rent": rent,
            "weight": _weights(frame["factor"]),
        }
    )


def _weights(raw: pd.Series) -> pd.Series:
    """Factores numéricos; los no numéricos valen 0 y no aportan a la ponderación."""
    return pd.to_numeric(raw.astype(str).str.strip(), errors="coerce").fillna(0.0)


def _levels(frame: pd.DataFrame, source: str, year: int, factor: float) -> list[BaselineRow]:
    """Genera las filas de los cuatro niveles para una fuente."""
    definitions: tuple[tuple[str, str | None], ...] = (
        ("nacional", None),
        ("entidad", "entidad"),
        ("tam_loc", "tam_loc"),
        ("tipo_viv", "tipo_viv"),
    )
    rows: list[BaselineRow] = []
    for level, column in definitions:
        if column is None:
            groups = [("nacional", frame)]
        else:
            groups = [(str(value), frame.loc[frame[column] == value]) for value in sorted(frame[column].unique())]
        for value, group in groups:
            row = _summarize(group, source, year, level, value, factor)
            if row is not None:
                rows.append(row)
    return rows


def _summarize(
    group: pd.DataFrame,
    source: str,
    year: int,
    level: str,
    value: str,
    factor: float,
) -> BaselineRow | None:
    """Resume un grupo; devuelve ``None`` si no hay renta válida o peso total."""
    total_weight = float(group["weight"].sum())
    if total_weight <= 0.0:
        return None
    sample = group.loc[group["valid_rent"]]
    n_muestra = len(sample)
    if n_muestra == 0:
        return None

    rents = sample["rent"].to_numpy(dtype=float)
    weights = sample["weight"].to_numpy(dtype=float)
    original = weighted_percentiles(rents, weights)
    adjusted = weighted_percentiles(rents * factor, weights)
    rented_weight = float(group.loc[group["rented"], "weight"].sum())
    return BaselineRow(
        source=source,
        year=year,
        level=level,
        level_value=value,
        p10=original["p10"],
        p25=original["p25"],
        p50=original["p50"],
        p75=original["p75"],
        p90=original["p90"],
        p10_jul2024=adjusted["p10"],
        p25_jul2024=adjusted["p25"],
        p50_jul2024=adjusted["p50"],
        p75_jul2024=adjusted["p75"],
        p90_jul2024=adjusted["p90"],
        rented_share=rented_weight / total_weight,
        n_muestra=n_muestra,
        suma_factores=float(weights.sum()),
        n_total=len(group),
        suma_factores_total=total_weight,
        deflactor=factor,
    )


def render_report(
    rows: list[BaselineRow] | tuple[BaselineRow, ...], deflators: Deflators, index: dict[str, float]
) -> str:
    """Reporte Markdown con la tabla resumen y la nota de pesos de julio 2024."""
    lines = [
        "# Bases nacionales de renta — ML-02 T03",
        "",
        "Generado por `uv run python -m thehouse_ml prepare`. Fuente: INEGI, Encuesta Nacional de Vivienda (ENVI) 2020",
        "y Encuesta Nacional de Ingresos y Gastos de los Hogares (ENIGH) 2022, citadas como autor del producto.",
        "INEGI no respalda este proyecto.",
        "",
        "## Pesos de julio de 2024",
        "",
        "- Todas las rentas en pesos ajustados están expresadas en **pesos de julio de 2024**;",
        "  es el último mes disponible en el INPC de datos abiertos (límite documentado, M-ML-014).",
        f"- Índice general INPC ({REFERENCE_MONTH}) = {index[REFERENCE_MONTH]:.3f} (base 2Q jul 2018 = 100).",
        f"- ENVI 2020: promedio oct-dic 2020 = {_period_average(index, ENVI_SURVEY_MONTHS):.3f};",
        f"  factor = {deflators.envi:.6f}.",
        f"- ENIGH 2022: promedio ago-nov 2022 = {_period_average(index, ENIGH_SURVEY_MONTHS):.3f};",
        f"  factor = {deflators.enigh:.6f}.",
        "",
        "> Limitación: julio de 2024 es el último mes del INPC publicado en el archivo de datos abiertos.",
        "> Si se ejecuta más adelante con una serie más reciente, hay que actualizar el mes de referencia.",
        "",
        "## Resumen: nacional, Hidalgo (13) y Estado de México (15)",
        "",
        "Los percentiles de la tabla están en **pesos de julio de 2024**. `p50 (año)` es la mediana",
        "en pesos del año del levantamiento (2020 u 2022). `% rentadas` es la proporción ponderada.",
        "",
        "| Fuente | Ámbito | p50 (año) | p10 | p25 | p50 | p75 | p90 | % rentadas | n | Σ factores |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _summary_rows(rows):
        source_label = "ENVI 2020" if row.source == ENVI_SOURCE else "ENIGH 2022"
        scope = "Nacional" if row.level == "nacional" else ENTITY_NAMES.get(row.level_value, row.level_value)
        values = (row.p10_jul2024, row.p25_jul2024, row.p50_jul2024, row.p75_jul2024, row.p90_jul2024)
        cells = " | ".join(f"{value:,.0f}" for value in values)
        lines.append(
            f"| {source_label} | {scope} | {row.p50:,.0f} | {cells} | "
            f"{row.rented_share * 100:.1f} % | {row.n_muestra:,} | {row.suma_factores:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Método",
            "",
            "- **Percentil ponderado:** el menor valor cuya suma acumulada de factores (valores ordenados de",
            "  menor a mayor, empatados agrupados) alcanza `q * suma de factores`; se usa `q` = 0.10, 0.25, 0.50,",
            "  0.75 y 0.90. Los montos de 2024 se obtienen multiplicando cada renta por el factor del deflactor.",
            '- **Viviendas rentadas:** ENVI `P5_1 == "1"`; ENIGH `tenencia == "1"`. La proporción usa como',
            "  denominador el total de viviendas del grupo, también ponderado.",
            "- **Renta válida:** ENVI `P5_2` entre 100 y 50000; se excluyen 0 (`Nada`), 999999888 (`No responde`),",
            "  999999999 (`No sabe`) y blanks. ENIGH `renta > 0` entre quienes pagan renta.",
            "- **Niveles:** nacional, entidad (`ENT` / primeros 2 dígitos de `ubica_geo`), tamaño de localidad",
            "  (`TLOC` / `tam_loc`) y clase o tipo de vivienda (`P4_3` / `tipo_viv`).",
            "- **`n_muestra`** cuenta las viviendas rentadas con renta válida (sin ponderar); `Σ factores` es la",
            "  suma de sus factores. `n_total` y `Σ factores total` son el denominador de la proporción.",
            "- Los grupos sin renta válida o sin peso total no generan fila.",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_rows(rows: list[BaselineRow] | tuple[BaselineRow, ...]) -> list[BaselineRow]:
    """Selecciona y ordena las seis filas de la tabla resumen."""
    selected = [
        row for row in rows if row.level == "nacional" or (row.level == "entidad" and row.level_value in _SUMMARY_ORDER)
    ]
    source_order = {ENVI_SOURCE: 0, ENIGH_SOURCE: 1}
    return sorted(
        selected,
        key=lambda row: (source_order.get(row.source, 9), _SUMMARY_ORDER.index(row.level_value)),
    )


def _period_average(index: dict[str, float], months: tuple[str, ...]) -> float:
    return sum(index[month] for month in months) / len(months)
