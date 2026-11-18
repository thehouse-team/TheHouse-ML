"""Etapa ``prepare`` (parte regional): calibración de los 7 municipios (ML-02 T04).

Fuentes: Censo 2020, muestra del cuestionario ampliado (solo Hidalgo 13 y Estado
de México 15, tabla ``Viviendas``) para tenencia y servicios; ENVI 2020 desde
``national_baselines.parquet`` (T03) para el monto de renta, que el Censo no
pregunta.

Salidas: ``data/processed/regional_calibration.parquet`` (fuera de git) y
``reports/regional_calibration.md`` (en git).

Varianza (linealización de Taylor para una razón, UPM con reemplazo dentro de
cada estrato). Con pesos ``w``, validez ``x`` (1 si el código no es 9/blanco) y
respuesta afirmativa ``y``:

    p = sum(w_i * y_i) / sum(w_i * x_i)
    l_i = w_i * (y_i - p * x_i) / sum(w_j * x_j)
    L_hj = sum(l_i) sobre las viviendas del estrato h y la UPM j
    Var(p) = sum_h [ n_h / (n_h - 1) * sum_j (L_hj - media(L_h*))^2 ]
    ee = sqrt(Var(p)) ; cv = ee / p

Los estratos con una sola UPM no aportan varianza (el estimador con reemplazo
no está definido) y se cuentan en ``n_strata_single_psu``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import pandas as pd

from thehouse_ml.csvio import CsvReadError, read_zip_csv
from thehouse_ml.national import ENIGH_SOURCE, ENVI_SOURCE
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.sources import SourceSpec, load_sources

CENSO_SOURCE = "inegi_censo_2020_viviendas_ca"
CENSO_COLUMNS: tuple[str, ...] = (
    "ENT",
    "MUN",
    "FACTOR",
    "ESTRATO",
    "UPM",
    "TENENCIA",
    "AGUA_ENTUBADA",
    "DRENAJE",
    "ELECTRICIDAD",
    "INTERNET",
    "COMBUSTIBLE",
)

MIN_RENTADAS = 30
MAX_CV = 0.30
UNSPECIFIED = frozenset({"9", ""})

# Códigos confirmados con el diccionario oficial del Censo (M-ML-015).
CODE_TENENCIA_RENTADA = "2"
CODE_INTERNET_SI = "7"
CODE_ELECTRICIDAD_SI = "1"
CODE_AGUA_DENTRO = "1"
CODE_AGUA_PATIO = "2"
CODIGOS_DRENAJE_SI = frozenset({"1", "2", "3", "4"})
CODE_COMBUSTIBLE_GAS = "2"

REGIONAL_COLUMNS: tuple[str, ...] = (
    "municipio_clave",
    "municipio_nombre",
    "entidad",
    "indicador",
    "level",
    "source",
    "year",
    "n_muestra",
    "suma_factores",
    "n_rentadas",
    "n_total",
    "suma_factores_total",
    "estimate",
    "se",
    "cv",
    "n_psu",
    "n_strata",
    "n_strata_single_psu",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
)


class RegionalError(RuntimeError):
    """Falta una fuente, un archivo o una estimación válida para la calibración."""


@dataclass(frozen=True)
class Municipality:
    """Uno de los 7 municipios de las universidades del seed."""

    entidad: str
    mun: str
    nombre: str

    @property
    def clave(self) -> str:
        """Clave ENT + MUN con ceros a la izquierda (por ejemplo ``13070``)."""
        return f"{self.entidad}{self.mun}"


MUNICIPIOS: tuple[Municipality, ...] = (
    Municipality("13", "013", "Atotonilco de Tula"),
    Municipality("13", "041", "Mixquiahuala de Juárez"),
    Municipality("13", "063", "Tepeji del Río de Ocampo"),
    Municipality("13", "070", "Tlahuelilpan"),
    Municipality("13", "076", "Tula de Allende"),
    Municipality("15", "045", "Jilotepec"),
    Municipality("15", "112", "Villa del Carbón"),
)


@dataclass(frozen=True)
class Indicator:
    """Indicador del Censo y códigos que cuentan como respuesta afirmativa."""

    key: str
    variable: str
    affirmative: frozenset[str]
    title: str


INDICATORS: tuple[Indicator, ...] = (
    Indicator("rentadas", "TENENCIA", frozenset({CODE_TENENCIA_RENTADA}), "Viviendas rentadas"),
    Indicator("internet", "INTERNET", frozenset({CODE_INTERNET_SI}), "Con internet"),
    Indicator("electricidad", "ELECTRICIDAD", frozenset({CODE_ELECTRICIDAD_SI}), "Con electricidad"),
    Indicator(
        "agua_entubada",
        "AGUA_ENTUBADA",
        frozenset({CODE_AGUA_DENTRO, CODE_AGUA_PATIO}),
        "Con agua entubada (dentro o patio)",
    ),
    Indicator("agua_dentro", "AGUA_ENTUBADA", frozenset({CODE_AGUA_DENTRO}), "Con agua dentro de la vivienda"),
    Indicator("drenaje", "DRENAJE", CODIGOS_DRENAJE_SI, "Con drenaje"),
    Indicator("gas", "COMBUSTIBLE", frozenset({CODE_COMBUSTIBLE_GAS}), "Cocina con gas"),
)


@dataclass(frozen=True)
class ProportionEstimate:
    """Proporción ponderada, error estándar y tamaños de muestra de un nivel."""

    estimate: float
    se: float
    cv: float
    n_muestra: int
    suma_factores: float
    n_total: int
    suma_factores_total: float
    n_psu: int
    n_strata: int
    n_strata_single_psu: int


@dataclass(frozen=True)
class LevelChoice:
    """Estimación seleccionada por la regla de suficiencia y su procedencia."""

    estimate: ProportionEstimate
    level: str
    source: str
    n_rentadas: int


@dataclass(frozen=True)
class RegionalRow:
    """Una fila de ``regional_calibration.parquet``."""

    municipio_clave: str
    municipio_nombre: str
    entidad: str
    indicador: str
    level: str
    source: str
    year: int
    n_muestra: int
    suma_factores: float
    n_rentadas: int
    n_total: int
    suma_factores_total: float
    estimate: float
    se: float
    cv: float
    n_psu: int
    n_strata: int
    n_strata_single_psu: int
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float


def municipality_key(entidad: object, mun: object) -> str:
    """Compone la clave ENT + MUN normalizando ceros a la izquierda."""
    return _padded(entidad, 2) + _padded(mun, 3)


def normalize_code(value: object) -> str:
    """Código de INEGI sin espacios ni ceros a la izquierda superfluos."""
    text = str(value).strip()
    if text.isdigit():
        return str(int(text))
    return text


def weighted_proportion(
    estrato: Sequence[str],
    upm: Sequence[str],
    weight: Sequence[float],
    code: Sequence[str],
    indicator: Indicator,
) -> ProportionEstimate:
    """Proporción ponderada con varianza de Taylor y UPM con reemplazo."""
    if not (len(estrato) == len(upm) == len(weight) == len(code)):
        raise ValueError("estrato, upm, weight y code deben tener la misma longitud")
    if not estrato:
        raise ValueError("No hay viviendas para calcular la proporción")

    denominator = 0.0
    numerator = 0.0
    total_weight = 0.0
    n_muestra = 0
    valid_flags: list[bool] = []
    affirmative_flags: list[bool] = []
    for raw_weight, raw_code in zip(weight, code, strict=True):
        normalized = normalize_code(raw_code)
        valid = normalized not in UNSPECIFIED
        affirmative = valid and normalized in indicator.affirmative
        valid_flags.append(valid)
        affirmative_flags.append(affirmative)
        w = float(raw_weight)
        total_weight += w
        if valid:
            denominator += w
            n_muestra += 1
            if affirmative:
                numerator += w
    if denominator <= 0.0:
        raise ValueError("La suma de factores válidos debe ser positiva")

    estimate = numerator / denominator
    linearized: dict[tuple[str, str], float] = {}
    psus: dict[str, set[str]] = {}
    for stratum, unit, raw_weight, valid, affirmative in zip(
        estrato, upm, weight, valid_flags, affirmative_flags, strict=True
    ):
        y = 1.0 if affirmative else 0.0
        x = 1.0 if valid else 0.0
        value = float(raw_weight) * (y - estimate * x) / denominator
        key = (stratum, unit)
        linearized[key] = linearized.get(key, 0.0) + value
        psus.setdefault(stratum, set()).add(unit)

    variance = 0.0
    single_psu = 0
    for stratum, units in psus.items():
        n_h = len(units)
        if n_h < 2:
            single_psu += 1
            continue
        values = [linearized[(stratum, unit)] for unit in units]
        mean = sum(values) / n_h
        variance += (n_h / (n_h - 1)) * sum((value - mean) ** 2 for value in values)

    se = math.sqrt(variance)
    cv = se / estimate if estimate > 0.0 else (0.0 if se == 0.0 else math.inf)
    return ProportionEstimate(
        estimate=estimate,
        se=se,
        cv=cv,
        n_muestra=n_muestra,
        suma_factores=denominator,
        n_total=len(estrato),
        suma_factores_total=total_weight,
        n_psu=sum(len(units) for units in psus.values()),
        n_strata=len(psus),
        n_strata_single_psu=single_psu,
    )


def choose_level(
    municipio: ProportionEstimate | None,
    estado: ProportionEstimate | None,
    nacional: ProportionEstimate | None,
    *,
    n_rentadas_municipio: int,
    n_rentadas_estado: int,
    nacional_source: str = "ENVI",
) -> LevelChoice:
    """Aplica la regla de suficiencia: municipio → estado → nacional."""
    if municipio is not None and n_rentadas_municipio >= MIN_RENTADAS and municipio.cv <= MAX_CV:
        return LevelChoice(municipio, "municipio", "Censo", n_rentadas_municipio)
    if estado is not None and n_rentadas_estado >= MIN_RENTADAS and estado.cv <= MAX_CV:
        return LevelChoice(estado, "estado", "Censo", n_rentadas_estado)
    if nacional is not None and nacional.cv <= MAX_CV:
        return LevelChoice(nacional, "nacional", nacional_source, nacional.n_muestra)
    raise RegionalError("Ningún nivel cumple la regla de suficiencia (>= 30 rentadas y CV <= 30 %)")


def run_regional(paths: ProjectPaths) -> str:
    """Genera el Parquet regional y su reporte con los archivos descargados."""
    sources = {source.id: source for source in load_sources(paths.sources_toml)}
    frame = _load_censo(paths, sources)
    baselines = _load_baselines(paths)
    rows = build_rows(frame, baselines)

    parquet_path = paths.data_processed / "regional_calibration.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    rows_to_frame(rows).to_parquet(parquet_path, index=False)

    report_path = paths.reports / "regional_calibration.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(rows), encoding="utf-8")
    return f"prepare regional: {parquet_path.as_posix()} ({len(rows)} filas) - {report_path.as_posix()}"


def build_rows(frame: pd.DataFrame, baselines: pd.DataFrame) -> list[RegionalRow]:
    """Construye las filas de los 7 municipios por 7 indicadores más renta."""
    rows: list[RegionalRow] = []
    state_cache: dict[tuple[str, str], ProportionEstimate | None] = {}
    for municipality in MUNICIPIOS:
        municipal = frame.loc[frame["clave"] == municipality.clave]
        state = frame.loc[frame["entidad"] == municipality.entidad]
        n_rentadas_municipio = int(municipal["rented"].sum())
        n_rentadas_estado = int(state["rented"].sum())

        for indicator in INDICATORS:
            municipal_estimate = _estimate(municipal, indicator)
            cache_key = (municipality.entidad, indicator.key)
            if cache_key not in state_cache:
                state_cache[cache_key] = _estimate(state, indicator)
            state_estimate = state_cache[cache_key]
            nacional_estimate = _national_estimate(indicator, baselines)
            choice = choose_level(
                municipal_estimate,
                state_estimate,
                nacional_estimate,
                n_rentadas_municipio=n_rentadas_municipio,
                n_rentadas_estado=n_rentadas_estado,
            )
            rows.append(_ratio_row(municipality, indicator, choice))

        rows.append(_rent_row(municipality, baselines))
    return rows


def rows_to_frame(rows: list[RegionalRow] | tuple[RegionalRow, ...]) -> pd.DataFrame:
    """Convierte las filas en un ``DataFrame`` con columnas estables."""
    frame = pd.DataFrame([asdict(row) for row in rows], columns=list(REGIONAL_COLUMNS))
    return frame.sort_values(["municipio_clave", "indicador"], kind="stable").reset_index(drop=True)


def render_report(rows: list[RegionalRow] | tuple[RegionalRow, ...]) -> str:
    """Reporte Markdown con la tabla de los 7 municipios y la fórmula aplicada."""
    by_municipio: dict[str, dict[str, RegionalRow]] = {}
    for row in rows:
        by_municipio.setdefault(row.municipio_clave, {})[row.indicador] = row

    lines = [
        "# Calibración regional — ML-02 T04",
        "",
        "Generado por `uv run python -m thehouse_ml prepare`. Fuente: INEGI, Censo de Población y Vivienda 2020,",
        "muestra del cuestionario ampliado (tabla Viviendas, solo Hidalgo 13 y Estado de México 15), citado como",
        "autor del producto. INEGI no respalda este proyecto. La renta procede de la ENVI 2020 (M-ML-015).",
        "",
        "## Los 7 municipios",
        "",
        "Cada celda muestra el valor y el nivel del que salió: `municipio`, `estado` (entidad completa) o `nacional`.",
        "Los porcentajes están ponderados con `FACTOR`; se excluyen los códigos 9 y blanco del denominador.",
        "",
        "| Clave | Municipio | Rentadas | Internet | Electricidad | Agua entubada | Agua dentro | "
        "Drenaje | Gas | Renta p50 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for municipality in MUNICIPIOS:
        cells = [municipality.clave, municipality.nombre]
        for indicator in INDICATORS:
            cells.append(_format_cell(by_municipio[municipality.clave][indicator.key], percent=True))
        cells.append(_format_cell(by_municipio[municipality.clave]["renta"], percent=False))
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Detalle de la estimación",
            "",
            "`n` es el conteo sin ponderar de viviendas válidas para el indicador; `Σ w` es la suma de sus factores.",
            "`ee` es el error estándar y `cv = ee / estimación`. La renta no trae `ee`/`cv` porque la ENVI 2020",
            "de T03 no publica varianza para sus percentiles; se marcan con `—`.",
            "",
            "| Municipio | Indicador | Nivel | Fuente | n | Σ w | Rentadas | Estimación | ee | cv |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (item.municipio_clave, item.indicador)):
        estimate = f"{row.estimate * 100:.1f} %" if row.indicador != "renta" else f"{row.estimate:,.0f}"
        se = "—" if math.isnan(row.se) else f"{row.se:.4f}"
        cv = "—" if math.isnan(row.cv) else f"{row.cv * 100:.1f} %"
        lines.append(
            f"| {row.municipio_nombre} | {row.indicador} | {row.level} | {row.source} | "
            f"{row.n_muestra:,} | {row.suma_factores:,.0f} | {row.n_rentadas:,} | {estimate} | {se} | {cv} |"
        )

    lines.extend(
        [
            "",
            "## Método",
            "",
            "- **Proporción:** `p = Σ(w_i y_i) / Σ(w_i x_i)`, con `x_i = 1` si el código no es 9/blanco y",
            "  `y_i = 1` si además está en los códigos afirmativos confirmados (M-ML-015).",
            "- **Varianza (linealización de Taylor para razón, UPM con reemplazo):**",
            "  `l_i = w_i (y_i - p x_i) / Σ(w_j x_j)`; agregando por estrato `h` y UPM `j`, `L_hj = Σ l_i`,",
            "  `Var(p) = Σ_h [ n_h / (n_h - 1) · Σ_j (L_hj - media(L_h*))² ]`, `ee = √Var(p)` y `cv = ee / p`.",
            "- Los estratos con una sola UPM no aportan varianza y se cuentan en la columna `n_strata_single_psu`",
            "  del Parquet; es una limitación documentada del diseño con reemplazo.",
            "- **Suficiencia:** el nivel municipal solo se usa con `>= 30` viviendas rentadas en la muestra y",
            "  `cv <= 30 %`; si no, se usa la entidad completa (`13`/`15`). El respaldo nacional de ENVI/ENIGH de",
            "  T03 existe para la proporción de rentadas; T03 no publica prevalencias de servicios (M-ML-014(c)),",
            "  así que para esos indicadores la entidad es el último nivel disponible. Cada fila registra su nivel.",
            "- **Renta:** el Censo no pregunta el monto. Para cada municipio se copian los percentiles p10-p90 en",
            "  pesos de julio de 2024 de la fila estatal de ENVI 2020 en `national_baselines.parquet` (T03),",
            "  con `level = estado` y `source = ENVI` (M-ML-014, M-ML-015).",
            "",
        ]
    )
    return "\n".join(lines)


def _padded(value: object, width: int) -> str:
    text = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value).strip()
    if text.isdigit() and len(text) < width:
        return text.zfill(width)
    return text


def _format_cell(row: RegionalRow, *, percent: bool) -> str:
    if math.isnan(row.estimate):
        return "—"
    value = f"{row.estimate * 100:.1f} %" if percent else f"${row.estimate:,.0f}"
    label = {"municipio": "mun", "estado": "edo", "nacional": "nac"}.get(row.level, row.level)
    return f"{value} ({label})"


def _load_censo(paths: ProjectPaths, sources: dict[str, SourceSpec]) -> pd.DataFrame:
    source = sources.get(CENSO_SOURCE)
    if source is None:
        raise RegionalError(f"Fuente no registrada: {CENSO_SOURCE}")
    data_files = [file for file in source.download_url if file.kind == "data"]
    if not data_files:
        raise RegionalError(f"{CENSO_SOURCE}: sin archivos de datos registrados")
    frames: list[pd.DataFrame] = []
    for file in data_files:
        local = paths.data_raw / source.id / file.filename
        if not local.is_file():
            raise RegionalError(f"{CENSO_SOURCE}: falta el archivo {local.as_posix()}; ejecute primero `download`")
        try:
            frames.append(read_zip_csv(local, file.table, file.encoding, CENSO_COLUMNS))
        except CsvReadError as exc:
            raise RegionalError(str(exc)) from exc
    return _normalize_censo(pd.concat(frames, ignore_index=True))


def _normalize_censo(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "entidad": [_padded(value, 2) for value in frame["ENT"]],
            "mun": [_padded(value, 3) for value in frame["MUN"]],
            "weight": pd.to_numeric(frame["FACTOR"].astype(str).str.strip(), errors="coerce").fillna(0.0),
            "estrato": frame["ESTRATO"].astype(str).str.strip(),
            "upm": frame["UPM"].astype(str).str.strip(),
            "rented": [normalize_code(value) == CODE_TENENCIA_RENTADA for value in frame["TENENCIA"]],
        }
    )
    for indicator in INDICATORS:
        out[indicator.key] = [normalize_code(value) for value in frame[indicator.variable]]
    out["clave"] = out["entidad"] + out["mun"]
    return out


def _load_baselines(paths: ProjectPaths) -> pd.DataFrame:
    path = paths.data_processed / "national_baselines.parquet"
    if not path.is_file():
        raise RegionalError(f"Falta {path.as_posix()}; ejecute primero la parte nacional de `prepare`")
    return pd.read_parquet(path)


def _estimate(frame: pd.DataFrame, indicator: Indicator) -> ProportionEstimate | None:
    if frame.empty:
        return None
    try:
        return weighted_proportion(
            estrato=frame["estrato"].tolist(),
            upm=frame["upm"].tolist(),
            weight=frame["weight"].tolist(),
            code=frame[indicator.key].tolist(),
            indicator=indicator,
        )
    except ValueError:
        return None


def _national_estimate(indicator: Indicator, baselines: pd.DataFrame) -> ProportionEstimate | None:
    """Respaldo nacional desde T03; solo existe para la proporción de rentadas."""
    if indicator.key != "rentadas":
        return None
    row = _baseline_row(baselines, ENVI_SOURCE, "nacional", "nacional")
    if row is None:
        return None
    n_muestra = int(row["n_muestra"])
    # T03 no publica varianza: el nivel nacional se usa como respaldo con cv = 0.
    return ProportionEstimate(
        estimate=float(row["rented_share"]),
        se=0.0,
        cv=0.0,
        n_muestra=n_muestra,
        suma_factores=float(row["suma_factores_total"]),
        n_total=int(row["n_total"]),
        suma_factores_total=float(row["suma_factores_total"]),
        n_psu=0,
        n_strata=0,
        n_strata_single_psu=0,
    )


def _ratio_row(municipality: Municipality, indicator: Indicator, choice: LevelChoice) -> RegionalRow:
    estimate = choice.estimate
    return RegionalRow(
        municipio_clave=municipality.clave,
        municipio_nombre=municipality.nombre,
        entidad=municipality.entidad,
        indicador=indicator.key,
        level=choice.level,
        source=choice.source,
        year=2020,
        n_muestra=estimate.n_muestra,
        suma_factores=estimate.suma_factores,
        n_rentadas=choice.n_rentadas,
        n_total=estimate.n_total,
        suma_factores_total=estimate.suma_factores_total,
        estimate=estimate.estimate,
        se=estimate.se,
        cv=estimate.cv,
        n_psu=estimate.n_psu,
        n_strata=estimate.n_strata,
        n_strata_single_psu=estimate.n_strata_single_psu,
        p10=math.nan,
        p25=math.nan,
        p50=math.nan,
        p75=math.nan,
        p90=math.nan,
    )


def _rent_row(municipality: Municipality, baselines: pd.DataFrame) -> RegionalRow:
    candidates: tuple[tuple[str, str, str, str], ...] = (
        (ENVI_SOURCE, "entidad", municipality.entidad, "estado"),
        (ENVI_SOURCE, "nacional", "nacional", "nacional"),
        (ENIGH_SOURCE, "nacional", "nacional", "nacional"),
    )
    chosen: tuple[str, str, pd.Series] | None = None
    for source_id, level, value, out_level in candidates:
        row = _baseline_row(baselines, source_id, level, value)
        if row is not None:
            chosen = (out_level, source_id, row)
            break
    if chosen is None:
        raise RegionalError(f"{municipality.clave}: sin percentiles de renta válidos en T03")

    level, source_id, row = chosen
    return RegionalRow(
        municipio_clave=municipality.clave,
        municipio_nombre=municipality.nombre,
        entidad=municipality.entidad,
        indicador="renta",
        level=level,
        source="ENVI" if source_id == ENVI_SOURCE else "ENIGH",
        year=2020 if source_id == ENVI_SOURCE else 2022,
        n_muestra=int(row["n_muestra"]),
        suma_factores=float(row["suma_factores"]),
        n_rentadas=int(row["n_muestra"]),
        n_total=int(row["n_total"]),
        suma_factores_total=float(row["suma_factores_total"]),
        estimate=float(row["p50_jul2024"]),
        se=math.nan,
        cv=math.nan,
        n_psu=0,
        n_strata=0,
        n_strata_single_psu=0,
        p10=float(row["p10_jul2024"]),
        p25=float(row["p25_jul2024"]),
        p50=float(row["p50_jul2024"]),
        p75=float(row["p75_jul2024"]),
        p90=float(row["p90_jul2024"]),
    )


def _baseline_row(baselines: pd.DataFrame, source: str, level: str, value: str) -> pd.Series | None:
    selected = baselines.loc[
        (baselines["source"] == source) & (baselines["level"] == level) & (baselines["level_value"] == value)
    ]
    return selected.iloc[0] if len(selected) == 1 else None
