"""ML-02 T05: entorno de los campus con el DENUE 05/2026 (M-ML-018).

Fuentes: DENUE de Hidalgo (13) y Estado de México (15). Se leen **solo** las
variables registradas en ``data/sources.toml`` (``usecols``) y por chunks, de
modo que nombres, teléfonos, correos y direcciones nunca llegan a memoria ni a
las salidas. Solo se publican conteos agregados por campus, radio y categoría.

Salidas: ``data/processed/campus_context.parquet`` (fuera de git, nombre de la
spec ML-02 §6.2) y ``reports/denue_campus.md`` (en git).

Atribución: "Fuente: INEGI, DENUE 05/2026".
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from thehouse_ml.csvio import CsvReadError, iter_zip_csv
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.sources import load_sources
from thehouse_ml.universities import haversine_m

DENUE_SOURCE = "inegi_denue_vigente"

# Radio y categorías en un solo lugar: el reporte, el Parquet y las pruebas los
# consumen de aquí (no hay literales repetidos). ``total`` (prefijos vacíos)
# cuenta todos los establecimientos con coordenada válida.
RADII_M: tuple[int, ...] = (1000, 2000, 5000)
CATEGORIES: dict[str, tuple[str, ...]] = {
    "comida": ("722",),
    "abarrotes_super": ("4611", "4621"),
    "farmacia": ("464111", "464112"),
    "salud": ("621", "622"),
    "papeleria_copias": ("465311", "561431"),
    "lavanderia": ("812210",),
    "gimnasio": ("713943",),
    "transporte": ("485",),
    "bancos": ("522",),
    "total": (),
}
ALL_CATEGORIES: tuple[str, ...] = tuple(CATEGORIES)

# Únicas columnas que se leen del DENUE. El registro de fuentes es la lista
# autorizada; esta tupla es su copia en código para el lector por chunks.
DENUE_COLUMNS: tuple[str, ...] = (
    "id",
    "codigo_act",
    "per_ocu",
    "cve_ent",
    "cve_mun",
    "latitud",
    "longitud",
    "fecha_alta",
)
# 5 km son ~0.045° de latitud y ~0.048° de longitud en esta región; 0.07° es
# una cota generosa que evita calcular haversine para la mayoría de los renglones.
BOUND_DEGREES = 0.07
COUNT_CHUNK_SIZE = 100_000
UNVERIFIED_STATUS = "seed_sin_verificar"

CAMPUS_COLUMNS = ("code", "name", "final_lat", "final_lon", "final_status")
DENUE_PARQUET_COLUMNS = ("code", "radius_m", "category", "count", "final_status")
ATTRIBUTION = "Fuente: INEGI, DENUE 05/2026"


class DenueError(RuntimeError):
    """Falta el DENUE, el CSV de universidades o una coordenada final válida."""


@dataclass(frozen=True)
class Campus:
    """Universidad con su coordenada final de la T07."""

    code: str
    name: str
    lat: float
    lon: float
    final_status: str


@dataclass(frozen=True)
class DenueRow:
    """Una fila de ``campus_context.parquet``: conteo agregado, nunca un negocio."""

    code: str
    radius_m: int
    category: str
    count: int
    final_status: str


DENUE_SKIP_MESSAGE = "denue: omitido (sin archivos DENUE; ejecute download --source inegi_denue_vigente)"


def categorize(code: object) -> tuple[str, ...]:
    """Devuelve las categorías cuyo prefijo SCIAN coincide con ``code``."""
    normalized = str(code).strip()
    return tuple(name for name, prefixes in CATEGORIES.items() if not prefixes or normalized.startswith(prefixes))


def load_campuses(path: Path) -> tuple[Campus, ...]:
    """Lee ``data/universities.csv`` y valida las columnas finales de la T07."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = set(CAMPUS_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise DenueError(f"{path.name}: faltan columnas {', '.join(sorted(missing))}")
            campuses: list[Campus] = []
            for row in reader:
                code = (row.get("code") or "").strip()
                if not code:
                    continue
                lat = _coordinate(row.get("final_lat"), code, "final_lat")
                lon = _coordinate(row.get("final_lon"), code, "final_lon")
                campuses.append(
                    Campus(
                        code=code,
                        name=(row.get("name") or "").strip(),
                        lat=lat,
                        lon=lon,
                        final_status=(row.get("final_status") or "").strip(),
                    )
                )
    except OSError as exc:
        raise DenueError(f"No se puede leer {path}: {exc}") from exc
    if not campuses:
        raise DenueError(f"{path.name}: no hay universidades")
    return tuple(campuses)


def accumulate_counts(
    frame: pd.DataFrame,
    campuses: Sequence[Campus],
    counts: dict[tuple[str, int, str], int],
) -> int:
    """Suma los conteos de un chunk; devuelve cuántos renglones son válidos."""
    businesses = _businesses(frame)
    for campus in campuses:
        nearby = [
            (lat, lon, categories)
            for lat, lon, categories in businesses
            if abs(lat - campus.lat) <= BOUND_DEGREES and abs(lon - campus.lon) <= BOUND_DEGREES
        ]
        for lat, lon, categories in nearby:
            distance = haversine_m(campus.lat, campus.lon, lat, lon)
            for radius in RADII_M:
                if distance > radius:  # la frontera exacta SÍ cuenta (<=)
                    continue
                for category in categories:
                    key = (campus.code, radius, category)
                    counts[key] = counts.get(key, 0) + 1
    return len(businesses)


def build_rows(
    campuses: Sequence[Campus],
    counts: Mapping[tuple[str, int, str], int],
) -> list[DenueRow]:
    """Materializa los conteos en filas con esquema estable."""
    rows = [
        DenueRow(
            code=campus.code,
            radius_m=radius,
            category=category,
            count=counts.get((campus.code, radius, category), 0),
            final_status=campus.final_status,
        )
        for campus in campuses
        for radius in RADII_M
        for category in ALL_CATEGORIES
    ]
    category_order = {name: index for index, name in enumerate(ALL_CATEGORIES)}
    return sorted(rows, key=lambda row: (row.code, row.radius_m, category_order[row.category]))


def rows_to_frame(rows: Sequence[DenueRow]) -> pd.DataFrame:
    """Convierte las filas en un ``DataFrame`` con columnas estables."""
    return pd.DataFrame([asdict(row) for row in rows], columns=list(DENUE_PARQUET_COLUMNS))


def render_report(rows: Sequence[DenueRow], campuses: Sequence[Campus]) -> str:
    """Reporte con resumen a 2 km, una tabla por campus y la sección sin verificar."""
    by_key = {(row.code, row.radius_m, row.category): row.count for row in rows}
    header = "| Radio (m) | " + " | ".join(ALL_CATEGORIES) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in ALL_CATEGORIES) + " |"

    lines = [
        "# Entorno de campus — ML-02 T05",
        "",
        f'Generado por `uv run python -m thehouse_ml prepare`. **Atribución: "{ATTRIBUTION}".**',
        "Salida de datos: `data/processed/campus_context.parquet` (fuera de git). INEGI no respalda este proyecto.",
        "",
        "Conteos **agregados** de establecimientos a 1 000, 2 000 y 5 000 m de cada campus,",
        "con distancia haversine (el límite exacto del radio sí cuenta). Se usan los dos estados,",
        "Hidalgo (13) y Estado de México (15), porque Tepeji y Jilotepec están cerca de la frontera.",
        "Solo se publican conteos: ni nombres, ni teléfonos, ni correos, ni direcciones.",
        "",
        "Categorías por prefijo SCIAN (`codigo_act`): "
        + "; ".join(
            f"`{name}` = {', '.join(prefixes) if prefixes else 'todos'}" for name, prefixes in CATEGORIES.items()
        )
        + ".",
        "",
        "## Conteos a 2 km por universidad y categoría",
        "",
        "| Universidad | Código | " + " | ".join(ALL_CATEGORIES) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in ALL_CATEGORIES) + " |",
    ]
    for campus in campuses:
        cells = [f"{by_key.get((campus.code, 2000, category), 0):,}" for category in ALL_CATEGORIES]
        lines.append(f"| {campus.name} | `{campus.code}` | " + " | ".join(cells) + " |")

    verified = [campus for campus in campuses if campus.final_status != UNVERIFIED_STATUS]
    unverified = [campus for campus in campuses if campus.final_status == UNVERIFIED_STATUS]
    lines.extend(["", _section("Universidades con coordenada verificada", verified, header, separator, by_key)])
    if unverified:
        lines.extend(
            [
                "",
                _section(
                    "Universidades con coordenada sin verificar (`seed_sin_verificar`)",
                    unverified,
                    header,
                    separator,
                    by_key,
                ),
                "",
                "> **Advertencia:** la coordenada de estas universidades no está verificada. Se mantienen las",
                "> del seed y no deben usarse como coordenadas verificadas hasta que el backend confirme el",
                "> nombre y la ubicación (en particular `UCAN`). Sus conteos son solo orientativos.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def run_denue(paths: ProjectPaths) -> str:
    """Genera el Parquet y el reporte con los ZIP descargados del DENUE."""
    sources = {source.id: source for source in load_sources(paths.sources_toml)}
    source = sources.get(DENUE_SOURCE)
    if source is None:
        raise DenueError(f"Fuente no registrada: {DENUE_SOURCE}")
    data_files = [file for file in source.download_url if file.kind == "data"]
    if not data_files:
        return DENUE_SKIP_MESSAGE
    archives = [paths.data_raw / source.id / file.filename for file in data_files]
    missing = [path for path in archives if not path.is_file()]
    if len(missing) == len(archives):
        return DENUE_SKIP_MESSAGE
    if missing:
        raise DenueError(
            f"{DENUE_SOURCE}: faltan {', '.join(path.name for path in missing)}; "
            "ejecute `download --source inegi_denue_vigente`"
        )

    campuses = load_campuses(paths.universities_csv)
    counts: dict[tuple[str, int, str], int] = {}
    n_rows = 0
    for file, archive in zip(data_files, archives, strict=True):
        try:
            for chunk in iter_zip_csv(archive, file.table, file.encoding, DENUE_COLUMNS, COUNT_CHUNK_SIZE):
                n_rows += accumulate_counts(chunk, campuses, counts)
        except CsvReadError as exc:
            raise DenueError(str(exc)) from exc

    rows = build_rows(campuses, counts)
    parquet_path = paths.data_processed / "campus_context.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    rows_to_frame(rows).to_parquet(parquet_path, index=False)

    report_path = paths.reports / "denue_campus.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(rows, campuses), encoding="utf-8", newline="\n")
    return (
        f"denue: {parquet_path.as_posix()} ({len(rows)} filas, {n_rows:,} establecimientos) - {report_path.as_posix()}"
    )


def _section(
    title: str,
    campuses: Sequence[Campus],
    header: str,
    separator: str,
    by_key: Mapping[tuple[str, int, str], int],
) -> str:
    lines = [f"## {title}", ""]
    for campus in campuses:
        lines.extend(
            [
                f"### `{campus.code}` — {campus.name}",
                "",
                f"- `final_status`: `{campus.final_status}`; coordenada final: `{campus.lat}, {campus.lon}`.",
                "",
                header,
                separator,
            ]
        )
        for radius in RADII_M:
            cells = [f"{by_key.get((campus.code, radius, category), 0):,}" for category in ALL_CATEGORIES]
            lines.append(f"| {radius:,} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines).rstrip()


def _businesses(frame: pd.DataFrame) -> list[tuple[float, float, tuple[str, ...]]]:
    """Convierte un chunk en pares (lat, lon, categorías), sin columnas privadas."""
    out: list[tuple[float, float, tuple[str, ...]]] = []
    for lat_raw, lon_raw, code in frame[["latitud", "longitud", "codigo_act"]].itertuples(index=False, name=None):
        lat, lon = _to_float(lat_raw), _to_float(lon_raw)
        if lat is None or lon is None or (lat == 0.0 and lon == 0.0):
            continue
        out.append((lat, lon, categorize(code)))
    return out


def _to_float(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _coordinate(value: object, code: str, field: str) -> float:
    number = _to_float(value)
    if number is None:
        raise DenueError(f"{code}: {field} no es numérico ({value!r})")
    return number


__all__ = [
    "ALL_CATEGORIES",
    "ATTRIBUTION",
    "CAMPUS_COLUMNS",
    "CATEGORIES",
    "DENUE_COLUMNS",
    "DENUE_PARQUET_COLUMNS",
    "DENUE_SKIP_MESSAGE",
    "DENUE_SOURCE",
    "RADII_M",
    "UNVERIFIED_STATUS",
    "Campus",
    "DenueError",
    "DenueRow",
    "accumulate_counts",
    "build_rows",
    "categorize",
    "load_campuses",
    "render_report",
    "rows_to_frame",
    "run_denue",
]
