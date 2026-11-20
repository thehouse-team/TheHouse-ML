"""Ayudas compartidas: fixtures sintéticos y constructores de registro sin red."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

from thehouse_ml.manifests import Manifest, ManifestFile, write_manifest
from thehouse_ml.sources import FileSpec, SourceSpec
from thehouse_ml.universities import UniversitySeed, build_query_url

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]
OSM_DOWNLOADED_AT = "2026-11-17"


def fixture_bytes(name: str) -> bytes:
    """Contenido de un fixture."""
    return (FIXTURES / name).read_bytes()


def fixture_sha256(name: str) -> str:
    """SHA-256 de un fixture."""
    return hashlib.sha256(fixture_bytes(name)).hexdigest()


def make_file(
    name: str,
    *,
    table: str | None = None,
    encoding: str = "utf-8",
    kind: str = "data",
    size_bytes: int | None = None,
    sha256: str | None = None,
    with_checks: bool = True,
) -> FileSpec:
    """Construye un FileSpec; por defecto usa el tamaño y hash reales del fixture."""
    path = FIXTURES / name
    return FileSpec(
        url=f"https://example.test/{name}",
        filename=name,
        size_bytes=path.stat().st_size if with_checks and size_bytes is None else size_bytes,
        sha256=fixture_sha256(name) if with_checks and sha256 is None else sha256,
        kind=kind,
        table=table,
        encoding=encoding,
    )


def make_source(
    source_id: str,
    variables: tuple[str, ...],
    files: tuple[FileSpec, ...] = (),
    *,
    status: str = "confirmed",
    reference: str | None = None,
    license_text: str = "Licencia de prueba",
) -> SourceSpec:
    """Construye un SourceSpec para pruebas."""
    return SourceSpec(
        id=source_id,
        name=f"Fuente de prueba {source_id}",
        license=license_text,
        attribution="Atribucion de prueba",
        variables_status=status,
        variables_used=variables,
        download_url=files,
        status_reference=reference,
    )


def fixture_fetcher() -> Callable[[str, Path], None]:
    """Simula la red copiando el fixture cuyo nombre pide la descarga."""

    def fetch(_url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FIXTURES / destination.name, destination)

    return fetch


def write_osm_project(root: Path, *, with_responses: bool = True) -> tuple[str, ...]:
    """Copia el seed real y escribe respuestas jsonv2 inventadas para las 10 universidades.

    Por defecto `UMB-VDC` queda a ~555 m (`revisar`) y `UCAN` solo tiene un
    elemento no educativo (`sin_coincidencia`); el resto quedan en `ok`.
    """
    seed_source = REPO_ROOT / "data" / "seed" / "universities_seed.csv"
    seed_local = root / "data" / "seed" / "universities_seed.csv"
    seed_local.parent.mkdir(parents=True, exist_ok=True)
    seed_local.write_bytes(seed_source.read_bytes())
    manual_source = REPO_ROOT / "data" / "seed" / "universities_manual.csv"
    (root / "data" / "seed" / "universities_manual.csv").write_bytes(manual_source.read_bytes())

    with seed_source.open("r", encoding="utf-8", newline="") as stream:
        seeds = list(csv.DictReader(stream))
    codes = tuple(str(row["code"]) for row in seeds)
    if not with_responses:
        return codes

    folder = root / "data" / "raw" / "openstreetmap"
    folder.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(seeds):
        code = str(row["code"])
        lat = float(str(row["latitude"]))
        lon = float(str(row["longitude"]))
        if code in {"UAEH-ESTE", "UAEH-ESTL", "UAEH-ESAT", "TESJI", "UMB-JILO"}:
            payload: list[dict[str, object]] = []  # sin coincidencia, como en los datos reales
        else:
            category, kind = "amenity", "university"
            if code == "UMB-VDC":
                lat += 0.005  # ~556 m al norte: fuera del umbral de 200 m.
            if code == "ITSOEH":
                lat += 0.03  # ~3.3 km, como en los datos reales.
            if code == "UCAN":
                category, kind = "building", "university"  # No es educativo segun la regla de la T07.
            payload = [
                {
                    "place_id": 1000 + index,
                    "osm_type": "node",
                    "osm_id": 2000 + index,
                    "lat": f"{lat:.5f}",
                    "lon": f"{lon:.5f}",
                    "name": str(row["name"]),
                    "display_name": str(row["name"]),
                    "class": category,
                    "type": kind,
                }
            ]
        (folder / f"{code}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # Manifiesto OSM inventado: las fechas de `universities.csv` salen de aquí
    # (filas ok_osm) y del CSV manual (resto), nunca del reloj del sistema.
    files = tuple(
        ManifestFile(
            url=build_query_url(
                UniversitySeed(
                    code=str(row["code"]),
                    name=str(row["name"]),
                    city=str(row["city"]),
                    latitude=str(row["latitude"]),
                    longitude=str(row["longitude"]),
                )
            ),
            filename=f"{row['code']}.json",
            downloaded_at=OSM_DOWNLOADED_AT,
            size_bytes=(folder / f"{row['code']}.json").stat().st_size,
            sha256=hashlib.sha256((folder / f"{row['code']}.json").read_bytes()).hexdigest(),
            license="Open Database License (ODbL) 1.0",
        )
        for row in seeds
    )
    write_manifest(root / "data" / "manifests" / "openstreetmap.json", Manifest("openstreetmap", files))
    return codes


# ---------------------------------------------------------------------------
# ML-03 T06: estudiantes y cuartos sintéticos desde insumos versionados (sin red ni data/processed)
# ---------------------------------------------------------------------------

# Renta p50 estatal de la ENVI 2020 en pesos de jul-2024 (reports/regional_calibration.md, M-ML-015).
STATE_P50 = {"13": 1872.167721, "15": 2496.223627}
# Histogramas agrupados (≥ 30 viviendas por celda) que publica reports/data_card.md §7, en pesos de jul-2024.
STATE_HISTOGRAM_BINS = {
    "13": ((500, 1500, 32597), (1500, 2000, 16127), (2000, 7500, 32551)),
    "15": ((500, 1500, 165766), (1500, 2000, 137990), (2000, 2500, 129732), (2500, 3500, 121903), (3500, None, 183989)),
}
CENSUS_COLUMNS = {"wifi": "Internet", "electricity": "Electricidad", "water": "Agua entubada", "gas": "Gas"}


def published_census() -> dict[str, dict[str, float]]:
    """Proporciones del Censo por municipio, tal como las publica reports/regional_calibration.md."""
    lines = (REPO_ROOT / "reports" / "regional_calibration.md").read_text(encoding="utf-8").splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("| Clave | Municipio"))
    header = [cell.strip() for cell in lines[header_index].strip("|").split("|")]
    result: dict[str, dict[str, float]] = {}
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        cells = dict(zip(header, (cell.strip() for cell in line.strip("|").split("|")), strict=True))
        result[cells["Clave"]] = {
            service: float(cells[column].split("%")[0]) / 100.0 for service, column in CENSUS_COLUMNS.items()
        }
    return result


@cache
def versioned_synthetic_data() -> tuple[tuple[Any, ...], tuple[Any, ...], dict[str, Any]]:
    """Los 2,000 estudiantes (T04) y los 3,000 cuartos (T05) generados solo con insumos versionados."""
    from thehouse_ml.features import default_schema, load_campuses
    from thehouse_ml.synthetic import (
        CampusInputs,
        HistogramBin,
        generate_rooms,
        generate_students,
        load_params,
        load_room_params,
    )

    schema = default_schema()
    campuses = load_campuses(REPO_ROOT / "data" / "universities.csv")
    student_params = load_params(schema)
    rent_p50 = {code: STATE_P50[clave[:2]] for code, clave in student_params.municipio_by_campus.items()}
    students = generate_students(student_params, rent_p50, campuses, schema)
    room_params = load_room_params(schema)
    census = published_census()
    inputs = {
        code: CampusInputs(
            STATE_P50[clave[:2]],
            tuple(HistogramBin(lower, upper, weight) for lower, upper, weight in STATE_HISTOGRAM_BINS[clave[:2]]),
            census[clave],
        )
        for code, clave in room_params.municipio_by_campus.items()
    }
    rooms = generate_rooms(room_params, inputs, campuses, schema)
    return tuple(students), tuple(rooms), campuses
