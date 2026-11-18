"""Ayudas compartidas: fixtures sintéticos y constructores de registro sin red."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path

from thehouse_ml.sources import FileSpec, SourceSpec

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]


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
    return codes
