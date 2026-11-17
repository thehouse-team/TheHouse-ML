"""Ayudas compartidas: fixtures sintéticos y constructores de registro sin red."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path

from thehouse_ml.sources import FileSpec, SourceSpec

FIXTURES = Path(__file__).resolve().parent / "fixtures"


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
