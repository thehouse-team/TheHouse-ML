"""Etapa ``download``: descarga verificable con manifiesto SHA-256.

Modos de operación por archivo:

* si el archivo ya está en ``data/raw/<id>/`` (lo puso a mano el humano), no
  se descarga: solo se verifica su SHA-256 y se escribe el manifiesto;
* si no está, se descarga y se validan tamaño y hash;
* una fuente con ``download_url`` vacío se salta con el aviso
  ``download_url pendiente``.

Un hash distinto al del registro o al del manifiesto anterior falla con un
mensaje explícito y no sobrescribe el archivo ni el manifiesto.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from thehouse_ml.manifests import Manifest, ManifestFile, load_manifest, manifest_path, write_manifest
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.sources import FileSpec, SourceSpec, load_sources, select_sources
from thehouse_ml.variables import VariableCheckError, verify_sources, write_variables_report

Fetcher = Callable[[str, Path], None]
CHUNK_SIZE = 1024 * 1024


class DownloadError(RuntimeError):
    """La descarga, el tamaño o el SHA-256 no coincide con lo registrado."""


@dataclass(frozen=True)
class DownloadOutcome:
    """Resultado de procesar una fuente."""

    source_id: str
    status: str
    message: str

    def render(self) -> str:
        """Línea para la consola."""
        return f"{self.source_id}: {self.status} - {self.message}"


def default_fetch(url: str, destination: Path) -> None:
    """Descarga ``url`` a ``destination`` con la biblioteca estándar."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,  # noqa: S310 - URL del registro verificado
                partial.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DownloadError(f"No se pudo descargar {url}: {exc}") from exc
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()


def download_source(
    source: SourceSpec,
    paths: ProjectPaths,
    fetcher: Fetcher | None = None,
    today: str | None = None,
) -> DownloadOutcome:
    """Descarga (o verifica) los archivos de una fuente y escribe su manifiesto."""
    if not source.download_url:
        return DownloadOutcome(source.id, "pendiente", "download_url pendiente")

    previous = load_manifest(manifest_path(paths, source.id))
    previous_files = {file.filename: file for file in previous.files} if previous else {}
    download_date = today or date.today().isoformat()
    entries: list[ManifestFile] = []
    downloaded = 0
    manual = 0

    for file in source.download_url:
        destination = paths.data_raw / source.id / file.filename
        if destination.is_file():
            manual += 1
        else:
            (fetcher or default_fetch)(file.url, destination)
            downloaded += 1
        digest, size = _digest_and_size(destination)
        _check_file(file, digest, size, previous_files.get(file.filename), destination)
        entries.append(
            ManifestFile(
                url=file.url,
                filename=file.filename,
                downloaded_at=download_date,
                size_bytes=size,
                sha256=digest,
                license=source.license,
            )
        )

    write_manifest(manifest_path(paths, source.id), Manifest(source.id, tuple(entries)))
    if downloaded and manual:
        status, message = "mixto", f"{len(entries)} archivo(s) verificados"
    elif downloaded:
        status, message = "descargado", f"{len(entries)} archivo(s) verificados"
    else:
        status, message = "manual", f"{len(entries)} archivo(s) verificados"
    return DownloadOutcome(source.id, status, message)


def download_all(
    paths: ProjectPaths,
    source_id: str | None = None,
    fetcher: Fetcher | None = None,
    today: str | None = None,
) -> list[DownloadOutcome]:
    """Descarga todas las fuentes registradas, o solo ``source_id``."""
    sources = select_sources(load_sources(paths.sources_toml), source_id)
    return [download_source(source, paths, fetcher, today) for source in sources]


def run_download(
    paths: ProjectPaths,
    source_id: str | None = None,
    fetcher: Fetcher | None = None,
) -> str:
    """Ejecuta la etapa completa: descarga, manifiestos y reporte de variables."""
    sources = select_sources(load_sources(paths.sources_toml), source_id)
    outcomes = [download_source(source, paths, fetcher) for source in sources]
    report = verify_sources(sources, paths)
    report_path = paths.reports / "variables_check.md"
    write_variables_report(report_path, report)

    lines = [outcome.render() for outcome in outcomes]
    lines.append(f"variables_check: {report_path.as_posix()}")
    if report.failures:
        raise VariableCheckError("Variables confirmadas ausentes: " + "; ".join(report.failures))
    return "\n".join(lines)


def _digest_and_size(path: Path) -> tuple[str, int]:
    """Calcula SHA-256 y tamaño sin cargar el archivo completo en memoria."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _check_file(
    file: FileSpec,
    digest: str,
    size: int,
    previous: ManifestFile | None,
    destination: Path,
) -> None:
    if file.size_bytes is not None and size != file.size_bytes:
        raise DownloadError(f"{destination.name}: tamaño esperado {file.size_bytes} bytes, obtenido {size} bytes")
    if file.sha256 is not None and digest != file.sha256:
        raise DownloadError(f"{destination.name}: SHA-256 no coincide. Esperado {file.sha256}, obtenido {digest}")
    if previous is not None and previous.sha256 != digest:
        raise DownloadError(
            f"{destination.name}: el manifiesto existente registra SHA-256 {previous.sha256}, obtenido {digest}"
        )
