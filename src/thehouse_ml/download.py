"""Etapa ``download``: descarga verificable con manifiesto SHA-256.

Modos de operación por archivo:

* si el archivo ya está en ``data/raw/<id>/`` (lo puso a mano el humano), no
  se descarga: solo se verifica su SHA-256 y se escribe el manifiesto;
* si no está, se descarga y se validan tamaño y hash;
* una fuente descartada (``variables_status = "discarded"``) se omite sin
  descargar ni verificar, con el aviso ``descartada: <status_reference>``
  (M-ML-019);
* una fuente con ``download_url`` vacío se salta con el aviso
  ``download_url pendiente``;
* ``openstreetmap`` es una fuente **bajo demanda** (ML-02 T07): solo se consulta
  con ``download --source openstreetmap``. Una descarga sin filtro la omite con
  el aviso ``omitido: ejecute download --source openstreetmap`` para no repetir
  consultas a Nominatim en CI (M-ML-016). Si se consulta, es una petición por
  universidad, con User-Agent propio, ≤ 1 req/s y cacheo por archivo JSON.

Un hash distinto al del registro o al del manifiesto anterior falla con un
mensaje explícito y no sobrescribe el archivo ni el manifiesto.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from thehouse_ml.manifests import Manifest, ManifestFile, load_manifest, manifest_path, write_manifest
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.sources import FileSpec, SourceSpec, load_sources, select_sources
from thehouse_ml.universities import OSM_SOURCE, download_osm
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
    if source.variables_status == "discarded":
        return DownloadOutcome(source.id, "omitida", f"descartada: {source.status_reference}")
    if source.id == OSM_SOURCE:
        outcome = download_osm(source.id, paths, fetcher, today)
        return DownloadOutcome(outcome.source_id, outcome.status, outcome.message)
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
        was_local = destination.is_file()
        if was_local:
            manual += 1
        else:
            (fetcher or default_fetch)(file.url, destination)
            downloaded += 1
        digest, size = _digest_and_size(destination)
        old = previous_files.get(file.filename)
        _check_file(file, digest, size, old, destination)
        # Un archivo que ya estaba (modo manual) y cuyo SHA-256 y tamaño no
        # cambiaron no es una descarga nueva: conserva su fecha original
        # (hallazgo 21). Solo se estampa una fecha nueva si se descargó o si
        # el contenido cambió con respecto al manifiesto anterior.
        recorded_at = (
            old.downloaded_at
            if was_local and old is not None and old.sha256 == digest and old.size_bytes == size
            else download_date
        )
        entries.append(
            ManifestFile(
                url=file.url,
                filename=file.filename,
                downloaded_at=recorded_at,
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


def _sources_to_download(all_sources: Sequence[SourceSpec], source_id: str | None) -> tuple[list[SourceSpec], bool]:
    """Selecciona fuentes y avisa si OSM se omite por ser bajo demanda.

    ``select_sources`` valida que un ``source_id`` explícito exista. Sin filtro,
    OpenStreetMap se excluye para no consultar Nominatim en una máquina limpia.
    """
    selected = select_sources(all_sources, source_id)
    if source_id is None:
        return [source for source in selected if source.id != OSM_SOURCE], True
    return selected, False


def _outcomes_in_registry_order(
    all_sources: Sequence[SourceSpec],
    to_download: Sequence[SourceSpec],
    paths: ProjectPaths,
    fetcher: Fetcher | None,
    today: str | None,
    omit_osm: bool,
) -> list[DownloadOutcome]:
    """Produce los resultados en el orden del registro, con el aviso de OSM si aplica."""
    selected_ids = {source.id for source in to_download}
    outcomes: list[DownloadOutcome] = []
    for source in all_sources:
        if omit_osm and source.id == OSM_SOURCE:
            outcomes.append(DownloadOutcome(OSM_SOURCE, "omitido", "ejecute download --source openstreetmap"))
        elif source.id in selected_ids:
            outcomes.append(download_source(source, paths, fetcher, today))
    return outcomes


def download_all(
    paths: ProjectPaths,
    source_id: str | None = None,
    fetcher: Fetcher | None = None,
    today: str | None = None,
) -> list[DownloadOutcome]:
    """Descarga todas las fuentes registradas, o solo ``source_id``; OSM es bajo demanda."""
    all_sources = load_sources(paths.sources_toml)
    to_download, omit_osm = _sources_to_download(all_sources, source_id)
    return _outcomes_in_registry_order(all_sources, to_download, paths, fetcher, today, omit_osm)


def run_download(
    paths: ProjectPaths,
    source_id: str | None = None,
    fetcher: Fetcher | None = None,
) -> str:
    """Ejecuta la etapa completa: descarga filtrada y reporte de TODAS las fuentes.

    ``source_id`` limita únicamente qué fuentes se descargan. Sin filtro,
    OpenStreetMap se omite (fuente bajo demanda) para no consultar Nominatim en
    CI. El reporte de variables cubre siempre todas las fuentes de
    ``data/sources.toml``: las descendadas se verifican leyendo sus archivos en
    ``data/raw`` y las que no tienen archivo se marcan como sin verificar, pero
    ninguna se omite del reporte.
    """
    all_sources = load_sources(paths.sources_toml)
    to_download, omit_osm = _sources_to_download(all_sources, source_id)
    outcomes = _outcomes_in_registry_order(all_sources, to_download, paths, fetcher, None, omit_osm)
    report = verify_sources(all_sources, paths)
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
