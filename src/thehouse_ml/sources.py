"""Registro de fuentes: lectura y validación de ``data/sources.toml``.

El archivo es la contract de ML-02 (R01/R02). Una entrada de ``download_url``
puede ser una URL simple o una tabla con url, nombre, tamaño, SHA-256, tipo,
tabla CSV dentro del ZIP y codificación.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_STATUS = frozenset({"pending", "confirmed"})
ALLOWED_KINDS = frozenset({"data", "dictionary"})


class SourceRegistryError(ValueError):
    """El registro no existe, no es válido o no contiene el id solicitado."""


@dataclass(frozen=True)
class FileSpec:
    """Un archivo declarado en ``download_url``."""

    url: str
    filename: str
    size_bytes: int | None
    sha256: str | None
    kind: str
    table: str | None
    encoding: str

    @property
    def is_zip(self) -> bool:
        """Indica si el archivo es un ZIP del que se lee un miembro CSV."""
        return self.filename.lower().endswith(".zip")


@dataclass(frozen=True)
class SourceSpec:
    """Una fuente del registro con sus archivos y variables."""

    id: str
    name: str
    license: str
    attribution: str
    variables_status: str
    variables_used: tuple[str, ...]
    download_url: tuple[FileSpec, ...]


def load_sources(path: Path) -> list[SourceSpec]:
    """Lee y valida el registro. Lanza ``SourceRegistryError`` si algo falta."""
    try:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
    except OSError as exc:
        raise SourceRegistryError(f"No se puede leer {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise SourceRegistryError(f"{path} no es TOML válido: {exc}") from exc

    entries = document.get("sources")
    if not isinstance(entries, list) or not entries:
        raise SourceRegistryError("sources.toml debe definir al menos una entrada [[sources]]")
    return [_parse_source(entry, index) for index, entry in enumerate(entries)]


def select_sources(sources: Sequence[SourceSpec], source_id: str | None) -> list[SourceSpec]:
    """Filtra por id; ``None`` devuelve todas. Un id desconocido es un error."""
    if source_id is None:
        return list(sources)
    selected = [source for source in sources if source.id == source_id]
    if not selected:
        raise SourceRegistryError(f"Fuente desconocida: {source_id!r}")
    return selected


def _parse_source(entry: object, index: int) -> SourceSpec:
    where = f"sources[{index}]"
    if not isinstance(entry, dict):
        raise SourceRegistryError(f"{where}: cada fuente debe ser una tabla")
    source_id = _required_str(entry, "id", where)
    if not SOURCE_ID_PATTERN.fullmatch(source_id):
        raise SourceRegistryError(f"{where}: id inválido {source_id!r} (solo letras, dígitos, - y _)")
    where = f"fuente {source_id!r}"

    status = _required_str(entry, "variables_status", where)
    if status not in ALLOWED_STATUS:
        raise SourceRegistryError(f"{where}: variables_status inválido {status!r}")

    variables = _required_str_list(entry, "variables_used", where)
    files = _parse_download_url(entry.get("download_url"), where)
    return SourceSpec(
        id=source_id,
        name=_required_str(entry, "name", where),
        license=_required_str(entry, "license", where),
        attribution=_required_str(entry, "attribution", where),
        variables_status=status,
        variables_used=variables,
        download_url=files,
    )


def _parse_download_url(value: object, where: str) -> tuple[FileSpec, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SourceRegistryError(f"{where}: download_url debe ser una lista de archivos")
    files: list[FileSpec] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            files.append(_file_from_url(item, where, index))
        elif isinstance(item, dict):
            files.append(_parse_file(item, where, index))
        else:
            raise SourceRegistryError(f"{where}: download_url[{index}] debe ser texto o tabla")
    return tuple(files)


def _file_from_url(url: str, where: str, index: int) -> FileSpec:
    return _parse_file({"url": url}, where, index)


def _parse_file(entry: dict[str, object], where: str, index: int) -> FileSpec:
    label = f"{where}: download_url[{index}]"
    url = _required_str(entry, "url", label)
    filename = _optional_str(entry, "filename", label) or _filename_from_url(url)
    size_bytes = _optional_int(entry, "size_bytes", label)
    sha256 = _optional_str(entry, "sha256", label)
    if sha256 is not None:
        sha256 = sha256.lower()
        if not SHA256_PATTERN.fullmatch(sha256):
            raise SourceRegistryError(f"{label}: sha256 debe tener 64 caracteres hexadecimales")
    kind = _optional_str(entry, "kind", label) or "data"
    if kind not in ALLOWED_KINDS:
        raise SourceRegistryError(f"{label}: kind inválido {kind!r}")
    return FileSpec(
        url=url,
        filename=filename,
        size_bytes=size_bytes,
        sha256=sha256,
        kind=kind,
        table=_optional_str(entry, "table", label),
        encoding=_optional_str(entry, "encoding", label) or "utf-8",
    )


def _filename_from_url(url: str) -> str:
    filename = Path(unquote(urlsplit(url).path)).name
    if not filename or filename in {".", ".."}:
        raise SourceRegistryError(f"No se puede derivar el nombre del archivo de {url!r}")
    return filename


def _required_str(entry: dict[str, object], field: str, where: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SourceRegistryError(f"{where}: {field!r} debe ser texto no vacío")
    return value.strip()


def _required_str_list(entry: dict[str, object], field: str, where: str) -> tuple[str, ...]:
    value = entry.get(field)
    if not isinstance(value, list) or not value:
        raise SourceRegistryError(f"{where}: {field!r} debe ser una lista no vacía")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise SourceRegistryError(f"{where}: {field!r} solo admite textos no vacíos")
    return tuple(str(item).strip() for item in value)


def _optional_str(entry: dict[str, object], field: str, where: str) -> str | None:
    value = entry.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SourceRegistryError(f"{where}: {field!r} debe ser texto no vacío")
    return value.strip()


def _optional_int(entry: dict[str, object], field: str, where: str) -> int | None:
    value = entry.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceRegistryError(f"{where}: {field!r} debe ser un entero no negativo")
    return value
