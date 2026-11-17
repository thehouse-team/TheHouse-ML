"""Manifiestos de datos: un JSON por fuente con URL, fecha, tamaño, SHA-256 y licencia."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from thehouse_ml.paths import ProjectPaths


class ManifestError(ValueError):
    """El manifiesto existente no tiene el formato esperado."""


@dataclass(frozen=True)
class ManifestFile:
    """Un archivo descargado y verificado."""

    url: str
    filename: str
    downloaded_at: str
    size_bytes: int
    sha256: str
    license: str

    def to_dict(self) -> dict[str, object]:
        """Representación JSON del archivo."""
        return {
            "url": self.url,
            "filename": self.filename,
            "downloaded_at": self.downloaded_at,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "license": self.license,
        }

    @classmethod
    def from_dict(cls, value: object, path: Path) -> ManifestFile:
        """Reconstruye un archivo y valida sus campos."""
        record = _as_record(value, path)
        return cls(
            url=_field_str(record, "url", path),
            filename=_field_str(record, "filename", path),
            downloaded_at=_field_str(record, "downloaded_at", path),
            size_bytes=_field_int(record, "size_bytes", path),
            sha256=_field_str(record, "sha256", path),
            license=_field_str(record, "license", path),
        )


@dataclass(frozen=True)
class Manifest:
    """Manifiesto completo de una fuente."""

    source_id: str
    files: tuple[ManifestFile, ...]

    def to_dict(self) -> dict[str, object]:
        """Representación JSON del manifiesto."""
        return {"source_id": self.source_id, "files": [file.to_dict() for file in self.files]}

    @classmethod
    def from_dict(cls, value: object, path: Path) -> Manifest:
        """Reconstruye un manifiesto y valida sus campos."""
        record = _as_record(value, path)
        files = record.get("files")
        if not isinstance(files, list):
            raise ManifestError(f"{path}: 'files' debe ser una lista")
        return cls(
            source_id=_field_str(record, "source_id", path),
            files=tuple(ManifestFile.from_dict(file, path) for file in files),
        )


def manifest_path(paths: ProjectPaths, source_id: str) -> Path:
    """Ruta del manifiesto de una fuente."""
    return paths.data_manifests / f"{source_id}.json"


def load_manifest(path: Path) -> Manifest | None:
    """Devuelve el manifiesto existente o ``None`` si aún no hay uno."""
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"No se puede leer el manifiesto {path}: {exc}") from exc
    return Manifest.from_dict(document, path)


def write_manifest(path: Path, manifest: Manifest) -> None:
    """Escribe el manifiesto, creando la carpeta si hace falta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _as_record(value: object, path: Path) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ManifestError(f"{path}: se esperaba un objeto JSON")
    return value


def _field_str(record: dict[str, object], field: str, path: Path) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{path}: {field!r} debe ser texto no vacío")
    return value


def _field_int(record: dict[str, object], field: str, path: Path) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(f"{path}: {field!r} debe ser un entero no negativo")
    return value
