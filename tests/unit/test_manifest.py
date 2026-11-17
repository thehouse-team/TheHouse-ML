"""ML-02 T02: manifiestos con url, fecha, tamaño, SHA-256 y licencia."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thehouse_ml.manifests import Manifest, ManifestError, ManifestFile, load_manifest, write_manifest
from thehouse_ml.paths import ProjectPaths


def _file() -> ManifestFile:
    return ManifestFile(
        url="https://example.test/a.zip",
        filename="a.zip",
        downloaded_at="2026-11-17",
        size_bytes=123,
        sha256="a" * 64,
        license="Licencia de prueba",
    )


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifests" / "prueba.json"
    write_manifest(path, Manifest("prueba", (_file(),)))

    loaded = load_manifest(path)

    assert loaded is not None
    assert loaded.source_id == "prueba"
    assert loaded.files == (_file(),)


def test_manifest_json_has_required_keys(tmp_path: Path) -> None:
    path = tmp_path / "manifests" / "prueba.json"
    write_manifest(path, Manifest("prueba", (_file(),)))

    document = json.loads(path.read_text(encoding="utf-8"))

    assert set(document["files"][0]) == {"url", "filename", "downloaded_at", "size_bytes", "sha256", "license"}


def test_manifest_path_uses_source_id(tmp_path: Path) -> None:
    from thehouse_ml.manifests import manifest_path

    paths = ProjectPaths(root=tmp_path)
    assert manifest_path(paths, "inegi_envi_2020") == tmp_path / "data" / "manifests" / "inegi_envi_2020.json"


def test_missing_manifest_returns_none(tmp_path: Path) -> None:
    assert load_manifest(tmp_path / "no_existe.json") is None


@pytest.mark.parametrize("content", ["{", '{"source_id": "x", "files": {}}', '{"source_id": 3, "files": []}'])
def test_malformed_manifest_is_rejected(content: str, tmp_path: Path) -> None:
    path = tmp_path / "mal.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ManifestError):
        load_manifest(path)
