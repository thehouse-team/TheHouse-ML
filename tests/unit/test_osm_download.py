"""ML-02 T07: descarga simulada, manifiesto SHA-256, cacheo y política de Nominatim."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import write_osm_project
from thehouse_ml.manifests import manifest_path
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.universities import (
    OSM_SOURCE,
    RATE_LIMIT_SECONDS,
    USER_AGENT,
    OsmError,
    download_osm,
    fetch_osm,
)

PAYLOAD = b"[]"


def _recording_fetcher(calls: list[str], payload: bytes = PAYLOAD):
    def fetch(url: str, destination: Path) -> None:
        calls.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    return fetch


def test_download_writes_ten_responses_and_their_manifest(tmp_path: Path) -> None:
    write_osm_project(tmp_path, with_responses=False)
    paths = ProjectPaths(root=tmp_path)
    calls: list[str] = []

    outcome = download_osm(OSM_SOURCE, paths, _recording_fetcher(calls), today="2026-11-17", sleeper=lambda _: None)

    assert outcome.status == "descargado"
    assert len(calls) == 10
    assert "format=jsonv2" in calls[0]
    assert len(list((paths.data_raw / OSM_SOURCE).glob("*.json"))) == 10
    manifest = json.loads(manifest_path(paths, OSM_SOURCE).read_text(encoding="utf-8"))
    assert manifest["source_id"] == OSM_SOURCE
    assert len(manifest["files"]) == 10
    assert manifest["files"][0]["filename"] == "UTTT.json"
    assert manifest["files"][0]["downloaded_at"] == "2026-11-17"
    assert len(manifest["files"][0]["sha256"]) == 64
    assert "ODbL" in manifest["files"][0]["license"]


def test_cached_responses_keep_their_original_downloaded_at(tmp_path: Path) -> None:
    """Hallazgo 21 también aplica al manifiesto OSM (fuente bajo demanda)."""
    write_osm_project(tmp_path)
    paths = ProjectPaths(root=tmp_path)
    manifest = manifest_path(paths, OSM_SOURCE)
    first = manifest.read_bytes()
    assert json.loads(first)["files"][0]["downloaded_at"] == "2026-11-17"

    def forbidden(_url: str, _destination: Path) -> None:
        raise AssertionError("Las respuestas ya están cacheadas: no se debe usar la red")

    outcome = download_osm(OSM_SOURCE, paths, forbidden, today="2026-11-20", sleeper=lambda _: None)

    assert outcome.status == "manual"
    assert manifest.read_bytes() == first


def test_existing_json_is_not_requested_again(tmp_path: Path) -> None:
    write_osm_project(tmp_path, with_responses=False)
    paths = ProjectPaths(root=tmp_path)
    cached = paths.data_raw / OSM_SOURCE / "UTTT.json"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(PAYLOAD)
    fetched: list[str] = []

    def fetch(_url: str, destination: Path) -> None:
        fetched.append(destination.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(PAYLOAD)

    outcome = download_osm(OSM_SOURCE, paths, fetch, sleeper=lambda _: None)

    assert outcome.status == "mixto"
    assert len(fetched) == 9
    assert "UTTT.json" not in fetched

    fetched.clear()
    outcome = download_osm(OSM_SOURCE, paths, fetch, sleeper=lambda _: None)

    assert outcome.status == "manual"
    assert fetched == []


def test_rate_limit_sleeps_between_consecutive_requests(tmp_path: Path) -> None:
    write_osm_project(tmp_path, with_responses=False)
    paths = ProjectPaths(root=tmp_path)
    sleeps: list[float] = []

    download_osm(OSM_SOURCE, paths, _recording_fetcher([]), sleeper=sleeps.append)

    assert sleeps == [RATE_LIMIT_SECONDS] * 9


def test_manifest_mismatch_is_reported_without_overwriting(tmp_path: Path) -> None:
    write_osm_project(tmp_path, with_responses=False)
    paths = ProjectPaths(root=tmp_path)
    download_osm(OSM_SOURCE, paths, _recording_fetcher([]), sleeper=lambda _: None)
    local = paths.data_raw / OSM_SOURCE / "UTTT.json"
    local.write_bytes(b"[]\nmodificado")

    with pytest.raises(OsmError, match="manifiesto existente registra SHA-256"):
        download_osm(OSM_SOURCE, paths, _recording_fetcher([]), sleeper=lambda _: None)


def test_default_fetch_sends_the_project_user_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def read(self) -> bytes:
            return PAYLOAD

    def fake_urlopen(request: object, timeout: int | None = None) -> _Response:
        captured["user_agent"] = str(request.get_header("User-agent"))  # type: ignore[attr-defined]
        captured["timeout"] = str(timeout)
        return _Response()

    monkeypatch.setattr("thehouse_ml.universities.urllib.request.urlopen", fake_urlopen)
    destination = tmp_path / "raw" / "openstreetmap" / "UTTT.json"

    fetch_osm("https://example.test/search?q=prueba", destination)

    assert captured["user_agent"] == USER_AGENT
    assert captured["timeout"] == "60"
    assert destination.read_bytes() == PAYLOAD
