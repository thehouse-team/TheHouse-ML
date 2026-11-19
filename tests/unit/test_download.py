"""ML-02 T02: descarga simulada, manifiesto, hash distinto y modo manual."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import fixture_fetcher, make_file, make_source
from thehouse_ml.download import DownloadError, default_fetch, download_source
from thehouse_ml.manifests import manifest_path
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.sources import SourceRegistryError, load_sources


def _paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths(root=tmp_path)


def test_download_writes_files_and_manifest(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    source = make_source(
        "envi",
        ("P5_1",),
        (
            make_file("envi_2020_base_de_datos_csv.zip", table="Bases de datos/TVIVIENDA.csv", encoding="latin-1"),
            make_file("envi_2020_fd.xlsx", kind="dictionary"),
        ),
        license_text="Licencia INEGI",
    )

    outcome = download_source(source, paths, fixture_fetcher(), today="2026-11-17")

    assert outcome.status == "descargado"
    assert (paths.data_raw / "envi" / "envi_2020_base_de_datos_csv.zip").is_file()
    manifest = json.loads(manifest_path(paths, "envi").read_text(encoding="utf-8"))
    assert manifest["source_id"] == "envi"
    assert len(manifest["files"]) == 2
    assert manifest["files"][0]["downloaded_at"] == "2026-11-17"
    assert manifest["files"][0]["license"] == "Licencia INEGI"
    assert manifest["files"][0]["sha256"] == source.download_url[0].sha256


def test_manual_mode_uses_local_file_without_fetching(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    source = make_source("manual", ("P5_1",), (make_file("envi_2020_base_de_datos_csv.zip"),))
    local = paths.data_raw / "manual" / "envi_2020_base_de_datos_csv.zip"
    local.parent.mkdir(parents=True)
    local.write_bytes((Path(__file__).resolve().parents[1] / "fixtures" / local.name).read_bytes())

    def forbidden_fetch(_url: str, _destination: Path) -> None:
        raise AssertionError("El modo manual no debe descargar")

    outcome = download_source(source, paths, forbidden_fetch)

    assert outcome.status == "manual"
    assert manifest_path(paths, "manual").is_file()


def test_wrong_sha256_in_registry_fails_clearly(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    source = make_source("envi", ("P5_1",), (make_file("envi_2020_base_de_datos_csv.zip", sha256="0" * 64),))

    with pytest.raises(DownloadError, match="SHA-256 no coincide"):
        download_source(source, paths, fixture_fetcher())

    assert not manifest_path(paths, "envi").exists()


def test_manifest_hash_mismatch_fails_clearly(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    source = make_source(
        "envi",
        ("P5_1",),
        (make_file("envi_2020_base_de_datos_csv.zip", with_checks=False),),
    )
    download_source(source, paths, fixture_fetcher())
    local = paths.data_raw / "envi" / "envi_2020_base_de_datos_csv.zip"
    local.write_bytes(b"contenido distinto")

    with pytest.raises(DownloadError, match="manifiesto existente registra SHA-256"):
        download_source(source, paths, fixture_fetcher())


def test_size_mismatch_fails(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    source = make_source("envi", ("P5_1",), (make_file("envi_2020_base_de_datos_csv.zip", size_bytes=1),))

    with pytest.raises(DownloadError, match="tamaño esperado 1 bytes"):
        download_source(source, paths, fixture_fetcher())


def test_source_with_empty_download_url_is_skipped(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    source = make_source("sin_url", ("X",), status="pending")

    outcome = download_source(source, paths, fixture_fetcher())

    assert outcome.status == "pendiente"
    assert outcome.message == "download_url pendiente"
    assert not manifest_path(paths, "sin_url").exists()


def test_discarded_source_is_skipped_without_fetching(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    source = make_source("inside_airbnb", ("room_type",), status="discarded", reference="M-ML-019")

    def forbidden_fetch(_url: str, _destination: Path) -> None:
        raise AssertionError("Una fuente descartada no debe descargarse")

    outcome = download_source(source, paths, forbidden_fetch)

    assert outcome.status == "omitida"
    assert outcome.message == "descartada: M-ML-019"
    assert outcome.render() == "inside_airbnb: omitida - descartada: M-ML-019"
    assert not manifest_path(paths, "inside_airbnb").exists()
    assert not (paths.data_raw / "inside_airbnb").exists()


def test_download_all_uses_the_registry_and_rejects_unknown_sources(tmp_path: Path) -> None:
    import shutil

    from thehouse_ml.download import download_all

    paths = _paths(tmp_path)
    registry = Path(__file__).resolve().parents[2] / "data" / "sources.toml"
    paths.sources_toml.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(registry, paths.sources_toml)
    assert load_sources(paths.sources_toml)

    with pytest.raises(SourceRegistryError, match="Fuente desconocida"):
        download_all(paths, source_id="no_existe", fetcher=fixture_fetcher())


def test_default_fetch_copies_a_local_file_uri_without_network(tmp_path: Path) -> None:
    from conftest import FIXTURES

    destination = tmp_path / "raw" / "copia.zip"
    default_fetch((FIXTURES / "envi_2020_base_de_datos_csv.zip").as_uri(), destination)

    assert destination.read_bytes() == (FIXTURES / "envi_2020_base_de_datos_csv.zip").read_bytes()


def test_default_fetch_reports_errors(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="No se pudo descargar"):
        default_fetch("file:///archivo/que/no/existe.zip", tmp_path / "copia.zip")
