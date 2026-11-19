"""ML-02 T02: verificación de variables leyendo solo encabezados de CSV en ZIP."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from conftest import fixture_bytes, make_file, make_source, write_osm_project
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.universities import OSM_SOURCE
from thehouse_ml.variables import (
    VariableCheckError,
    read_csv_header,
    read_zip_header,
    verify_sources,
    write_variables_report,
)

ENVI_FILE = "envi_2020_base_de_datos_csv.zip"
ENIGH_FILE = "enigh2022_ns_viviendas_csv.zip"


def _place(paths: ProjectPaths, source_id: str, filename: str, content: bytes) -> Path:
    destination = paths.data_raw / source_id / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def test_read_zip_header_reads_nested_latin1_member(tmp_path: Path) -> None:
    archive = Path(__file__).resolve().parents[1] / "fixtures" / ENVI_FILE

    header = read_zip_header(archive, "Bases de datos/TVIVIENDA.csv", "latin-1")

    assert "P5_1" in header
    assert "P5_2" in header
    assert "P5_5" in header
    assert "P5_8" in header


def test_read_zip_header_strips_utf8_bom(tmp_path: Path) -> None:
    archive = Path(__file__).resolve().parents[1] / "fixtures" / ENIGH_FILE

    header = read_zip_header(archive, "viviendas.csv", "utf-8-sig")

    assert header[0] == "tenencia"


def test_read_zip_header_joins_every_csv_matching_a_glob(tmp_path: Path) -> None:
    archive = tmp_path / "denue_15.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "conjunto_de_datos/denue_inegi_15_.csv", "id,codigo_act,latitud\n1,722111,20.0\n".encode("latin-1")
        )
        bundle.writestr("conjunto_de_datos/otro.csv", "id,codigo_act,longitud\n2,522110,-99.6\n".encode("latin-1"))
        bundle.writestr("catalogos/no.csv", "x,y\n".encode("latin-1"))

    header = read_zip_header(archive, "conjunto_de_datos/*.csv", "latin-1")

    assert set(header) == {"id", "codigo_act", "latitud", "longitud"}


def test_read_zip_header_rejects_unknown_member(tmp_path: Path) -> None:
    archive = Path(__file__).resolve().parents[1] / "fixtures" / ENVI_FILE

    with pytest.raises(VariableCheckError, match="no contiene el miembro"):
        read_zip_header(archive, "no_existe.csv", "latin-1")


def test_read_zip_header_rejects_a_file_that_is_not_a_zip(tmp_path: Path) -> None:
    archive = tmp_path / "malo.zip"
    archive.write_bytes(b"no soy un zip")

    with pytest.raises(VariableCheckError, match="no es un ZIP válido"):
        read_zip_header(archive, "x.csv")


def test_read_csv_header_reads_only_the_first_line(tmp_path: Path) -> None:
    path = tmp_path / "solo_encabezado.csv"
    path.write_text("A,B\ndato1,dato2\n", encoding="utf-8")

    assert read_csv_header(path) == ["A", "B"]


def test_verify_sources_finds_all_confirmed_variables(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    envi = make_source(
        "envi",
        ("P5_1", "P5_2", "P5_5"),
        (make_file(ENVI_FILE, table="Bases de datos/TVIVIENDA.csv", encoding="latin-1"),),
    )
    enigh = make_source(
        "enigh",
        ("tenencia", "renta", "factor"),
        (make_file(ENIGH_FILE, table="viviendas.csv", encoding="utf-8-sig"),),
    )
    _place(paths, "envi", ENVI_FILE, fixture_bytes(ENVI_FILE))
    _place(paths, "enigh", ENIGH_FILE, fixture_bytes(ENIGH_FILE))

    report = verify_sources([envi, enigh], paths)

    assert report.failures == []
    assert all(check.found for check in report.checks)
    assert all(check.missing == () for check in report.checks)


def test_missing_confirmed_variable_is_a_failure(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    source = make_source(
        "envi",
        ("P5_1", "P5_2", "P5_5"),
        (make_file(ENVI_FILE, table="Bases de datos/TVIVIENDA.csv", encoding="latin-1"),),
    )
    _place(paths, "envi", ENVI_FILE, fixture_bytes("envi_sin_p5_5.zip"))

    report = verify_sources([source], paths)

    assert report.checks[0].missing == ("P5_5",)
    assert any("P5_5" in failure for failure in report.failures)
    assert "`P5_5`" in report.render()


def test_source_without_url_is_skipped_not_failed(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    source = make_source("sin_url", ("X",), status="pending")

    report = verify_sources([source], paths)

    assert "download_url pendiente" in report.checks[0].note
    assert report.failures == []


def test_confirmed_source_without_url_is_skipped_not_failed(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    source = make_source("sin_url_confirmado", ("X",))

    report = verify_sources([source], paths)

    assert report.checks[0].verified is False
    assert report.failures == []
    assert "(sin verificar)" in report.render()


def test_downloaded_but_missing_file_is_not_failed(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    source = make_source("sin_archivo", ("X",), (make_file(ENVI_FILE),), status="pending")

    report = verify_sources([source], paths)

    assert "sin verificar" in report.checks[0].note
    assert report.failures == []


def test_osm_variables_are_verified_from_the_jsonv2_responses(tmp_path: Path) -> None:
    write_osm_project(tmp_path)
    paths = ProjectPaths(root=tmp_path)
    osm = make_source(OSM_SOURCE, ("osm_id", "osm_type", "lat", "lon", "name"))

    report = verify_sources([osm], paths)

    assert report.checks[0].verified is True
    assert report.checks[0].missing == ()
    assert "respuestas jsonv2 verificadas" in report.checks[0].note
    assert report.failures == []


def test_osm_without_responses_is_not_failed(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    osm = make_source(OSM_SOURCE, ("osm_id", "osm_type", "lat", "lon", "name"))

    report = verify_sources([osm], paths)

    assert report.checks[0].verified is False
    assert "sin respuestas descargadas" in report.checks[0].note
    assert report.failures == []


def test_discarded_source_is_shown_as_discarded_and_other_sections_are_kept(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    envi = make_source(
        "envi",
        ("P5_1", "P5_2", "P5_5"),
        (make_file(ENVI_FILE, table="Bases de datos/TVIVIENDA.csv", encoding="latin-1"),),
    )
    discarded = make_source("inside_airbnb", ("room_type",), status="discarded", reference="M-ML-019")
    _place(paths, "envi", ENVI_FILE, fixture_bytes(ENVI_FILE))

    report = verify_sources([envi, discarded], paths)
    rendered = report.render()

    assert "## envi" in rendered
    assert "## inside_airbnb" in rendered
    assert "- estado: `discarded`" in rendered
    assert "descartada: M-ML-019" in rendered
    discarded_check = next(check for check in report.checks if check.source_id == "inside_airbnb")
    assert discarded_check.verified is False
    assert report.failures == []


def test_write_variables_report_creates_the_reports_folder(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    report = verify_sources([make_source("sin_url", ("X",), status="pending")], paths)

    write_variables_report(paths.reports / "variables_check.md", report)

    text = (paths.reports / "variables_check.md").read_text(encoding="utf-8")
    assert "# Verificacion de variables" in text
    assert "## sin_url" in text
