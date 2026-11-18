"""ML-02 T02 (integración): etapa download completa con datos sintéticos y sin red."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import fixture_bytes, fixture_fetcher, fixture_sha256
from thehouse_ml.cli import main
from thehouse_ml.download import run_download
from thehouse_ml.paths import ProjectPaths

pytestmark = pytest.mark.integration

ENVI_ZIP = "envi_2020_base_de_datos_csv.zip"
ENVI_XLSX = "envi_2020_fd.xlsx"
ENIGH_ZIP = "enigh2022_ns_viviendas_csv.zip"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _file_entry(name: str, with_hashes: bool, *, kind: str, table: str, encoding: str) -> str:
    fields = [f'url = "https://example.test/{name}"', f'filename = "{name}"']
    if with_hashes:
        digest = fixture_sha256(name)
        size = (FIXTURES / name).stat().st_size
        fields.append(f'sha256 = "{digest}", size_bytes = {size}')
    if kind != "data":
        fields.append(f'kind = "{kind}"')
    if table:
        fields.append(f'table = "{table}"')
    if encoding:
        fields.append(f'encoding = "{encoding}"')
    return "  { " + ", ".join(fields) + " },"


def _variables(names: tuple[str, ...]) -> str:
    return ", ".join(f'"{name}"' for name in names)


def _write_registry(tmp_path: Path, *, with_hashes: bool = True) -> None:
    envi_files = "\n".join(
        [
            _file_entry(
                ENVI_ZIP,
                with_hashes,
                kind="data",
                table="Bases de datos/TVIVIENDA.csv",
                encoding="latin-1",
            ),
            _file_entry(ENVI_XLSX, with_hashes, kind="dictionary", table="", encoding=""),
        ]
    )
    enigh_file = _file_entry(ENIGH_ZIP, with_hashes, kind="data", table="viviendas.csv", encoding="utf-8-sig")
    envi_variables = _variables(("P5_1", "P5_2", "P5_5", "ENT", "TLOC", "FACTOR", "EST_DIS", "UPM_DIS"))
    enigh_variables = _variables(
        (
            "tenencia",
            "renta",
            "factor",
            "ubica_geo",
            "tam_loc",
            "est_dis",
            "upm",
            "disp_agua",
            "drenaje",
            "disp_elect",
            "combustible",
        )
    )
    document = f"""[[sources]]
id = "inegi_envi_2020"
name = "ENVI de prueba"
license = "Términos de libre uso de INEGI"
attribution = "INEGI"
variables_status = "confirmed"
variables_used = [{envi_variables}]
download_url = [
{envi_files}
]

[[sources]]
id = "inegi_enigh_2022"
name = "ENIGH de prueba"
license = "Términos de libre uso de INEGI"
attribution = "INEGI"
variables_status = "confirmed"
variables_used = [{enigh_variables}]
download_url = [
{enigh_file}
]

[[sources]]
id = "inegi_censo_2020_viviendas_ca"
name = "Censo de prueba"
license = "Términos de libre uso de INEGI"
attribution = "INEGI"
variables_status = "pending"
variables_used = ["TENENCIA", "FACTOR"]
download_url = []
"""
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "sources.toml").write_text(document, encoding="utf-8")


def _manifest_for(tmp_path: Path, source_id: str) -> dict[str, object]:
    return json.loads((tmp_path / "data" / "manifests" / f"{source_id}.json").read_text(encoding="utf-8"))


def test_download_stage_creates_manifests_and_variable_report(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    paths = ProjectPaths(root=tmp_path)

    output = run_download(paths, fetcher=fixture_fetcher())

    assert "inegi_envi_2020: descargado" in output
    assert "inegi_enigh_2022: descargado" in output
    assert "download_url pendiente" in output
    assert len(_manifest_for(tmp_path, "inegi_envi_2020")["files"]) == 2  # type: ignore[arg-type]
    report = (tmp_path / "reports" / "variables_check.md").read_text(encoding="utf-8")
    assert "`P5_5`" in report
    assert "P5_8" not in report
    assert "variables faltantes: (ninguna)" in report


def test_second_run_uses_manual_mode_without_network(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    paths = ProjectPaths(root=tmp_path)
    run_download(paths, fetcher=fixture_fetcher())

    def forbidden_fetch(_url: str, _destination: Path) -> None:
        raise AssertionError("Los archivos ya existen: no se debe usar la red")

    output = run_download(paths, fetcher=forbidden_fetch)

    assert "inegi_envi_2020: manual" in output
    assert "inegi_enigh_2022: manual" in output


def test_source_filter_limits_the_download_but_not_the_report(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    paths = ProjectPaths(root=tmp_path)
    run_download(paths, fetcher=fixture_fetcher())

    output = run_download(paths, source_id="inegi_enigh_2022", fetcher=fixture_fetcher())
    report = (tmp_path / "reports" / "variables_check.md").read_text(encoding="utf-8")

    # El filtro solo limita las líneas de descarga de la consola.
    assert "inegi_enigh_2022:" in output
    assert "inegi_envi_2020:" not in output
    # El reporte cubre siempre todas las fuentes del registro.
    assert "## inegi_envi_2020" in report
    assert "## inegi_enigh_2022" in report
    assert "## inegi_censo_2020_viviendas_ca" in report


def test_download_without_source_omits_osm_and_never_calls_the_fetcher(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "sources.toml").write_text(
        """[[sources]]
id = "inegi_envi_2020"
name = "ENVI de prueba"
license = "Libre uso"
attribution = "INEGI"
variables_status = "pending"
variables_used = ["P5_1"]
download_url = []

[[sources]]
id = "openstreetmap"
name = "OpenStreetMap"
license = "ODbL"
attribution = "OSM"
variables_status = "confirmed"
variables_used = ["osm_id"]
download_url = []
""",
        encoding="utf-8",
    )
    paths = ProjectPaths(root=tmp_path)
    calls: list[str] = []

    def fetch(url: str, _destination: Path) -> None:
        calls.append(url)

    output = run_download(paths, fetcher=fetch)

    assert "openstreetmap: omitido - ejecute download --source openstreetmap" in output
    assert calls == []
    assert not (paths.data_raw / "openstreetmap").exists()


def test_cli_download_returns_zero_with_preplaced_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_registry(tmp_path)
    paths = ProjectPaths(root=tmp_path)
    run_download(paths, fetcher=fixture_fetcher())

    code = main(["--root", str(tmp_path), "download", "--source", "inegi_envi_2020"])

    assert code == 0
    assert "inegi_envi_2020: manual" in capsys.readouterr().out


def test_cli_fails_when_a_confirmed_variable_is_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_registry(tmp_path, with_hashes=False)
    paths = ProjectPaths(root=tmp_path)
    local = paths.data_raw / "inegi_envi_2020" / ENVI_ZIP
    local.parent.mkdir(parents=True)
    local.write_bytes(fixture_bytes("envi_sin_p5_5.zip"))
    local = paths.data_raw / "inegi_envi_2020" / ENVI_XLSX
    local.write_bytes(fixture_bytes(ENVI_XLSX))
    local = paths.data_raw / "inegi_enigh_2022" / ENIGH_ZIP
    local.parent.mkdir(parents=True)
    local.write_bytes(fixture_bytes(ENIGH_ZIP))

    code = main(["--root", str(tmp_path), "download", "--source", "inegi_envi_2020"])

    assert code == 1
    assert "Variables confirmadas ausentes" in capsys.readouterr().err
    assert "`P5_5`" in (tmp_path / "reports" / "variables_check.md").read_text(encoding="utf-8")


def test_cli_fails_when_the_hash_does_not_match(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_registry(tmp_path)
    paths = ProjectPaths(root=tmp_path)
    altered = bytearray(fixture_bytes(ENVI_ZIP))
    altered[0] ^= 0xFF
    local = paths.data_raw / "inegi_envi_2020" / ENVI_ZIP
    local.parent.mkdir(parents=True)
    local.write_bytes(altered)

    code = main(["--root", str(tmp_path), "download", "--source", "inegi_envi_2020"])

    assert code == 1
    assert "SHA-256 no coincide" in capsys.readouterr().err
