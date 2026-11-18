"""ML-02 T03 (integración): etapa prepare completa con datos inventados y sin red."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from conftest import write_osm_project
from thehouse_ml.cli import main
from thehouse_ml.national import BASELINE_COLUMNS, ENIGH_SOURCE, ENVI_SOURCE, run_prepare
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.stages import run_stage

pytestmark = pytest.mark.integration

ENVI_ZIP = "envi_de_prueba.zip"
ENIGH_ZIP = "enigh_de_prueba.zip"
INPC_ZIP = "inpc_de_prueba.zip"

CENSO_HGO_ZIP = "censo_hgo_de_prueba.zip"
CENSO_MEX_ZIP = "censo_mex_de_prueba.zip"
CENSO_COLUMNS = [
    "ENT",
    "MUN",
    "FACTOR",
    "ESTRATO",
    "UPM",
    "TENENCIA",
    "AGUA_ENTUBADA",
    "DRENAJE",
    "ELECTRICIDAD",
    "INTERNET",
    "COMBUSTIBLE",
]

GENERAL = "Índice nacional de precios al consumidor, Resumen, Precios al Consumidor (INPC)"
SUBYACENTE = "Índice nacional de precios al consumidor, Resumen, Subyacente"


def _write_zip(path: Path, member: str, encoding: str, rows: list[list[str]]) -> None:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(member, buffer.getvalue().encode(encoding))


def _write_project(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    registry = f"""[[sources]]
id = "{ENVI_SOURCE}"
name = "ENVI de prueba"
license = "Términos de libre uso de INEGI"
attribution = "INEGI"
variables_status = "confirmed"
variables_used = ["P5_1", "P5_2", "P4_3", "ENT", "TLOC", "FACTOR"]

[[sources.download_url]]
url = "https://example.test/{ENVI_ZIP}"
filename = "{ENVI_ZIP}"
kind = "data"
table = "Bases de datos/TVIVIENDA.csv"
encoding = "latin-1"

[[sources]]
id = "{ENIGH_SOURCE}"
name = "ENIGH de prueba"
license = "Términos de libre uso de INEGI"
attribution = "INEGI"
variables_status = "confirmed"
variables_used = ["tenencia", "renta", "factor", "ubica_geo", "tam_loc", "tipo_viv"]

[[sources.download_url]]
url = "https://example.test/{ENIGH_ZIP}"
filename = "{ENIGH_ZIP}"
kind = "data"
table = "viviendas.csv"
encoding = "utf-8-sig"

[[sources]]
id = "inegi_censo_2020_viviendas_ca"
name = "Censo de prueba"
license = "Términos de libre uso de INEGI"
attribution = "INEGI"
variables_status = "confirmed"
variables_used = {CENSO_COLUMNS!r}

[[sources.download_url]]
url = "https://example.test/{CENSO_HGO_ZIP}"
filename = "{CENSO_HGO_ZIP}"
kind = "data"
table = "Viviendas13.CSV"
encoding = "latin-1"

[[sources.download_url]]
url = "https://example.test/{CENSO_MEX_ZIP}"
filename = "{CENSO_MEX_ZIP}"
kind = "data"
table = "Viviendas15.CSV"
encoding = "latin-1"

[[sources]]
id = "inegi_inpc"
name = "INPC de prueba"
license = "Términos de libre uso de INEGI"
attribution = "INEGI"
variables_status = "confirmed"
variables_used = ["FECHA", "CONCEPTO", "VALOR"]

[[sources.download_url]]
url = "https://example.test/{INPC_ZIP}"
filename = "{INPC_ZIP}"
kind = "data"
table = "conjunto_de_datos/inpc.csv"
encoding = "utf-8"

[[sources]]
id = "inegi_denue_vigente"
name = "DENUE de prueba"
license = "Términos de Libre Uso de la Información del INEGI"
attribution = "Fuente: INEGI, DENUE 05/2026"
dictionary_url = "https://example.test"
variables_status = "confirmed"
variables_used = ["id", "codigo_act", "latitud", "longitud"]
"""
    (tmp_path / "data" / "sources.toml").write_text(registry, encoding="utf-8")

    envi_rows = [
        ["P5_1", "P5_2", "P4_3", "ENT", "TLOC", "FACTOR"],
        ["1", "000001000", "1", "13", "1", "10"],
        ["1", "000003000", "2", "13", "1", "30"],
        ["1", "999999999", "1", "13", "1", "10"],
        ["1", "000000000", "1", "13", "1", "10"],
        ["1", "", "1", "13", "1", "10"],
        ["2", "", "1", "13", "1", "40"],
        ["1", "000002000", "1", "15", "2", "20"],
        ["2", "", "1", "15", "2", "20"],
    ]
    enigh_rows = [
        ["tenencia", "renta", "factor", "ubica_geo", "tam_loc", "tipo_viv"],
        ["1", "1000", "10", "13001", "1", "1"],
        ["1", "3000", "30", "13002", "1", "2"],
        ["2", "", "20", "13003", "1", "1"],
        ["1", "2000", "20", "15001", "2", "1"],
        ["2", "", "20", "15002", "2", "1"],
    ]
    inpc_rows = [["FECHA", "CONCEPTO", "VALOR"]]
    for month, value in (
        ("2020-10", "100"),
        ("2020-11", "102"),
        ("2020-12", "104"),
        ("2022-08", "110"),
        ("2022-09", "112"),
        ("2022-10", "114"),
        ("2022-11", "116"),
        ("2024-07", "136"),
    ):
        inpc_rows.append([month + "-01", GENERAL, value])
        inpc_rows.append([month + "-01", SUBYACENTE, "999"])

    raw = tmp_path / "data" / "raw"
    _write_zip(raw / ENVI_SOURCE / ENVI_ZIP, "Bases de datos/TVIVIENDA.csv", "latin-1", envi_rows)
    _write_zip(raw / ENIGH_SOURCE / ENIGH_ZIP, "viviendas.csv", "utf-8-sig", enigh_rows)
    _write_zip(raw / "inegi_inpc" / INPC_ZIP, "conjunto_de_datos/inpc.csv", "utf-8", inpc_rows)

    censo_raw = raw / "inegi_censo_2020_viviendas_ca"
    hidalgo = [CENSO_COLUMNS]
    for index in range(40):
        hidalgo.append(["13", "070", "10", "1", "A" if index % 2 == 0 else "B", "2", "1", "1", "1", "7", "2"])
    for mun in ("013", "041", "063", "076"):
        hidalgo.append(["13", mun, "10", "1", "A", "1", "3", "5", "3", "8", "5"])
    mexico = [CENSO_COLUMNS]
    for index in range(40):
        mexico.append(["15", "045", "10", "1", "A" if index % 2 == 0 else "B", "2", "1", "1", "1", "7", "2"])
    mexico.append(["15", "112", "10", "1", "A", "1", "3", "5", "3", "8", "5"])
    _write_zip(censo_raw / CENSO_HGO_ZIP, "Viviendas13.CSV", "latin-1", hidalgo)
    _write_zip(censo_raw / CENSO_MEX_ZIP, "Viviendas15.CSV", "latin-1", mexico)

    write_osm_project(tmp_path)  # ML-02 T07: seed y respuestas OSM para la etapa completa.


def _frame(tmp_path: Path) -> pd.DataFrame:
    return pd.read_parquet(tmp_path / "data" / "processed" / "national_baselines.parquet")


def _row(frame: pd.DataFrame, source: str, level_value: str) -> pd.Series:
    selected = frame.loc[
        (frame["source"] == source) & (frame["level"] == "entidad") & (frame["level_value"] == level_value)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_prepare_writes_parquet_with_documented_schema(tmp_path: Path) -> None:
    _write_project(tmp_path)

    message = run_prepare(ProjectPaths(root=tmp_path))

    frame = _frame(tmp_path)
    assert list(frame.columns) == list(BASELINE_COLUMNS)
    assert "national_baselines.parquet" in message
    assert not (tmp_path / "data" / "processed" / "national_baselines.parquet").is_symlink()
    assert "ENVI 2020" in (tmp_path / "reports" / "national_baselines.md").read_text(encoding="utf-8")


def test_percentiles_are_ordered_and_shares_are_in_range(tmp_path: Path) -> None:
    _write_project(tmp_path)
    run_prepare(ProjectPaths(root=tmp_path))

    frame = _frame(tmp_path)
    assert not frame.empty
    for _, row in frame.iterrows():
        assert row["p10"] <= row["p25"] <= row["p50"] <= row["p75"] <= row["p90"]
        adjusted = tuple(
            row[name] for name in ("p10_jul2024", "p25_jul2024", "p50_jul2024", "p75_jul2024", "p90_jul2024")
        )
        assert adjusted == tuple(sorted(adjusted))
        assert 0.0 <= row["rented_share"] <= 1.0


def test_national_and_state_medians_use_the_invented_data(tmp_path: Path) -> None:
    _write_project(tmp_path)
    run_prepare(ProjectPaths(root=tmp_path))
    frame = _frame(tmp_path)

    envi_nacional = frame.loc[(frame["source"] == ENVI_SOURCE) & (frame["level"] == "nacional")].iloc[0]
    assert envi_nacional["p50"] == 2000.0
    assert envi_nacional["p50_jul2024"] == pytest.approx(2000.0 * 136.0 / 102.0)
    assert envi_nacional["n_muestra"] == 3
    assert envi_nacional["rented_share"] == pytest.approx(0.6)

    envi_hidalgo = _row(frame, ENVI_SOURCE, "13")
    assert envi_hidalgo["p50"] == 3000.0
    assert envi_hidalgo["p50_jul2024"] == pytest.approx(3000.0 * 136.0 / 102.0)
    assert envi_hidalgo["n_muestra"] == 2  # los cuatro códigos especiales se excluyeron
    assert envi_hidalgo["rented_share"] == pytest.approx(70.0 / 110.0)

    enigh_nacional = frame.loc[(frame["source"] == ENIGH_SOURCE) & (frame["level"] == "nacional")].iloc[0]
    assert enigh_nacional["p50"] == 2000.0
    assert enigh_nacional["p50_jul2024"] == pytest.approx(2000.0 * 136.0 / 113.0)

    enigh_edomex = _row(frame, ENIGH_SOURCE, "15")
    assert enigh_edomex["p50"] == 2000.0
    assert enigh_edomex["rented_share"] == pytest.approx(0.5)


def test_prepare_report_has_reference_month_note_and_summary_rows(tmp_path: Path) -> None:
    _write_project(tmp_path)
    run_prepare(ProjectPaths(root=tmp_path))

    report = (tmp_path / "reports" / "national_baselines.md").read_text(encoding="utf-8")

    assert "pesos de julio de 2024" in report
    assert "Hidalgo" in report
    assert "Estado de México" in report
    assert report.count("| ENVI 2020 |") == 3
    assert report.count("| ENIGH 2022 |") == 3


def test_cli_prepare_returns_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_project(tmp_path)

    code = main(["--root", str(tmp_path), "prepare"])

    assert code == 0
    assert "national_baselines.parquet" in capsys.readouterr().out


def test_cli_prepare_fails_clearly_when_a_file_is_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_project(tmp_path)
    (tmp_path / "data" / "raw" / "inegi_inpc" / INPC_ZIP).unlink()

    code = main(["--root", str(tmp_path), "prepare"])

    assert code == 1
    assert "ejecute primero `download`" in capsys.readouterr().err


def test_run_stage_dispatches_prepare(tmp_path: Path) -> None:
    _write_project(tmp_path)

    output = run_stage("prepare", ProjectPaths(root=tmp_path))

    assert output.startswith("prepare: ")
    assert "universities: " in output
    assert "denue: " in output
