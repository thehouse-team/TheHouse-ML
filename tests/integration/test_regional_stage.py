"""ML-02 T04 (integración): etapa regional completa con datos inventados y sin red."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from thehouse_ml.national import (
    BASELINE_COLUMNS,
    ENVI_SOURCE,
    BaselineRow,
    rows_to_frame,
)
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.regional import CENSO_SOURCE, REGIONAL_COLUMNS, run_regional

pytestmark = pytest.mark.integration

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

HGO_ZIP = "censo_hgo_de_prueba.zip"
MEX_ZIP = "censo_mex_de_prueba.zip"


def _write_zip(path: Path, member: str, rows: list[list[str]]) -> None:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(member, buffer.getvalue().encode("latin-1"))


def _censo_rows(entidad: str, mun: str, rented: bool, size: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for index in range(size):
        rows.append(
            [
                entidad,
                mun,
                "10",
                "1",
                "A" if index % 2 == 0 else "B",
                "2" if rented else "1",
                "1" if rented else "3",
                "1" if rented else "5",
                "1" if rented else "3",
                "7" if rented else "8",
                "2" if rented else "5",
            ]
        )
    return rows


def _write_project(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    registry = f"""[[sources]]
id = "{CENSO_SOURCE}"
name = "Censo de prueba"
license = "Terminos de libre uso de INEGI"
attribution = "INEGI"
variables_status = "confirmed"
variables_used = {CENSO_COLUMNS}

[[sources.download_url]]
url = "https://example.test/{HGO_ZIP}"
filename = "{HGO_ZIP}"
kind = "data"
table = "Viviendas13.CSV"
encoding = "latin-1"

[[sources.download_url]]
url = "https://example.test/{MEX_ZIP}"
filename = "{MEX_ZIP}"
kind = "data"
table = "Viviendas15.CSV"
encoding = "latin-1"
"""
    (tmp_path / "data" / "sources.toml").write_text(registry, encoding="utf-8")

    hidalgo = [CENSO_COLUMNS]
    hidalgo.extend(_censo_rows("13", "070", True, 40))
    for mun in ("013", "041", "063", "076"):
        hidalgo.extend(_censo_rows("13", mun, False, 1))
    mexico = [CENSO_COLUMNS]
    mexico.extend(_censo_rows("15", "045", True, 40))
    mexico.extend(_censo_rows("15", "112", False, 1))
    raw = tmp_path / "data" / "raw" / CENSO_SOURCE
    _write_zip(raw / HGO_ZIP, "Viviendas13.CSV", hidalgo)
    _write_zip(raw / MEX_ZIP, "Viviendas15.CSV", mexico)

    baselines = [
        BaselineRow(
            source=source,
            year=2020,
            level=level,
            level_value=value,
            p10=1000.0,
            p25=1500.0,
            p50=2000.0,
            p75=2500.0,
            p90=3000.0,
            p10_jul2024=1250.0,
            p25_jul2024=1875.0,
            p50_jul2024=2500.0,
            p75_jul2024=3125.0,
            p90_jul2024=3750.0,
            rented_share=0.4,
            n_muestra=200,
            suma_factores=100_000.0,
            n_total=1000,
            suma_factores_total=250_000.0,
            deflactor=1.25,
        )
        for source, level, value in (
            (ENVI_SOURCE, "nacional", "nacional"),
            (ENVI_SOURCE, "entidad", "13"),
            (ENVI_SOURCE, "entidad", "15"),
        )
    ]
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    frame = rows_to_frame(baselines)
    assert list(frame.columns) == list(BASELINE_COLUMNS)
    frame.to_parquet(processed / "national_baselines.parquet", index=False)


def _frame(tmp_path: Path) -> pd.DataFrame:
    return pd.read_parquet(tmp_path / "data" / "processed" / "regional_calibration.parquet")


def test_run_regional_writes_schema_and_all_seven_municipalities(tmp_path: Path) -> None:
    _write_project(tmp_path)

    message = run_regional(ProjectPaths(root=tmp_path))

    frame = _frame(tmp_path)
    assert list(frame.columns) == list(REGIONAL_COLUMNS)
    assert "regional_calibration.parquet" in message
    assert len(frame) == 56
    assert frame["municipio_clave"].nunique() == 7
    assert set(frame["level"]) <= {"municipio", "estado", "nacional"}
    assert "Atotonilco de Tula" in (tmp_path / "reports" / "regional_calibration.md").read_text(encoding="utf-8")


def test_rent_rows_copy_state_envi_percentiles(tmp_path: Path) -> None:
    _write_project(tmp_path)
    run_regional(ProjectPaths(root=tmp_path))

    frame = _frame(tmp_path)
    rent = frame.loc[frame["indicador"] == "renta"]
    assert len(rent) == 7
    assert (rent["level"] == "estado").all()
    assert (rent["source"] == "ENVI").all()
    assert (rent["p10"] == 1250.0).all()
    assert (rent["p50"] == 2500.0).all()
    assert (rent["p90"] == 3750.0).all()
    assert rent["se"].isna().all()
    assert rent["cv"].isna().all()


def test_sufficient_municipality_is_kept_and_small_one_falls_back(tmp_path: Path) -> None:
    _write_project(tmp_path)
    run_regional(ProjectPaths(root=tmp_path))

    frame = _frame(tmp_path)
    tlahuelilpan = frame.loc[(frame["municipio_clave"] == "13070") & (frame["indicador"] == "rentadas")].iloc[0]
    atotonilco = frame.loc[(frame["municipio_clave"] == "13013") & (frame["indicador"] == "rentadas")].iloc[0]
    assert tlahuelilpan["level"] == "municipio"
    assert tlahuelilpan["n_muestra"] == 40
    assert tlahuelilpan["estimate"] == pytest.approx(1.0)
    assert atotonilco["level"] == "estado"
    assert atotonilco["source"] == "Censo"


def test_weights_of_valid_dwellings_are_the_denominator(tmp_path: Path) -> None:
    _write_project(tmp_path)
    run_regional(ProjectPaths(root=tmp_path))

    frame = _frame(tmp_path)
    gas = frame.loc[(frame["municipio_clave"] == "13070") & (frame["indicador"] == "gas")].iloc[0]
    assert gas["suma_factores"] == pytest.approx(400.0)
    assert gas["n_muestra"] == 40
