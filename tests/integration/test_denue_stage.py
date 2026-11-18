"""ML-02 T05 (integración): etapa DENUE con dos ZIP inventados y sin red."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from thehouse_ml.denue import (
    ALL_CATEGORIES,
    DENUE_COLUMNS,
    DENUE_PARQUET_COLUMNS,
    UNVERIFIED_STATUS,
    run_denue,
)
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.stages import run_stage

pytestmark = pytest.mark.integration

PRIVATE_HEADER = ["nom_estab", "raz_social", "telefono", "correoelec", "www", "tipo_vial", "nom_vial", "numero_ext"]
HEADER = [*DENUE_COLUMNS, *PRIVATE_HEADER]


def _row(index: int, lat: float, lon: float, code: str) -> list[str]:
    return [
        str(index),
        code,
        "1",
        "13",
        "001",
        f"{lat:.6f}",
        f"{lon:.6f}",
        "2020-01-01",
        *["SECRETO" for _ in PRIVATE_HEADER],
    ]


def _north(lat: float, meters: float) -> float:
    """Latitud a ``meters`` al norte (suficiente para radios de 1/2/5 km)."""
    return lat + meters / 111_320.0


def _write_zip(path: Path, members: dict[str, list[list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, rows in members.items():
            buffer = io.StringIO(newline="")
            csv.writer(buffer, lineterminator="\n").writerows(rows)
            bundle.writestr(name, buffer.getvalue().encode("latin-1"))


def _write_project(tmp_path: Path) -> ProjectPaths:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "sources.toml").write_text(
        """[[sources]]
id = "inegi_denue_vigente"
name = "DENUE de prueba"
license = "Términos de Libre Uso de la Información del INEGI"
attribution = "Fuente: INEGI, DENUE 05/2026"
dictionary_url = "https://example.test"
variables_status = "confirmed"
variables_used = ["id", "codigo_act", "per_ocu", "cve_ent", "cve_mun", "latitud", "longitud", "fecha_alta"]

[[sources.download_url]]
url = "https://example.test/denue_13_csv.zip"
filename = "denue_13_csv.zip"
kind = "data"
table = "conjunto_de_datos/denue_inegi_13_.csv"
encoding = "latin-1"

[[sources.download_url]]
url = "https://example.test/denue_15_csv.zip"
filename = "denue_15_csv.zip"
kind = "data"
table = "conjunto_de_datos/*.csv"
encoding = "latin-1"
""",
        encoding="utf-8",
    )
    with (tmp_path / "data" / "universities.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["code", "name", "final_lat", "final_lon", "final_status"])
        writer.writeheader()
        writer.writerow(
            {"code": "A", "name": "Campus A", "final_lat": "20.0", "final_lon": "-99.0", "final_status": "ok_osm"}
        )
        writer.writerow(
            {
                "code": "B",
                "name": "Campus B sin verificar",
                "final_lat": "19.9",
                "final_lon": "-99.6",
                "final_status": UNVERIFIED_STATUS,
            }
        )

    raw = tmp_path / "data" / "raw" / "inegi_denue_vigente"
    # Hidalgo: negocios a 200 m (comida), 1.2 km (abarrotes), 3 km (farmacia) y 6 km (fuera).
    _write_zip(
        raw / "denue_13_csv.zip",
        {
            "conjunto_de_datos/denue_inegi_13_.csv": [
                HEADER,
                _row(1, _north(20.0, 200), -99.0, "722111"),
                _row(2, _north(20.0, 1200), -99.0, "461110"),
                _row(3, _north(20.0, 3000), -99.0, "464111"),
                _row(4, _north(20.0, 6000), -99.0, "722111"),
            ]
        },
    )
    # Estado de México: dos CSV dentro de conjunto_de_datos/, uno con banco a 500 m de B
    # y otro con un negocio lejano que no debe contarse.
    _write_zip(
        raw / "denue_15_csv.zip",
        {
            "conjunto_de_datos/denue_inegi_15_.csv": [HEADER, _row(5, _north(19.9, 500), -99.6, "522110")],
            "conjunto_de_datos/denue_inegi_15_b.csv": [HEADER, _row(6, _north(10.0, 100), -99.0, "722111")],
        },
    )
    return ProjectPaths(root=tmp_path)


def _frame(tmp_path: Path) -> pd.DataFrame:
    return pd.read_parquet(tmp_path / "data" / "processed" / "campus_context.parquet")


def test_prepare_writes_the_denue_parquet_with_the_documented_schema(tmp_path: Path) -> None:
    paths = _write_project(tmp_path)

    message = run_denue(paths)

    frame = _frame(tmp_path)
    assert list(frame.columns) == list(DENUE_PARQUET_COLUMNS)
    assert len(frame) == 2 * 3 * len(ALL_CATEGORIES)
    assert "campus_context.parquet" in message
    assert "6 establecimientos" in message


def test_counts_are_monotonic_and_use_both_states(tmp_path: Path) -> None:
    paths = _write_project(tmp_path)
    run_denue(paths)

    frame = _frame(tmp_path)

    def count(code: str, radius: int, category: str) -> int:
        selected = frame.loc[(frame["code"] == code) & (frame["radius_m"] == radius) & (frame["category"] == category)]
        assert len(selected) == 1
        return int(selected.iloc[0]["count"])

    assert count("A", 1000, "comida") == 1
    assert count("A", 2000, "comida") == 1
    assert count("A", 5000, "comida") == 1  # el de 6 km queda fuera
    assert count("A", 1000, "abarrotes_super") == 0
    assert count("A", 2000, "abarrotes_super") == 1
    assert count("A", 5000, "total") == 3
    assert count("B", 1000, "bancos") == 1  # viene del ZIP de Estado de México
    assert count("B", 1000, "total") == 1
    for code in ("A", "B"):
        for category in ALL_CATEGORIES:
            values = [count(code, radius, category) for radius in (1000, 2000, 5000)]
            assert values == sorted(values)
            assert all(value >= 0 for value in values)


def test_report_has_attribution_a_table_per_campus_and_the_warning(tmp_path: Path) -> None:
    paths = _write_project(tmp_path)
    run_denue(paths)

    report = (paths.reports / "denue_campus.md").read_text(encoding="utf-8")

    assert "Fuente: INEGI, DENUE 05/2026" in report
    assert "### `A` — Campus A" in report
    assert "### `B` — Campus B sin verificar" in report
    assert "Universidades con coordenada sin verificar" in report
    assert "Advertencia" in report
    assert (
        report.index("### `A`") < report.index("Universidades con coordenada sin verificar") < report.index("### `B`")
    )


def test_final_status_is_copied_to_every_row(tmp_path: Path) -> None:
    paths = _write_project(tmp_path)
    run_denue(paths)

    frame = _frame(tmp_path)

    assert set(frame.loc[frame["code"] == "A", "final_status"]) == {"ok_osm"}
    assert set(frame.loc[frame["code"] == "B", "final_status"]) == {UNVERIFIED_STATUS}
    assert "SECRETO" not in frame.to_string()


def test_run_stage_dispatches_the_denue_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_project(tmp_path)
    for name in ("run_prepare", "run_regional", "run_universities"):
        monkeypatch.setattr(f"thehouse_ml.stages.{name}", lambda _paths: "parte previa: ok")

    output = run_stage("prepare", paths)

    assert "denue: " in output
    assert "campus_context.parquet" in output
