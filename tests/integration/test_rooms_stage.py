"""ML-03 T05 (integración): etapa del catálogo sintético de cuartos con una calibración inventada y sin red."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from thehouse_ml.cli import main
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.regional import MUNICIPIOS
from thehouse_ml.synthetic.rooms_stage import (
    HISTOGRAM_PARQUET,
    ROOMS_PARQUET,
    ROOMS_REPORT,
    ROOMS_SKIP_MESSAGE,
    RoomsError,
    run_rooms,
)

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
# Renta p50 inventada por entidad (pesos) y proporciones del Censo inventadas.
INVENTED_P50 = {"13": 2000.0, "15": 3000.0}
INVENTED_CENSUS = {"agua_entubada": 0.95, "electricidad": 0.99, "gas": 0.90, "internet": 0.40, "drenaje": 0.97}


def _write_project(
    tmp_path: Path, *, skip_indicator: str | None = None, histogram_scopes: tuple[str, ...] = ("13", "15")
) -> ProjectPaths:
    paths = ProjectPaths(root=tmp_path)
    paths.data_processed.mkdir(parents=True)
    shutil.copyfile(REPO / "data" / "universities.csv", tmp_path / "data" / "universities.csv")
    rows = []
    for municipality in MUNICIPIOS:
        for indicator in ("renta", *INVENTED_CENSUS):
            if indicator == skip_indicator and municipality.clave == "13076":
                continue
            rows.append(
                {
                    "municipio_clave": municipality.clave,
                    "municipio_nombre": municipality.nombre,
                    "entidad": municipality.entidad,
                    "indicador": indicator,
                    "level": "estado" if indicator == "renta" else "municipio",
                    "estimate": float("nan") if indicator == "renta" else INVENTED_CENSUS[indicator],
                    "p50": INVENTED_P50[municipality.entidad] if indicator == "renta" else float("nan"),
                }
            )
    pd.DataFrame(rows).to_parquet(paths.data_processed / "regional_calibration.parquet", index=False)
    bins = [(0.0, 1000.0, 10.0), (1000.0, 2000.0, 30.0), (2000.0, 4000.0, 40.0), (10000.0, None, 5.0)]
    histogram = pd.DataFrame(
        [
            {"scope": scope, "lower": lower, "upper": upper, "frecuencia_expandida": weight}
            for scope in ("nacional", *histogram_scopes)
            for lower, upper, weight in bins
        ]
    )
    histogram.to_parquet(paths.data_processed / HISTOGRAM_PARQUET, index=False)
    return paths


def test_run_rooms_writes_the_parquet_and_the_report(tmp_path: Path) -> None:
    paths = _write_project(tmp_path)

    message = run_rooms(paths)

    assert message.startswith("rooms: 3000 cuartos sintéticos")
    frame = pd.read_parquet(paths.data_processed / ROOMS_PARQUET)
    assert len(frame) == 3000
    assert bool(frame["synthetic"].all())
    assert frame.groupby("university_code").size().eq(300).all()
    assert frame["campus_distance_km"].max() <= 30.0
    assert set(frame.loc[~frame["seed_verified"], "university_code"]) == {"TESJI", "UMB-VDC", "UCAN"}
    for code, group in frame.groupby("university_code"):
        target = 3000.0 if str(code) in {"TESJI", "UMB-JILO", "UMB-VDC"} else 2000.0
        assert abs(float(group["price_mxn"].median()) / target - 1.0) <= 0.10, code
    report = (paths.reports / ROOMS_REPORT).read_text(encoding="utf-8")
    assert report.startswith("# DATOS SINTÉTICOS")
    for heading in (
        "Calibración de la renta",
        "Renta por tipo de lugar",
        "Distancia al campus",
        "Servicios frente al Censo 2020",
        "placeType",
        "propertyType",
        "Mascotas",
        "livingWith",
        "Recámaras",
        "Privacidad",
    ):
        assert heading in report
    assert "| UTTT | Tula de Allende | estado | $2,000 |" in report
    assert "| UTTT | `wifi` | `internet` | municipio | 40.0 % |" in report
    assert "**no**" in report  # campus sin verificar marcados en la tabla de distancia
    calibration = report.split("## 2.")[1].split("## 3.")[0]
    assert "**no**" not in calibration  # toda la calibración dentro de la tolerancia
    assert str(tmp_path) not in report


def test_running_twice_does_not_change_any_file(tmp_path: Path) -> None:
    paths = _write_project(tmp_path)
    run_rooms(paths)
    report = (paths.reports / ROOMS_REPORT).read_bytes()
    parquet = (paths.data_processed / ROOMS_PARQUET).read_bytes()

    run_rooms(paths)

    assert (paths.reports / ROOMS_REPORT).read_bytes() == report
    assert (paths.data_processed / ROOMS_PARQUET).read_bytes() == parquet


def test_without_inputs_the_stage_is_skipped(tmp_path: Path) -> None:
    assert run_rooms(ProjectPaths(root=tmp_path)) == ROOMS_SKIP_MESSAGE
    paths = _write_project(tmp_path)
    (paths.data_processed / HISTOGRAM_PARQUET).unlink()
    assert run_rooms(paths) == ROOMS_SKIP_MESSAGE
    assert not (tmp_path / "reports").exists()


def test_missing_census_indicator_is_an_error(tmp_path: Path) -> None:
    paths = _write_project(tmp_path, skip_indicator="internet")
    with pytest.raises(RoomsError, match="internet"):
        run_rooms(paths)


def test_missing_rent_is_an_error(tmp_path: Path) -> None:
    paths = _write_project(tmp_path, skip_indicator="renta")
    with pytest.raises(RoomsError, match="13076"):
        run_rooms(paths)


def test_missing_state_histogram_is_an_error(tmp_path: Path) -> None:
    paths = _write_project(tmp_path, histogram_scopes=("13",))
    with pytest.raises(RoomsError, match="entidad 15"):
        run_rooms(paths)


def test_missing_campus_in_universities_csv_is_an_error(tmp_path: Path) -> None:
    paths = _write_project(tmp_path)
    lines = paths.universities_csv.read_text(encoding="utf-8").splitlines()
    paths.universities_csv.write_text(
        "\n".join(line for line in lines if not line.startswith("UTTT,")) + "\n", encoding="utf-8"
    )
    with pytest.raises(RoomsError, match="UTTT"):
        run_rooms(paths)


def test_cli_reports_rooms_errors_with_exit_code_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_project(tmp_path, histogram_scopes=("13",))

    def no_op(_paths: ProjectPaths) -> str:
        return "ok"

    from thehouse_ml import stages

    for name in ("run_prepare", "run_regional", "run_universities", "run_denue", "run_students", "run_data_card"):
        monkeypatch.setattr(stages, name, no_op)

    assert main(["--root", str(tmp_path), "prepare"]) == 1
    assert "entidad 15" in capsys.readouterr().err
