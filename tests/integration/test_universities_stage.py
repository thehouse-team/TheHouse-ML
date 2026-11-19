"""ML-02 T07 (integración): etapa de universidades completa con respuestas inventadas y sin red."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from conftest import OSM_DOWNLOADED_AT, REPO_ROOT, write_osm_project
from thehouse_ml.cli import main
from thehouse_ml.download import run_download
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.universities import OSM_SKIP_MESSAGE, OSM_SOURCE, UNIVERSITY_COLUMNS, run_universities

pytestmark = pytest.mark.integration

VERIFIED_AT = "2026-11-17"


def _rows(paths: ProjectPaths) -> list[dict[str, str]]:
    with paths.universities_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == UNIVERSITY_COLUMNS
        return list(reader)


def test_prepare_writes_the_ten_universities_with_all_three_statuses(tmp_path: Path) -> None:
    write_osm_project(tmp_path)
    paths = ProjectPaths(root=tmp_path)

    message = run_universities(paths, today=VERIFIED_AT)

    rows = _rows(paths)
    assert "universities:" in message
    assert len(rows) == 10
    assert {row["code"] for row in rows} == {
        "UTTT",
        "UAEH-ESTE",
        "UAEH-ESTL",
        "UAEH-ESAT",
        "ITSOEH",
        "TESJI",
        "UMB-JILO",
        "UMB-VDC",
        "UCAN",
        "UNE-TULA",
    }
    statuses = {row["code"]: row["status"] for row in rows}
    assert "ok" in statuses.values()
    assert statuses["UMB-VDC"] == "revisar"
    assert statuses["UCAN"] == "sin_coincidencia"
    by_code = {row["code"]: row for row in rows}
    # Hallazgo 24: ok_osm toma la fecha del manifiesto OSM; las 8 filas con
    # revisión humana toman `verified_at` del CSV manual (2026-11-18).
    assert by_code["UTTT"]["verified_at"] == OSM_DOWNLOADED_AT
    assert by_code["UNE-TULA"]["verified_at"] == OSM_DOWNLOADED_AT
    for code in ("UAEH-ESTE", "UAEH-ESTL", "UAEH-ESAT", "ITSOEH", "TESJI", "UMB-JILO", "UMB-VDC", "UCAN"):
        assert by_code[code]["verified_at"] == "2026-11-18"
    assert all(row["final_lat"] and row["final_lon"] for row in rows)
    assert all(row["final_status"] and row["final_source"] for row in rows)


def test_prepare_on_two_different_days_does_not_change_the_csv(tmp_path: Path) -> None:
    """Hallazgo 24: dos fechas simuladas producen exactamente el mismo archivo."""
    write_osm_project(tmp_path)
    paths = ProjectPaths(root=tmp_path)
    report_path = paths.reports / "universities_check.md"

    run_universities(paths, today="2026-11-19")
    csv_first = paths.universities_csv.read_bytes()
    report_first = report_path.read_bytes()
    run_universities(paths, today="2026-11-20")

    assert paths.universities_csv.read_bytes() == csv_first
    assert report_path.read_bytes() == report_first
    rows = {row["code"]: row for row in _rows(paths)}
    assert rows["UTTT"]["verified_at"] == OSM_DOWNLOADED_AT
    assert rows["UCAN"]["verified_at"] == "2026-11-18"


def test_all_ten_rows_have_final_coordinates_and_the_expected_final_statuses(tmp_path: Path) -> None:
    write_osm_project(tmp_path)
    paths = ProjectPaths(root=tmp_path)

    run_universities(paths, today=VERIFIED_AT)
    rows = {row["code"]: row for row in _rows(paths)}

    assert len(rows) == 10
    assert rows["UTTT"]["final_status"] == "ok_osm"
    assert rows["UTTT"]["final_lat"] == "20.00903"  # Candidato inventado sobre el seed.
    assert rows["UTTT"]["final_source"] == "osm:node/2000"
    assert rows["UAEH-ESTE"]["final_status"] == "ok_manual"
    assert rows["UAEH-ESTE"]["final_lat"] == "19.91703"
    assert rows["ITSOEH"]["final_status"] == "corregida_manual"
    assert (rows["ITSOEH"]["final_lat"], rows["ITSOEH"]["final_lon"]) == ("20.2333739", "-99.2096330")
    assert rows["UAEH-ESTL"]["final_status"] == "corregida_manual"
    assert (rows["UAEH-ESTL"]["final_lat"], rows["UAEH-ESTL"]["final_lon"]) == ("20.132062", "-99.235535")
    for code in ("TESJI", "UMB-VDC", "UCAN"):
        assert rows[code]["final_status"] == "seed_sin_verificar"
        assert rows[code]["final_source"] == "seed"


def test_prepare_writes_the_osm_columns_and_haversine_distance(tmp_path: Path) -> None:
    write_osm_project(tmp_path)
    paths = ProjectPaths(root=tmp_path)

    run_universities(paths, today=VERIFIED_AT)
    rows = {row["code"]: row for row in _rows(paths)}

    assert rows["UTTT"]["osm_type"] == "node"
    assert rows["UTTT"]["osm_id"] == "2000"
    assert rows["UTTT"]["osm_lat"] == "20.00903"
    assert float(rows["UTTT"]["distance_m"]) == pytest.approx(0.0, abs=0.01)
    assert rows["UCAN"]["osm_type"] == ""
    assert rows["UCAN"]["osm_lat"] == ""
    assert rows["UCAN"]["distance_m"] == ""


def test_report_has_attribution_and_the_list_to_fix_nv11(tmp_path: Path) -> None:
    write_osm_project(tmp_path)
    paths = ProjectPaths(root=tmp_path)

    run_universities(paths, today=VERIFIED_AT)
    report = (paths.reports / "universities_check.md").read_text(encoding="utf-8")

    assert "© OpenStreetMap contributors (ODbL)" in report
    assert "b7d75955513320a0a97aa5daf6b3bbe92e9514d2" in report
    assert "## Universidades a corregir en NV-11" in report
    assert "`UMB-VDC`" in report
    assert "`UCAN`" in report
    assert report.count("| `UTTT` |") == 1
    assert "## Verificación humana (openstreetmap.org)" in report
    assert "## Correcciones para NV-11 (backend)" in report
    assert "Instituto Canadiense A.C." in report
    assert "Google Maps" in report
    corrections = report.split("## Correcciones para NV-11 (backend)", maxsplit=1)[1]
    for code in ("ITSOEH", "UAEH-ESTL", "TESJI", "UMB-VDC", "UCAN"):
        assert corrections.count(f"- `{code}` —") == 1


def test_prepare_without_osm_responses_keeps_the_versioned_files(tmp_path: Path) -> None:
    write_osm_project(tmp_path, with_responses=False)
    paths = ProjectPaths(root=tmp_path)
    shutil.copyfile(REPO_ROOT / "data" / "universities.csv", paths.universities_csv)
    report = paths.reports / "universities_check.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / "reports" / "universities_check.md", report)
    csv_before = paths.universities_csv.read_bytes()
    report_before = report.read_bytes()

    message = run_universities(paths, today=VERIFIED_AT)

    assert message == OSM_SKIP_MESSAGE
    assert paths.universities_csv.read_bytes() == csv_before
    assert report.read_bytes() == report_before
    assert not (paths.data_raw / OSM_SOURCE).exists()


def test_filtered_download_keeps_every_source_in_the_variables_report(tmp_path: Path) -> None:
    write_osm_project(tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / "data" / "sources.toml", tmp_path / "data" / "sources.toml")
    paths = ProjectPaths(root=tmp_path)

    def forbidden(_url: str, _destination: Path) -> None:
        raise AssertionError("Las respuestas OSM ya existen: no se debe usar la red")

    output = run_download(paths, source_id=OSM_SOURCE, fetcher=forbidden)
    report = (paths.reports / "variables_check.md").read_text(encoding="utf-8")

    assert "openstreetmap: manual" in output
    for source_id in (
        "inegi_censo_2020_viviendas_ca",
        "inegi_envi_2020",
        "inegi_enigh_2022",
        "inegi_denue_vigente",
        "inegi_inpc",
        "inside_airbnb_cdmx_2026_09_23",
        "openstreetmap",
        "thehouse_seed_nv07_nv08_nv11",
    ):
        assert f"## {source_id}" in report, source_id
    assert report.count("## ") == 8


def test_cli_reports_osm_errors_with_exit_code_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / "data" / "sources.toml", tmp_path / "data" / "sources.toml")

    code = main(["--root", str(tmp_path), "download", "--source", "openstreetmap"])

    assert code == 1
    assert "No se puede leer el seed" in capsys.readouterr().err
