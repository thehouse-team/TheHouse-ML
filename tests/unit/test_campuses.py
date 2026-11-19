"""ML-03 T02 (R02): coordenadas de campus desde `data/universities.csv`."""

from __future__ import annotations

from pathlib import Path

import pytest

from thehouse_ml.features import Campus, CampusError, default_schema, load_campuses

REPO = Path(__file__).resolve().parents[2]
HEADER = "code,final_lat,final_lon,final_status\n"


def test_the_real_csv_gives_ten_campuses_with_the_unverified_ones_marked() -> None:
    campuses = load_campuses(REPO / "data" / "universities.csv")
    schema = default_schema()
    assert set(campuses) == set(schema.catalog("university_code"))
    unverified = {code for code, campus in campuses.items() if not campus.seed_verified}
    assert unverified == set(schema.unverified_university_codes)
    assert campuses["UTTT"] == Campus("UTTT", 20.0084106, -99.3449608, seed_verified=True)


def test_blank_codes_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "u.csv"
    path.write_text(HEADER + "A,20.0,-99.0,ok_osm\n,,,\n", encoding="utf-8")
    assert list(load_campuses(path)) == ["A"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("code,final_lat\nA,20\n", "faltan columnas"),
        (HEADER + "A,20,-99,ok_osm\nA,21,-99,ok_osm\n", "duplicado"),
        (HEADER + "A,,-99,ok_osm\n", "coordenadas finales"),
        (HEADER + "A,95,-99,ok_osm\n", "latitud"),
        (HEADER + "A,20,-190,ok_osm\n", "longitud"),
    ],
)
def test_malformed_csv_is_rejected(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "u.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CampusError, match=message):
        load_campuses(path)


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CampusError, match="No se pudo leer"):
        load_campuses(tmp_path / "missing.csv")


def test_a_campus_needs_a_code() -> None:
    with pytest.raises(CampusError, match="sin código"):
        Campus("", 20.0, -99.0)
