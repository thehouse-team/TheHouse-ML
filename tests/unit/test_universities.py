"""ML-02 T07: haversine, selección de candidato, estados y consulta a Nominatim."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from conftest import REPO_ROOT
from thehouse_ml.universities import (
    COUNTRY_CODES,
    DISTANCE_THRESHOLD_M,
    NOMINATIM_URL,
    RESULT_LIMIT,
    USER_AGENT,
    ManualVerification,
    OsmCandidate,
    OsmError,
    UniversityRow,
    UniversitySeed,
    apply_manual,
    build_query_url,
    build_rows,
    finalize_row,
    haversine_m,
    load_manual,
    load_seed,
    parse_candidates,
    select_candidate,
)

SEED_PATH = REPO_ROOT / "data" / "seed" / "universities_seed.csv"
MANUAL_PATH = REPO_ROOT / "data" / "seed" / "universities_manual.csv"


def _seed(lat: str = "20.0", lon: str = "-99.0", code: str = "UTTT") -> UniversitySeed:
    return UniversitySeed(code=code, name="Universidad de prueba", city="Tula", latitude=lat, longitude=lon)


def _candidate(lat: str, lon: str, *, category: str = "amenity", kind: str = "university") -> OsmCandidate:
    return OsmCandidate(
        osm_type="node",
        osm_id="123",
        osm_name="Campus",
        osm_lat=lat,
        osm_lon=lon,
        category=category,
        kind=kind,
    )


def test_haversine_matches_a_known_equator_degree() -> None:
    assert haversine_m(0.0, 0.0, 0.0, 1.0) == pytest.approx(111_194.93, abs=0.05)


def test_haversine_is_zero_for_the_same_point() -> None:
    assert haversine_m(20.00903, -99.34471, 20.00903, -99.34471) == pytest.approx(0.0)


def test_haversine_is_symmetric() -> None:
    direct = haversine_m(20.0, -99.0, 20.1, -99.2)
    reverse = haversine_m(20.1, -99.2, 20.0, -99.0)

    assert direct == pytest.approx(reverse)


@pytest.mark.parametrize(
    ("category", "kind", "expected"),
    [
        ("amenity", "university", True),
        ("amenity", "college", True),
        ("amenity", "school", True),
        ("building", "university", False),
        ("amenity", "restaurant", False),
        ("highway", "residential", False),
    ],
)
def test_is_educational_accepts_only_educational_amenities(category: str, kind: str, expected: bool) -> None:
    assert _candidate("20.0", "-99.0", category=category, kind=kind).is_educational is expected


def test_select_candidate_chooses_the_closest_educational_element() -> None:
    far = _candidate("20.05", "-99.0")  # ~5.5 km
    near = _candidate("20.0001", "-99.0001")  # ~15 m
    non_educational = _candidate("20.00001", "-99.00001", category="building")

    chosen = select_candidate([far, non_educational, near], _seed())

    assert chosen is near


def test_select_candidate_returns_none_when_there_is_no_educational_element() -> None:
    candidates = [_candidate("20.0", "-99.0", category="building")]

    assert select_candidate(candidates, _seed()) is None


def test_parse_candidates_ignores_malformed_entries() -> None:
    payload = [{"osm_type": "way", "osm_id": 5, "lat": "20", "lon": "-99", "class": "amenity", "type": "college"}, 7]

    assert len(parse_candidates(payload, "UTTT")) == 1


def test_parse_candidates_reads_category_of_the_current_nominatim_response() -> None:
    payload = [{"osm_type": "way", "osm_id": 9, "lat": "20", "lon": "-99", "category": "amenity", "type": "college"}]

    candidates = parse_candidates(payload, "UTTT")

    assert candidates[0].is_educational is True


def test_parse_candidates_rejects_a_non_list_payload() -> None:
    with pytest.raises(OsmError, match="se esperaba una lista"):
        parse_candidates({"error": "ubo"}, "UTTT")


def test_build_rows_marks_ok_at_the_threshold() -> None:
    seed = _seed()
    exact = haversine_m(seed.latitude_float, seed.longitude_float, 20.0, -99.0)
    assert exact == pytest.approx(0.0)

    rows = build_rows([seed], {seed.code: [_candidate("20.0", "-99.0")]}, "2026-11-17")

    assert rows[0].status == "ok"
    assert rows[0].distance_m == "0.00"
    assert rows[0].verified_at == "2026-11-17"
    assert DISTANCE_THRESHOLD_M == 200.0


def test_build_rows_marks_revisar_beyond_the_threshold() -> None:
    seed = _seed()

    rows = build_rows([seed], {seed.code: [_candidate("20.005", "-99.0")]}, "2026-11-17")

    assert rows[0].status == "revisar"
    assert float(rows[0].distance_m) > DISTANCE_THRESHOLD_M


def test_build_rows_marks_sin_coincidencia_and_leaves_osm_columns_empty() -> None:
    seed = _seed()

    rows = build_rows([seed], {seed.code: []}, "2026-11-17")

    assert rows[0].status == "sin_coincidencia"
    assert rows[0].osm_type == rows[0].osm_id == rows[0].osm_name == rows[0].osm_lat == rows[0].osm_lon == ""
    assert rows[0].distance_m == ""


def test_build_query_url_contains_the_nominatim_policy_parameters() -> None:
    url = build_query_url(_seed("20.00903", "-99.34471"))
    split = urlsplit(url)
    query = parse_qs(split.query)

    assert f"{split.scheme}://{split.netloc}{split.path}" == NOMINATIM_URL
    assert query["format"] == ["jsonv2"]
    assert query["limit"] == [str(RESULT_LIMIT)]
    assert query["countrycodes"] == [COUNTRY_CODES]
    assert query["bounded"] == ["1"]
    assert query["q"] == ["Universidad de prueba"]
    assert query["viewbox"] == ["-99.444710,20.109030,-99.244710,19.909030"]


def test_user_agent_identifies_the_project() -> None:
    assert USER_AGENT == "TheHouse-ML/0.1 (proyecto escolar UTTT)"


def test_load_seed_reads_the_ten_universities() -> None:
    seeds = load_seed(SEED_PATH)

    assert len(seeds) == 10
    assert seeds[0].code == "UTTT"
    assert seeds[0].latitude == "20.00903"
    assert seeds[-1].code == "UNE-TULA"
    assert any("," in seed.name for seed in seeds)  # Nombres con comas citados correctamente.


def test_load_seed_rejects_a_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "mal.csv"
    path.write_text("code,name\nUTTT,Prueba\n", encoding="utf-8")

    with pytest.raises(OsmError, match="faltan columnas"):
        load_seed(path)


def test_load_seed_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OsmError, match="No se puede leer el seed"):
        load_seed(tmp_path / "no_existe.csv")


def _manual(status: str, lat: str = "", lon: str = "", osm_ref: str = "") -> ManualVerification:
    return ManualVerification(
        code="UTTT",
        manual_status=status,
        lat=lat,
        lon=lon,
        osm_ref=osm_ref,
        verified_by="César Yael López Trujillo",
        verified_at="2026-11-18",
        note="Nota de prueba",
    )


def _row(status: str = "sin_coincidencia") -> UniversityRow:
    return UniversityRow(
        code="UTTT",
        name="Universidad de prueba",
        seed_lat="20.00903",
        seed_lon="-99.34471",
        osm_type="way",
        osm_id="386380501",
        osm_lat="20.0084106",
        osm_lon="-99.3449608",
        status=status,
    )


def test_final_status_ok_osm_uses_the_osm_coordinates() -> None:
    row = finalize_row(_row("ok"), {})

    assert (row.final_lat, row.final_lon) == ("20.0084106", "-99.3449608")
    assert row.final_status == "ok_osm"
    assert row.final_source == "osm:way/386380501"


def test_final_status_ok_manual_uses_the_seed_coordinates() -> None:
    row = finalize_row(_row(), {"UTTT": _manual("ok_seed")})

    assert (row.final_lat, row.final_lon) == ("20.00903", "-99.34471")
    assert row.final_status == "ok_manual"
    assert row.final_source == "seed"


def test_final_status_corregida_manual_uses_the_human_coordinates() -> None:
    entry = _manual("corregida", "20.132062", "-99.235535", "way/550950316")

    row = finalize_row(_row("revisar"), {"UTTT": entry})

    assert (row.final_lat, row.final_lon) == ("20.132062", "-99.235535")
    assert row.final_status == "corregida_manual"
    assert row.final_source == "way/550950316"


def test_final_status_seed_sin_verificar_uses_the_seed_coordinates() -> None:
    row = finalize_row(_row(), {"UTTT": _manual("no_localizada")})

    assert (row.final_lat, row.final_lon) == ("20.00903", "-99.34471")
    assert row.final_status == "seed_sin_verificar"
    assert row.final_source == "seed"


def test_apply_manual_keeps_the_original_columns() -> None:
    (row,) = apply_manual([_row("ok")], {})

    assert row.seed_lat == "20.00903"
    assert row.osm_id == "386380501"
    assert row.status == "ok"
    assert row.distance_m == ""


def test_load_manual_reads_the_eight_human_decisions() -> None:
    manual = load_manual(MANUAL_PATH)

    assert len(manual) == 8
    assert manual["ITSOEH"].manual_status == "corregida"
    assert manual["ITSOEH"].osm_ref == "nominatim T07"
    assert manual["UAEH-ESTL"].osm_ref == "way/550950316"
    assert manual["UCAN"].osm_ref == ""
    assert "Instituto Canadiense A.C." in manual["UCAN"].note
    assert manual["UAEH-ESTE"].verified_at == "2026-11-18"


def test_load_manual_rejects_an_unknown_status(tmp_path: Path) -> None:
    path = tmp_path / "manual.csv"
    path.write_text(
        "code,manual_status,lat,lon,osm_ref,verified_by,verified_at,note\nUTTT,desconocido,,,,César,2026-11-18,nota\n",
        encoding="utf-8",
    )

    with pytest.raises(OsmError, match="manual_status inválido"):
        load_manual(path)


def test_load_manual_rejects_a_correction_without_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "manual.csv"
    path.write_text(
        "code,manual_status,lat,lon,osm_ref,verified_by,verified_at,note\nUTTT,corregida,,,,César,2026-11-18,nota\n",
        encoding="utf-8",
    )

    with pytest.raises(OsmError, match="necesita lat y lon"):
        load_manual(path)
