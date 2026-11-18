"""ML-02 T05 (unidad): categorías SCIAN, radios, privacidad y ZIP del DENUE."""

from __future__ import annotations

import csv
import io
import math
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from thehouse_ml.csvio import iter_zip_csv
from thehouse_ml.denue import (
    ALL_CATEGORIES,
    CATEGORIES,
    DENUE_COLUMNS,
    DENUE_SKIP_MESSAGE,
    RADII_M,
    UNVERIFIED_STATUS,
    Campus,
    DenueError,
    accumulate_counts,
    build_rows,
    categorize,
    load_campuses,
    render_report,
    run_denue,
)
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.sources import load_sources

PRIVATE_COLUMNS = ("nom_estab", "raz_social", "telefono", "correoelec", "www", "tipo_vial", "nom_vial", "numero_ext")
PRIVATE_SENTINEL = "SECRETO_NO_DEBE_APARECER"


def _zip_with_csv(path: Path, members: dict[str, list[list[str]]], encoding: str = "latin-1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, rows in members.items():
            buffer = io.StringIO(newline="")
            csv.writer(buffer, lineterminator="\n").writerows(rows)
            bundle.writestr(name, buffer.getvalue().encode(encoding))


def _campus(code: str = "A", status: str = "ok_osm") -> Campus:
    return Campus(code=code, name=f"Universidad {code}", lat=20.0, lon=-99.0, final_status=status)


def test_categories_and_radii_are_declared_in_one_constant() -> None:
    assert RADII_M == (1000, 2000, 5000)
    assert list(CATEGORIES) == [
        "comida",
        "abarrotes_super",
        "farmacia",
        "salud",
        "papeleria_copias",
        "lavanderia",
        "gimnasio",
        "transporte",
        "bancos",
        "total",
    ]
    assert tuple(CATEGORIES) == ALL_CATEGORIES
    assert CATEGORIES["total"] == ()


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("722111", ("comida", "total")),
        ("461110", ("abarrotes_super", "total")),
        ("462110", ("abarrotes_super", "total")),
        ("464111", ("farmacia", "total")),
        ("464112", ("farmacia", "total")),
        ("621111", ("salud", "total")),
        ("622111", ("salud", "total")),
        ("465311", ("papeleria_copias", "total")),
        ("561431", ("papeleria_copias", "total")),
        ("812210", ("lavanderia", "total")),
        ("713943", ("gimnasio", "total")),
        ("485111", ("transporte", "total")),
        ("522110", ("bancos", "total")),
        ("111111", ("total",)),
    ],
)
def test_categorize_matches_scian_prefixes(code: str, expected: tuple[str, ...]) -> None:
    assert categorize(code) == expected


def test_prefix_does_not_match_a_longer_scian_code() -> None:
    # 4641 no está en farmacia: solo 464111 y 464112.
    assert categorize("464100") == ("total",)


def test_total_counts_every_business_even_with_a_blank_code() -> None:
    assert categorize("  ") == ("total",)


def test_radius_boundary_is_inclusive() -> None:
    campus = _campus()
    # 1 micrómetro dentro del límite (evita el error de redondeo de haversine
    # en exactamente 1000 m) y 1 mm fuera: el límite se compara con <=.
    inside = math.degrees(999.999999 / 6_371_000.0)
    outside = math.degrees(1000.001 / 6_371_000.0)
    frame = pd.DataFrame(
        {
            "latitud": [repr(campus.lat + inside), repr(campus.lat + outside)],
            "longitud": [str(campus.lon), str(campus.lon)],
            "codigo_act": ["722111", "722111"],
        }
    )
    counts: dict[tuple[str, int, str], int] = {}

    accumulate_counts(frame, [campus], counts)

    assert counts[("A", 1000, "comida")] == 1  # el de 1000 m exacto sí cuenta
    assert counts[("A", 1000, "total")] == 1
    assert counts[("A", 2000, "comida")] == 2
    assert counts[("A", 5000, "total")] == 2


def test_reader_uses_only_usecols_and_never_surfaces_private_columns(tmp_path: Path) -> None:
    header = [*DENUE_COLUMNS, *PRIVATE_COLUMNS]
    row = ["1", "722111", "1", "13", "001", "20.0", "-99.0", "2020-01-01", *([PRIVATE_SENTINEL] * len(PRIVATE_COLUMNS))]
    archive = tmp_path / "denue.zip"
    _zip_with_csv(archive, {"conjunto_de_datos/denue.csv": [header, row, row, row]})

    chunks = list(iter_zip_csv(archive, "conjunto_de_datos/denue.csv", "latin-1", DENUE_COLUMNS, chunksize=2))

    assert len(chunks) == 2  # lectura por chunks, no un único DataFrame gigante
    assert all(list(chunk.columns) == list(DENUE_COLUMNS) for chunk in chunks)
    assert all(PRIVATE_SENTINEL not in chunk.to_numpy() for chunk in chunks)


def test_reader_reads_every_csv_inside_the_folder(tmp_path: Path) -> None:
    header = ["id", "codigo_act", "latitud", "longitud"]
    archive = tmp_path / "denue_15.zip"
    _zip_with_csv(
        archive,
        {
            "conjunto_de_datos/denue_inegi_15_.csv": [header, ["1", "722111", "20.0", "-99.0"]],
            "conjunto_de_datos/otro.csv": [header, ["2", "522110", "19.9", "-99.6"]],
            "catalogos/catalogo.csv": [["cve", "desc"], ["x", "y"]],  # fuera del patrón
        },
    )

    frame = pd.concat(list(iter_zip_csv(archive, "conjunto_de_datos/*.csv", "latin-1", header, chunksize=10)))

    assert len(frame) == 2
    assert set(frame["codigo_act"]) == {"722111", "522110"}


def test_build_rows_covers_every_campus_radius_and_category() -> None:
    campuses = (_campus("A"), _campus("B", UNVERIFIED_STATUS))
    counts = {("A", 2000, "comida"): 7}

    rows = build_rows(campuses, counts)

    assert len(rows) == 2 * len(RADII_M) * len(ALL_CATEGORIES)
    assert next(row for row in rows if (row.code, row.radius_m, row.category) == ("A", 2000, "comida")).count == 7
    assert {row.final_status for row in rows if row.code == "B"} == {UNVERIFIED_STATUS}


def test_render_report_separates_unverified_campuses() -> None:
    campuses = (_campus("UTTT"), _campus("UCAN", UNVERIFIED_STATUS))
    rows = build_rows(campuses, {})

    report = render_report(rows, campuses)

    warning = report.index("Universidades con coordenada sin verificar")
    assert "Advertencia" in report
    assert report.index("### `UTTT`") < warning
    assert report.index("### `UCAN`") > warning
    assert "no está verificada" in report
    assert "Fuente: INEGI, DENUE 05/2026" in report


def test_load_campuses_reads_the_final_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "universities.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["code", "name", "final_lat", "final_lon", "final_status"])
        writer.writeheader()
        writer.writerow(
            {"code": "A", "name": "Uno", "final_lat": "20.0", "final_lon": "-99.0", "final_status": "ok_osm"}
        )

    campuses = load_campuses(path)

    assert len(campuses) == 1
    assert campuses[0].code == "A"
    assert campuses[0].lat == 20.0
    assert campuses[0].lon == -99.0
    assert campuses[0].final_status == "ok_osm"


def test_run_denue_skips_when_no_archive_is_downloaded(tmp_path: Path) -> None:
    paths = _empty_project(tmp_path)

    assert run_denue(paths) == DENUE_SKIP_MESSAGE
    assert not (paths.data_processed / "campus_context.parquet").exists()


def test_run_denue_fails_when_only_one_state_is_present(tmp_path: Path) -> None:
    paths = _empty_project(tmp_path)
    (paths.data_raw / "inegi_denue_vigente" / "denue_13_csv.zip").parent.mkdir(parents=True, exist_ok=True)
    (paths.data_raw / "inegi_denue_vigente" / "denue_13_csv.zip").write_bytes(b"x")

    with pytest.raises(DenueError, match=r"denue_15_csv\.zip"):
        run_denue(paths)


def _empty_project(tmp_path: Path) -> ProjectPaths:
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
            {"code": "A", "name": "Uno", "final_lat": "20.0", "final_lon": "-99.0", "final_status": "ok_osm"}
        )
    return ProjectPaths(root=tmp_path)


def test_real_registry_declares_the_two_states_and_confirmed_variables() -> None:
    registry = Path(__file__).resolve().parents[2] / "data" / "sources.toml"
    denue = next(source for source in load_sources(registry) if source.id == "inegi_denue_vigente")

    assert denue.variables_status == "confirmed"
    assert denue.variables_used == DENUE_COLUMNS
    assert [file.filename for file in denue.download_url] == [
        "denue_13_csv.zip",
        "denue_15_1_csv.zip",
        "denue_15_2_csv.zip",
    ]
    assert denue.download_url[0].sha256 == "d0747958bc364177ea8d281047334cffd4d3505247f43967e08da2ca9c1d9ca5"
    assert all(file.sha256 is None for file in denue.download_url[1:])  # lo registra `download`
