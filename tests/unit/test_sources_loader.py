"""ML-02 T02: lectura y validación del registro ampliado con download_url."""

from __future__ import annotations

from pathlib import Path

import pytest

from thehouse_ml.sources import SourceRegistryError, SourceSpec, load_sources, select_sources

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES_PATH = REPO_ROOT / "data" / "sources.toml"

ENVI_VARIABLES = ("P5_1", "P5_2", "P5_5", "ENT", "TLOC", "FACTOR", "EST_DIS", "UPM_DIS")
ENIGH_VARIABLES = (
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


def _by_id() -> dict[str, SourceSpec]:
    return {source.id: source for source in load_sources(SOURCES_PATH)}


def test_envi_registers_two_verified_files() -> None:
    envi = _by_id()["inegi_envi_2020"]

    files = envi.download_url
    assert len(files) == 2
    assert files[0].filename == "envi_2020_base_de_datos_csv.zip"
    assert files[0].size_bytes == 7_322_530
    assert files[0].sha256 == "36b8e3d04f37722974e0d48edb2b4708b4a5e55c9c3faa551b9b75fda95cc40b"
    assert files[0].table == "Bases de datos/TVIVIENDA.csv"
    assert files[0].encoding == "latin-1"
    assert files[1].kind == "dictionary"
    assert files[1].sha256 == "208fc3d2c70a4cc2b346479813fd98c0481fa5c490d5c8bc2ca8970694befbcf"


def test_envi_and_enigh_variables_are_confirmed_without_p5_8() -> None:
    sources = _by_id()
    envi = sources["inegi_envi_2020"]
    enigh = sources["inegi_enigh_2022"]

    assert envi.variables_status == "confirmed"
    assert set(envi.variables_used) == set(ENVI_VARIABLES)
    assert "P5_8" not in envi.variables_used
    assert enigh.variables_status == "confirmed"
    assert set(enigh.variables_used) == set(ENIGH_VARIABLES)


def test_enigh_registers_one_verified_zip_with_bom() -> None:
    enigh = _by_id()["inegi_enigh_2022"]

    files = enigh.download_url
    assert len(files) == 1
    assert files[0].size_bytes == 1_712_569
    assert files[0].sha256 == "6ebd189feb30657dda1ebc95ab8815d13717dac4022d95672ec17ccf56095bd4"
    assert files[0].table == "viviendas.csv"
    assert files[0].encoding == "utf-8-sig"


@pytest.mark.parametrize("source_id", ["inegi_censo_2020_viviendas_ca", "inegi_denue_vigente", "inegi_inpc"])
def test_pending_sources_have_no_download_url(source_id: str) -> None:
    source = _by_id()[source_id]

    assert source.download_url == ()


def test_select_sources_filters_and_rejects_unknown_ids() -> None:
    sources = load_sources(SOURCES_PATH)

    assert [source.id for source in select_sources(sources, "inegi_envi_2020")] == ["inegi_envi_2020"]
    with pytest.raises(SourceRegistryError, match="Fuente desconocida"):
        select_sources(sources, "no_existe")


def test_hash_is_lowercased_and_filename_derived_from_url(tmp_path: Path) -> None:
    digest = "A" * 64
    path = tmp_path / "sources.toml"
    path.write_text(
        f"""[[sources]]
id = "prueba"
name = "Prueba"
license = "Licencia"
attribution = "Atribucion"
variables_status = "pending"
variables_used = ["x"]
download_url = [{{ url = "https://example.test/archivo.zip", sha256 = "{digest}" }}]
""",
        encoding="utf-8",
    )

    source = load_sources(path)[0]

    assert source.download_url[0].sha256 == "a" * 64
    assert source.download_url[0].filename == "archivo.zip"
    assert source.download_url[0].kind == "data"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sha256", '"abc"', "64 caracteres"),
        ("kind", '"otro"', "kind inválido"),
        ("size_bytes", "-1", "entero no negativo"),
    ],
)
def test_invalid_registry_entries_fail(field: str, value: str, message: str, tmp_path: Path) -> None:
    path = tmp_path / "sources.toml"
    path.write_text(
        f"""[[sources]]
id = "prueba"
name = "Prueba"
license = "Licencia"
attribution = "Atribucion"
variables_status = "pending"
variables_used = ["x"]
download_url = [{{ url = "https://example.test/a.zip", {field} = {value} }}]
""",
        encoding="utf-8",
    )

    with pytest.raises(SourceRegistryError, match=message):
        load_sources(path)


def test_invalid_source_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sources.toml"
    path.write_text(
        """[[sources]]
id = "../mal"
name = "Prueba"
license = "Licencia"
attribution = "Atribucion"
variables_status = "pending"
variables_used = ["x"]
""",
        encoding="utf-8",
    )

    with pytest.raises(SourceRegistryError, match="id inválido"):
        load_sources(path)


def test_missing_required_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sources.toml"
    path.write_text(
        """[[sources]]
id = "prueba"
name = "Prueba"
variables_status = "pending"
variables_used = ["x"]
""",
        encoding="utf-8",
    )

    with pytest.raises(SourceRegistryError, match="'license'"):
        load_sources(path)


def test_missing_registry_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SourceRegistryError, match="No se puede leer"):
        load_sources(tmp_path / "sin_archivo.toml")
