"""ML-02: resolución de miembros CSV y lectura por chunks sin datos personales."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from thehouse_ml.csvio import CsvReadError, iter_zip_csv, read_zip_csv, resolve_zip_members


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
    return path


def test_resolve_zip_members_requires_exactly_one_csv_when_no_member_is_given() -> None:
    with pytest.raises(CsvReadError, match="Se esperaba un CSV"):
        resolve_zip_members(["leeme.txt", "otro.txt"], None)
    with pytest.raises(CsvReadError, match="Se esperaba un CSV"):
        resolve_zip_members(["a.csv", "b.csv"], None)


def test_resolve_zip_members_with_no_member_returns_the_only_csv() -> None:
    assert resolve_zip_members(["leeme.txt", "datos.CSV"], None) == ("datos.CSV",)


def test_resolve_zip_members_glob_rejects_a_pattern_without_matches() -> None:
    with pytest.raises(CsvReadError, match="No contiene el miembro"):
        resolve_zip_members(["datos.csv"], "conjunto_de_datos/*.csv")


def test_resolve_zip_members_returns_the_exact_name_when_present() -> None:
    names = ["conjunto_de_datos\\Viviendas13.CSV", "datos.csv"]
    assert resolve_zip_members(names, "conjunto_de_datos\\Viviendas13.CSV") == ("conjunto_de_datos\\Viviendas13.CSV",)


def test_resolve_zip_members_matches_a_relative_path_ignoring_case() -> None:
    names = ["conjunto_de_datos/denue_inegi_13_.csv"]
    assert resolve_zip_members(names, "CONJUNTO_DE_DATOS/DENUE_INEGI_13_.CSV") == tuple(names)


def test_resolve_zip_members_rejects_an_ambiguous_relative_name() -> None:
    names = ["13/x.csv", "15/x.csv"]
    with pytest.raises(CsvReadError, match="coincidencias: 2"):
        resolve_zip_members(names, "x.csv")


def test_iter_zip_csv_uses_only_usecols_and_reads_every_matching_csv(tmp_path: Path) -> None:
    archive = _zip(
        tmp_path / "denue.zip",
        {
            "conjunto_de_datos/denue_inegi_13_.csv": "id,nombre_estab,latitud\n1,Secreto,20.0\n",
            "conjunto_de_datos/denue_inegi_15_.csv": "id,nombre_estab,latitud\n2,Otro secreto,19.0\n",
        },
    )

    frame = read_zip_csv(archive, "conjunto_de_datos/*.csv", "utf-8", usecols=["id", "latitud"])

    assert list(frame.columns) == ["id", "latitud"]
    assert frame["id"].tolist() == ["1", "2"]


def test_iter_zip_csv_yields_one_dataframe_when_the_reader_is_not_chunked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _zip(tmp_path / "single.zip", {"datos.csv": "id\n1\n"})
    real_read_csv = pd.read_csv

    def fake_read_csv(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("chunksize", None)
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr("thehouse_ml.csvio.pd.read_csv", fake_read_csv)

    frames = list(iter_zip_csv(archive, None, "utf-8"))

    assert len(frames) == 1
    assert frames[0]["id"].tolist() == ["1"]


def test_iter_zip_csv_rejects_a_file_that_is_not_a_zip(tmp_path: Path) -> None:
    archive = tmp_path / "roto.zip"
    archive.write_bytes(b"esto no es un ZIP")

    with pytest.raises(CsvReadError, match="no es un ZIP válido"):
        list(iter_zip_csv(archive, None, "utf-8"))


def test_iter_zip_csv_reports_a_decoding_error(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "latin.zip", {"datos.csv": "nombre\nCésar\n"})

    with pytest.raises(CsvReadError, match="no se pudo leer el CSV"):
        list(iter_zip_csv(archive, None, "ascii"))


def test_read_zip_csv_returns_an_empty_frame_when_there_are_no_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _zip(tmp_path / "vacio.zip", {"datos.csv": "id\n"})

    monkeypatch.setattr("thehouse_ml.csvio.iter_zip_csv", lambda *_args, **_kwargs: iter(()))

    frame = read_zip_csv(archive, None, "utf-8", usecols=["id"])

    assert frame.empty
    assert list(frame.columns) == ["id"]
