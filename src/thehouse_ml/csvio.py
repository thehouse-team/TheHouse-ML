"""Lectura de un CSV dentro de un ZIP con la codificación declarada.

Se usan cadenas para conservar los ceros a la izquierda de los códigos de
INEGI (por ejemplo ``000003000`` en ``P5_2`` o ``01001`` en ``ubica_geo``).
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


class CsvReadError(RuntimeError):
    """El ZIP no contiene el CSV esperado o no se puede decodificar."""


def read_zip_csv(
    archive: Path,
    member: str | None,
    encoding: str,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Lee un miembro CSV de ``archive`` como un ``DataFrame`` de texto."""
    try:
        with zipfile.ZipFile(archive) as bundle:
            chosen = _resolve_member(bundle.namelist(), member)
            with bundle.open(chosen) as stream:
                return pd.read_csv(
                    stream,
                    encoding=encoding,
                    dtype=str,
                    keep_default_na=False,
                    usecols=list(usecols) if usecols is not None else None,
                )
    except zipfile.BadZipFile as exc:
        raise CsvReadError(f"{archive.name}: no es un ZIP válido") from exc
    except (KeyError, OSError, UnicodeError, ValueError) as exc:
        raise CsvReadError(f"{archive.name}: no se pudo leer el CSV {member!r}: {exc}") from exc


def _resolve_member(names: list[str], member: str | None) -> str:
    """Resuelve el nombre del CSV, tolerando mayúsculas y barras invertidas."""
    if member is not None and member in names:
        return member
    csv_names = [name for name in names if name.lower().endswith(".csv")]
    if member is None:
        if len(csv_names) != 1:
            raise CsvReadError(f"Se esperaba un CSV y se encontraron {len(csv_names)}")
        return csv_names[0]
    wanted = member.lower().replace("\\", "/")
    matches = [name for name in csv_names if name.lower().replace("\\", "/") == wanted]
    if not matches:
        matches = [name for name in csv_names if name.lower().replace("\\", "/").endswith("/" + wanted)]
    if len(matches) != 1:
        raise CsvReadError(f"No se encontró un único CSV para {member!r}")
    return matches[0]
