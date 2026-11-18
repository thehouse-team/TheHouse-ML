"""Lectura de CSV dentro de un ZIP con la codificación declarada.

Se usan cadenas para conservar los ceros a la izquierda de los códigos de
INEGI (por ejemplo ``000003000`` en ``P5_2`` o ``01001`` en ``ubica_geo``).

Un ``table`` con comodín (por ejemplo ``conjunto_de_datos/*.csv``) lee **todos**
los CSV que coinciden; es el caso del DENUE de Estado de México (M-ML-018).
``iter_zip_csv`` lee por chunks para no cargar en memoria un microdato grande.
"""

from __future__ import annotations

import fnmatch
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd

CHUNK_SIZE = 100_000


class CsvReadError(RuntimeError):
    """El ZIP no contiene el CSV esperado o no se puede decodificar."""


def resolve_zip_members(names: Sequence[str], member: str | None) -> tuple[str, ...]:
    """Resuelve uno o varios miembros CSV de un ZIP.

    * ``member=None`` exige un único CSV en el archivo (comportamiento histórico);
    * un patrón con ``*`` devuelve todos los CSV que coinciden (DENUE 15);
    * un nombre concreto admite mayúsculas, ``\\`` y rutas relativas.
    """
    csv_names = [name for name in names if name.lower().endswith(".csv")]
    if member is None:
        if len(csv_names) != 1:
            raise CsvReadError(f"Se esperaba un CSV y se encontraron {len(csv_names)}")
        return (csv_names[0],)

    wanted = member.lower().replace("\\", "/")
    if "*" in wanted or "?" in wanted:
        matches = [name for name in csv_names if fnmatch.fnmatchcase(name.lower().replace("\\", "/"), wanted)]
        if not matches:
            raise CsvReadError(f"No contiene el miembro {member!r}")
        return tuple(sorted(matches))

    if member in names:
        return (member,)
    matches = [name for name in csv_names if name.lower().replace("\\", "/") == wanted]
    if not matches:
        matches = [name for name in csv_names if name.lower().replace("\\", "/").endswith("/" + wanted)]
    if len(matches) != 1:
        raise CsvReadError(f"no contiene el miembro {member!r} (coincidencias: {len(matches)})")
    return (matches[0],)


def iter_zip_csv(
    archive: Path,
    member: str | None,
    encoding: str,
    usecols: Sequence[str] | None = None,
    chunksize: int = CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Itera por chunks el CSV (o los CSV) de ``archive``, solo con ``usecols``.

    ``usecols`` evita que las columnas privadas de la fuente (nombres, teléfonos,
    correos, direcciones) lleguen a memoria: el lector nunca las materializa.
    """
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = resolve_zip_members(bundle.namelist(), member)
            for chosen in members:
                with bundle.open(chosen) as stream:
                    reader = pd.read_csv(
                        stream,
                        encoding=encoding,
                        dtype=str,
                        keep_default_na=False,
                        usecols=list(usecols) if usecols is not None else None,
                        chunksize=chunksize,
                    )
                    if isinstance(reader, pd.DataFrame):
                        yield reader
                    else:
                        yield from reader
    except zipfile.BadZipFile as exc:
        raise CsvReadError(f"{archive.name}: no es un ZIP válido") from exc
    except (KeyError, OSError, UnicodeError, ValueError) as exc:
        raise CsvReadError(f"{archive.name}: no se pudo leer el CSV {member!r}: {exc}") from exc


def read_zip_csv(
    archive: Path,
    member: str | None,
    encoding: str,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Lee uno o varios miembros CSV de ``archive`` como ``DataFrame`` de texto."""
    frames = list(iter_zip_csv(archive, member, encoding, usecols))
    if not frames:
        return pd.DataFrame(columns=list(usecols) if usecols is not None else [])
    return pd.concat(frames, ignore_index=True)
