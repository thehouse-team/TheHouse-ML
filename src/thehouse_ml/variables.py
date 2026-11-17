"""Verificación de variables contra los encabezados de los CSV descargados.

Solo se lee la primera línea de cada CSV dentro de los ZIP: nunca se carga el
cuerpo de los microdatos (ML-02 T02 / M-ML-012).
"""

from __future__ import annotations

import csv
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from thehouse_ml.paths import ProjectPaths
from thehouse_ml.sources import SourceSpec


class VariableCheckError(RuntimeError):
    """Una variable confirmada no aparece en el archivo descargado."""


class _BinaryLineReader(Protocol):
    def readline(self) -> bytes:
        """Devuelve la siguiente línea en bytes."""


@dataclass(frozen=True)
class SourceVariableCheck:
    """Resultado de la verificación de una fuente."""

    source_id: str
    variables_status: str
    found: tuple[str, ...]
    missing: tuple[str, ...]
    note: str = ""
    verified: bool = True


@dataclass(frozen=True)
class VariablesReport:
    """Reporte agregado de todas las fuentes verificadas."""

    checks: tuple[SourceVariableCheck, ...]

    @property
    def failures(self) -> list[str]:
        """Fuentes confirmadas con variables ausentes."""
        return [
            f"{check.source_id}: {', '.join(check.missing)}"
            for check in self.checks
            if check.verified and check.variables_status == "confirmed" and check.missing
        ]

    def render(self) -> str:
        """Renderiza el reporte en Markdown."""
        lines = [
            "# Verificacion de variables - ML-02 T02",
            "",
            "Generado por `thehouse-ml download`. Solo se lee el encabezado de los CSV dentro de los ZIP;",
            "no se carga el cuerpo de los microdatos. Una variable confirmada ausente hace fallar el comando.",
            "",
        ]
        for check in self.checks:
            missing = _format_variables(check.missing) if check.verified else "(sin verificar)"
            lines.extend(
                [
                    f"## {check.source_id}",
                    "",
                    f"- estado: `{check.variables_status}`",
                    f"- verificado: {'si' if check.verified else 'no'}",
                    f"- variables encontradas: {_format_variables(check.found)}",
                    f"- variables faltantes: {missing}",
                    f"- nota: {check.note or '-'}",
                    "",
                ]
            )
        return "\n".join(lines)


def verify_sources(sources: Sequence[SourceSpec], paths: ProjectPaths) -> VariablesReport:
    """Compara las variables registradas con los encabezados disponibles."""
    checks: list[SourceVariableCheck] = []
    for source in sources:
        headers: set[str] = set()
        note = ""
        data_files = [file for file in source.download_url if file.kind == "data"]
        present_files = [file for file in data_files if (paths.data_raw / source.id / file.filename).is_file()]
        if not data_files:
            note = "download_url pendiente (sin verificar)"
        elif not present_files:
            note = "archivo no descargado (sin verificar)"
        for file in present_files:
            local = paths.data_raw / source.id / file.filename
            try:
                if file.is_zip:
                    header = read_zip_header(local, file.table, file.encoding)
                else:
                    header = read_csv_header(local, file.encoding)
            except VariableCheckError as exc:
                note = f"{file.filename}: {exc}"
                continue
            headers.update(header)

        found = tuple(variable for variable in source.variables_used if variable in headers)
        missing = tuple(variable for variable in source.variables_used if variable not in headers)
        checks.append(
            SourceVariableCheck(
                source.id,
                source.variables_status,
                found,
                missing,
                note,
                verified=bool(present_files),
            )
        )
    return VariablesReport(tuple(checks))


def read_zip_header(archive: Path, member: str | None, encoding: str = "utf-8") -> list[str]:
    """Lee solo el encabezado de un CSV dentro de un ZIP."""
    try:
        with zipfile.ZipFile(archive) as bundle:
            names = sorted(bundle.namelist())
            chosen: str
            if member is None:
                candidates = [name for name in names if name.lower().endswith(".csv")]
                if len(candidates) != 1:
                    raise VariableCheckError(f"{archive.name}: se esperaba un CSV y se encontraron {len(candidates)}")
                chosen = candidates[0]
            else:
                resolved = _resolve_member(names, member)
                if resolved is None:
                    raise VariableCheckError(f"{archive.name}: no contiene el miembro {member!r}")
                chosen = resolved
            with bundle.open(chosen) as stream:
                return _first_row(stream, encoding)
    except zipfile.BadZipFile as exc:
        raise VariableCheckError(f"{archive.name}: no es un ZIP válido") from exc
    except (OSError, UnicodeError) as exc:
        raise VariableCheckError(f"{archive.name}: no se pudo leer el encabezado: {exc}") from exc


def read_csv_header(path: Path, encoding: str = "utf-8") -> list[str]:
    """Lee solo el encabezado de un CSV suelto."""
    try:
        with path.open("rb") as stream:
            return _first_row(stream, encoding)
    except (OSError, UnicodeError) as exc:
        raise VariableCheckError(f"{path.name}: no se pudo leer el encabezado: {exc}") from exc


def write_variables_report(path: Path, report: VariablesReport) -> None:
    """Escribe el reporte Markdown en ``reports/``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.render(), encoding="utf-8")


def _resolve_member(names: list[str], member: str) -> str | None:
    if member in names:
        return member
    wanted = member.lower().replace("\\", "/")
    normalized = [(name, name.lower().replace("\\", "/")) for name in names]
    exact = [name for name, value in normalized if value == wanted]
    if len(exact) == 1:
        return exact[0]
    suffix = [name for name, value in normalized if value.endswith("/" + wanted) or value == wanted]
    return suffix[0] if len(suffix) == 1 else None


def _first_row(stream: _BinaryLineReader, encoding: str) -> list[str]:
    line = stream.readline()
    if not line:
        raise VariableCheckError("CSV sin encabezado")
    row = next(csv.reader([line.decode(encoding)]), [])
    return [cell.strip().lstrip("\ufeff") for cell in row if cell.strip()]


def _format_variables(variables: tuple[str, ...]) -> str:
    return ", ".join(f"`{variable}`" for variable in variables) if variables else "(ninguna)"
