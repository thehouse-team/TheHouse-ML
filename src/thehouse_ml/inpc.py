"""INPC general y deflactor a pesos de julio de 2024 (M-ML-014).

El archivo de datos abiertos trae siete conceptos por mes. Solo se usa la fila
cuyo ``CONCEPTO`` termina en ``Precios al Consumidor (INPC)`` (índice general),
no la subyacente ni la no subyacente. Aunque el diccionario dice que ``VALOR``
es "variación porcentual anual", los valores son niveles del índice base 2Q
jul 2018 = 100 (verificado por el responsable el 2026-11-17).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

GENERAL_CONCEPT_SUFFIX = "Precios al Consumidor (INPC)"
REFERENCE_MONTH = "2024-07"
ENVI_SURVEY_MONTHS: tuple[str, ...] = ("2020-10", "2020-11", "2020-12")
ENIGH_SURVEY_MONTHS: tuple[str, ...] = ("2022-08", "2022-09", "2022-10", "2022-11")


class InpcError(ValueError):
    """El índice general falta en algún mes necesario para el deflactor."""


@dataclass(frozen=True)
class Deflators:
    """Factores para llevar cada encuesta a pesos de julio de 2024."""

    envi: float
    enigh: float


def month_of(fecha: str) -> str:
    """Convierte ``2024-07-01`` o ``2024-07`` en ``2024-07``."""
    parts = fecha.strip().split("-")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise InpcError(f"FECHA con formato inesperado: {fecha!r}")
    try:
        month = int(parts[1])
    except ValueError as exc:
        raise InpcError(f"FECHA con mes inválido: {fecha!r}") from exc
    if not 1 <= month <= 12:
        raise InpcError(f"FECHA con mes fuera de rango: {fecha!r}")
    return f"{parts[0]}-{month:02d}"


def general_index(records: Iterable[Mapping[str, str]]) -> dict[str, float]:
    """Devuelve ``{YYYY-MM: valor}`` solo del concepto del índice general."""
    index: dict[str, float] = {}
    for record in records:
        concept = str(record.get("CONCEPTO", "")).strip()
        if not concept.endswith(GENERAL_CONCEPT_SUFFIX):
            continue
        month = month_of(str(record.get("FECHA", "")))
        raw_value = str(record.get("VALOR", "")).strip()
        try:
            index[month] = float(raw_value)
        except ValueError as exc:
            raise InpcError(f"VALOR no numérico en {month}: {raw_value!r}") from exc
    return index


def deflactor(index: Mapping[str, float], survey_months: Iterable[str]) -> float:
    """``INPC(2024-07) / promedio(INPC de los meses del levantamiento)``."""
    months = tuple(survey_months)
    if not months:
        raise InpcError("El periodo del levantamiento no puede estar vacío")
    missing = [month for month in months if month not in index]
    if missing:
        raise InpcError("Faltan meses del índice general: " + ", ".join(missing))
    if REFERENCE_MONTH not in index:
        raise InpcError(f"Falta el índice general de {REFERENCE_MONTH}")
    period_average = sum(index[month] for month in months) / len(months)
    if period_average <= 0.0:
        raise InpcError("El promedio del índice general debe ser positivo")
    return index[REFERENCE_MONTH] / period_average


def build_deflators(index: Mapping[str, float]) -> Deflators:
    """Calcula los dos factores: ENVI 2020 (oct-dic) y ENIGH 2022 (ago-nov)."""
    return Deflators(
        envi=deflactor(index, ENVI_SURVEY_MONTHS),
        enigh=deflactor(index, ENIGH_SURVEY_MONTHS),
    )
