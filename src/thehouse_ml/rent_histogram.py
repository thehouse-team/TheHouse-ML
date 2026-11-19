"""Histograma de la renta ponderada de ENVI para la data card (ML-02 T08).

A diferencia de ``national_baselines.parquet`` (que solo guarda percentiles),
aquí se conserva la distribución completa en intervalos fijos de 500 pesos
hasta 10,000 y un último intervalo abierto ``10,000+``. Cada intervalo guarda
el conteo sin ponderar de viviendas y la frecuencia expandida (suma de
``FACTOR``), ya en pesos de julio de 2024.

La supresión de celdas con menos de 30 viviendas en muestra se aplica al
renderizar (`:func:`group_small_bins`): los intervalos pequeños se agrupan con
los adyacentes, de modo que la suma de frecuencias expandidas no cambia.

Salida intermedia: ``data/processed/rent_histogram.parquet`` (fuera de git),
generada por la parte nacional de ``prepare`` y leída por ``data_card``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import pandas as pd

from thehouse_ml.thresholds import MIN_DWELLINGS

ENVI_SOURCE = "inegi_envi_2020"

BIN_WIDTH = 500.0
BIN_MAX = 10_000.0
BIN_COUNT = int(BIN_MAX / BIN_WIDTH) + 1  # 0..9,500 (20 cerrados) + 10,000+ (1 abierto)

RENT_HISTOGRAM_COLUMNS: tuple[str, ...] = (
    "scope",
    "label",
    "lower",
    "upper",
    "n_muestra",
    "frecuencia_expandida",
    "source",
    "deflactor",
)


class RentHistogramError(ValueError):
    """Los datos del histograma no son válidos."""


@dataclass(frozen=True)
class RentBin:
    """Un intervalo del histograma de renta de un ámbito (scope)."""

    scope: str
    lower: float
    upper: float | None  # None = intervalo abierto (10,000+)
    n_muestra: int
    frecuencia_expandida: float
    source: str = ENVI_SOURCE
    deflactor: float = 1.0

    @property
    def label(self) -> str:
        """Etiqueta legible, por ejemplo ``0-499`` o ``10,000+``."""
        if self.upper is None:
            return f"{self.lower:,.0f}+"
        return f"{self.lower:,.0f}-{self.upper - 1:,.0f}"


def build_bins(
    scope: str,
    rents: Sequence[float],
    weights: Sequence[float],
    deflator: float,
    *,
    source: str = ENVI_SOURCE,
) -> list[RentBin]:
    """Construye los intervalos fijos a partir de rentas y factores de expansión.

    ``rents`` están en pesos del año de la encuesta; se multiplican por
    ``deflator`` para expresarlos en pesos de julio de 2024. Las rentas no
    finitas o negativas se ignoran (la normalización de T03 ya las excluye).
    """
    if len(rents) != len(weights):
        raise RentHistogramError("rents y weights deben tener la misma longitud")
    if deflator <= 0.0 or not math.isfinite(deflator):
        raise RentHistogramError("el deflactor debe ser finito y positivo")

    counts = [0] * BIN_COUNT
    expanded = [0.0] * BIN_COUNT
    for raw_rent, raw_weight in zip(rents, weights, strict=True):
        try:
            rent = float(raw_rent)
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(rent) or not math.isfinite(weight) or rent < 0.0 or weight < 0.0:
            continue
        value = rent * deflator
        index = min(math.floor(value / BIN_WIDTH), BIN_COUNT - 1)
        counts[index] += 1
        expanded[index] += weight

    return [
        RentBin(
            scope=scope,
            lower=index * BIN_WIDTH,
            upper=None if index == BIN_COUNT - 1 else (index + 1) * BIN_WIDTH,
            n_muestra=counts[index],
            frecuencia_expandida=expanded[index],
            source=source,
            deflactor=deflator,
        )
        for index in range(BIN_COUNT)
    ]


def group_small_bins(
    bins: Sequence[RentBin],
    min_sample: int = MIN_DWELLINGS,
) -> list[RentBin]:
    """Agrupa intervalos adyacentes hasta que cada celda tenga ``min_sample``.

    La suma de ``frecuencia_expandida`` se conserva. Si el total de la muestra
    es menor que ``min_sample``, no se publica ninguna celda (se devuelve una
    lista vacía) para no revelar intervalos con pocas viviendas.
    """
    if bins and any(bin_.scope != bins[0].scope for bin_ in bins[1:]):
        raise RentHistogramError("todos los intervalos deben pertenecer al mismo scope")
    grouped: list[RentBin] = []
    pending: list[RentBin] = []
    for bin_ in bins:
        if bin_.n_muestra == 0:
            continue  # un intervalo vacío no aporta celdas ni frecuencia
        pending.append(bin_)
        if sum(item.n_muestra for item in pending) >= min_sample:
            grouped.append(_merge(pending))
            pending = []
    if pending:
        if not grouped:
            return []
        grouped[-1] = _merge([grouped[-1], *pending])
    return grouped


def rows_to_frame(bins: Sequence[RentBin]) -> pd.DataFrame:
    """Materializa los intervalos con un esquema estable de Parquet."""
    records = []
    for bin_ in bins:
        records.append(
            {
                "scope": bin_.scope,
                "label": bin_.label,
                "lower": bin_.lower,
                "upper": math.nan if bin_.upper is None else bin_.upper,
                "n_muestra": bin_.n_muestra,
                "frecuencia_expandida": bin_.frecuencia_expandida,
                "source": bin_.source,
                "deflactor": bin_.deflactor,
            }
        )
    return pd.DataFrame(records, columns=list(RENT_HISTOGRAM_COLUMNS))


def bins_from_frame(frame: pd.DataFrame) -> list[RentBin]:
    """Reconstruye los intervalos a partir del Parquet (``upper`` nulo = abierto)."""
    result: list[RentBin] = []
    for record in frame.to_dict("records"):
        upper = record["upper"]
        result.append(
            RentBin(
                scope=str(record["scope"]),
                lower=float(record["lower"]),
                upper=None if pd.isna(upper) else float(upper),
                n_muestra=int(record["n_muestra"]),
                frecuencia_expandida=float(record["frecuencia_expandida"]),
                source=str(record["source"]),
                deflactor=float(record["deflactor"]),
            )
        )
    return result


def _merge(bins: Sequence[RentBin]) -> RentBin:
    """Fusiona intervalos contiguos conservando pesos y deflactor."""
    first, last = bins[0], bins[-1]
    return replace(
        first,
        lower=first.lower,
        upper=last.upper,
        n_muestra=sum(bin_.n_muestra for bin_ in bins),
        frecuencia_expandida=sum(bin_.frecuencia_expandida for bin_ in bins),
    )


__all__ = [
    "BIN_COUNT",
    "BIN_MAX",
    "BIN_WIDTH",
    "ENVI_SOURCE",
    "RENT_HISTOGRAM_COLUMNS",
    "RentBin",
    "RentHistogramError",
    "bins_from_frame",
    "build_bins",
    "group_small_bins",
    "rows_to_frame",
]
