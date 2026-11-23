"""Métricas de ranking por estudiante y bootstrap (ML-04 T06, ML-04-R06; reutilizable por ML-05).

Funciones puras (sin I/O ni estado global). Un *ranking* es la lista de
relevancias graduadas (enteros ≥ 0) de los candidatos de un estudiante en el
orden en que los puso un ranker (posición 1 primero). La lista completa de
candidatos define el orden ideal, así que el ranking debe traer **todos** los
candidatos del estudiante, no solo los primeros k.

Definiciones (constitución §10.2, spec ML-04 §6.6, M-ML-034):

- **DCG@k** = Σ_{i=1..min(k, n)} (2^rel_i - 1) / log2(i + 1)  (ganancia graduada, descuento log2).
- **NDCG@k** = DCG@k / IDCG@k, con IDCG@k el DCG@k del orden por relevancia
  descendente. Si IDCG@k = 0 (ningún candidato con relevancia > 0) el NDCG
  **no está definido** y se devuelve `None`.
- **Precision@k** = (relevantes en las primeras k posiciones) / k, con
  relevante = `rel ≥ threshold`. El denominador es siempre k (como trec_eval):
  con menos de k candidatos el máximo alcanzable es n / k.
- **AP** = (1 / R) Σ_{i: rel_i ≥ threshold} Precision@i sobre la lista
  completa, con R = relevantes del estudiante. Si R = 0 el AP **no está
  definido** y se devuelve `None`. **MAP** = media de AP entre estudiantes.

Tratamiento de un estudiante sin ningún cuarto relevante: `ndcg_at_k`
(si además no hay ganancia) y `average_precision` devuelven `None` y
`student_metrics` también; quien evalúa decide. ML-04 T06 **excluye** a esos
estudiantes del conjunto evaluado (igual que a los de menos de 2 candidatos) y
reporta cuántos son, para que todas las métricas usen los mismos estudiantes.
Con las etiquetas de T04 no hay ninguno: todo estudiante con candidatos tiene
al menos un 3 (M-ML-032).

Bootstrap por estudiante: se remuestrean **estudiantes** con reemplazo (no
pares), se calcula la media de cada serie en cada remuestreo con los **mismos**
índices para todas las series (remuestreo pareado: sirve para diferencias
entre rankers) y el intervalo es el percentil (1 - c)/2 y (1 + c)/2 de esas
medias con interpolación lineal.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

METRIC_NAMES: tuple[str, ...] = ("ndcg@10", "ndcg@5", "precision@5", "map")


class RankingInputError(ValueError):
    """Relevancias, k, umbral o parámetros del bootstrap inválidos."""


def _check_relevances(relevances: Sequence[int]) -> None:
    for value in relevances:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RankingInputError(f"relevancia inválida {value!r}: se esperaba un entero ≥ 0")


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise RankingInputError(f"k inválido {k!r}: se esperaba un entero ≥ 1")


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    """DCG@k con ganancia 2^rel - 1 y descuento log2(posición + 1)."""
    _check_relevances(relevances)
    _check_k(k)
    total = 0.0
    for position, rel in enumerate(relevances[:k], start=1):
        total += (2**rel - 1) / math.log2(position + 1)
    return total


def ndcg_at_k(relevances: Sequence[int], k: int) -> float | None:
    """NDCG@k; `None` si el orden ideal no tiene ganancia (todas las relevancias en 0 o lista vacía)."""
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal == 0.0:
        return None
    return dcg_at_k(relevances, k) / ideal


def precision_at_k(relevances: Sequence[int], k: int, threshold: int) -> float:
    """Fracción de relevantes (`rel ≥ threshold`) en las primeras k posiciones, siempre sobre k."""
    _check_relevances(relevances)
    _check_k(k)
    _check_threshold(threshold)
    return sum(1 for rel in relevances[:k] if rel >= threshold) / k


def average_precision(relevances: Sequence[int], threshold: int) -> float | None:
    """AP sobre la lista completa; `None` si no hay ningún relevante (`rel ≥ threshold`)."""
    _check_relevances(relevances)
    _check_threshold(threshold)
    hits = 0
    total = 0.0
    for position, rel in enumerate(relevances, start=1):
        if rel >= threshold:
            hits += 1
            total += hits / position
    if hits == 0:
        return None
    return total / hits


def _check_threshold(threshold: int) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
        raise RankingInputError(f"umbral de relevancia inválido {threshold!r}: se esperaba un entero ≥ 1")


@dataclass(frozen=True)
class StudentMetrics:
    """Métricas de un estudiante, en el orden de `METRIC_NAMES`."""

    ndcg_10: float
    ndcg_5: float
    precision_5: float
    average_precision: float

    def as_dict(self) -> dict[str, float]:
        return dict(zip(METRIC_NAMES, self.values(), strict=True))

    def values(self) -> tuple[float, float, float, float]:
        return (self.ndcg_10, self.ndcg_5, self.precision_5, self.average_precision)


def student_metrics(relevances: Sequence[int], threshold: int) -> StudentMetrics | None:
    """NDCG@10, NDCG@5, Precision@5 y AP de un ranking; `None` si alguna no está definida."""
    ndcg_10 = ndcg_at_k(relevances, 10)
    ndcg_5 = ndcg_at_k(relevances, 5)
    ap = average_precision(relevances, threshold)
    if ndcg_10 is None or ndcg_5 is None or ap is None:
        return None
    return StudentMetrics(ndcg_10, ndcg_5, precision_at_k(relevances, 5, threshold), ap)


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Percentil `q` ∈ [0, 1] con interpolación lineal entre posiciones (`numpy` «linear»)."""
    if not sorted_values:
        raise RankingInputError("percentile: lista vacía")
    if not 0.0 <= q <= 1.0:
        raise RankingInputError(f"percentile: q debe estar en [0, 1], llegó {q!r}")
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = position - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * fraction


@dataclass(frozen=True)
class Interval:
    """Media observada y su intervalo percentil por bootstrap."""

    mean: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "ci_low": self.low, "ci_high": self.high}


def bootstrap_means(
    series: Mapping[str, Sequence[float]],
    *,
    resamples: int,
    confidence: float,
    rng: random.Random,
) -> dict[str, Interval]:
    """Media de cada serie (un valor por estudiante) con intervalo percentil por bootstrap pareado.

    Todas las series deben tener la misma longitud n ≥ 1 (un valor por
    estudiante en el mismo orden). En cada remuestreo se eligen n índices con
    reemplazo con `rng.choices` y se usan para **todas** las series. El
    resultado depende solo del estado de `rng`, del orden de los estudiantes y
    del orden de las series.
    """
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise RankingInputError(f"resamples inválido {resamples!r}: se esperaba un entero ≥ 1")
    if not 0.0 < confidence < 1.0:
        raise RankingInputError(f"confidence inválido {confidence!r}: se esperaba un valor en (0, 1)")
    lengths = {len(values) for values in series.values()}
    if len(lengths) != 1 or 0 in lengths:
        raise RankingInputError("bootstrap_means: las series deben tener la misma longitud ≥ 1")
    n = lengths.pop()
    names = list(series)
    columns = [[float(value) for value in series[name]] for name in names]
    population = range(n)
    means: list[list[float]] = [[] for _ in names]
    for _ in range(resamples):
        indices = rng.choices(population, k=n)
        for column, bucket in zip(columns, means, strict=True):
            bucket.append(math.fsum(column[i] for i in indices) / n)
    tail = (1.0 - confidence) / 2.0
    result: dict[str, Interval] = {}
    for name, column, bucket in zip(names, columns, means, strict=True):
        bucket.sort()
        result[name] = Interval(
            mean=math.fsum(column) / n,
            low=percentile(bucket, tail),
            high=percentile(bucket, 1.0 - tail),
        )
    return result


__all__ = [
    "METRIC_NAMES",
    "Interval",
    "RankingInputError",
    "StudentMetrics",
    "average_precision",
    "bootstrap_means",
    "dcg_at_k",
    "ndcg_at_k",
    "percentile",
    "precision_at_k",
    "student_metrics",
]
