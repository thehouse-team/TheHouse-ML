"""Evaluación de rankings por estudiante (ML-04 T06; reutilizable por ML-05).

`ranking` tiene las métricas puras (NDCG@k, Precision@k, AP/MAP) y el
bootstrap por estudiante con intervalo percentil. No lee archivos ni conoce el
modelo base: recibe listas de relevancias en el orden que produjo un ranker.
"""

from __future__ import annotations

from thehouse_ml.evaluation.ranking import (
    METRIC_NAMES,
    Interval,
    RankingInputError,
    StudentMetrics,
    average_precision,
    bootstrap_means,
    dcg_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    student_metrics,
)

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
