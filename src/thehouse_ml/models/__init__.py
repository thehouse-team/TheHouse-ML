"""Modelos de compatibilidad (ML-04: modelo base por reglas; ML-05: modelos entrenados).

ML-04 T01 (ML-04-R01): configuración versionada `baseline-1.0.0`
(`baseline_params.json`, empaquetado junto a este módulo) y su cargador.
ML-04 T03 (ML-04-R03): puntaje de 0 a 100 con desglose por factor (`baseline_score`).
"""

from __future__ import annotations

from thehouse_ml.models.baseline_config import (
    CONFIG_RESOURCE,
    BaselineConfig,
    BaselineConfigError,
    Curve,
    Factor,
    NonScoringFeature,
    RoundingRules,
    default_config,
    load_config,
    parse_config,
    schema_is_compatible,
)
from thehouse_ml.models.baseline_score import (
    ExcludedPair,
    FactorScore,
    ScoredRoom,
    ScoreInputError,
    ScoreResult,
    factor_points,
    match_score,
    rank,
    round_half_up,
    score_features,
    score_pair,
)

__all__ = [
    "CONFIG_RESOURCE",
    "BaselineConfig",
    "BaselineConfigError",
    "Curve",
    "ExcludedPair",
    "Factor",
    "FactorScore",
    "NonScoringFeature",
    "RoundingRules",
    "ScoreInputError",
    "ScoreResult",
    "ScoredRoom",
    "default_config",
    "factor_points",
    "load_config",
    "match_score",
    "parse_config",
    "rank",
    "round_half_up",
    "schema_is_compatible",
    "score_features",
    "score_pair",
]
