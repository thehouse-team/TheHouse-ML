"""Modelos de compatibilidad (ML-04: modelo base por reglas; ML-05: modelos entrenados).

ML-04 T01 (ML-04-R01): configuración versionada `baseline-1.0.0`
(`baseline_params.json`, empaquetado junto a este módulo) y su cargador.
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

__all__ = [
    "CONFIG_RESOURCE",
    "BaselineConfig",
    "BaselineConfigError",
    "Curve",
    "Factor",
    "NonScoringFeature",
    "RoundingRules",
    "default_config",
    "load_config",
    "parse_config",
    "schema_is_compatible",
]
