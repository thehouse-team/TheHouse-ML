"""Esquema de features compartido con el backend (ML-03 T01, ML-03-R01).

`feature_schema.json` vive junto a este módulo y se empaqueta con él.
"""

from __future__ import annotations

from functools import cache

from thehouse_ml.features.loader import (
    SCHEMA_RESOURCE,
    FeatureSchema,
    FieldSpec,
    HardConstraint,
    Metadata,
    MissingRule,
    PairFeature,
    SchemaError,
    ValueRange,
    load_schema,
    parse_schema,
    schema_transport_factors,
    validate,
)


@cache
def default_schema() -> FeatureSchema:
    """Esquema empaquetado, cargado y validado una sola vez."""
    return load_schema()


def schema_version() -> str:
    """Versión semántica del esquema empaquetado."""
    return default_schema().version


__all__ = [
    "SCHEMA_RESOURCE",
    "FeatureSchema",
    "FieldSpec",
    "HardConstraint",
    "Metadata",
    "MissingRule",
    "PairFeature",
    "SchemaError",
    "ValueRange",
    "default_schema",
    "load_schema",
    "parse_schema",
    "schema_transport_factors",
    "schema_version",
    "validate",
]
