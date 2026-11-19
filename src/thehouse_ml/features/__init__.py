"""Esquema de features compartido con el backend (ML-03 T01, ML-03-R01).

Incluye las features del par (T02) y las restricciones duras (T03).

`feature_schema.json` vive junto a este módulo y se empaqueta con él.
"""

from __future__ import annotations

from functools import cache

from thehouse_ml.features.campuses import Campus, CampusError, load_campuses
from thehouse_ml.features.constraints import (
    ConstraintResult,
    FilterResult,
    check_hard_constraints,
    filter_rooms,
    implemented_constraints,
)
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
from thehouse_ml.features.pairs import (
    IncompleteProfileError,
    PairFeatures,
    PairInputError,
    ResolvedStudent,
    Room,
    StudentProfile,
    compute_pair_features,
    feature_vector,
    resolve_student,
    room_from_record,
    student_from_record,
    validate_room,
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
    "Campus",
    "CampusError",
    "ConstraintResult",
    "FeatureSchema",
    "FieldSpec",
    "FilterResult",
    "HardConstraint",
    "IncompleteProfileError",
    "Metadata",
    "MissingRule",
    "PairFeature",
    "PairFeatures",
    "PairInputError",
    "ResolvedStudent",
    "Room",
    "SchemaError",
    "StudentProfile",
    "ValueRange",
    "check_hard_constraints",
    "compute_pair_features",
    "default_schema",
    "feature_vector",
    "filter_rooms",
    "implemented_constraints",
    "load_campuses",
    "load_schema",
    "parse_schema",
    "resolve_student",
    "room_from_record",
    "schema_transport_factors",
    "schema_version",
    "student_from_record",
    "validate",
    "validate_room",
]
