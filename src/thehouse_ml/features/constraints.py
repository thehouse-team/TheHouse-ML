"""Restricciones duras del par estudiante-cuarto (ML-03 T03, ML-03-R03).

Implementación de referencia, pura (sin I/O, sin red ni aleatoriedad), de las
restricciones `hard_constraints` de `feature_schema.json` (constitution §11):
un cuarto que viola alguna no se recomienda, sin importar el puntaje.

Pasos:

1. `resolve_student` (la misma de `compute_pair_features`, una sola
   implementación) aplica las reglas para faltantes y valida el perfil; un
   quick setup incompleto lanza `IncompleteProfileError`. `validate_room`
   valida el cuarto.
2. Se evalúan **las restricciones que define el esquema, en su orden**. Cada
   id del esquema necesita una implementación aquí; si falta, la función falla.

La distancia **no** es restricción dura (M-ML-023): un cuarto a más de
`maxDistanceKm` o de 30 km no se excluye aquí; ML-04 decide el puntaje.

Privacidad (constitution §11): el perfil no tiene ningún campo de género.
`HC02_living_with` solo compara la convivencia que elige el estudiante
(`livingWith`) con la que declara el cuarto.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from thehouse_ml.features.loader import FeatureSchema
from thehouse_ml.features.pairs import (
    PairInputError,
    ResolvedStudent,
    Room,
    StudentProfile,
    resolve_student,
    validate_room,
)

ANY_LIVING_WITH = "any"

# Id de la restricción del esquema → predicado que devuelve True si el par la VIOLA.
_Predicate = Callable[[ResolvedStudent, Room], bool]
_VIOLATES: Mapping[str, _Predicate] = {
    # room.price_mxn > student.budget.max (renta igual al presupuesto sí pasa).
    "HC01_budget": lambda student, room: room.price_mxn > student.budget_max,
    # student.livingWith != "any" and room.livingWith != student.livingWith.
    "HC02_living_with": lambda student, room: (
        student.living_with != ANY_LIVING_WITH and room.living_with != student.living_with
    ),
    # student.pets = true and room.pets_allowed = false (pets nulo → false por la regla para faltantes).
    "HC03_pets": lambda student, room: student.pets and not room.pets_allowed,
}


def _default_schema() -> FeatureSchema:
    from thehouse_ml.features import default_schema  # import local: evita el ciclo con __init__

    return default_schema()


@dataclass(frozen=True)
class ConstraintResult:
    """Resultado de las restricciones duras para un par.

    `violated` lista **todas** las restricciones violadas, en el orden de
    `hard_constraints` del esquema; `allowed` es verdadero solo si está vacía.
    """

    allowed: bool
    violated: tuple[str, ...]


@dataclass(frozen=True)
class FilterResult:
    """Cuartos permitidos para un estudiante y exclusiones por restricción.

    `exclusions` tiene una entrada por cada id del esquema (en su orden, con 0
    si no excluyó nada). Un cuarto que viola varias restricciones cuenta en
    cada una, así que la suma puede ser mayor que `excluded`.
    """

    allowed: tuple[Room, ...]
    exclusions: Mapping[str, int]
    excluded: int


def _predicates(schema: FeatureSchema) -> tuple[tuple[str, _Predicate], ...]:
    """Predicados en el orden del esquema; falla si alguna restricción no tiene implementación."""
    unknown = [constraint.id for constraint in schema.hard_constraints if constraint.id not in _VIOLATES]
    if unknown:
        raise PairInputError(f"hard_constraints: restricciones sin implementación de referencia {unknown}")
    return tuple((constraint.id, _VIOLATES[constraint.id]) for constraint in schema.hard_constraints)


def _evaluate(predicates: tuple[tuple[str, _Predicate], ...], student: ResolvedStudent, room: Room) -> ConstraintResult:
    violated = tuple(constraint_id for constraint_id, violates in predicates if violates(student, room))
    return ConstraintResult(allowed=not violated, violated=violated)


def check_hard_constraints(
    student: StudentProfile,
    room: Room,
    schema: FeatureSchema | None = None,
) -> ConstraintResult:
    """Evalúa las restricciones duras del esquema para un estudiante y un cuarto.

    Primero aplica las reglas para faltantes (`resolve_student`, la misma de
    `compute_pair_features`) y valida el cuarto; lanza `IncompleteProfileError`
    si falta un dato del quick setup y `PairInputError` si un valor es inválido
    o si el esquema trae una restricción sin implementación.
    """
    schema = schema or _default_schema()
    predicates = _predicates(schema)
    resolved = resolve_student(student, schema)
    validate_room(room, schema)
    return _evaluate(predicates, resolved, room)


def filter_rooms(
    student: StudentProfile,
    rooms: Iterable[Room],
    schema: FeatureSchema | None = None,
) -> FilterResult:
    """Filtra los cuartos de un estudiante con las restricciones duras del esquema.

    Devuelve los cuartos permitidos (en el orden de entrada) y el conteo de
    exclusiones por id de restricción. El perfil se resuelve una sola vez.
    """
    schema = schema or _default_schema()
    predicates = _predicates(schema)
    resolved = resolve_student(student, schema)
    allowed: list[Room] = []
    exclusions = {constraint_id: 0 for constraint_id, _ in predicates}
    excluded = 0
    for room in rooms:
        validate_room(room, schema)
        result = _evaluate(predicates, resolved, room)
        if result.allowed:
            allowed.append(room)
            continue
        excluded += 1
        for constraint_id in result.violated:
            exclusions[constraint_id] += 1
    return FilterResult(allowed=tuple(allowed), exclusions=exclusions, excluded=excluded)


def implemented_constraints() -> tuple[str, ...]:
    """Ids de las restricciones con implementación de referencia."""
    return tuple(_VIOLATES)


__all__ = [
    "ConstraintResult",
    "FilterResult",
    "check_hard_constraints",
    "filter_rooms",
    "implemented_constraints",
]
