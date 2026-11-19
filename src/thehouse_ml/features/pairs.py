"""Features del par estudiante-cuarto: implementación de referencia (ML-03 T02, ML-03-R02).

Una sola implementación (spec ML-03 §7 regla 2) que también produce los casos
dorados del backend. Es pura: no lee archivos, no usa red ni aleatoriedad; el
esquema y las coordenadas de los campus llegan como argumentos.

Pasos:

1. `resolve_student` aplica las reglas para faltantes de `feature_schema.json`
   (`default`, `reject`, `accept_all`) y valida tipos, rangos y catálogos.
2. `compute_pair_features` calcula las features en el orden de
   `feature_order` del esquema.

La distancia **no** es restricción dura: no se recorta ni se excluye nada por
pasar de 30 km o de `maxDistanceKm`; el valor real se devuelve y ML-04 decide
el puntaje. Las restricciones duras (HC01 a HC03) son T03.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from thehouse_ml.features.campuses import Campus
from thehouse_ml.features.loader import FeatureSchema, FieldSpec, schema_transport_factors
from thehouse_ml.universities import haversine_m

# Nombre del esquema (NV-11 / cuarto) → atributo de la dataclass.
STUDENT_ATTRIBUTES: Mapping[str, str] = {
    "budget.min": "budget_min",
    "budget.max": "budget_max",
    "maxDistanceKm": "max_distance_km",
    "essentialServices": "essential_services",
    "livingWith": "living_with",
    "placeType": "place_type",
    "pets": "pets",
    "transport": "transport",
    "propertyTypes": "property_types",
    "university_code": "university_code",
}
ROOM_ATTRIBUTES: Mapping[str, str] = {
    "price_mxn": "price_mxn",
    "lat": "lat",
    "lon": "lon",
    "services": "services",
    "placeType": "place_type",
    "propertyType": "property_type",
    "pets_allowed": "pets_allowed",
    "livingWith": "living_with",
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",
    "area_m2": "area_m2",
    "university_code": "university_code",
    "synthetic": "synthetic",
}

PairFeatures = dict[str, float]


class PairInputError(ValueError):
    """El estudiante o el cuarto traen un valor fuera de tipo, rango o catálogo."""


class IncompleteProfileError(PairInputError):
    """Falta un dato del quick setup (`missing.action = "reject"`): no se calcula el par."""

    def __init__(self, missing: Iterable[str]) -> None:
        self.missing = tuple(missing)
        super().__init__(
            "Perfil incompleto: faltan datos del quick setup "
            f"({', '.join(self.missing)}); no se calculan las features del par."
        )


@dataclass(frozen=True)
class StudentProfile:
    """Perfil de búsqueda tal como llega de NV-11 (todo nulable salvo la universidad)."""

    university_code: str
    budget_min: int | None = None
    budget_max: int | None = None
    max_distance_km: float | None = None
    essential_services: frozenset[str] | None = None
    living_with: str | None = None
    place_type: str | None = None
    pets: bool | None = None
    transport: str | None = None
    property_types: frozenset[str] | None = None


@dataclass(frozen=True)
class ResolvedStudent:
    """Perfil tras aplicar las reglas para faltantes: sin nulos.

    `property_types` vacío significa «acepta todos» (`accept_all`).
    """

    university_code: str
    budget_min: int
    budget_max: int
    max_distance_km: float
    essential_services: frozenset[str]
    living_with: str
    place_type: str
    pets: bool
    transport: str
    property_types: frozenset[str]


@dataclass(frozen=True)
class Room:
    """Cuarto con los campos de `room` del esquema (ninguno nulable)."""

    price_mxn: int
    lat: float
    lon: float
    services: frozenset[str]
    place_type: str
    property_type: str
    pets_allowed: bool
    living_with: str
    bedrooms: int
    bathrooms: int
    area_m2: float
    university_code: str
    synthetic: bool


def _default_schema() -> FeatureSchema:
    from thehouse_ml.features import default_schema  # import local: evita el ciclo con __init__

    return default_schema()


def _check_value(schema: FeatureSchema, spec: FieldSpec, value: object, where: str) -> None:
    """Valida tipo, rango y catálogo de un valor ya sin nulos."""
    kind = spec.type
    if kind == "boolean":
        if not isinstance(value, bool):
            raise PairInputError(f"{where}: se esperaba un booleano, llegó {value!r}")
        return
    if kind in {"integer", "number"}:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise PairInputError(f"{where}: se esperaba un número, llegó {value!r}")
        if kind == "integer" and not isinstance(value, int):
            raise PairInputError(f"{where}: se esperaba un entero, llegó {value!r}")
        if not math.isfinite(value):
            raise PairInputError(f"{where}: valor no finito {value!r}")
        if spec.range_is_assumption:
            # Rango SUPUESTO del catálogo sintético: en la entrada solo tipo y no negativo (M-ML-023).
            if value < 0:
                raise PairInputError(f"{where}: {value!r} es negativo")
            return
        if spec.range is not None and not spec.range.contains(float(value)):
            raise PairInputError(f"{where}: {value!r} fuera del rango {spec.range.render()}")
        return
    if spec.catalog is None:  # pragma: no cover - el esquema validado no lo permite
        raise PairInputError(f"{where}: tipo {kind!r} sin catálogo")
    catalog = schema.catalog(spec.catalog)
    if kind == "category":
        if value not in catalog:
            raise PairInputError(f"{where}: {value!r} no está en el catálogo {spec.catalog!r}")
        return
    if not isinstance(value, frozenset) or not all(isinstance(item, str) for item in value):
        raise PairInputError(f"{where}: se esperaba un conjunto de textos, llegó {value!r}")
    unknown = sorted(value - set(catalog))
    if unknown:
        raise PairInputError(f"{where}: {unknown} no están en el catálogo {spec.catalog!r}")


def resolve_student(student: StudentProfile, schema: FeatureSchema | None = None) -> ResolvedStudent:
    """Aplica las reglas para faltantes del esquema y valida el perfil.

    Lanza `IncompleteProfileError` con todos los campos del quick setup que
    faltan (`budget.max`, `livingWith`, `placeType`) y `PairInputError` si un
    valor está fuera de tipo, rango o catálogo.
    """
    schema = schema or _default_schema()
    values: dict[str, Any] = {}
    rejected: list[str] = []
    for spec in schema.student:
        attribute = STUDENT_ATTRIBUTES[spec.name]
        value = getattr(student, attribute)
        if value is None:
            rule = spec.missing
            if rule is None:
                raise PairInputError(f"student.{spec.name}: obligatorio y no nulable")
            if rule.action == "reject":
                rejected.append(spec.name)
                continue
            if rule.action == "accept_all":
                value = frozenset()
            else:  # "default"
                value = frozenset(rule.value) if spec.type == "category_set" else rule.value
        values[attribute] = value
    if rejected:
        raise IncompleteProfileError(rejected)
    for spec in schema.student:
        _check_value(schema, spec, values[STUDENT_ATTRIBUTES[spec.name]], f"student.{spec.name}")
    if values["budget_min"] > values["budget_max"]:
        raise PairInputError(
            f"student: budget.min ({values['budget_min']}) mayor que budget.max ({values['budget_max']})"
        )
    values["max_distance_km"] = float(values["max_distance_km"])
    return ResolvedStudent(**values)


def validate_room(room: Room, schema: FeatureSchema | None = None) -> None:
    """Valida el cuarto contra `room` del esquema (tipos, rangos y catálogos).

    Los rangos SUPUESTOS (`range_basis`: `bedrooms`, `bathrooms`, `area_m2`) no
    rechazan el cuarto: solo se exige el tipo y un valor no negativo (M-ML-023).
    """
    schema = schema or _default_schema()
    for spec in schema.room:
        value = getattr(room, ROOM_ATTRIBUTES[spec.name])
        if value is None:
            raise PairInputError(f"room.{spec.name}: no es nulable")
        _check_value(schema, spec, value, f"room.{spec.name}")


def _distance_km(student: ResolvedStudent, room: Room, campuses: Mapping[str, Campus]) -> float:
    campus = campuses.get(student.university_code)
    if campus is None:
        raise PairInputError(f"student.university_code: {student.university_code!r} sin coordenadas de campus")
    return haversine_m(campus.lat, campus.lon, room.lat, room.lon) / 1000.0


def _price_to_budget(student: ResolvedStudent, room: Room) -> float:
    # Esquema 1.1.0 (M-ML-023): una sola fórmula, sin casos especiales ni división entre cero.
    return room.price_mxn / max(student.budget_max, 1)


def _services_coverage(student: ResolvedStudent, room: Room) -> float:
    if not student.essential_services:
        return 1.0
    return len(student.essential_services & room.services) / len(student.essential_services)


def compute_pair_features(
    student: StudentProfile,
    room: Room,
    campuses: Mapping[str, Campus],
    schema: FeatureSchema | None = None,
) -> PairFeatures:
    """Features del par en el orden exacto de `schema.feature_order`.

    Primero aplica las reglas para faltantes (`resolve_student`) y valida el
    cuarto; después calcula cada feature con la fórmula del esquema. Los
    valores de las features `integer` son 0 o 1.
    """
    schema = schema or _default_schema()
    resolved = resolve_student(student, schema)
    validate_room(room, schema)
    factors = schema_transport_factors(schema)
    if factors is None:
        raise PairInputError("feature_schema.json no define transport_factors")
    distance = _distance_km(resolved, room, campuses)
    formulas: dict[str, Callable[[], float]] = {
        "distance_km": lambda: distance,
        "effective_distance_km": lambda: distance * factors[resolved.transport],
        "price_to_budget": lambda: _price_to_budget(resolved, room),
        "services_coverage": lambda: _services_coverage(resolved, room),
        "place_type_match": lambda: int(resolved.place_type == room.place_type),
        "property_type_match": lambda: int(
            not resolved.property_types or room.property_type in resolved.property_types
        ),
        "pets_compatible": lambda: int(not (resolved.pets and not room.pets_allowed)),
    }
    unknown = [name for name in schema.feature_order if name not in formulas]
    if unknown:
        raise PairInputError(f"feature_order: features sin implementación de referencia {unknown}")
    return {name: formulas[name]() for name in schema.feature_order}


def feature_vector(features: PairFeatures, schema: FeatureSchema | None = None) -> tuple[float, ...]:
    """Vector de entrada del modelo en el orden del esquema."""
    schema = schema or _default_schema()
    return tuple(float(features[name]) for name in schema.feature_order)


def _optional_set(value: object, where: str) -> frozenset[str] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple | set | frozenset):
        raise PairInputError(f"{where}: se esperaba una lista, llegó {value!r}")
    return frozenset(value)


def student_from_record(record: Mapping[str, Any]) -> StudentProfile:
    """Construye el perfil desde un registro con los nombres del esquema (clave ausente = nulo)."""
    unknown = sorted(set(record) - set(STUDENT_ATTRIBUTES))
    if unknown:
        raise PairInputError(f"student: campos desconocidos {unknown}")
    code = record.get("university_code")
    if not isinstance(code, str) or not code:
        raise PairInputError("student.university_code: obligatorio y no nulable")
    return StudentProfile(
        university_code=code,
        budget_min=record.get("budget.min"),
        budget_max=record.get("budget.max"),
        max_distance_km=record.get("maxDistanceKm"),
        essential_services=_optional_set(record.get("essentialServices"), "student.essentialServices"),
        living_with=record.get("livingWith"),
        place_type=record.get("placeType"),
        pets=record.get("pets"),
        transport=record.get("transport"),
        property_types=_optional_set(record.get("propertyTypes"), "student.propertyTypes"),
    )


def room_from_record(record: Mapping[str, Any]) -> Room:
    """Construye el cuarto desde un registro con los nombres del esquema (todos obligatorios)."""
    keys = set(record)
    unknown = sorted(keys - set(ROOM_ATTRIBUTES))
    missing = sorted(set(ROOM_ATTRIBUTES) - keys)
    if unknown or missing:
        raise PairInputError(f"room: campos desconocidos {unknown}, faltantes {missing}")
    services = _optional_set(record["services"], "room.services")
    values = {attribute: record[name] for name, attribute in ROOM_ATTRIBUTES.items()}
    values["services"] = services
    return Room(**values)


__all__ = [
    "ROOM_ATTRIBUTES",
    "STUDENT_ATTRIBUTES",
    "IncompleteProfileError",
    "PairFeatures",
    "PairInputError",
    "ResolvedStudent",
    "Room",
    "StudentProfile",
    "compute_pair_features",
    "feature_vector",
    "resolve_student",
    "room_from_record",
    "student_from_record",
    "validate_room",
]
