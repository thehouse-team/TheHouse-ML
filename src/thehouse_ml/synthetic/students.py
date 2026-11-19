"""Generador de estudiantes sintéticos (ML-03 T04, ML-03-R04).

Es puro: no lee archivos, no usa red ni el reloj. Los parámetros
(`students_params.json`), las anclas de renta por campus y los campus llegan
como argumentos. Cada campus usa su propio `random.Random("<seed>:<code>")`
y las extracciones siguen un orden fijo, así que la misma semilla produce los
mismos estudiantes byte por byte.

Todo lo que sale de aquí es **sintético** (constitución §8): ids sin datos
personales (`S0001`) y ningún campo de género (constitución §11); `livingWith`
es la elección del estudiante, no su género.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from thehouse_ml.features.campuses import Campus
from thehouse_ml.features.loader import FeatureSchema
from thehouse_ml.features.pairs import StudentProfile, resolve_student

PARAMS_PACKAGE = "thehouse_ml.synthetic"
PARAMS_RESOURCE = "students_params.json"
BASIS_PREFIXES = ("calibrado:", "SUPUESTO documentado")

# Campos opcionales de NV-11 con probabilidad de nulo (nombre del esquema → parámetro).
OPTIONAL_NULL_PARAMS: Mapping[str, str] = {
    "budget.min": "budget.min.null_probability",
    "maxDistanceKm": "maxDistanceKm.null_probability",
    "essentialServices": "essentialServices.null_probability",
    "pets": "pets.null_probability",
    "transport": "transport.null_probability",
    "propertyTypes": "propertyTypes.null_probability",
}
QUICK_SETUP_FIELDS = ("budget.max", "livingWith", "placeType")


class StudentParamsError(ValueError):
    """`students_params.json` no tiene la estructura esperada; lista todos los problemas."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("students_params.json inválido:\n- " + "\n- ".join(self.problems))


@dataclass(frozen=True)
class Parameter:
    """Un parámetro del generador con su base documental."""

    name: str
    value: Any
    basis: str

    @property
    def is_assumption(self) -> bool:
        return self.basis.startswith("SUPUESTO")


@dataclass(frozen=True)
class StudentParams:
    """Parámetros validados del generador."""

    metadata: Mapping[str, Any]
    parameters: Mapping[str, Parameter]

    def value(self, name: str) -> Any:
        return self.parameters[name].value

    @property
    def seed(self) -> int:
        return int(self.value("seed"))

    @property
    def n_students(self) -> int:
        return int(self.value("n_students"))

    @property
    def per_campus(self) -> int:
        return int(self.value("universities.per_campus"))

    @property
    def municipio_by_campus(self) -> Mapping[str, str]:
        return dict(self.value("universities.municipio_clave"))

    @property
    def median_factor(self) -> float:
        return float(self.value("budget.max.median_factor"))

    @property
    def calibration_tolerance(self) -> float:
        return float(self.value("budget.max.calibration_tolerance"))

    def with_seed(self, seed: int) -> StudentParams:
        """Copia con otra semilla (para pruebas de reproducibilidad)."""
        parameters = dict(self.parameters)
        parameters["seed"] = Parameter("seed", seed, parameters["seed"].basis)
        return StudentParams(metadata=self.metadata, parameters=parameters)


@dataclass(frozen=True)
class SyntheticStudent:
    """Estudiante sintético: id, campus verificado o no y perfil NV-11."""

    student_id: str
    seed_verified: bool
    profile: StudentProfile


REQUIRED_PARAMETERS = (
    "seed",
    "n_students",
    "student_id.prefix",
    "student_id.width",
    "universities.per_campus",
    "universities.municipio_clave",
    "budget.max.median_factor",
    "budget.max.log_sigma",
    "budget.max.round_to",
    "budget.max.clip",
    "budget.max.calibration_tolerance",
    "budget.min.null_probability",
    "budget.min.fraction",
    "transport.probabilities",
    "transport.null_probability",
    "maxDistanceKm.by_transport",
    "maxDistanceKm.null_probability",
    "essentialServices.probabilities",
    "essentialServices.null_probability",
    "livingWith.probabilities",
    "placeType.probabilities",
    "propertyTypes.inclusion_probabilities",
    "propertyTypes.null_probability",
    "pets.true_probability",
    "pets.null_probability",
)


def parse_params(data: object, schema: FeatureSchema) -> StudentParams:
    """Convierte el JSON decodificado en `StudentParams` y lo valida contra el esquema."""
    problems: list[str] = []
    if not isinstance(data, dict):
        raise StudentParamsError(["raíz: se esperaba un objeto"])
    metadata = data.get("metadata")
    raw = data.get("parameters")
    if not isinstance(metadata, dict) or metadata.get("synthetic") is not True:
        problems.append("metadata.synthetic: los estudiantes son sintéticos y deben declararlo (constitución §8)")
    if not isinstance(raw, dict):
        raise StudentParamsError([*problems, "parameters: se esperaba un objeto"])
    parameters: dict[str, Parameter] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict) or "value" not in entry:
            problems.append(f"{name}: se esperaba {{value, basis}}")
            continue
        basis = entry.get("basis")
        if not isinstance(basis, str) or not basis.startswith(BASIS_PREFIXES):
            problems.append(f"{name}.basis: debe empezar con «calibrado:» o «SUPUESTO documentado»")
            continue
        parameters[name] = Parameter(name=name, value=entry["value"], basis=basis)
    missing = [name for name in REQUIRED_PARAMETERS if name not in raw]
    if missing:
        problems.append(f"parameters: faltan {missing}")
    unknown = sorted(set(raw) - set(REQUIRED_PARAMETERS))
    if unknown:
        problems.append(f"parameters: desconocidos {unknown}")
    if not problems:
        problems.extend(_validate_values(parameters, schema))
    if problems:
        raise StudentParamsError(problems)
    return StudentParams(metadata=dict(metadata or {}), parameters=parameters)


def _is_probability(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0


def _check_distribution(name: str, value: object, catalog: Sequence[str]) -> list[str]:
    if not isinstance(value, dict) or set(value) != set(catalog):
        return [f"{name}: debe cubrir exactamente el catálogo {list(catalog)}"]
    if not all(_is_probability(item) for item in value.values()):
        return [f"{name}: probabilidades fuera de [0, 1]"]
    if not math.isclose(sum(float(item) for item in value.values()), 1.0, abs_tol=1e-9):
        return [f"{name}: las probabilidades deben sumar 1"]
    return []


def _validate_values(parameters: Mapping[str, Parameter], schema: FeatureSchema) -> list[str]:
    problems: list[str] = []

    def value(name: str) -> Any:
        return parameters[name].value

    campuses = schema.catalog("university_code")
    n_students, per_campus = value("n_students"), value("universities.per_campus")
    if not (isinstance(n_students, int) and isinstance(per_campus, int) and per_campus * len(campuses) == n_students):
        problems.append("n_students: debe ser universities.per_campus por el número de campus del esquema")
    if not isinstance(value("seed"), int):
        problems.append("seed: se esperaba un entero")
    if set(value("universities.municipio_clave")) != set(campuses):
        problems.append("universities.municipio_clave: debe cubrir exactamente el catálogo university_code")
    budget = schema.student_field("budget.max").range
    clip = value("budget.max.clip")
    if budget is None or clip != [int(budget.min), int(budget.max or 0)]:
        problems.append("budget.max.clip: debe ser el rango de presupuesto de NV-11")
    if not (isinstance(value("budget.max.median_factor"), int | float) and value("budget.max.median_factor") > 0):
        problems.append("budget.max.median_factor: debe ser positivo")
    if not (isinstance(value("budget.max.log_sigma"), int | float) and value("budget.max.log_sigma") >= 0):
        problems.append("budget.max.log_sigma: no puede ser negativo")
    low, high = value("budget.min.fraction")
    if not (0.0 <= low <= high <= 1.0):
        problems.append("budget.min.fraction: se esperaba 0 ≤ min ≤ max ≤ 1")
    for name in (*OPTIONAL_NULL_PARAMS.values(), "pets.true_probability", "budget.max.calibration_tolerance"):
        if not _is_probability(value(name)):
            problems.append(f"{name}: se esperaba una probabilidad en [0, 1]")
    problems.extend(_check_distribution("transport.probabilities", value("transport.probabilities"), _t(schema)))
    problems.extend(
        _check_distribution("livingWith.probabilities", value("livingWith.probabilities"), schema.catalog("livingWith"))
    )
    problems.extend(
        _check_distribution("placeType.probabilities", value("placeType.probabilities"), schema.catalog("placeType"))
    )
    for name, catalog in (
        ("essentialServices.probabilities", schema.catalog("essentialServices")),
        ("propertyTypes.inclusion_probabilities", schema.catalog("propertyTypes")),
    ):
        entry = value(name)
        if not isinstance(entry, dict) or set(entry) != set(catalog) or not all(map(_is_probability, entry.values())):
            problems.append(f"{name}: una probabilidad por elemento del catálogo")
    distance = schema.student_field("maxDistanceKm").range
    by_transport = value("maxDistanceKm.by_transport")
    if not isinstance(by_transport, dict) or set(by_transport) != set(_t(schema)):
        problems.append("maxDistanceKm.by_transport: una distribución por transporte del catálogo")
    else:
        for mode, entry in by_transport.items():
            values, weights = entry.get("values", []), entry.get("weights", [])
            if len(values) != len(weights) or not values:
                problems.append(f"maxDistanceKm.by_transport.{mode}: values y weights del mismo largo")
            elif distance is None or not all(distance.contains(float(item)) for item in values):
                problems.append(f"maxDistanceKm.by_transport.{mode}: valores fuera del rango de NV-11")
            elif not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
                problems.append(f"maxDistanceKm.by_transport.{mode}: los pesos deben sumar 1")
    return problems


def _t(schema: FeatureSchema) -> tuple[str, ...]:
    return schema.catalog("transport")


def load_params(schema: FeatureSchema, path: Path | None = None) -> StudentParams:
    """Carga `students_params.json` empaquetado (o `path`) y lo valida."""
    if path is None:
        text = resources.files(PARAMS_PACKAGE).joinpath(PARAMS_RESOURCE).read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise StudentParamsError([f"JSON inválido: {error}"]) from error
    return parse_params(data, schema)


def budget_targets(params: StudentParams, rent_p50: Mapping[str, float]) -> dict[str, float]:
    """Mediana objetivo de `budget.max` por campus: renta p50 regional por el factor."""
    return {code: rent_p50[code] * params.median_factor for code in params.municipio_by_campus}


def _weighted(rng: random.Random, distribution: Mapping[str, float], catalog: Sequence[str]) -> str:
    return rng.choices(list(catalog), weights=[float(distribution[item]) for item in catalog])[0]


def _round_to(value: float, step: int) -> int:
    return math.floor(value / step + 0.5) * step


def _student(
    rng: random.Random,
    params: StudentParams,
    schema: FeatureSchema,
    code: str,
    target: float,
) -> StudentProfile:
    value = params.value
    # Orden fijo de extracciones: cambiarlo cambia los datos (y la versión de los parámetros).
    low, high = value("budget.max.clip")
    step = int(value("budget.max.round_to"))
    raw_max = rng.lognormvariate(math.log(target), float(value("budget.max.log_sigma"))) if target > 0 else 0.0
    budget_max = min(max(_round_to(raw_max, step), int(low)), int(high))
    fraction_low, fraction_high = value("budget.min.fraction")
    fraction = rng.uniform(float(fraction_low), float(fraction_high))
    budget_min = math.floor(budget_max * fraction / step) * step
    transport = _weighted(rng, value("transport.probabilities"), _t(schema))
    distances = value("maxDistanceKm.by_transport")[transport]
    max_distance = float(rng.choices(distances["values"], weights=distances["weights"])[0])
    services_catalog = schema.catalog("essentialServices")
    service_probability = value("essentialServices.probabilities")
    services = frozenset(item for item in services_catalog if rng.random() < float(service_probability[item]))
    living_with = _weighted(rng, value("livingWith.probabilities"), schema.catalog("livingWith"))
    place_type = _weighted(rng, value("placeType.probabilities"), schema.catalog("placeType"))
    property_catalog = schema.catalog("propertyTypes")
    inclusion = value("propertyTypes.inclusion_probabilities")
    property_types = frozenset(item for item in property_catalog if rng.random() < float(inclusion[item]))
    fallback = rng.choice(list(property_catalog))
    if not property_types:
        property_types = frozenset({fallback})
    pets = rng.random() < float(value("pets.true_probability"))
    # Nulos de los opcionales: una extracción por campo, siempre (aunque no se use).
    nulls = {field: rng.random() < float(value(name)) for field, name in OPTIONAL_NULL_PARAMS.items()}
    return StudentProfile(
        university_code=code,
        budget_min=None if nulls["budget.min"] else budget_min,
        budget_max=budget_max,
        max_distance_km=None if nulls["maxDistanceKm"] else max_distance,
        essential_services=None if nulls["essentialServices"] else services,
        living_with=living_with,
        place_type=place_type,
        pets=None if nulls["pets"] else pets,
        transport=None if nulls["transport"] else transport,
        property_types=None if nulls["propertyTypes"] else property_types,
    )


def generate_students(
    params: StudentParams,
    rent_p50: Mapping[str, float],
    campuses: Mapping[str, Campus],
    schema: FeatureSchema,
) -> list[SyntheticStudent]:
    """Genera `n_students` estudiantes: `per_campus` por campus en el orden del catálogo.

    `rent_p50` es la renta p50 regional por código de campus (pesos de jul-2024).
    Cada perfil se valida con `resolve_student` (reglas para faltantes y catálogos
    de NV-11) antes de devolverlo.
    """
    codes = schema.catalog("university_code")
    missing = [code for code in codes if code not in rent_p50 or code not in campuses]
    if missing:
        raise StudentParamsError([f"campus sin renta p50 o sin coordenadas: {missing}"])
    targets = budget_targets(params, rent_p50)
    prefix, width = str(params.value("student_id.prefix")), int(params.value("student_id.width"))
    students: list[SyntheticStudent] = []
    for code in codes:
        rng = random.Random(f"{params.seed}:{code}")  # noqa: S311 - simulación reproducible, no criptografía
        for _ in range(params.per_campus):
            profile = _student(rng, params, schema, code, targets[code])
            resolve_student(profile, schema)
            student_id = f"{prefix}{len(students) + 1:0{width}d}"
            students.append(SyntheticStudent(student_id, campuses[code].seed_verified, profile))
    return students


def student_record(student: SyntheticStudent, schema: FeatureSchema) -> dict[str, Any]:
    """Registro plano con los nombres del esquema; los conjuntos van en el orden del catálogo."""
    profile = student.profile

    def ordered(values: frozenset[str] | None, catalog: str) -> list[str] | None:
        return None if values is None else [item for item in schema.catalog(catalog) if item in values]

    return {
        "student_id": student.student_id,
        "university_code": profile.university_code,
        "seed_verified": student.seed_verified,
        "budget.min": profile.budget_min,
        "budget.max": profile.budget_max,
        "maxDistanceKm": profile.max_distance_km,
        "essentialServices": ordered(profile.essential_services, "essentialServices"),
        "livingWith": profile.living_with,
        "placeType": profile.place_type,
        "pets": profile.pets,
        "transport": profile.transport,
        "propertyTypes": ordered(profile.property_types, "propertyTypes"),
        "synthetic": True,
    }


__all__ = [
    "OPTIONAL_NULL_PARAMS",
    "PARAMS_RESOURCE",
    "QUICK_SETUP_FIELDS",
    "Parameter",
    "StudentParams",
    "StudentParamsError",
    "SyntheticStudent",
    "budget_targets",
    "generate_students",
    "load_params",
    "parse_params",
    "student_record",
]
