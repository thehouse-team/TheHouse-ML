"""Carga y validación de `feature_schema.json` (ML-03 T01, ML-03-R01).

El esquema es el contrato con el backend (constitution §2.4 y §13): entradas
del estudiante (NV-11), del cuarto y features del par, con tipos, rangos,
catálogos, reglas para faltantes, restricciones duras y el orden fijo de las
features. Se valida solo con la biblioteca estándar.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, TypeGuard

SCHEMA_PACKAGE = "thehouse_ml.features"
SCHEMA_RESOURCE = "feature_schema.json"
FIELD_TYPES = frozenset({"integer", "number", "boolean", "string", "category", "category_set"})
NUMERIC_TYPES = frozenset({"integer", "number"})
CATALOG_TYPES = frozenset({"category", "category_set"})
MISSING_ACTIONS = frozenset({"default", "reject", "accept_all"})
BACKEND_STATUSES = frozenset({"present", "missing_in_property", "ml_only"})
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


class SchemaError(ValueError):
    """El esquema no cumple la estructura esperada; lista todos los problemas."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("feature_schema.json inválido:\n- " + "\n- ".join(problems))


@dataclass(frozen=True)
class ValueRange:
    """Rango cerrado [min, max]."""

    min: float
    max: float

    def contains(self, value: float) -> bool:
        return self.min <= value <= self.max


@dataclass(frozen=True)
class MissingRule:
    """Qué hacer cuando un campo llega nulo."""

    action: str
    value: Any
    description: str


@dataclass(frozen=True)
class FieldSpec:
    """Campo de entrada del estudiante o del cuarto."""

    name: str
    type: str
    nullable: bool
    description: str
    range: ValueRange | None = None
    catalog: str | None = None
    missing: MissingRule | None = None
    required_for_scoring: bool = False
    backend_status: str | None = None
    backend_field: str | None = None
    unit: str | None = None


@dataclass(frozen=True)
class PairFeature:
    """Feature del par estudiante-cuarto (definición; el cálculo es T02)."""

    name: str
    type: str
    range: ValueRange
    formula: str
    depends_on: tuple[str, ...]
    parameters: Mapping[str, Any]
    unit: str | None = None


@dataclass(frozen=True)
class HardConstraint:
    """Regla que excluye el cuarto sin importar el puntaje (constitution §11)."""

    id: str
    description: str
    excludes_when: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class Metadata:
    """Procedencia y versión del esquema."""

    schema_version: str
    spec: str
    nv11_source: Mapping[str, str]
    generated_by: str
    synthetic_training_data: bool


@dataclass(frozen=True)
class FeatureSchema:
    """Esquema de features tipado."""

    metadata: Metadata
    currency: str
    nv11_catalogs: Mapping[str, tuple[str, ...]]
    ml_catalogs: Mapping[str, tuple[str, ...]]
    unverified_university_codes: tuple[str, ...]
    student_constraints: tuple[str, ...]
    student: tuple[FieldSpec, ...]
    room: tuple[FieldSpec, ...]
    pair_features: tuple[PairFeature, ...]
    hard_constraints: tuple[HardConstraint, ...]
    feature_order: tuple[str, ...]

    @property
    def version(self) -> str:
        return self.metadata.schema_version

    def catalog(self, name: str) -> tuple[str, ...]:
        """Catálogo por nombre (primero NV-11, después los propios de ML)."""
        if name in self.nv11_catalogs:
            return self.nv11_catalogs[name]
        return self.ml_catalogs[name]

    def student_field(self, name: str) -> FieldSpec:
        return _by_name(self.student, name)

    def room_field(self, name: str) -> FieldSpec:
        return _by_name(self.room, name)


def _by_name(fields: tuple[FieldSpec, ...], name: str) -> FieldSpec:
    for spec in fields:
        if spec.name == name:
            return spec
    raise KeyError(name)


class _Reader:
    """Lee el JSON acumulando problemas en lugar de fallar en el primero."""

    def __init__(self) -> None:
        self.problems: list[str] = []

    def obj(self, value: object, where: str) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        self.problems.append(f"{where}: se esperaba un objeto")
        return {}

    def lst(self, value: object, where: str) -> list[Any]:
        if isinstance(value, list):
            return value
        self.problems.append(f"{where}: se esperaba una lista")
        return []

    def text(self, data: Mapping[str, Any], key: str, where: str) -> str:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
        self.problems.append(f"{where}.{key}: se esperaba un texto no vacío")
        return ""

    def optional_text(self, data: Mapping[str, Any], key: str, where: str) -> str | None:
        if key not in data:
            return None
        return self.text(data, key, where)

    def flag(self, data: Mapping[str, Any], key: str, where: str, default: bool | None = None) -> bool:
        value = data.get(key, default)
        if isinstance(value, bool):
            return value
        self.problems.append(f"{where}.{key}: se esperaba un booleano")
        return False

    def strings(self, value: object, where: str) -> tuple[str, ...]:
        items = self.lst(value, where)
        if not all(isinstance(item, str) and item for item in items):
            self.problems.append(f"{where}: todos los elementos deben ser textos no vacíos")
            return ()
        if len(set(items)) != len(items):
            self.problems.append(f"{where}: elementos duplicados")
        return tuple(items)

    def value_range(self, value: object, where: str) -> ValueRange | None:
        data = self.obj(value, where)
        low, high = data.get("min"), data.get("max")
        if not _is_number(low) or not _is_number(high):
            self.problems.append(f"{where}: min y max deben ser numéricos")
            return None
        if low > high:
            self.problems.append(f"{where}: min ({low}) mayor que max ({high})")
        return ValueRange(min=float(low), max=float(high))

    def catalogs(self, value: object, where: str) -> dict[str, tuple[str, ...]]:
        result: dict[str, tuple[str, ...]] = {}
        for name, items in self.obj(value, where).items():
            values = self.strings(items, f"{where}.{name}")
            if not values:
                self.problems.append(f"{where}.{name}: catálogo vacío")
            result[name] = values
        return result


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _read_field(reader: _Reader, raw: object, where: str, *, room: bool) -> FieldSpec:
    data = reader.obj(raw, where)
    name = reader.text(data, "name", where)
    where = f"{where}[{name}]" if name else where
    kind = reader.text(data, "type", where)
    if kind and kind not in FIELD_TYPES:
        reader.problems.append(f"{where}.type: tipo desconocido {kind!r}")
    value_range = reader.value_range(data["range"], f"{where}.range") if "range" in data else None
    if kind in NUMERIC_TYPES and value_range is None:
        reader.problems.append(f"{where}: un campo numérico necesita range")
    catalog = reader.optional_text(data, "catalog", where)
    if kind in CATALOG_TYPES and catalog is None:
        reader.problems.append(f"{where}: un campo de catálogo necesita catalog")
    missing = None
    if "missing" in data:
        rule = reader.obj(data["missing"], f"{where}.missing")
        action = reader.text(rule, "action", f"{where}.missing")
        if action and action not in MISSING_ACTIONS:
            reader.problems.append(f"{where}.missing.action: acción desconocida {action!r}")
        missing = MissingRule(
            action=action, value=rule.get("value"), description=reader.text(rule, "description", f"{where}.missing")
        )
    backend_status = None
    if room:
        backend_status = reader.text(data, "backend_status", where)
        if backend_status and backend_status not in BACKEND_STATUSES:
            reader.problems.append(f"{where}.backend_status: valor desconocido {backend_status!r}")
    return FieldSpec(
        name=name,
        type=kind,
        nullable=reader.flag(data, "nullable", where),
        description=reader.text(data, "description", where),
        range=value_range,
        catalog=catalog,
        missing=missing,
        required_for_scoring=reader.flag(data, "required_for_scoring", where, default=False),
        backend_status=backend_status,
        backend_field=reader.optional_text(data, "backend_field", where),
        unit=reader.optional_text(data, "unit", where),
    )


def _read_pair_feature(reader: _Reader, raw: object, where: str) -> PairFeature:
    data = reader.obj(raw, where)
    name = reader.text(data, "name", where)
    where = f"{where}[{name}]" if name else where
    kind = reader.text(data, "type", where)
    if kind and kind not in NUMERIC_TYPES:
        reader.problems.append(f"{where}.type: una feature del par debe ser numérica")
    value_range = reader.value_range(data.get("range"), f"{where}.range") or ValueRange(0.0, 0.0)
    return PairFeature(
        name=name,
        type=kind,
        range=value_range,
        formula=reader.text(data, "formula", where),
        depends_on=reader.strings(data.get("depends_on"), f"{where}.depends_on"),
        parameters=reader.obj(data.get("parameters", {}), f"{where}.parameters"),
        unit=reader.optional_text(data, "unit", where),
    )


def _read_constraint(reader: _Reader, raw: object, where: str) -> HardConstraint:
    data = reader.obj(raw, where)
    constraint_id = reader.text(data, "id", where)
    where = f"{where}[{constraint_id}]" if constraint_id else where
    return HardConstraint(
        id=constraint_id,
        description=reader.text(data, "description", where),
        excludes_when=reader.text(data, "excludes_when", where),
        fields=reader.strings(data.get("fields"), f"{where}.fields"),
    )


def parse_schema(data: object) -> FeatureSchema:
    """Convierte el JSON ya decodificado en `FeatureSchema` y lo valida."""
    reader = _Reader()
    root = reader.obj(data, "raíz")
    meta = reader.obj(root.get("metadata"), "metadata")
    source = reader.obj(meta.get("nv11_source"), "metadata.nv11_source")
    metadata = Metadata(
        schema_version=reader.text(meta, "schema_version", "metadata"),
        spec=reader.text(meta, "spec", "metadata"),
        nv11_source={key: str(value) for key, value in source.items()},
        generated_by=reader.text(meta, "generated_by", "metadata"),
        synthetic_training_data=reader.flag(meta, "synthetic_training_data", "metadata"),
    )
    reader.text(source, "commit", "metadata.nv11_source")
    catalogs = reader.obj(root.get("catalogs"), "catalogs")
    student = reader.obj(root.get("student"), "student")
    room = reader.obj(root.get("room"), "room")
    universities = reader.obj(root.get("universities"), "universities")
    schema = FeatureSchema(
        metadata=metadata,
        currency=reader.text(root, "currency", "raíz"),
        nv11_catalogs=reader.catalogs(catalogs.get("nv11"), "catalogs.nv11"),
        ml_catalogs=reader.catalogs(catalogs.get("ml"), "catalogs.ml"),
        unverified_university_codes=reader.strings(
            universities.get("unverified_codes"), "universities.unverified_codes"
        ),
        student_constraints=reader.strings(student.get("constraints", []), "student.constraints"),
        student=tuple(
            _read_field(reader, raw, f"student.fields[{i}]", room=False)
            for i, raw in enumerate(reader.lst(student.get("fields"), "student.fields"))
        ),
        room=tuple(
            _read_field(reader, raw, f"room.fields[{i}]", room=True)
            for i, raw in enumerate(reader.lst(room.get("fields"), "room.fields"))
        ),
        pair_features=tuple(
            _read_pair_feature(reader, raw, f"pair_features[{i}]")
            for i, raw in enumerate(reader.lst(root.get("pair_features"), "pair_features"))
        ),
        hard_constraints=tuple(
            _read_constraint(reader, raw, f"hard_constraints[{i}]")
            for i, raw in enumerate(reader.lst(root.get("hard_constraints"), "hard_constraints"))
        ),
        feature_order=reader.strings(root.get("feature_order"), "feature_order"),
    )
    problems = reader.problems + validate(schema)
    if problems:
        raise SchemaError(problems)
    return schema


def validate(schema: FeatureSchema) -> list[str]:
    """Reglas semánticas del esquema; devuelve la lista de problemas (vacía si es válido)."""
    problems: list[str] = []
    if not _SEMVER.match(schema.metadata.schema_version):
        problems.append(f"metadata.schema_version: {schema.metadata.schema_version!r} no es semver X.Y.Z")
    if not schema.metadata.synthetic_training_data:
        problems.append("metadata.synthetic_training_data: los datos de entrenamiento de ML-03 son sintéticos")
    overlap = set(schema.nv11_catalogs) & set(schema.ml_catalogs)
    if overlap:
        problems.append(f"catalogs: nombres repetidos entre nv11 y ml: {sorted(overlap)}")
    known_catalogs = set(schema.nv11_catalogs) | set(schema.ml_catalogs)
    universities = set(schema.ml_catalogs.get("university_code", ()))
    for code in schema.unverified_university_codes:
        if code not in universities:
            problems.append(f"universities.unverified_codes: {code!r} no está en catalogs.ml.university_code")
    for group, fields in (("student", schema.student), ("room", schema.room)):
        names = [spec.name for spec in fields]
        for name in sorted({name for name in names if names.count(name) > 1}):
            problems.append(f"{group}: campo duplicado {name!r}")
        for spec in fields:
            problems.extend(_validate_field(schema, group, spec, known_catalogs))
    problems.extend(_validate_pairs(schema))
    return problems


def _validate_field(schema: FeatureSchema, group: str, spec: FieldSpec, known_catalogs: set[str]) -> list[str]:
    where = f"{group}.{spec.name}"
    problems: list[str] = []
    if spec.type in CATALOG_TYPES and spec.catalog is not None and spec.catalog not in known_catalogs:
        problems.append(f"{where}: catálogo desconocido {spec.catalog!r}")
        return problems
    if spec.nullable and spec.missing is None:
        problems.append(f"{where}: campo nullable sin regla para faltantes")
    if not spec.nullable and spec.missing is not None:
        problems.append(f"{where}: regla para faltantes en un campo no nullable")
    if spec.missing is None:
        return problems
    rule = spec.missing
    if spec.required_for_scoring and rule.action != "reject":
        problems.append(f"{where}: un campo obligatorio para el puntaje solo admite missing.action = 'reject'")
    if rule.action == "reject" and not spec.required_for_scoring:
        problems.append(f"{where}: 'reject' solo aplica a campos obligatorios para el puntaje")
    if rule.action == "accept_all" and spec.type != "category_set":
        problems.append(f"{where}: 'accept_all' solo aplica a category_set")
    if rule.action == "default" and not _default_fits(schema, spec, rule.value):
        problems.append(f"{where}: valor por defecto {rule.value!r} fuera de tipo, rango o catálogo")
    return problems


def _default_fits(schema: FeatureSchema, spec: FieldSpec, value: object) -> bool:
    if spec.type == "boolean":
        return isinstance(value, bool)
    if spec.type in NUMERIC_TYPES:
        if not _is_number(value) or spec.range is None:
            return False
        if spec.type == "integer" and not isinstance(value, int):
            return False
        return spec.range.contains(float(value))
    if spec.type == "category" and spec.catalog is not None:
        return value in schema.catalog(spec.catalog)
    if spec.type == "category_set" and spec.catalog is not None:
        catalog = schema.catalog(spec.catalog)
        return isinstance(value, list) and all(item in catalog for item in value)
    return isinstance(value, str)


def _validate_pairs(schema: FeatureSchema) -> list[str]:
    problems: list[str] = []
    names = [feature.name for feature in schema.pair_features]
    for name in sorted({name for name in names if names.count(name) > 1}):
        problems.append(f"pair_features: feature duplicada {name!r}")
    references = {f"student.{spec.name}" for spec in schema.student} | {f"room.{spec.name}" for spec in schema.room}
    for feature in schema.pair_features:
        for dependency in feature.depends_on:
            if dependency not in references and dependency not in names:
                problems.append(f"pair_features.{feature.name}: dependencia desconocida {dependency!r}")
    factors = schema_transport_factors(schema)
    if factors is not None and set(factors) != set(schema.nv11_catalogs.get("transport", ())):
        problems.append("pair_features.effective_distance_km: transport_factors no cubre exactamente el catálogo")
    ids = [constraint.id for constraint in schema.hard_constraints]
    if not ids:
        problems.append("hard_constraints: se esperaba al menos una restricción")
    for constraint_id in sorted({cid for cid in ids if ids.count(cid) > 1}):
        problems.append(f"hard_constraints: id duplicado {constraint_id!r}")
    for constraint in schema.hard_constraints:
        for field in constraint.fields:
            if field not in references:
                problems.append(f"hard_constraints.{constraint.id}: campo desconocido {field!r}")
    if not schema.feature_order:
        problems.append("feature_order: lista vacía")
    if set(schema.feature_order) != set(names):
        problems.append("feature_order: debe contener exactamente las features de pair_features")
    return problems


def schema_transport_factors(schema: FeatureSchema) -> Mapping[str, float] | None:
    """Factores de distancia efectiva por transporte, si el esquema los define."""
    for feature in schema.pair_features:
        raw = feature.parameters.get("transport_factors")
        if isinstance(raw, dict):
            return {str(key): float(value) for key, value in raw.items()}
    return None


def load_schema(path: Path | None = None) -> FeatureSchema:
    """Carga el esquema empaquetado (o `path`) y lo valida."""
    if path is None:
        text = resources.files(SCHEMA_PACKAGE).joinpath(SCHEMA_RESOURCE).read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise SchemaError([f"JSON inválido: {error}"]) from error
    return parse_schema(data)


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
    "load_schema",
    "parse_schema",
    "schema_transport_factors",
    "validate",
]
