"""Conjunto de pares estudiante-cuarto candidatos (ML-03 T06, ML-03-R06).

Es puro: no lee archivos, no usa red ni el reloj. Recibe los estudiantes (T04),
los cuartos (T05), los campus y los parámetros (`pairs_params.json`).

Reglas (M-ML-027, aclaración de M-ML-021):

1. **Candidatos:** los cuartos del campus del estudiante que pasan las
   restricciones duras (`filter_rooms`, M-ML-024), hasta
   `candidates.max_per_student`; si hay más, muestra uniforme sin reemplazo con
   `random.Random("pairs:candidates:<seed>:<student_id>")`. Los cuartos que no
   pasan las restricciones no entran. La distancia **no** es restricción dura
   (M-ML-023): los cuartos más lejos que `maxDistanceKm` sí entran.
2. **Features:** las del esquema con `compute_pair_features` (T02, una sola
   implementación), en el orden de `feature_order`.
3. **Partición por estudiante**, estratificada por campus, con
   `random.Random("pairs:split:<seed>:<code>")`: cada estudiante queda en una
   sola partición (sin fuga); los cuartos son el catálogo y pueden repetirse
   entre particiones. La partición no depende de los candidatos: un estudiante
   sin candidatos tiene partición, pero no genera pares.
4. **Sin etiquetas:** ni relevancia ni puntaje (ML-04/ML-05).
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
from thehouse_ml.features.constraints import filter_rooms
from thehouse_ml.features.loader import FeatureSchema
from thehouse_ml.features.pairs import PairFeatures, compute_pair_features, resolve_student
from thehouse_ml.synthetic.rooms import SyntheticRoom
from thehouse_ml.synthetic.students import BASIS_PREFIXES, PARAMS_PACKAGE, Parameter, SyntheticStudent

PAIRS_PARAMS_RESOURCE = "pairs_params.json"
SPLITS = ("train", "validation", "test")
STRATIFY_BY = "university_code"
REQUIRED_PARAMETERS = (
    "seed",
    "candidates.max_per_student",
    "split.fractions",
    "split.stratify_by",
    "coverage.min_share",
)


class PairParamsError(ValueError):
    """`pairs_params.json` no tiene la estructura esperada; lista todos los problemas."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("pairs_params.json inválido:\n- " + "\n- ".join(self.problems))


class PairSetError(ValueError):
    """Los estudiantes o los cuartos no permiten construir el conjunto de pares."""


@dataclass(frozen=True)
class PairParams:
    """Parámetros validados del conjunto de pares."""

    metadata: Mapping[str, Any]
    parameters: Mapping[str, Parameter]

    def value(self, name: str) -> Any:
        return self.parameters[name].value

    @property
    def seed(self) -> int:
        return int(self.value("seed"))

    @property
    def max_candidates(self) -> int:
        return int(self.value("candidates.max_per_student"))

    @property
    def fractions(self) -> dict[str, float]:
        fractions = self.value("split.fractions")
        return {split: float(fractions[split]) for split in SPLITS}

    @property
    def min_share(self) -> float:
        return float(self.value("coverage.min_share"))

    def with_seed(self, seed: int) -> PairParams:
        """Copia con otra semilla (para pruebas de reproducibilidad)."""
        parameters = dict(self.parameters)
        parameters["seed"] = Parameter("seed", seed, parameters["seed"].basis)
        return PairParams(metadata=self.metadata, parameters=parameters)

    def with_max_candidates(self, maximum: int) -> PairParams:
        """Copia con otro máximo de candidatos (para pruebas del muestreo)."""
        name = "candidates.max_per_student"
        parameters = dict(self.parameters)
        parameters[name] = Parameter(name, maximum, parameters[name].basis)
        return PairParams(metadata=self.metadata, parameters=parameters)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_values(parameters: Mapping[str, Parameter]) -> list[str]:
    problems: list[str] = []
    seed = parameters["seed"].value
    if not _is_int(seed) or seed < 0:
        problems.append("seed: entero no negativo")
    maximum = parameters["candidates.max_per_student"].value
    if not _is_int(maximum) or maximum < 1:
        problems.append("candidates.max_per_student: entero ≥ 1")
    fractions = parameters["split.fractions"].value
    if not isinstance(fractions, dict) or set(fractions) != set(SPLITS):
        problems.append(f"split.fractions: se esperaban exactamente {list(SPLITS)}")
    elif not all(_is_number(value) and 0.0 < float(value) < 1.0 for value in fractions.values()):
        problems.append("split.fractions: cada fracción en (0, 1)")
    elif abs(sum(float(value) for value in fractions.values()) - 1.0) > 1e-9:
        problems.append("split.fractions: las fracciones deben sumar 1")
    if parameters["split.stratify_by"].value != STRATIFY_BY:
        problems.append(f"split.stratify_by: solo se admite {STRATIFY_BY!r}")
    share = parameters["coverage.min_share"].value
    if not _is_number(share) or not 0.0 < float(share) <= 1.0:
        problems.append("coverage.min_share: número en (0, 1]")
    return problems


def parse_params(data: object) -> PairParams:
    """Convierte el JSON decodificado en `PairParams` y lo valida."""
    if not isinstance(data, dict):
        raise PairParamsError(["raíz: se esperaba un objeto"])
    problems: list[str] = []
    metadata = data.get("metadata")
    raw = data.get("parameters")
    if not isinstance(metadata, dict) or metadata.get("synthetic") is not True:
        problems.append("metadata.synthetic: los pares son sintéticos y deben declararlo (constitución §8)")
    if not isinstance(raw, dict):
        raise PairParamsError([*problems, "parameters: se esperaba un objeto"])
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
        problems.extend(_validate_values(parameters))
    if problems:
        raise PairParamsError(problems)
    return PairParams(metadata=dict(metadata or {}), parameters=parameters)


def load_params(path: Path | None = None) -> PairParams:
    """Carga `pairs_params.json` empaquetado (o `path`) y lo valida."""
    if path is None:
        text = resources.files(PARAMS_PACKAGE).joinpath(PAIRS_PARAMS_RESOURCE).read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise PairParamsError([f"JSON inválido: {error}"]) from error
    return parse_params(data)


# ---------------------------------------------------------------------------
# Partición por estudiante
# ---------------------------------------------------------------------------


def split_sizes(n: int, fractions: Mapping[str, float]) -> dict[str, int]:
    """Cuotas de una estrato con el método del mayor residuo, al menos 1 por partición.

    Los empates del residuo se resuelven en el orden de `SPLITS`. Si una
    partición queda en 0, toma un estudiante de la más grande. Con menos de
    3 estudiantes no se pueden llenar las tres particiones.
    """
    if n < len(SPLITS):
        raise PairSetError(f"se necesitan al menos {len(SPLITS)} estudiantes por campus para estratificar; hay {n}")
    quotas = {split: fractions[split] * n for split in SPLITS}
    sizes = {split: math.floor(quotas[split] + 1e-9) for split in SPLITS}
    remainder = n - sum(sizes.values())
    by_residue = sorted(SPLITS, key=lambda split: (-(quotas[split] - sizes[split]), SPLITS.index(split)))
    for split in by_residue[:remainder]:
        sizes[split] += 1
    for split in SPLITS:
        if sizes[split] == 0:
            largest = max(SPLITS, key=lambda name: (sizes[name], -SPLITS.index(name)))
            sizes[largest] -= 1
            sizes[split] = 1
    return sizes


def assign_splits(students: Sequence[SyntheticStudent], params: PairParams, schema: FeatureSchema) -> dict[str, str]:
    """Partición de cada estudiante (`student_id` → `train`/`validation`/`test`), estratificada por campus."""
    result: dict[str, str] = {}
    for code in schema.catalog("university_code"):
        ids = sorted(student.student_id for student in students if student.profile.university_code == code)
        if not ids:
            continue
        rng = random.Random(f"pairs:split:{params.seed}:{code}")  # noqa: S311 - simulación reproducible
        rng.shuffle(ids)
        sizes = split_sizes(len(ids), params.fractions)
        start = 0
        for split in SPLITS:
            for student_id in ids[start : start + sizes[split]]:
                result[student_id] = split
            start += sizes[split]
    return result


# ---------------------------------------------------------------------------
# Candidatos y features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentCandidates:
    """Candidatos de un estudiante y conteos para el reporte.

    `compatible` es el número de cuartos de su campus que pasan las
    restricciones duras (antes del máximo); `room_ids` los candidatos elegidos,
    en orden de `room_id`. `exclusions` cuenta por id de restricción (un cuarto
    que viola varias cuenta en cada una) y `excluded` los cuartos excluidos.
    `beyond_max_distance` cuenta los candidatos más lejos que su
    `maxDistanceKm` resuelto.
    """

    student_id: str
    university_code: str
    seed_verified: bool
    split: str
    campus_rooms: int
    compatible: int
    room_ids: tuple[str, ...]
    exclusions: Mapping[str, int]
    excluded: int
    beyond_max_distance: int

    @property
    def sampled(self) -> bool:
        return self.compatible > len(self.room_ids)


@dataclass(frozen=True)
class Pair:
    """Un par candidato con sus features en el orden del esquema (sin etiqueta)."""

    student_id: str
    room_id: str
    university_code: str
    seed_verified: bool
    split: str
    features: PairFeatures


@dataclass(frozen=True)
class PairSet:
    """Conjunto de pares y el resumen por estudiante."""

    students: tuple[StudentCandidates, ...]
    pairs: tuple[Pair, ...]


def choose_candidates(student_id: str, allowed: Sequence[str], params: PairParams) -> tuple[str, ...]:
    """Todos los permitidos si caben; si no, muestra uniforme con semilla fija por estudiante, en orden."""
    if len(allowed) <= params.max_candidates:
        return tuple(allowed)
    rng = random.Random(f"pairs:candidates:{params.seed}:{student_id}")  # noqa: S311 - simulación reproducible
    chosen = sorted(rng.sample(range(len(allowed)), params.max_candidates))
    return tuple(allowed[index] for index in chosen)


def _rooms_by_campus(rooms: Sequence[SyntheticRoom], schema: FeatureSchema) -> dict[str, list[SyntheticRoom]]:
    ids = [item.room_id for item in rooms]
    if len(set(ids)) != len(ids):
        raise PairSetError("room_id duplicado en el catálogo de cuartos")
    result: dict[str, list[SyntheticRoom]] = {code: [] for code in schema.catalog("university_code")}
    for item in sorted(rooms, key=lambda room: room.room_id):
        code = item.room.university_code
        if code not in result:
            raise PairSetError(f"{item.room_id}: campus {code!r} fuera del catálogo")
        result[code].append(item)
    return result


def build_pairs(
    students: Sequence[SyntheticStudent],
    rooms: Sequence[SyntheticRoom],
    campuses: Mapping[str, Campus],
    params: PairParams,
    schema: FeatureSchema,
) -> PairSet:
    """Construye los pares candidatos con sus features y la partición por estudiante.

    Los estudiantes y los pares salen en orden de `student_id` y, dentro de
    cada estudiante, de `room_id`.
    """
    ids = [student.student_id for student in students]
    if len(set(ids)) != len(ids):
        raise PairSetError("student_id duplicado en los estudiantes")
    splits = assign_splits(students, params, schema)
    by_campus = _rooms_by_campus(rooms, schema)
    summaries: list[StudentCandidates] = []
    pairs: list[Pair] = []
    for student in sorted(students, key=lambda item: item.student_id):
        profile = student.profile
        if profile.university_code not in by_campus:
            raise PairSetError(f"{student.student_id}: campus {profile.university_code!r} fuera del catálogo")
        campus_rooms = by_campus[profile.university_code]
        room_ids = {id(item.room): item.room_id for item in campus_rooms}
        by_id = {item.room_id: item for item in campus_rooms}
        result = filter_rooms(profile, [item.room for item in campus_rooms], schema)
        allowed = [room_ids[id(room)] for room in result.allowed]
        chosen = choose_candidates(student.student_id, allowed, params)
        max_distance = resolve_student(profile, schema).max_distance_km
        split = splits[student.student_id]
        beyond = 0
        for room_id in chosen:
            features = compute_pair_features(profile, by_id[room_id].room, campuses, schema)
            beyond += int(features["distance_km"] > max_distance)
            pairs.append(
                Pair(
                    student_id=student.student_id,
                    room_id=room_id,
                    university_code=profile.university_code,
                    seed_verified=student.seed_verified,
                    split=split,
                    features=features,
                )
            )
        summaries.append(
            StudentCandidates(
                student_id=student.student_id,
                university_code=profile.university_code,
                seed_verified=student.seed_verified,
                split=split,
                campus_rooms=len(campus_rooms),
                compatible=len(allowed),
                room_ids=chosen,
                exclusions=dict(result.exclusions),
                excluded=result.excluded,
                beyond_max_distance=beyond,
            )
        )
    return PairSet(students=tuple(summaries), pairs=tuple(pairs))


# ---------------------------------------------------------------------------
# Umbral de cobertura (M-ML-021): ≥ 90 % con al menos un candidato
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Coverage:
    """Estudiantes con al menos un cuarto que pasa las restricciones duras en un grupo."""

    label: str
    students: int
    with_candidates: int
    min_share: float

    @property
    def share(self) -> float:
        return self.with_candidates / self.students if self.students else 0.0

    @property
    def meets(self) -> bool:
        return self.students > 0 and self.share >= self.min_share


def coverage(
    summaries: Sequence[StudentCandidates], params: PairParams, schema: FeatureSchema, *, verified_only: bool = False
) -> list[Coverage]:
    """Cobertura por campus (orden del catálogo) y total; con `verified_only`, sin las `seed_sin_verificar`."""
    selected = [item for item in summaries if item.seed_verified or not verified_only]
    rows: list[Coverage] = []
    for code in schema.catalog("university_code"):
        group = [item for item in selected if item.university_code == code]
        if group:
            rows.append(Coverage(code, len(group), sum(1 for item in group if item.compatible), params.min_share))
    rows.append(Coverage("Total", len(selected), sum(1 for item in selected if item.compatible), params.min_share))
    return rows


__all__ = [
    "PAIRS_PARAMS_RESOURCE",
    "SPLITS",
    "Coverage",
    "Pair",
    "PairParams",
    "PairParamsError",
    "PairSet",
    "PairSetError",
    "StudentCandidates",
    "assign_splits",
    "build_pairs",
    "choose_candidates",
    "coverage",
    "load_params",
    "parse_params",
    "split_sizes",
]
