"""ML-03 T06 (ML-03-R06): conjunto de pares estudiante-cuarto, sin red ni datos fuera de git.

Los estudiantes y los cuartos se generan con los insumos versionados de
`conftest.versioned_synthetic_data` (renta p50 de `reports/regional_calibration.md`
e histogramas agrupados de `reports/data_card.md`). La etapa `prepare` repite
la comprobación del umbral sobre `students.parquet` y `rooms.parquet` reales.
"""

from __future__ import annotations

import copy
import io
import json
import re
from collections import defaultdict
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from conftest import versioned_synthetic_data
from thehouse_ml.features import (
    Campus,
    FeatureSchema,
    StudentProfile,
    check_hard_constraints,
    compute_pair_features,
    default_schema,
    filter_rooms,
    resolve_student,
)
from thehouse_ml.synthetic import load_params as load_student_params
from thehouse_ml.synthetic import load_room_params
from thehouse_ml.synthetic.pairs import (
    PAIRS_PARAMS_RESOURCE,
    SPLITS,
    PairParams,
    PairParamsError,
    PairSet,
    PairSetError,
    assign_splits,
    build_pairs,
    choose_candidates,
    coverage,
    load_params,
    parse_params,
    split_sizes,
)
from thehouse_ml.synthetic.pairs_stage import (
    pair_columns,
    pairs_frame,
    render_report,
    rooms_from_frame,
    students_from_frame,
)
from thehouse_ml.synthetic.rooms import SyntheticRoom
from thehouse_ml.synthetic.rooms_stage import rooms_frame
from thehouse_ml.synthetic.students import BASIS_PREFIXES, PARAMS_PACKAGE, SyntheticStudent
from thehouse_ml.synthetic.students_stage import students_frame

LABEL_WORDS = re.compile(r"label|relevan|score|puntaje|target|etiqueta|rank", re.IGNORECASE)


def _raw_params() -> dict[str, Any]:
    text = resources.files(PARAMS_PACKAGE).joinpath(PAIRS_PARAMS_RESOURCE).read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


@pytest.fixture(scope="module")
def schema() -> FeatureSchema:
    return default_schema()


@pytest.fixture(scope="module")
def params() -> PairParams:
    return load_params()


@pytest.fixture(scope="module")
def data() -> tuple[tuple[SyntheticStudent, ...], tuple[SyntheticRoom, ...], dict[str, Campus]]:
    return versioned_synthetic_data()


@pytest.fixture(scope="module")
def students(data: tuple[Any, ...]) -> tuple[SyntheticStudent, ...]:
    return data[0]  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def rooms(data: tuple[Any, ...]) -> tuple[SyntheticRoom, ...]:
    return data[1]  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def campuses(data: tuple[Any, ...]) -> dict[str, Campus]:
    return data[2]  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def pair_set(
    students: tuple[SyntheticStudent, ...],
    rooms: tuple[SyntheticRoom, ...],
    campuses: dict[str, Campus],
    params: PairParams,
    schema: FeatureSchema,
) -> PairSet:
    return build_pairs(students, rooms, campuses, params, schema)


def _subset(
    students: tuple[SyntheticStudent, ...], rooms: tuple[SyntheticRoom, ...], codes: set[str]
) -> tuple[list[SyntheticStudent], list[SyntheticRoom]]:
    return (
        [item for item in students if item.profile.university_code in codes],
        [item for item in rooms if item.room.university_code in codes],
    )


def _room(base: SyntheticRoom, room_id: str, **changes: Any) -> SyntheticRoom:
    return SyntheticRoom(room_id, base.seed_verified, base.campus_distance_km, replace(base.room, **changes))


# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------


def test_every_parameter_has_a_value_and_a_documented_basis() -> None:
    for name, entry in _raw_params()["parameters"].items():
        assert set(entry) == {"value", "basis"}, name
        assert entry["basis"].startswith(BASIS_PREFIXES), name


def test_parameters_are_marked_synthetic_and_approved_as_m_ml_027(params: PairParams) -> None:
    assert params.metadata["synthetic"] is True
    assert params.metadata["decision"] == "M-ML-027 (APROBADO)"
    assert params.metadata["params_version"] == "1.0.0"


def test_requested_values_are_in_the_params(params: PairParams, schema: FeatureSchema) -> None:
    assert params.max_candidates == 200
    assert params.fractions == {"train": 0.70, "validation": 0.15, "test": 0.15}
    assert params.value("split.stratify_by") == "university_code"
    assert params.min_share == 0.90
    # Semilla propia, distinta de la de estudiantes y de cuartos.
    assert params.seed not in {load_student_params(schema).seed, load_room_params(schema).seed}


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("seed", -1, "seed"),
        ("seed", 1.5, "seed"),
        ("candidates.max_per_student", 0, "max_per_student"),
        ("candidates.max_per_student", True, "max_per_student"),
        ("split.fractions", {"train": 0.7, "validation": 0.3}, "exactamente"),
        ("split.fractions", {"train": 0.7, "validation": 0.2, "test": 0.2}, "sumar 1"),
        ("split.fractions", {"train": 1.0, "validation": 0.0, "test": 0.0}, "(0, 1)"),
        ("split.fractions", [0.7, 0.15, 0.15], "exactamente"),
        ("split.stratify_by", "livingWith", "stratify_by"),
        ("coverage.min_share", 0, "min_share"),
        ("coverage.min_share", 1.5, "min_share"),
        ("coverage.min_share", "0.9", "min_share"),
    ],
)
def test_invalid_parameters_are_rejected(name: str, value: object, message: str) -> None:
    data = copy.deepcopy(_raw_params())
    data["parameters"][name]["value"] = value
    with pytest.raises(PairParamsError, match=re.escape(message)):
        parse_params(data)


def test_missing_unknown_and_unexplained_parameters_are_rejected() -> None:
    data = copy.deepcopy(_raw_params())
    del data["parameters"]["seed"]
    data["parameters"]["extra"] = {"value": 1, "basis": "SUPUESTO documentado: prueba"}
    data["parameters"]["coverage.min_share"]["basis"] = "sin base"
    data["parameters"]["split.stratify_by"] = "university_code"
    data["metadata"]["synthetic"] = False
    with pytest.raises(PairParamsError) as error:
        parse_params(data)
    text = str(error.value)
    for fragment in ("faltan ['seed']", "desconocidos ['extra']", "coverage.min_share.basis", "{value, basis}"):
        assert fragment in text
    assert "metadata.synthetic" in text
    with pytest.raises(PairParamsError, match="raíz"):
        parse_params([])
    with pytest.raises(PairParamsError, match="parameters"):
        parse_params({"metadata": {"synthetic": True}, "parameters": []})


def test_load_params_reads_an_explicit_path_and_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "pairs_params.json"
    path.write_text(json.dumps(_raw_params()), encoding="utf-8")
    assert load_params(path).seed == load_params().seed
    path.write_text("{", encoding="utf-8")
    with pytest.raises(PairParamsError, match="JSON inválido"):
        load_params(path)


# ---------------------------------------------------------------------------
# Partición por estudiante (sin fuga)
# ---------------------------------------------------------------------------


def test_split_sizes_use_the_largest_remainder_and_fill_every_split(params: PairParams) -> None:
    assert split_sizes(200, params.fractions) == {"train": 140, "validation": 30, "test": 30}
    assert split_sizes(3, params.fractions) == {"train": 1, "validation": 1, "test": 1}
    for n in range(3, 120):
        sizes = split_sizes(n, params.fractions)
        assert sum(sizes.values()) == n
        assert min(sizes.values()) >= 1
        assert abs(sizes["train"] - 0.70 * n) <= 2
    with pytest.raises(PairSetError, match="al menos 3"):
        split_sizes(2, params.fractions)


def test_no_student_appears_in_two_splits(
    pair_set: PairSet, students: tuple[SyntheticStudent, ...], params: PairParams, schema: FeatureSchema
) -> None:
    assigned = assign_splits(students, params, schema)
    assert set(assigned) == {student.student_id for student in students}
    by_split = {split: {s for s, value in assigned.items() if value == split} for split in SPLITS}
    assert not by_split["train"] & by_split["validation"]
    assert not by_split["train"] & by_split["test"]
    assert not by_split["validation"] & by_split["test"]
    splits_of_student: dict[str, set[str]] = defaultdict(set)
    for pair in pair_set.pairs:
        splits_of_student[pair.student_id].add(pair.split)
        assert pair.split == assigned[pair.student_id]
    assert all(len(splits) == 1 for splits in splits_of_student.values())
    assert {item.student_id: item.split for item in pair_set.students} == assigned


def test_every_campus_appears_in_the_three_splits_with_140_30_30(pair_set: PairSet, schema: FeatureSchema) -> None:
    for code in schema.catalog("university_code"):
        group = [item for item in pair_set.students if item.university_code == code]
        counts = {split: sum(1 for item in group if item.split == split) for split in SPLITS}
        assert counts == {"train": 140, "validation": 30, "test": 30}, code
        assert {pair.split for pair in pair_set.pairs if pair.university_code == code} == set(SPLITS), code


def test_rooms_may_repeat_across_splits_because_they_are_the_catalog(pair_set: PairSet) -> None:
    splits_of_room: dict[str, set[str]] = defaultdict(set)
    for pair in pair_set.pairs:
        splits_of_room[pair.room_id].add(pair.split)
    assert any(len(splits) > 1 for splits in splits_of_room.values())


def test_split_does_not_depend_on_rooms_and_changes_with_the_seed(
    students: tuple[SyntheticStudent, ...], params: PairParams, schema: FeatureSchema
) -> None:
    assigned = assign_splits(students, params, schema)
    assert assign_splits(list(reversed(students)), params, schema) == assigned
    assert assign_splits(students, params.with_seed(params.seed + 1), schema) != assigned


# ---------------------------------------------------------------------------
# Umbral formal de M-ML-021 / M-ML-025: ≥ 90 % con al menos un cuarto que pasa las restricciones duras
# ---------------------------------------------------------------------------


def test_at_least_ninety_percent_of_students_have_a_room_that_passes_the_hard_constraints(
    students: tuple[SyntheticStudent, ...],
    rooms: tuple[SyntheticRoom, ...],
    pair_set: PairSet,
    params: PairParams,
    schema: FeatureSchema,
) -> None:
    # Recuento independiente de build_pairs: filter_rooms (T03) con los cuartos del campus del estudiante.
    rooms_by_campus: dict[str, list[Any]] = defaultdict(list)
    for item in rooms:
        rooms_by_campus[item.room.university_code].append(item.room)
    hits: dict[str, list[bool]] = defaultdict(list)
    for student in students:
        code = student.profile.university_code
        allowed = filter_rooms(student.profile, rooms_by_campus[code], schema).allowed
        hits[code].append(bool(allowed))
        hits["__total__"].append(bool(allowed))
        if student.seed_verified:
            hits["__verified__"].append(bool(allowed))
    assert set(hits) == {*schema.catalog("university_code"), "__total__", "__verified__"}
    for code, values in hits.items():
        assert sum(values) / len(values) >= 0.90, code
    # La cobertura del conjunto de pares coincide con el recuento independiente.
    for verified_only in (False, True):
        for row in coverage(pair_set.students, params, schema, verified_only=verified_only):
            key = {"Total": "__verified__" if verified_only else "__total__"}.get(row.label, row.label)
            assert row.with_candidates == sum(hits[key]), row.label
            assert row.meets, row.label


def test_coverage_below_the_threshold_does_not_meet(params: PairParams, schema: FeatureSchema) -> None:
    from thehouse_ml.synthetic.pairs import Coverage

    assert not Coverage("X", 10, 8, params.min_share).meets
    assert Coverage("X", 10, 9, params.min_share).meets
    assert not Coverage("X", 0, 0, params.min_share).meets
    assert Coverage("X", 0, 0, params.min_share).share == 0.0


# ---------------------------------------------------------------------------
# Candidatos
# ---------------------------------------------------------------------------


def test_candidates_are_the_compatible_rooms_of_the_campus_up_to_the_maximum(
    students: tuple[SyntheticStudent, ...],
    rooms: tuple[SyntheticRoom, ...],
    pair_set: PairSet,
    params: PairParams,
    schema: FeatureSchema,
) -> None:
    by_campus: dict[str, list[SyntheticRoom]] = defaultdict(list)
    for item in rooms:
        by_campus[item.room.university_code].append(item)
    profiles = {student.student_id: student.profile for student in students}
    sampled = 0
    for summary in pair_set.students:
        campus_rooms = by_campus[summary.university_code]
        allowed = filter_rooms(profiles[summary.student_id], [item.room for item in campus_rooms], schema).allowed
        allowed_ids = [item.room_id for item in campus_rooms if any(item.room is room for room in allowed)]
        assert summary.compatible == len(allowed_ids)
        assert summary.campus_rooms == len(campus_rooms)
        assert list(summary.room_ids) == sorted(summary.room_ids)
        if len(allowed_ids) <= params.max_candidates:
            assert list(summary.room_ids) == allowed_ids
        else:
            sampled += 1
            assert len(summary.room_ids) == params.max_candidates
            assert set(summary.room_ids) <= set(allowed_ids)
    assert sampled > 0  # el muestreo se ejercita con los datos versionados


def test_every_pair_passes_the_hard_constraints_and_is_from_the_student_campus(
    students: tuple[SyntheticStudent, ...], rooms: tuple[SyntheticRoom, ...], pair_set: PairSet
) -> None:
    profiles = {student.student_id: student.profile for student in students}
    by_id = {item.room_id: item.room for item in rooms}
    for pair in pair_set.pairs[::97]:
        room = by_id[pair.room_id]
        assert room.university_code == pair.university_code == profiles[pair.student_id].university_code
        assert check_hard_constraints(profiles[pair.student_id], room).allowed


def test_rooms_farther_than_max_distance_are_candidates(
    rooms: tuple[SyntheticRoom, ...], campuses: dict[str, Campus], params: PairParams, schema: FeatureSchema
) -> None:
    far = max((item for item in rooms if item.room.university_code == "UTTT"), key=lambda item: item.campus_distance_km)
    room = _room(far, "R1", price_mxn=1000, living_with="any", pets_allowed=True)
    student = SyntheticStudent(
        "S1", True, StudentProfile("UTTT", budget_max=2000, living_with="any", place_type="room", max_distance_km=0.5)
    )
    others = [SyntheticStudent(f"S{i}", True, replace(student.profile)) for i in (2, 3)]
    result = build_pairs([student, *others], [room], campuses, params, schema)
    assert [pair.room_id for pair in result.pairs if pair.student_id == "S1"] == ["R1"]
    assert result.pairs[0].features["distance_km"] > 20.0
    assert result.students[0].beyond_max_distance == 1


def test_student_without_candidates_generates_no_pairs_but_is_counted(
    rooms: tuple[SyntheticRoom, ...], campuses: dict[str, Campus], params: PairParams, schema: FeatureSchema
) -> None:
    base = next(item for item in rooms if item.room.university_code == "UTTT")
    catalog = [_room(base, "R1", price_mxn=5000, living_with="men", pets_allowed=False)]
    profile = StudentProfile("UTTT", budget_max=1000, living_with="women", place_type="room", pets=True)
    students = [SyntheticStudent(f"S{i}", True, profile) for i in (1, 2, 3)]
    result = build_pairs(students, catalog, campuses, params, schema)
    assert result.pairs == ()
    assert [item.compatible for item in result.students] == [0, 0, 0]
    assert result.students[0].exclusions == {"HC01_budget": 1, "HC02_living_with": 1, "HC03_pets": 1}
    assert result.students[0].excluded == 1
    assert {item.split for item in result.students} == set(SPLITS)
    assert not coverage(result.students, params, schema)[-1].meets


def test_choose_candidates_is_a_fixed_uniform_sample_in_room_order(params: PairParams) -> None:
    allowed = [f"R{i:05d}" for i in range(1, 301)]
    first = choose_candidates("S0001", allowed, params)
    assert first == choose_candidates("S0001", allowed, params)
    assert len(first) == params.max_candidates and list(first) == sorted(first) and set(first) <= set(allowed)
    assert choose_candidates("S0002", allowed, params) != first
    assert choose_candidates("S0001", allowed, params.with_seed(1)) != first
    assert choose_candidates("S0001", allowed[:5], params) == tuple(allowed[:5])
    assert len(choose_candidates("S0001", allowed, params.with_max_candidates(10))) == 10


def test_invalid_inputs_are_errors(
    students: tuple[SyntheticStudent, ...],
    rooms: tuple[SyntheticRoom, ...],
    campuses: dict[str, Campus],
    params: PairParams,
    schema: FeatureSchema,
) -> None:
    few_students, few_rooms = _subset(students, rooms, {"UTTT"})
    with pytest.raises(PairSetError, match="student_id duplicado"):
        build_pairs([*few_students, few_students[0]], few_rooms, campuses, params, schema)
    with pytest.raises(PairSetError, match="room_id duplicado"):
        build_pairs(few_students, [*few_rooms, few_rooms[0]], campuses, params, schema)
    outside = _room(few_rooms[0], "R99999", university_code="XXX")
    with pytest.raises(PairSetError, match="fuera del catálogo"):
        build_pairs(few_students, [*few_rooms, outside], campuses, params, schema)
    with pytest.raises(PairSetError, match="al menos 3"):
        build_pairs(few_students[:2], few_rooms, campuses, params, schema)


# ---------------------------------------------------------------------------
# Features, columnas y reproducibilidad
# ---------------------------------------------------------------------------


def test_features_come_from_compute_pair_features_in_the_schema_order(
    students: tuple[SyntheticStudent, ...],
    rooms: tuple[SyntheticRoom, ...],
    campuses: dict[str, Campus],
    pair_set: PairSet,
    schema: FeatureSchema,
) -> None:
    profiles = {student.student_id: student.profile for student in students}
    by_id = {item.room_id: item.room for item in rooms}
    for pair in pair_set.pairs[::53]:
        expected = compute_pair_features(profiles[pair.student_id], by_id[pair.room_id], campuses, schema)
        assert list(pair.features) == list(schema.feature_order)
        assert pair.features == expected


def test_frame_has_the_ids_split_seven_features_and_no_labels(pair_set: PairSet, schema: FeatureSchema) -> None:
    frame = pairs_frame(pair_set, schema)
    assert list(frame.columns) == [
        "student_id",
        "room_id",
        "university_code",
        "seed_verified",
        "split",
        *schema.feature_order,
        "synthetic",
    ]
    assert len(schema.feature_order) == 7
    assert list(frame.columns) == list(pair_columns(schema))
    assert not [column for column in frame.columns if LABEL_WORDS.search(column)]
    assert len(frame) == len(pair_set.pairs)
    assert bool(frame["synthetic"].all())
    assert set(frame["split"]) == set(SPLITS)
    assert str(frame["place_type_match"].dtype) == "int64"
    assert str(frame["distance_km"].dtype) == "float64"
    assert not frame.duplicated(["student_id", "room_id"]).any()
    assert set(frame.loc[~frame["seed_verified"], "university_code"]) == {"TESJI", "UMB-VDC", "UCAN"}


def test_same_inputs_give_the_same_pairs_byte_for_byte(
    students: tuple[SyntheticStudent, ...],
    rooms: tuple[SyntheticRoom, ...],
    campuses: dict[str, Campus],
    params: PairParams,
    schema: FeatureSchema,
) -> None:
    few_students, few_rooms = _subset(students, rooms, {"UTTT", "UCAN"})

    def parquet(pair_set: PairSet) -> bytes:
        buffer = io.BytesIO()
        pairs_frame(pair_set, schema).to_parquet(buffer, index=False)
        return buffer.getvalue()

    first = build_pairs(few_students, few_rooms, campuses, params, schema)
    again = build_pairs(list(reversed(few_students)), list(reversed(few_rooms)), campuses, params, schema)
    assert parquet(first) == parquet(again)
    other = build_pairs(few_students, few_rooms, campuses, params.with_seed(params.seed + 1), schema)
    assert parquet(other) != parquet(first)


def test_parquet_round_trip_keeps_students_and_rooms(
    students: tuple[SyntheticStudent, ...], rooms: tuple[SyntheticRoom, ...], schema: FeatureSchema
) -> None:
    def through_parquet(frame: pd.DataFrame) -> pd.DataFrame:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        buffer.seek(0)
        return pd.read_parquet(buffer)

    assert students_from_frame(through_parquet(students_frame(students, schema))) == list(students)
    assert rooms_from_frame(through_parquet(rooms_frame(rooms, schema))) == list(rooms)
    # Los nulos de NV-11 se conservan y se resuelven con las reglas para faltantes.
    assert any(student.profile.budget_min is None for student in students)
    for student in students_from_frame(through_parquet(students_frame(students[:50], schema))):
        resolve_student(student.profile, schema)


def test_report_is_synthetic_and_shows_numbers_with_and_without_unverified_campuses(
    pair_set: PairSet, params: PairParams, schema: FeatureSchema
) -> None:
    report = render_report(pair_set, params, schema)
    assert report.startswith("# DATOS SINTÉTICOS")
    for heading in (
        "Regla de candidatos",
        "Partición por estudiante",
        "Candidatos por estudiante",
        "Exclusiones por restricción dura",
        "Estudiantes con al menos un candidato",
        "Features de los pares",
        "Sin etiquetas",
        "sí se repiten entre particiones",
    ):
        assert heading in report
    for constraint in ("HC01_budget", "HC02_living_with", "HC03_pets"):
        assert f"`{constraint}`" in report
    assert report.count("| **Total sin `seed_sin_verificar`** |") == 4
    assert report.count("| **Total** |") == 4
    assert "**cumple**" in report
    assert "M-ML-027" in report
    assert "| UTTT | sí | 140 | 30 | 30 |" in report
    assert "| UCAN | **no** | 140 | 30 | 30 |" in report


# ---------------------------------------------------------------------------
# Punto 6 del «EJECUTA» de T06: proporciones y comprobación exhaustiva de todos los pares
# ---------------------------------------------------------------------------

# Tolerancia documentada (evidence.md, T06): estudiantes por campus a menos de 1 estudiante de fracción por n
# (mayor residuo) y ±0.5 puntos en total; pares ±3 puntos (el número de candidatos varía por estudiante).
STUDENT_TOTAL_TOLERANCE = 0.005
PAIR_TOLERANCE = 0.03


def test_split_proportions_are_70_15_15_within_the_documented_tolerance(
    pair_set: PairSet, params: PairParams, schema: FeatureSchema
) -> None:
    for code in schema.catalog("university_code"):
        group = [item for item in pair_set.students if item.university_code == code]
        for split, fraction in params.fractions.items():
            count = sum(1 for item in group if item.split == split)
            assert abs(count - fraction * len(group)) < 1, (code, split)
    total_students, total_pairs = len(pair_set.students), len(pair_set.pairs)
    for split, fraction in params.fractions.items():
        students_share = sum(1 for item in pair_set.students if item.split == split) / total_students
        pairs_share = sum(1 for pair in pair_set.pairs if pair.split == split) / total_pairs
        assert abs(students_share - fraction) <= STUDENT_TOTAL_TOLERANCE, split
        assert abs(pairs_share - fraction) <= PAIR_TOLERANCE, split


def test_no_pair_violates_a_hard_constraint_every_candidate_is_from_its_campus_and_at_most_200(
    students: tuple[SyntheticStudent, ...], rooms: tuple[SyntheticRoom, ...], pair_set: PairSet, schema: FeatureSchema
) -> None:
    # Todos los pares (no una muestra): filter_rooms con los candidatos de cada estudiante no excluye ninguno.
    profiles = {student.student_id: student.profile for student in students}
    by_id = {item.room_id: item.room for item in rooms}
    candidates: dict[str, list[str]] = defaultdict(list)
    for pair in pair_set.pairs:
        candidates[pair.student_id].append(pair.room_id)
    assert max(len(room_ids) for room_ids in candidates.values()) <= 200
    assert all(len(item.room_ids) <= 200 for item in pair_set.students)
    for student_id, room_ids in candidates.items():
        profile = profiles[student_id]
        assert {by_id[room_id].university_code for room_id in room_ids} == {profile.university_code}
        result = filter_rooms(profile, [by_id[room_id] for room_id in room_ids], schema)
        assert result.excluded == 0, student_id
        assert len(result.allowed) == len(room_ids)
