"""ML-05 T01 (ML-05-R01): carga de pares y etiquetas para entrenar, sin red ni archivos de data/."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from thehouse_ml.features import default_schema
from thehouse_ml.models.training_data import (
    EXPECTED_LABELS_PARAMS_VERSION,
    MIN_CANDIDATES,
    TRAINING_INPUTS,
    ExcludedStudent,
    TrainingDataError,
    _read,
    build_training_data,
    load_training_data,
)
from thehouse_ml.paths import ProjectPaths

SCHEMA = default_schema()
ORDER = list(SCHEMA.feature_order)
# (student_id, split, university_code, seed_verified, budget.max, candidatos)
STUDENTS = (
    ("S0001", "train", "UTTT", True, 1500, 3),
    ("S0002", "validation", "UTTT", True, 2800, 2),
    ("S0003", "test", "UCAN", False, 4000, 4),
    ("S0004", "train", "UCAN", False, 2000, 1),  # se excluye: < 2 candidatos
)


def _feature_values(student_index: int, room_index: int) -> dict[str, float]:
    base = 0.01 * (10 * student_index + room_index)
    values = {name: round(base + 0.001 * position, 6) for position, name in enumerate(ORDER)}
    values.update(place_type_match=1, property_type_match=room_index % 2, pets_compatible=1)
    return values


def make_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows, label_rows = [], []
    for s_index, (student_id, split, code, verified, _budget, n) in enumerate(STUDENTS):
        for r_index in range(n):
            room_id = f"R{s_index}{r_index:03d}"
            pair_rows.append(
                {
                    "student_id": student_id,
                    "room_id": room_id,
                    "university_code": code,
                    "seed_verified": verified,
                    "split": split,
                    **_feature_values(s_index, r_index),
                    "synthetic": True,
                }
            )
            label_rows.append(
                {
                    "student_id": student_id,
                    "room_id": room_id,
                    "split": split,
                    "relevance": 3 if r_index == 0 else (r_index % 3),
                    "synthetic": True,
                    "labels_params_version": EXPECTED_LABELS_PARAMS_VERSION,
                }
            )
    pairs = pd.DataFrame(pair_rows)
    labels = pd.DataFrame(label_rows)
    students = pd.DataFrame(
        {
            "student_id": [row[0] for row in STUDENTS],
            "budget.max": [row[4] for row in STUDENTS],
            "livingWith": ["any"] * len(STUDENTS),  # columna ajena: no debe pasar a la salida
        }
    )
    return pairs, labels, students


def test_join_aligns_each_pair_with_its_label_even_if_labels_come_shuffled() -> None:
    pairs, labels, students = make_frames()
    shuffled = labels.sample(frac=1.0, random_state=7).reset_index(drop=True)
    data = build_training_data(pairs, shuffled, students, SCHEMA)

    expected = labels.set_index(["student_id", "room_id"])["relevance"]
    keys = list(zip(data.group, data.room_id, strict=True))
    assert data.relevance.tolist() == [int(expected[key]) for key in keys]
    assert keys == sorted(keys)  # orden (student_id, room_id)
    by_key = pairs.set_index(["student_id", "room_id"])
    for position, key in enumerate(keys):
        assert data.features.iloc[position].tolist() == pytest.approx(by_key.loc[key, ORDER].astype(float).tolist())
    assert data.split.tolist() == [by_key.loc[key, "split"] for key in keys]


def test_students_with_fewer_than_two_candidates_are_excluded_and_counted() -> None:
    data = build_training_data(*make_frames(), SCHEMA)

    assert MIN_CANDIDATES == 2
    assert data.excluded == (ExcludedStudent("S0004", "train", 1),)
    assert "S0004" not in set(data.group)
    assert data.summary() == {
        "train": {"pairs": 3, "students": 1, "excluded_students": 1},
        "validation": {"pairs": 2, "students": 1, "excluded_students": 0},
        "test": {"pairs": 4, "students": 1, "excluded_students": 0},
        "total": {"pairs": 9, "students": 3, "excluded_students": 1},
    }


def test_inputs_are_exactly_feature_order_without_hidden_or_extra_columns() -> None:
    pairs, labels, students = make_frames()
    # Aunque alguien deje columnas ajenas en los insumos, X solo trae feature_order.
    pairs = pairs.assign(utility=1.0, weight_budget=20.0)
    labels = labels.assign(utility=1.0)
    data = build_training_data(pairs, labels, students, SCHEMA)

    assert list(data.features.columns) == ORDER
    assert len(ORDER) == 10
    assert all(dtype == "float64" for dtype in data.features.dtypes)
    forbidden = {"utility", "relevance", "split", "student_id", "room_id", "university_code", "budget.max"}
    assert not forbidden & set(data.features.columns)
    assert not [column for column in data.features.columns if column.startswith("weight_")]


def test_student_metadata_for_segments() -> None:
    data = build_training_data(*make_frames(), SCHEMA)

    assert data.students.to_dict("records") == [
        {
            "student_id": "S0001",
            "split": "train",
            "university_code": "UTTT",
            "seed_verified": True,
            "budget_max": 1500,
            "candidates": 3,
        },
        {
            "student_id": "S0002",
            "split": "validation",
            "university_code": "UTTT",
            "seed_verified": True,
            "budget_max": 2800,
            "candidates": 2,
        },
        {
            "student_id": "S0003",
            "split": "test",
            "university_code": "UCAN",
            "seed_verified": False,
            "budget_max": 4000,
            "candidates": 4,
        },
    ]
    assert "livingWith" not in data.students.columns
    assert data.schema_version == "2.0.0"
    assert data.labels_params_version == "1.0.0"


@pytest.mark.parametrize("column", [*ORDER, "student_id", "room_id", "split", "university_code", "seed_verified"])
def test_missing_pair_column_is_a_clear_error(column: str) -> None:
    pairs, labels, students = make_frames()
    with pytest.raises(TrainingDataError, match=rf"pairs\.parquet: faltan las columnas \['{column}'\]"):
        build_training_data(pairs.drop(columns=[column]), labels, students, SCHEMA)


@pytest.mark.parametrize("column", ["relevance", "synthetic", "labels_params_version", "split"])
def test_missing_label_column_is_a_clear_error(column: str) -> None:
    pairs, labels, students = make_frames()
    with pytest.raises(TrainingDataError, match=rf"labels\.parquet: faltan las columnas \['{column}'\]"):
        build_training_data(pairs, labels.drop(columns=[column]), students, SCHEMA)


def test_missing_budget_column_is_a_clear_error() -> None:
    pairs, labels, students = make_frames()
    with pytest.raises(TrainingDataError, match=r"students\.parquet: faltan las columnas \['budget\.max'\]"):
        build_training_data(pairs, labels, students.drop(columns=["budget.max"]), SCHEMA)


def test_features_out_of_schema_order_is_an_error() -> None:
    pairs, labels, students = make_frames()
    columns = list(pairs.columns)
    a, b = columns.index(ORDER[0]), columns.index(ORDER[1])
    columns[a], columns[b] = columns[b], columns[a]
    with pytest.raises(TrainingDataError, match="no están en el orden de feature_order"):
        build_training_data(pairs[columns], labels, students, SCHEMA)


def test_changed_labels_params_version_is_an_error() -> None:
    pairs, labels, students = make_frames()
    labels.loc[0, "labels_params_version"] = "1.1.0"
    with pytest.raises(TrainingDataError, match=r"labels_params_version \['1\.0\.0', '1\.1\.0'\]"):
        build_training_data(pairs, labels, students, SCHEMA)


def test_changed_feature_schema_version_is_an_error() -> None:
    other = dataclasses.replace(SCHEMA, metadata=dataclasses.replace(SCHEMA.metadata, schema_version="3.0.0"))
    with pytest.raises(TrainingDataError, match=r"feature_schema\.json 3\.0\.0: ML-05 entrena con el esquema 2\.0\.0"):
        build_training_data(*make_frames(), other)


@pytest.mark.parametrize("frame", ["pairs", "labels"])
def test_non_synthetic_rows_are_an_error(frame: str) -> None:
    pairs, labels, students = make_frames()
    target = pairs if frame == "pairs" else labels
    target.loc[2, "synthetic"] = False
    with pytest.raises(TrainingDataError, match="synthetic = true"):
        build_training_data(pairs, labels, students, SCHEMA)


def test_pair_without_label_is_an_error() -> None:
    pairs, labels, students = make_frames()
    with pytest.raises(TrainingDataError, match="1 pares sin etiqueta"):
        build_training_data(pairs, labels.iloc[1:], students, SCHEMA)


def test_split_mismatch_between_pairs_and_labels_is_an_error() -> None:
    pairs, labels, students = make_frames()
    labels.loc[labels["student_id"] == "S0001", "split"] = "test"
    with pytest.raises(TrainingDataError, match="partición"):
        build_training_data(pairs, labels, students, SCHEMA)


def test_relevance_out_of_range_is_an_error() -> None:
    pairs, labels, students = make_frames()
    labels.loc[0, "relevance"] = 4
    with pytest.raises(TrainingDataError, match="relevancias fuera de 0-3"):
        build_training_data(pairs, labels, students, SCHEMA)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_feature_is_an_error(value: float) -> None:
    pairs, labels, students = make_frames()
    pairs.loc[0, ORDER[0]] = value
    with pytest.raises(TrainingDataError, match="no finitas"):
        build_training_data(pairs, labels, students, SCHEMA)


def test_student_missing_from_students_parquet_is_an_error() -> None:
    pairs, labels, students = make_frames()
    with pytest.raises(TrainingDataError, match=r"students\.parquet: faltan 1 estudiantes"):
        build_training_data(pairs, labels, students[students["student_id"] != "S0001"], SCHEMA)


def test_only_training_inputs_can_be_read(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    assert TRAINING_INPUTS == ("pairs.parquet", "labels.parquet", "students.parquet")
    with pytest.raises(TrainingDataError, match="no es una entrada del entrenamiento"):
        _read(paths, "labels_hidden_utility.parquet")
    with pytest.raises(TrainingDataError, match=r"falta data/processed/pairs\.parquet"):
        load_training_data(paths)


def test_load_reads_only_budget_from_students_and_matches_the_pure_build(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    paths.data_processed.mkdir(parents=True)
    pairs, labels, students = make_frames()
    pairs.to_parquet(paths.data_processed / "pairs.parquet", index=False)
    labels.to_parquet(paths.data_processed / "labels.parquet", index=False)
    students.to_parquet(paths.data_processed / "students.parquet", index=False)

    loaded = load_training_data(paths)
    built = build_training_data(pairs, labels, students, SCHEMA)
    pd.testing.assert_frame_equal(loaded.features, built.features)
    pd.testing.assert_frame_equal(loaded.students, built.students)
    assert loaded.relevance.tolist() == built.relevance.tolist()

    students.drop(columns=["budget.max"]).to_parquet(paths.data_processed / "students.parquet", index=False)
    with pytest.raises(TrainingDataError, match=r"students\.parquet: no se pudieron leer las columnas"):
        load_training_data(paths)


def test_unknown_split_is_an_error() -> None:
    pairs, labels, students = make_frames()
    pairs.loc[pairs["student_id"] == "S0001", "split"] = "holdout"
    labels.loc[labels["student_id"] == "S0001", "split"] = "holdout"
    with pytest.raises(TrainingDataError, match=r"particiones desconocidas \['holdout'\]"):
        build_training_data(pairs, labels, students, SCHEMA)


@pytest.mark.parametrize(("column", "value"), [("university_code", "UCAN"), ("seed_verified", False)])
def test_student_with_two_values_of_a_segment_column_is_an_error(column: str, value: object) -> None:
    pairs, labels, students = make_frames()
    pairs.loc[0, column] = value  # primer par de S0001 (UTTT, verificado)
    with pytest.raises(TrainingDataError, match=f"más de un valor de {column}"):
        build_training_data(pairs, labels, students, SCHEMA)


def test_duplicated_student_in_students_parquet_is_an_error() -> None:
    pairs, labels, students = make_frames()
    with pytest.raises(TrainingDataError, match="student_id duplicado"):
        build_training_data(pairs, labels, pd.concat([students, students.iloc[:1]]), SCHEMA)


def test_null_budget_is_an_error() -> None:
    pairs, labels, students = make_frames()
    students["budget.max"] = students["budget.max"].astype("Int64")
    students.loc[0, "budget.max"] = pd.NA
    with pytest.raises(TrainingDataError, match=r"budget\.max nulo"):
        build_training_data(pairs, labels, students, SCHEMA)


def test_no_student_with_two_candidates_is_an_error() -> None:
    pairs, labels, students = make_frames()
    only = pairs["student_id"] == "S0004"
    with pytest.raises(TrainingDataError, match="ningún estudiante tiene 2 o más candidatos"):
        build_training_data(pairs[only], labels[labels["student_id"] == "S0004"], students, SCHEMA)
