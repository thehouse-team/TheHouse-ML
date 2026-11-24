"""ML-05 T01 (integración): el cargador de entrenamiento sobre pares y etiquetas generados por `prepare`.

Prueba obligatoria de la decisión 1 de ML-05 (M-ML-036): el cargador **nunca** lee
`labels_hidden_utility.parquet` ni `labels_hidden_weights.parquet` y su salida no cambia
si esos archivos se borran.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from conftest import versioned_synthetic_data
from thehouse_ml.features import default_schema
from thehouse_ml.models.training_data import TrainingData, load_training_data
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.synthetic.labels_stage import (
    HIDDEN_UTILITY_PARQUET,
    HIDDEN_WEIGHTS_PARQUET,
    LABELS_PARQUET,
    run_labels,
)
from thehouse_ml.synthetic.pairs_stage import PAIRS_PARQUET, run_pairs
from thehouse_ml.synthetic.rooms_stage import ROOMS_PARQUET, rooms_frame
from thehouse_ml.synthetic.students_stage import STUDENTS_PARQUET, students_frame

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
CODES = {"UTTT", "UCAN"}  # Un campus verificado y uno sin verificar.
ML05_MODULES = (REPO / "src" / "thehouse_ml" / "models" / "training_data.py",)


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> ProjectPaths:
    """Proyecto con 2 campus: estudiantes, cuartos, pares (ML-03 T06) y etiquetas con archivos ocultos (ML-04 T04)."""
    students, rooms, _campuses = versioned_synthetic_data()
    schema = default_schema()
    paths = ProjectPaths(root=tmp_path_factory.mktemp("training_data"))
    paths.data_processed.mkdir(parents=True)
    shutil.copyfile(REPO / "data" / "universities.csv", paths.universities_csv)
    chosen_students = [item for item in students if item.profile.university_code in CODES]
    chosen_rooms = [item for item in rooms if item.room.university_code in CODES]
    students_frame(chosen_students, schema).to_parquet(paths.data_processed / STUDENTS_PARQUET, index=False)
    rooms_frame(chosen_rooms, schema).to_parquet(paths.data_processed / ROOMS_PARQUET, index=False)
    run_pairs(paths)
    run_labels(paths)
    assert (paths.data_processed / HIDDEN_UTILITY_PARQUET).is_file()
    assert (paths.data_processed / HIDDEN_WEIGHTS_PARQUET).is_file()
    return paths


def _assert_same(first: TrainingData, second: TrainingData) -> None:
    pd.testing.assert_frame_equal(first.features, second.features)
    pd.testing.assert_series_equal(first.relevance, second.relevance)
    pd.testing.assert_series_equal(first.group, second.group)
    pd.testing.assert_series_equal(first.room_id, second.room_id)
    pd.testing.assert_series_equal(first.split, second.split)
    pd.testing.assert_frame_equal(first.students, second.students)
    assert first.excluded == second.excluded


def test_loader_never_reads_the_hidden_label_files(project: ProjectPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    before = load_training_data(project)
    read: list[str] = []
    original = pd.read_parquet

    def spy(path: object, *args: Any, **kwargs: Any) -> pd.DataFrame:
        read.append(Path(str(path)).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    hidden = [project.data_processed / HIDDEN_UTILITY_PARQUET, project.data_processed / HIDDEN_WEIGHTS_PARQUET]
    stash = {path: path.read_bytes() for path in hidden}
    for path in hidden:
        path.unlink()
    try:
        after = load_training_data(project)
    finally:
        for path, content in stash.items():
            path.write_bytes(content)

    assert sorted(read) == sorted([PAIRS_PARQUET, LABELS_PARQUET, STUDENTS_PARQUET])
    assert HIDDEN_UTILITY_PARQUET not in read
    assert HIDDEN_WEIGHTS_PARQUET not in read
    _assert_same(before, after)  # sin los archivos ocultos la salida es idéntica
    for module in ML05_MODULES:
        source = module.read_text(encoding="utf-8")
        assert "HIDDEN_" not in source
        assert HIDDEN_UTILITY_PARQUET not in source
        assert HIDDEN_WEIGHTS_PARQUET not in source
        assert "labels_hidden" not in source


def test_model_inputs_are_exactly_feature_order_without_utility_or_weights(project: ProjectPaths) -> None:
    data = load_training_data(project)
    assert list(data.features.columns) == list(default_schema().feature_order)
    assert "utility" not in data.features.columns
    assert not [column for column in data.features.columns if column.startswith("weight_")]


def test_every_candidate_pair_is_loaded_except_students_with_one_candidate(project: ProjectPaths) -> None:
    data = load_training_data(project)
    pairs = pd.read_parquet(project.data_processed / PAIRS_PARQUET)
    labels = pd.read_parquet(project.data_processed / LABELS_PARQUET)
    counts = pairs.groupby("student_id").size()
    few = set(counts[counts < 2].index)

    assert {item.student_id for item in data.excluded} == few
    assert len(data.features) == int(counts[counts >= 2].sum())
    assert data.students["student_id"].tolist() == sorted(set(counts.index) - few)
    assert set(data.students["university_code"]) == CODES
    expected = labels.set_index(["student_id", "room_id"])["relevance"]
    keys = list(zip(data.group, data.room_id, strict=True))
    assert data.relevance.tolist() == [int(expected[key]) for key in keys]
    summary = data.summary()
    assert summary["total"]["pairs"] == len(data.features)
    assert sum(summary[split]["students"] for split in ("train", "validation", "test")) == len(data.students)
