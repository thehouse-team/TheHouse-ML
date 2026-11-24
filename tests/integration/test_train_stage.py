"""ML-05 T02 (integración): la etapa `train` sobre pares y etiquetas generados por `prepare`.

Proyecto temporal con 2 campus y una rejilla pequeña (la corrida completa con los datos
de ML-03/ML-04 queda en la evidencia). Comprueba la reproducibilidad (mismos SHA-256 y
mismos reportes en dos corridas), que los modelos no usan pickle, que test no entra y la
prueba obligatoria de la decisión 1: `train` nunca lee la utilidad ni los pesos ocultos.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import pytest

from conftest import copy_project, small_train_params
from thehouse_ml.features import default_schema
from thehouse_ml.models.train import (
    LIGHTGBM_MODEL,
    LOGREG_MODEL,
    TRAIN_CV_METRICS,
    TRAIN_MANIFEST,
    TRAIN_SKIP_MESSAGE,
    TRAINING_REPORT,
    LogRegModel,
    run_train,
)
from thehouse_ml.models.train_config import PROFILE_ENV_VAR, TrainParams, load_train_params
from thehouse_ml.models.training_data import TRAINING_INPUTS, load_training_data
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.stages import run_stage
from thehouse_ml.synthetic.labels_stage import HIDDEN_UTILITY_PARQUET, HIDDEN_WEIGHTS_PARQUET

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
CODES = {"UTTT", "UCAN"}
ML05_TRAIN_MODULES = tuple(
    REPO / "src" / "thehouse_ml" / "models" / name for name in ("train.py", "train_config.py", "train_report.py")
)
OUTPUTS = (f"models/{LOGREG_MODEL}", f"models/{LIGHTGBM_MODEL}", f"models/{TRAIN_MANIFEST}")
REPORTS = (f"reports/{TRAINING_REPORT}", f"reports/{TRAIN_CV_METRICS}")


def small_params() -> TrainParams:
    params: TrainParams = small_train_params()  # la rejilla de la plantilla de sesión (conftest)
    return params


@pytest.fixture(scope="module")
def project(two_campus_labels_template: Path, tmp_path_factory: pytest.TempPathFactory) -> ProjectPaths:
    """Proyecto con 2 campus: estudiantes, cuartos, pares (ML-03 T06) y etiquetas con archivos ocultos (ML-04 T04)."""
    paths: ProjectPaths = copy_project(two_campus_labels_template, tmp_path_factory.mktemp("train_stage") / "project")
    assert (paths.data_processed / HIDDEN_UTILITY_PARQUET).is_file()
    return paths


def _snapshot(paths: ProjectPaths) -> dict[str, str]:
    return {name: hashlib.sha256((paths.root / name).read_bytes()).hexdigest() for name in (*OUTPUTS, *REPORTS)}


@pytest.fixture(scope="module")
def first_run(project: ProjectPaths) -> dict[str, str]:
    message = run_train(project, small_params())
    assert message.startswith("train (perfil full): ")
    assert "DATOS SINTÉTICOS" in message
    return _snapshot(project)


def test_two_runs_produce_identical_models_and_reports(project: ProjectPaths, first_run: dict[str, str]) -> None:
    run_train(project, small_params())

    assert _snapshot(project) == first_run  # mismo SHA-256 del JSON de la LR, del texto de LightGBM y de los reportes


def test_train_never_reads_the_hidden_label_files(
    project: ProjectPaths, first_run: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
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
        run_train(project, small_params())
    finally:
        for path, content in stash.items():
            path.write_bytes(content)

    assert sorted(read) == sorted(TRAINING_INPUTS)
    assert _snapshot(project) == first_run  # resultado idéntico sin los archivos ocultos
    for module in ML05_TRAIN_MODULES:
        source = module.read_text(encoding="utf-8")
        for forbidden in ("HIDDEN_", "labels_hidden", HIDDEN_UTILITY_PARQUET, HIDDEN_WEIGHTS_PARQUET):
            assert forbidden not in source, (module.name, forbidden)


def test_models_are_json_and_native_text_and_reload_without_pickle(
    project: ProjectPaths, first_run: dict[str, str]
) -> None:
    del first_run
    document = json.loads((project.models / LOGREG_MODEL).read_text(encoding="utf-8"))
    model = LogRegModel.from_dict(document)
    assert document["synthetic"] is True
    assert document["feature_order"] == list(default_schema().feature_order)
    assert document["trained_on"]["splits"] == ["train", "validation"]

    text = (project.models / LIGHTGBM_MODEL).read_text(encoding="utf-8")
    assert text.startswith("tree\n")
    assert "objective=lambdarank" in text
    booster = lightgbm.Booster(model_file=str(project.models / LIGHTGBM_MODEL))
    data = load_training_data(project)
    sample = data.features.to_numpy()[:50]
    assert booster.predict(sample, num_threads=1).shape == (50,)
    assert np.isfinite(model.decision_function(sample)).all()

    manifest = json.loads((project.models / TRAIN_MANIFEST).read_text(encoding="utf-8"))
    for name in (LOGREG_MODEL, LIGHTGBM_MODEL):
        assert manifest["models"][name]["sha256"] == hashlib.sha256((project.models / name).read_bytes()).hexdigest()


def test_test_partition_is_not_in_folds_or_final_models(project: ProjectPaths, first_run: dict[str, str]) -> None:
    del first_run
    data = load_training_data(project)
    test_students = set(data.students.loc[data.students["split"] == "test", "student_id"])
    cv_students = set(data.students.loc[data.students["split"] != "test", "student_id"])
    manifest = json.loads((project.models / TRAIN_MANIFEST).read_text(encoding="utf-8"))
    metrics = json.loads((project.reports / TRAIN_CV_METRICS).read_text(encoding="utf-8"))

    assert set(manifest["folds"]) == cv_students
    assert not set(manifest["folds"]) & test_students
    cv_pairs = int((data.split != "test").sum())
    assert metrics["folds"]["pairs"] == cv_pairs
    assert metrics["final_models"]["trained_on"]["pairs"] == cv_pairs
    assert sum(item["students"] for item in metrics["folds"]["by_fold"]) == len(cv_students)
    assert metrics["synthetic"] is True
    assert all(len(item["ndcg@10_by_fold"]) == 3 for item in metrics["logreg_grid"] + metrics["lightgbm_grid"])
    assert len(metrics["baseline"]["ndcg@10_by_fold"]) == 3
    assert metrics["final_models"]["lightgbm_monotone_check"]["violations"] == dict.fromkeys(
        small_params().monotone_decreasing, 0
    )
    report = (project.reports / TRAINING_REPORT).read_text(encoding="utf-8")
    assert "DATOS SINTÉTICOS" in report
    assert "test no se evalúa aquí" in report


def test_stage_skips_without_inputs(tmp_path: Path) -> None:
    assert run_stage("train", ProjectPaths(root=tmp_path)) == TRAIN_SKIP_MESSAGE
    assert not (tmp_path / "models").exists()


# ---------------------------------------------------------------------------
# Perfiles (M-ML-038, T02-H1)
# ---------------------------------------------------------------------------


def test_ci_profile_does_not_modify_the_versioned_reports_or_official_models(
    project: ProjectPaths, first_run: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PROFILE_ENV_VAR, "ci")

    message = run_stage("train", project)  # el CI llama a la etapa con la variable de entorno

    assert message.startswith("train (perfil ci): ")
    assert _snapshot(project) == first_run  # reports/ y los modelos oficiales intactos
    ci_dir = project.root / load_train_params().ci_profile.output_dir
    for name in (LOGREG_MODEL, LIGHTGBM_MODEL, TRAIN_MANIFEST, TRAINING_REPORT, TRAIN_CV_METRICS):
        assert (ci_dir / name).is_file(), name
    metrics = json.loads((ci_dir / TRAIN_CV_METRICS).read_text(encoding="utf-8"))
    assert metrics["profile"] == "ci"
    assert metrics["folds"]["n_splits"] == 2
    assert [item["config_id"] for item in metrics["logreg_grid"]] == ["logreg:standard:C=0.01"]
    assert [item["config_id"] for item in metrics["lightgbm_grid"]] == ["lightgbm:leaves=15:lr=0.05:trees=300"]
    assert "Perfil: **`ci`**" in (ci_dir / TRAINING_REPORT).read_text(encoding="utf-8")


def test_default_profile_is_full_and_writes_the_versioned_reports(
    project: ProjectPaths, first_run: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    del first_run
    monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)
    (project.reports / TRAIN_CV_METRICS).unlink()

    message = run_train(project, small_params())

    assert message.startswith("train (perfil full): ")
    metrics = json.loads((project.reports / TRAIN_CV_METRICS).read_text(encoding="utf-8"))
    assert metrics["profile"] == "full"
    assert metrics["folds"]["n_splits"] == small_params().n_splits
    assert len(metrics["logreg_grid"]) == len(small_params().logreg_grid())
    assert "Perfil: **`full`**" in (project.reports / TRAINING_REPORT).read_text(encoding="utf-8")
