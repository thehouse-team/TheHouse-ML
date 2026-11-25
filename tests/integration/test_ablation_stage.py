"""ML-05 T04 (integración): métricas por pliegue, ablación, segmentos e hipótesis dentro de `evaluate`.

Proyecto temporal con 2 campus, `train` con una rejilla pequeña y `evaluate` del baseline; después
`run_trained_evaluate` (T03 + T04). Sin red.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import copy_project, small_train_params
from thehouse_ml.evaluation.ranking import METRIC_NAMES
from thehouse_ml.features import default_schema
from thehouse_ml.models import ablation as ablation_module
from thehouse_ml.models.ablation import ABLATION_METRICS, ABLATION_REPORT
from thehouse_ml.models.factor_groups import FACTOR_GROUPS
from thehouse_ml.models.train import TRAIN_CV_METRICS
from thehouse_ml.models.train_config import PROFILE_ENV_VAR, TrainParams, load_train_params
from thehouse_ml.models.trained_eval import TRAINED_METRICS, TRAINED_REPORT, run_trained_evaluate
from thehouse_ml.models.training_data import load_training_data
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.stages import run_stage
from thehouse_ml.synthetic.labels_stage import HIDDEN_UTILITY_PARQUET, HIDDEN_WEIGHTS_PARQUET

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
CODES = {"UTTT", "UCAN"}
T04_MODULES = tuple(
    REPO / "src" / "thehouse_ml" / "models" / name for name in ("ablation.py", "ablation_report.py", "factor_groups.py")
)
OUTPUTS = tuple(f"reports/{name}" for name in (TRAINED_METRICS, TRAINED_REPORT, ABLATION_METRICS, ABLATION_REPORT))


def small_params() -> TrainParams:
    params: TrainParams = small_train_params()  # la rejilla de la plantilla de sesión (conftest)
    return params


@pytest.fixture(scope="module")
def project(two_campus_trained_template: Path, tmp_path_factory: pytest.TempPathFactory) -> ProjectPaths:
    paths: ProjectPaths = copy_project(two_campus_trained_template, tmp_path_factory.mktemp("ablation") / "project")
    return paths


def _snapshot(paths: ProjectPaths, names: tuple[str, ...] = OUTPUTS) -> dict[str, str]:
    return {name: hashlib.sha256((paths.root / name).read_bytes()).hexdigest() for name in names}


@pytest.fixture(scope="module")
def first_run(project: ProjectPaths) -> dict[str, str]:
    message = run_trained_evaluate(project, small_params(), "full")
    assert "evaluate (ML-05 T04, perfil full): " in message
    message.encode("cp1252")
    return _snapshot(project)


def _json(paths: ProjectPaths, name: str, directory: str = "reports") -> dict[str, Any]:
    data: dict[str, Any] = json.loads((paths.root / directory / name).read_text(encoding="utf-8"))
    return data


def test_fold_metrics_have_the_four_metrics_and_reproduce_t02(project: ProjectPaths, first_run: dict[str, str]) -> None:
    del first_run
    ablation = _json(project, ABLATION_METRICS)
    train_cv = _json(project, TRAIN_CV_METRICS)
    metrics = ablation["cross_validation"]["metrics"]

    assert ablation["synthetic"] is True
    assert ablation["cross_validation"]["n_splits"] == small_params().n_splits
    expected = {
        "baseline": train_cv["baseline"]["ndcg@10_by_fold"],
        "logreg": train_cv["selected"]["logreg"]["ndcg@10_by_fold"],
        "lightgbm": train_cv["selected"]["lightgbm"]["ndcg@10_by_fold"],
    }
    for name, values in expected.items():
        folds = metrics[name]["by_fold"]
        assert [item["ndcg@10"] for item in folds] == pytest.approx(values, abs=1e-12)
        assert all(set(METRIC_NAMES) <= set(item) for item in folds)
        assert set(metrics[name]["mean"]) == set(metrics[name]["std"]) == set(METRIC_NAMES)
    report = (project.reports / TRAINED_REPORT).read_text(encoding="utf-8")
    assert "Las 4 métricas por pliegue (T04, traspaso T03-H2)" in report
    assert "| Precision@5 | LightGBM |" in report


def test_ablation_never_uses_test_and_each_group_removes_exactly_its_features(
    project: ProjectPaths, first_run: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    del first_run
    data = load_training_data(project)
    test_students = set(data.students.loc[data.students["split"] == "test", "student_id"])
    cv_pairs = int((data.split != "test").sum())
    fits: list[tuple[str, tuple[str, ...], int, set[str]]] = []
    fit_logreg = ablation_module.fit_logreg
    fit_lightgbm = ablation_module.fit_lightgbm

    def spy_logreg(features: np.ndarray, target: np.ndarray, config: Any, params: Any, bounds: Any, order: Any) -> Any:
        fits.append(("logreg", tuple(order), len(features), set()))
        return fit_logreg(features, target, config, params, bounds, order)

    def spy_lightgbm(
        features: np.ndarray, relevance: np.ndarray, group: np.ndarray, config: Any, params: Any, order: Any
    ) -> Any:
        fits.append(("lightgbm", tuple(order), len(features), set(group.tolist())))
        return fit_lightgbm(features, relevance, group, config, params, order)

    monkeypatch.setattr(ablation_module, "fit_logreg", spy_logreg)
    monkeypatch.setattr(ablation_module, "fit_lightgbm", spy_lightgbm)

    run_trained_evaluate(project, small_params(), "full")

    n_folds = small_params().n_splits
    full_order = tuple(default_schema().feature_order)
    for name in ("logreg", "lightgbm"):
        mine = [item for item in fits if item[0] == name]
        assert len(mine) == n_folds * (1 + len(FACTOR_GROUPS))  # modelo completo + 6 grupos, por pliegue
        orders = [item[1] for item in mine]
        assert orders[:n_folds] == [full_order] * n_folds
        for index, group in enumerate(FACTOR_GROUPS):
            chunk = orders[n_folds * (index + 1) : n_folds * (index + 2)]
            assert chunk == [tuple(f for f in full_order if f not in group.features)] * n_folds
        assert all(item[2] < cv_pairs for item in mine)  # siempre sin el pliegue retenido y sin test
        assert all(not (item[3] & test_students) for item in mine)


def test_segments_reproduce_t03_and_the_baseline_reference(project: ProjectPaths, first_run: dict[str, str]) -> None:
    del first_run
    ablation = _json(project, ABLATION_METRICS)
    trained = _json(project, TRAINED_METRICS)
    baseline = _json(project, "baseline_metrics.json")
    items = {item["id"]: item for item in ablation["segments"]["items"]}
    reference = {item["id"]: item for item in baseline["segments"]}

    total = items["total"]
    for name in ("logreg", "lightgbm"):
        assert total[name] == trained["test"][name]  # mismo total que T03, cifra por cifra
        assert total[f"diff_{name}_minus_baseline"] == trained["paired_difference_vs_baseline"][name]
    for segment_id, item in items.items():
        assert item["baseline"] == reference[segment_id]["baseline"], segment_id
    for kind in ("campus", "budget"):
        parts = [item for item in items.values() if item["kind"] == kind]
        assert sum(item["students"] for item in parts) == total["students"]
        for name in ("baseline", "logreg", "lightgbm"):
            for metric in METRIC_NAMES:
                weighted = sum(item[name][metric]["mean"] * item["students"] for item in parts) / total["students"]
                assert weighted == pytest.approx(total[name][metric]["mean"], abs=1e-12)

    verdict = ablation["hypothesis_budget_gt_3500"]
    assert verdict["verdict"] in ("confirmed", "refuted", "inconclusive")
    assert len(verdict["reasons"]) == 3
    report = (project.reports / ABLATION_REPORT).read_text(encoding="utf-8")
    for text in ("DATOS SINTÉTICOS", "Ablación por grupo", "Segmentos en test", "**Veredicto:", "`pets_compatible`"):
        assert text in report, text


def test_running_twice_does_not_change_any_file(project: ProjectPaths, first_run: dict[str, str]) -> None:
    run_trained_evaluate(project, small_params(), "full")

    assert _snapshot(project) == first_run


def test_t04_modules_never_name_the_hidden_label_files() -> None:
    # La lectura real se espía en test_trained_eval_stage.py::test_evaluate_never_reads_the_hidden_label_files,
    # que corre la etapa completa (T03 + T04).
    for module in T04_MODULES:
        source = module.read_text(encoding="utf-8")
        for forbidden in ("HIDDEN_", "labels_hidden", HIDDEN_UTILITY_PARQUET, HIDDEN_WEIGHTS_PARQUET):
            assert forbidden not in source, (module.name, forbidden)


def test_ci_profile_writes_only_in_models_ci_and_skips_lightgbm_ablation(
    project: ProjectPaths, first_run: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del first_run
    copy = ProjectPaths(root=tmp_path / "copy")
    shutil.copytree(project.root, copy.root)
    monkeypatch.setenv(PROFILE_ENV_VAR, "ci")
    run_stage("train", copy)
    versioned = (*OUTPUTS, "reports/baseline_metrics.json")
    before = _snapshot(copy, versioned)

    message = run_stage("evaluate", copy)

    assert "evaluate (ML-05 T04, perfil ci): " in message
    assert _snapshot(copy, versioned) == before  # reportes versionados intactos
    ci_dir = load_train_params().ci_profile.output_dir
    ablation = _json(copy, ABLATION_METRICS, ci_dir)
    assert ablation["profile"] == "ci"
    assert ablation["official"] is False
    assert ablation["ablation"]["models_ablated"] == ["logreg"]
    assert ablation["ablation"]["lightgbm"]["omitted"] is True
    assert len(ablation["ablation"]["logreg"]["groups"]) == len(FACTOR_GROUPS)
    assert "no oficial" in (copy.root / ci_dir / ABLATION_REPORT).read_text(encoding="utf-8")
