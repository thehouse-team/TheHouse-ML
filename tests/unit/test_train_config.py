"""ML-05 T02 (ML-05-R02): `train_params.json` 1.0.0 y su validación."""

from __future__ import annotations

import copy
import json
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from thehouse_ml.features import default_schema
from thehouse_ml.models.train_config import (
    DEFAULT_PROFILE,
    PROFILE_ENV_VAR,
    LightGBMConfig,
    LogRegConfig,
    TrainParamsError,
    load_train_params,
    parse_train_params,
    resolve_profile,
)

REPO = Path(__file__).resolve().parents[2]


def _raw() -> dict[str, Any]:
    text = resources.files("thehouse_ml.models").joinpath("train_params.json").read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


def test_packaged_params_match_the_spec_and_the_execute_order() -> None:
    params = load_train_params()

    assert params.version == "1.0.0"
    assert params.decision == "M-ML-038 (APROBADO)"
    assert params.seed == 20261124
    assert params.n_splits == 5
    assert params.cv_splits == ("train", "validation")
    assert params.stratify_by == "university_code"
    assert params.selection_metric == "ndcg@10"
    assert params.logreg_c == (0.01, 0.1, 1.0, 10.0)
    assert params.logreg_transforms == ("standard", "quantile_bins")
    assert params.n_bins == 5
    assert params.num_leaves == (15, 31)
    assert params.learning_rate == (0.05, 0.1)
    assert params.n_estimators == (100, 300)
    assert params.min_data_in_leaf == 50
    assert params.label_gain == (0.0, 1.0, 3.0, 7.0)
    assert set(params.monotone_decreasing) == {
        "price_to_budget",
        "distance_km",
        "effective_distance_km",
        "distance_ratio",
    }
    assert params.num_threads == 1
    assert params.deterministic is True


def test_every_parameter_has_a_documented_basis() -> None:
    params = load_train_params()
    raw = _raw()

    assert raw["metadata"]["synthetic"] is True
    assert set(params.bases) == set(raw["parameters"])
    for basis in params.bases.values():
        assert basis.startswith(("calibrado:", "SUPUESTO documentado"))


def test_grids_are_small_ordered_and_have_unique_ids() -> None:
    params = load_train_params()

    logreg = params.logreg_grid()
    lightgbm = params.lightgbm_grid()
    assert len(logreg) == 8
    assert len(lightgbm) == 8
    assert logreg[0] == LogRegConfig("standard", 0.01)  # la más simple primero
    assert lightgbm[0] == LightGBMConfig(15, 0.05, 100)
    assert len({item.id for item in logreg}) == 8
    assert len({item.id for item in lightgbm}) == 8


def test_monotone_constraints_follow_feature_order() -> None:
    params = load_train_params()
    order = default_schema().feature_order

    constraints = params.monotone_constraints(order)

    assert len(constraints) == 10
    for name, value in zip(order, constraints, strict=True):
        assert value == (-1 if name in params.monotone_decreasing else 0)
    params.check_schema(default_schema())


def test_unknown_monotone_feature_is_rejected_against_the_schema() -> None:
    params = load_train_params().with_values(monotone_decreasing=("rent",))

    with pytest.raises(TrainParamsError, match="rent"):
        params.check_schema(default_schema())


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("folds.splits", ["train", "test"], "test nunca"),
        ("folds.splits", ["train", "holdout"], "fuera de"),
        ("folds.stratify_by", "student_id", "university_code"),
        ("selection.metric", "map", "selection.metric"),
        ("logreg.C", [], "lista no vacía"),
        ("logreg.C", [1.0, 1.0], "repetidos"),
        ("logreg.C", [0.0], "> 0"),
        ("logreg.transforms", ["spline"], "fuera de"),
        ("logreg.solver", "saga", "lbfgs"),
        ("logreg.clip", "none", "training_range"),
        ("logreg.tol", 0, "logreg.tol"),
        ("lightgbm.num_leaves", [15.5], "enteros"),
        ("lightgbm.label_gain", [0, 1, 2, 3], "2\\^rel"),
        ("lightgbm.num_threads", 4, "un hilo"),
        ("lightgbm.deterministic", False, "deterministic"),
        ("seed", -1, "seed"),
        ("folds.n_splits", 1, "folds.n_splits"),
    ],
)
def test_invalid_values_are_rejected(name: str, value: object, message: str) -> None:
    raw = _raw()
    raw["parameters"][name]["value"] = value

    with pytest.raises(TrainParamsError, match=message):
        parse_train_params(raw)


def test_structure_errors_are_reported() -> None:
    raw = _raw()
    broken = copy.deepcopy(raw)
    broken["parameters"]["seed"] = {"value": 1}
    broken["parameters"]["extra"] = {"value": 1, "basis": "SUPUESTO documentado"}
    broken["parameters"]["logreg.C"]["basis"] = "porque sí"

    with pytest.raises(TrainParamsError) as error:
        parse_train_params(broken)

    text = str(error.value)
    assert "seed: se esperaba {value, basis}" in text
    assert "desconocidos: ['extra']" in text
    assert "logreg.C.basis" in text
    with pytest.raises(TrainParamsError, match="metadata"):
        parse_train_params([])
    with pytest.raises(TrainParamsError, match="parameters"):
        parse_train_params({"metadata": {"synthetic": True}})
    with pytest.raises(TrainParamsError, match="synthetic"):
        parse_train_params({**raw, "metadata": {"synthetic": False}})


def test_load_from_an_explicit_path(tmp_path: Any) -> None:
    path = tmp_path / "train_params.json"
    path.write_text(json.dumps(_raw()), encoding="utf-8")

    assert load_train_params(path) == load_train_params()


# ---------------------------------------------------------------------------
# Perfiles (M-ML-038, T02-H1)
# ---------------------------------------------------------------------------


def test_default_profile_is_the_full_official_grid() -> None:
    params = load_train_params()

    assert resolve_profile({}) == DEFAULT_PROFILE == "full"
    assert resolve_profile({PROFILE_ENV_VAR: "  "}) == "full"
    assert params.for_profile("full") is params
    assert len(params.for_profile("full").logreg_grid()) == 8
    assert len(params.for_profile("full").lightgbm_grid()) == 8
    assert params.for_profile("full").n_splits == 5


def test_ci_profile_uses_one_combination_per_model_and_two_folds() -> None:
    params = load_train_params()

    ci = params.for_profile("ci")

    assert resolve_profile({PROFILE_ENV_VAR: "ci"}) == "ci"
    assert ci.n_splits == 2
    assert ci.logreg_grid() == [LogRegConfig("standard", 0.01)]
    assert ci.lightgbm_grid() == [LightGBMConfig(15, 0.05, 300)]
    assert ci.seed == params.seed  # mismas reglas que el perfil completo
    assert ci.monotone_decreasing == params.monotone_decreasing
    assert params.ci_profile.output_dir == "models/ci"


def test_ci_profile_combinations_are_the_ones_chosen_by_the_official_run() -> None:
    metrics = json.loads((REPO / "reports" / "train_cv_metrics.json").read_text(encoding="utf-8"))
    params = load_train_params()

    assert metrics["profile"] == "full"
    assert params.ci_profile.logreg.id == metrics["selected"]["logreg"]["config_id"]
    assert params.ci_profile.lightgbm.id == metrics["selected"]["lightgbm"]["config_id"]


def test_unknown_profiles_are_rejected() -> None:
    with pytest.raises(TrainParamsError, match=PROFILE_ENV_VAR):
        resolve_profile({PROFILE_ENV_VAR: "fast"})
    with pytest.raises(TrainParamsError, match="desconocido"):
        load_train_params().for_profile("fast")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"logreg": {"transform": "standard", "C": 3.0}}, "no está en la rejilla"),
        ({"lightgbm": {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 300}}, "no está en la rejilla"),
        ({"n_splits": 9}, "no puede superar"),
        ({"n_splits": 1}, "profiles.ci.n_splits"),
        ({"output_dir": "reports"}, "subcarpeta relativa de models/"),
        ({"output_dir": "models"}, "subcarpeta relativa de models/"),
        ({"output_dir": "models/../reports"}, "subcarpeta relativa de models/"),
        ({"output_dir": "/models/ci"}, "subcarpeta relativa de models/"),
        ({"logreg": {"transform": "standard"}}, "profiles.ci.logreg"),
        ({"lightgbm": {"num_leaves": 15}}, "profiles.ci.lightgbm"),
        ({"extra": 1}, "profiles.ci: se esperaba"),
    ],
)
def test_invalid_ci_profiles_are_rejected(change: dict[str, object], message: str) -> None:
    raw = _raw()
    raw["parameters"]["profiles.ci"]["value"] = {**raw["parameters"]["profiles.ci"]["value"], **change}

    with pytest.raises(TrainParamsError, match=message):
        parse_train_params(raw)
