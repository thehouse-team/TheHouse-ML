"""ML-05 T01 (ML-05-R01): scikit-learn y LightGBM instalados desde uv.lock y funcionales en este sistema.

Corre en los dos jobs del CI (ubuntu y windows): si falta una biblioteca del sistema
(p. ej. `libgomp.so.1`, que el wheel de LightGBM para Linux no incluye), esta prueba
falla al importar o entrenar, no a mitad de T02. Sin red ni archivos de data/.
"""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def _locked_versions() -> dict[str, str]:
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    return {item["name"]: item["version"] for item in lock["package"] if "version" in item}


def test_installed_versions_are_the_locked_ones() -> None:
    locked = _locked_versions()
    for name in ("scikit-learn", "lightgbm", "scipy", "joblib", "threadpoolctl"):
        assert importlib.metadata.version(name) == locked[name], name


def test_dependencies_are_declared_in_pyproject() -> None:
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = {spec.split(">")[0].split("=")[0].split("<")[0].strip() for spec in project["dependencies"]}
    assert {"scikit-learn", "lightgbm"} <= names


def _toy() -> tuple[pd.DataFrame, list[int], list[int], list[str]]:
    rows, relevance, groups = [], [], []
    for student in range(12):
        for room in range(6):
            price = (room + student % 3) / 8
            rows.append({"price_to_budget": price, "services_coverage": ((room * 7 + student) % 5) / 4})
            relevance.append(3 if room == 0 else (1 if room == 1 else 0))
            groups.append(f"S{student:02d}")
    sizes = [6] * 12
    return pd.DataFrame(rows), relevance, sizes, groups


def test_lightgbm_lambdarank_trains_deterministically_with_monotone_constraints() -> None:
    import lightgbm

    features, relevance, sizes, _groups = _toy()
    params = {
        "objective": "lambdarank",
        "n_estimators": 10,
        "num_leaves": 4,
        "min_child_samples": 2,
        "monotone_constraints": [-1, 0],
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": 1,
        "random_state": 20261124,
        "verbose": -1,
    }
    first = lightgbm.LGBMRanker(**params).fit(features, relevance, group=sizes)
    second = lightgbm.LGBMRanker(**params).fit(features, relevance, group=sizes)
    assert first.booster_.model_to_string() == second.booster_.model_to_string()  # texto nativo, sin pickle
    assert list(first.predict(features)) == list(second.predict(features))


def test_sklearn_logistic_regression_and_grouped_stratified_folds() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features, relevance, _sizes, groups = _toy()
    target = [int(value >= 2) for value in relevance]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0)).fit(features, target)
    assert model.predict_proba(features).shape == (len(target), 2)

    campus = [int(group[1:]) % 2 for group in groups]
    folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261124)
    seen: set[str] = set()
    for _train, held in folds.split(features, campus, groups):
        held_groups = {groups[index] for index in held}
        assert not held_groups & seen  # ningún estudiante en dos pliegues
        seen |= held_groups
    assert seen == set(groups)
