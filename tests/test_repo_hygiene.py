"""ML-01 R03: datos y modelos nunca entran a git (constitution §6 y §7).

ML-06 T05 (spec ML-06 §6.6): el paquete de entrega `dist/` tampoco entra a git (va en el release).

ML-05 T01 (M-ML-037): el código de `src/thehouse_ml/` no usa pickle (constitution §13.1).
"""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = ("data/raw/", "data/processed/", "models/", "dist/")
FORBIDDEN_SUFFIXES = (".pkl", ".pickle", ".joblib", ".onnx")


def forbidden_files(tracked: list[str]) -> list[str]:
    """Archivos versionados que la constitution prohíbe."""
    return [
        path
        for path in tracked
        if path.replace("\\", "/").startswith(FORBIDDEN_PREFIXES) or path.lower().endswith(FORBIDDEN_SUFFIXES)
    ]


def tracked_files() -> list[str]:
    git = shutil.which("git")
    if git is None or not (REPO / ".git").exists():
        pytest.skip("Sin git o fuera de un repositorio")
    result = subprocess.run([git, "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/listings.csv.gz",
        "data/processed/rooms.parquet",
        "models/model.onnx",
        "notas/modelo.pkl",
        "x.joblib",
        "dist/model-v1.0.0/SHA256SUMS",
    ],
)
def test_detector_flags_forbidden_paths(path: str) -> None:
    assert forbidden_files([path]) == [path]


@pytest.mark.parametrize("path", ["data/manifests/censo2020.json", "reports/data_card.md", "src/thehouse_ml/cli.py"])
def test_detector_allows_versioned_paths(path: str) -> None:
    assert forbidden_files([path]) == []


def test_gitignore_excludes_data_and_models() -> None:
    lines = {line.strip() for line in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()}

    # `/models/` y `/dist/` anclados a la raíz: solo las carpetas de artefactos (M-ML-030, H1; spec ML-06 §6.6).
    assert {"data/raw/", "data/processed/", "/models/", "/dist/"} <= lines


def _git_ignored(paths: list[str]) -> set[str]:
    git = shutil.which("git")
    if git is None or not (REPO / ".git").exists():
        pytest.skip("Sin git o fuera de un repositorio")
    result = subprocess.run(
        [git, "check-ignore", "--no-index", *paths], cwd=REPO, capture_output=True, text=True, check=False
    )
    return {line.strip().replace("\\", "/") for line in result.stdout.splitlines()}


@pytest.mark.parametrize(
    "path",
    [
        "models/model.onnx",
        "models/baseline/model.onnx",
        "data/raw/censo.zip",
        "data/processed/pairs.parquet",
        # Paquete de entrega de ML-06 (spec §6.6): los 7 archivos y el aviso de no publicable.
        "dist/model-v1.0.0/model.onnx",
        "dist/model-v1.0.0/feature_schema.json",
        "dist/model-v1.0.0/golden_cases.json",
        "dist/model-v1.0.0/baseline_golden_cases.json",
        "dist/model-v1.0.0/model_card.md",
        "dist/model-v1.0.0/metrics.json",
        "dist/model-v1.0.0/SHA256SUMS",
        "dist/model-v1.0.0/NO_PUBLICABLE.txt",
        "dist/ci/model.onnx",
    ],
)
def test_artifacts_and_data_are_ignored_by_git(path: str) -> None:
    assert _git_ignored([path]) == {path}


@pytest.mark.parametrize(
    "path",
    [
        "src/thehouse_ml/models/baseline_params.json",
        "src/thehouse_ml/models/baseline_config.py",
        "src/thehouse_ml/models/__init__.py",
        # `/dist/` está anclado: el código de exportación no se ignora.
        "src/thehouse_ml/export/package.py",
    ],
)
def test_the_models_package_of_the_code_is_not_ignored(path: str) -> None:
    assert _git_ignored([path]) == set()


def test_onnx_files_are_flagged_wherever_they_are_tracked() -> None:
    # Un .onnx versionado fuera de /models/ también es un artefacto prohibido (constitution §13: va en el release).
    assert forbidden_files(["src/thehouse_ml/models/model.onnx", "reports/model.onnx"]) == [
        "src/thehouse_ml/models/model.onnx",
        "reports/model.onnx",
    ]


def test_no_data_or_model_files_are_tracked() -> None:
    assert forbidden_files(tracked_files()) == []


# ---------------------------------------------------------------------------
# Sin pickle en el código (constitution §13.1, M-ML-037): los modelos se guardan
# en JSON o en el formato de texto nativo de LightGBM; ML-06 produce el ONNX.
# ---------------------------------------------------------------------------

PICKLE_MODULES = frozenset({"pickle", "cloudpickle", "dill"})
JOBLIB_FORBIDDEN = frozenset({"dump", "load"})
# pandas serializa con pickle en `DataFrame.to_pickle`/`Series.to_pickle` y `pandas.read_pickle`.
PANDAS_PICKLE = frozenset({"to_pickle", "read_pickle"})
SRC = REPO / "src" / "thehouse_ml"


def _root(module: str) -> str:
    return module.split(".")[0]


def pickle_uses(source: str) -> list[str]:
    """Usos prohibidos en un módulo: importar pickle, cloudpickle o dill (o submódulos), joblib.dump/joblib.load
    y los métodos de pandas que usan pickle (`read_pickle`, `to_pickle`, sobre cualquier objeto o importados)."""
    found: list[str] = []
    joblib_aliases: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _root(alias.name) in PICKLE_MODULES:
                    found.append(f"línea {node.lineno}: import {alias.name}")
                elif alias.name == "joblib":
                    joblib_aliases.add(alias.asname or "joblib")
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            if _root(node.module) in PICKLE_MODULES:
                found.append(f"línea {node.lineno}: from {node.module} import …")
            elif _root(node.module) == "joblib":
                found += [
                    f"línea {node.lineno}: from {node.module} import {alias.name}"
                    for alias in node.names
                    if alias.name in JOBLIB_FORBIDDEN
                ]
            elif _root(node.module) == "pandas":
                found += [
                    f"línea {node.lineno}: from {node.module} import {alias.name}"
                    for alias in node.names
                    if alias.name in PANDAS_PICKLE
                ]
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            # __import__("pickle") o importlib.import_module("pickle")
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            value = node.args[0].value
            if name in {"__import__", "import_module"} and isinstance(value, str) and _root(value) in PICKLE_MODULES:
                found.append(f"línea {node.lineno}: {name}({value!r})")
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in JOBLIB_FORBIDDEN
            and isinstance(node.value, ast.Name)
            and node.value.id in joblib_aliases
        ):
            found.append(f"línea {node.lineno}: {node.value.id}.{node.attr}")
        elif isinstance(node, ast.Attribute) and node.attr in PANDAS_PICKLE:
            # pd.read_pickle, pandas.read_pickle, df.to_pickle, frame["x"].to_pickle, etc.
            found.append(f"línea {node.lineno}: .{node.attr}")
    return found


@pytest.mark.parametrize(
    "source",
    [
        "import pickle",
        "import pickle as pk",
        "import cloudpickle",
        "import dill.settings",
        "from pickle import dumps",
        "from cloudpickle import dump",
        "import os, dill",
        "import joblib\njoblib.dump(model, 'm.joblib')",
        "import joblib as jl\nmodel = jl.load('m.joblib')",
        "from joblib import dump",
        "from joblib import load as restore",
        "__import__('pickle')",
        "import importlib\nimportlib.import_module('cloudpickle')",
        "def save(m):\n    import pickle\n    return pickle.dumps(m)",
        "import pandas\nframe = pandas.read_pickle('pairs.pkl')",
        "import pandas as pd\nframe = pd.read_pickle('pairs.pkl')",
        "frame.to_pickle('pairs.pkl')",
        "frame['relevance'].to_pickle('relevance.pkl')",
        "from pandas import read_pickle",
        "from pandas import read_pickle as rp",
        "import pandas as pd\nreader = pd.read_pickle",
    ],
)
def test_pickle_detector_flags_forbidden_uses(source: str) -> None:
    assert pickle_uses(source)


@pytest.mark.parametrize(
    "source",
    [
        "import json\njson.dumps({})",
        "import joblib\njoblib.Parallel(n_jobs=1)",
        "from joblib import Parallel",
        "booster.save_model('model.txt')",
        "text = 'sin pickle (constitución §13.1)'",
        "import pandas as pd\npd.read_parquet('pairs.parquet')",
        "from .pickle_free import load",
        "frame.to_parquet('pairs.parquet', index=False)",
        "from pandas import read_parquet",
        "text = 'nunca usar to_pickle ni read_pickle'",
    ],
)
def test_pickle_detector_allows_safe_code(source: str) -> None:
    assert pickle_uses(source) == []


def test_no_module_of_the_package_uses_pickle() -> None:
    modules = sorted(SRC.rglob("*.py"))
    assert modules, "no se encontró src/thehouse_ml/"
    problems = {
        path.relative_to(REPO).as_posix(): uses
        for path in modules
        if (uses := pickle_uses(path.read_text(encoding="utf-8")))
    }
    assert problems == {}
