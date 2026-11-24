"""Parámetros de la etapa `train` de ML-05 (`train_params.json` 1.0.0; ML-05 T02, ML-05-R02).

Lee y valida los pliegues, la rejilla de hiperparámetros y los ajustes de
reproducibilidad. Cada parámetro lleva `value` y `basis` («calibrado: …» o
«SUPUESTO documentado»), como `baseline_eval_params.json` (M-ML-034). Las
rejillas se recorren en el orden del archivo (de la configuración más simple a
la más compleja), que es también el último criterio de desempate.

Perfiles (M-ML-038, T02-H1): `full` (por defecto) es la corrida oficial con la
rejilla completa; `ci` (variable de entorno `THEHOUSE_ML_TRAIN_PROFILE=ci`) es
una prueba de humo con 1 combinación por modelo y 2 pliegues que escribe en
`profiles.ci.output_dir` (fuera de git) y nunca en los reportes versionados.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path, PurePosixPath

from thehouse_ml.features.loader import FeatureSchema

TRAIN_PARAMS_RESOURCE = "train_params.json"
PARAMS_PACKAGE = "thehouse_ml.models"
BASIS_PREFIXES = ("calibrado:", "SUPUESTO documentado")
LOGREG_TRANSFORMS = ("standard", "quantile_bins")
SELECTION_METRICS = ("ndcg@10",)
KNOWN_SPLITS = ("train", "validation", "test")
PROFILE_ENV_VAR = "THEHOUSE_ML_TRAIN_PROFILE"
DEFAULT_PROFILE = "full"
PROFILES = (DEFAULT_PROFILE, "ci")
EXPECTED = (
    "profiles.ci",
    "seed",
    "folds.n_splits",
    "folds.splits",
    "folds.stratify_by",
    "selection.metric",
    "logreg.C",
    "logreg.transforms",
    "logreg.quantile_bins.n_bins",
    "logreg.solver",
    "logreg.max_iter",
    "logreg.tol",
    "logreg.clip",
    "lightgbm.num_leaves",
    "lightgbm.learning_rate",
    "lightgbm.n_estimators",
    "lightgbm.min_data_in_leaf",
    "lightgbm.label_gain",
    "lightgbm.monotone_decreasing",
    "lightgbm.num_threads",
    "lightgbm.deterministic",
)


class TrainParamsError(ValueError):
    """`train_params.json` inválido."""


@dataclass(frozen=True)
class LogRegConfig:
    """Una configuración de la rejilla de la regresión logística."""

    transform: str
    c: float

    @property
    def id(self) -> str:
        return f"logreg:{self.transform}:C={self.c:g}"

    def as_dict(self) -> dict[str, object]:
        return {"transform": self.transform, "C": self.c}


@dataclass(frozen=True)
class LightGBMConfig:
    """Una configuración de la rejilla de LightGBM LambdaRank."""

    num_leaves: int
    learning_rate: float
    n_estimators: int

    @property
    def id(self) -> str:
        return f"lightgbm:leaves={self.num_leaves}:lr={self.learning_rate:g}:trees={self.n_estimators}"

    def as_dict(self) -> dict[str, object]:
        return {"num_leaves": self.num_leaves, "learning_rate": self.learning_rate, "n_estimators": self.n_estimators}


@dataclass(frozen=True)
class CIProfile:
    """Perfil `ci`: 1 combinación por modelo, pocos pliegues y salidas fuera de git."""

    n_splits: int
    logreg: LogRegConfig
    lightgbm: LightGBMConfig
    output_dir: str


@dataclass(frozen=True)
class TrainParams:
    """Valores de `train_params.json`."""

    version: str
    decision: str
    seed: int
    n_splits: int
    cv_splits: tuple[str, ...]
    stratify_by: str
    selection_metric: str
    logreg_c: tuple[float, ...]
    logreg_transforms: tuple[str, ...]
    n_bins: int
    solver: str
    max_iter: int
    tol: float
    clip: str
    num_leaves: tuple[int, ...]
    learning_rate: tuple[float, ...]
    n_estimators: tuple[int, ...]
    min_data_in_leaf: int
    label_gain: tuple[float, ...]
    monotone_decreasing: tuple[str, ...]
    num_threads: int
    deterministic: bool
    ci_profile: CIProfile
    bases: Mapping[str, str]

    def logreg_grid(self) -> list[LogRegConfig]:
        """Transformación por C, en el orden del archivo (más simple primero)."""
        return [LogRegConfig(transform, c) for transform in self.logreg_transforms for c in self.logreg_c]

    def lightgbm_grid(self) -> list[LightGBMConfig]:
        """num_leaves por learning_rate por n_estimators, en el orden del archivo."""
        return [
            LightGBMConfig(leaves, rate, trees)
            for leaves in self.num_leaves
            for rate in self.learning_rate
            for trees in self.n_estimators
        ]

    def monotone_constraints(self, feature_order: tuple[str, ...]) -> list[int]:
        """-1 en las features restringidas y 0 en el resto, en el orden de `feature_order`."""
        return [-1 if name in self.monotone_decreasing else 0 for name in feature_order]

    def with_values(self, **changes: object) -> TrainParams:
        """Copia con otros valores (para pruebas con una rejilla pequeña)."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def for_profile(self, profile: str) -> TrainParams:
        """`full`: los mismos parámetros; `ci`: la rejilla reducida a las combinaciones del perfil y sus pliegues."""
        if profile == DEFAULT_PROFILE:
            return self
        if profile != "ci":
            raise TrainParamsError(f"perfil de entrenamiento desconocido {profile!r}; válidos: {list(PROFILES)}")
        ci = self.ci_profile
        return self.with_values(
            n_splits=ci.n_splits,
            logreg_transforms=(ci.logreg.transform,),
            logreg_c=(ci.logreg.c,),
            num_leaves=(ci.lightgbm.num_leaves,),
            learning_rate=(ci.lightgbm.learning_rate,),
            n_estimators=(ci.lightgbm.n_estimators,),
        )

    def check_schema(self, schema: FeatureSchema) -> None:
        """Las features restringidas deben existir en `feature_order`."""
        unknown = [name for name in self.monotone_decreasing if name not in schema.feature_order]
        if unknown:
            raise TrainParamsError(f"lightgbm.monotone_decreasing: features fuera de feature_order {unknown}")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _int(value: object, name: str, problems: list[str], minimum: int) -> int:
    if not _is_int(value) or int(str(value)) < minimum:
        problems.append(f"{name}: se esperaba un entero ≥ {minimum}, llegó {value!r}")
        return minimum
    return int(str(value))


def _positive_list(value: object, name: str, problems: list[str], *, integer: bool) -> tuple[float, ...]:
    check = _is_int if integer else _is_number
    if not isinstance(value, list) or not value or not all(check(item) and float(item) > 0 for item in value):
        kind = "enteros" if integer else "números"
        problems.append(f"{name}: se esperaba una lista no vacía de {kind} > 0, llegó {value!r}")
        return (1.0,)
    if len(set(value)) != len(value):
        problems.append(f"{name}: valores repetidos {value!r}")
    return tuple(float(item) for item in value)


def _str_list(value: object, name: str, problems: list[str], allowed: tuple[str, ...] | None = None) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        problems.append(f"{name}: se esperaba una lista no vacía de textos, llegó {value!r}")
        return ()
    if len(set(value)) != len(value):
        problems.append(f"{name}: valores repetidos {value!r}")
    if allowed is not None and not set(value) <= set(allowed):
        problems.append(f"{name}: valores fuera de {list(allowed)}: {sorted(set(value) - set(allowed))}")
    return tuple(value)


def _ci_profile(value: object, problems: list[str]) -> CIProfile:
    """Valida `profiles.ci`: sus combinaciones deben existir en la rejilla (se comprueba después)."""
    fallback = CIProfile(2, LogRegConfig("standard", 1.0), LightGBMConfig(1, 1.0, 1), "models/ci")
    expected = {"n_splits", "logreg", "lightgbm", "output_dir"}
    if not isinstance(value, dict) or set(value) != expected:
        problems.append(f"profiles.ci: se esperaba {{{', '.join(sorted(expected))}}}, llegó {value!r}")
        return fallback
    logreg, gbm, output_dir = value["logreg"], value["lightgbm"], value["output_dir"]
    n_splits = _int(value["n_splits"], "profiles.ci.n_splits", problems, 2)
    if not isinstance(logreg, dict) or set(logreg) != {"transform", "C"} or not _is_number(logreg.get("C")):
        problems.append(f"profiles.ci.logreg: se esperaba {{transform, C}}, llegó {logreg!r}")
        return fallback
    keys = {"num_leaves", "learning_rate", "n_estimators"}
    if (
        not isinstance(gbm, dict)
        or set(gbm) != keys
        or not _is_int(gbm["num_leaves"])
        or not _is_int(gbm["n_estimators"])
        or not _is_number(gbm["learning_rate"])
    ):
        problems.append(f"profiles.ci.lightgbm: se esperaba {{{', '.join(sorted(keys))}}}, llegó {gbm!r}")
        return fallback
    path = PurePosixPath(str(output_dir))
    if (
        not isinstance(output_dir, str)
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in output_dir
        or len(path.parts) < 2
        or path.parts[0] != "models"
    ):
        problems.append(
            f"profiles.ci.output_dir: debe ser una subcarpeta relativa de models/ (fuera de git), llegó {output_dir!r}"
        )
    return CIProfile(
        n_splits=n_splits,
        logreg=LogRegConfig(str(logreg["transform"]), float(logreg["C"])),
        lightgbm=LightGBMConfig(int(gbm["num_leaves"]), float(gbm["learning_rate"]), int(gbm["n_estimators"])),
        output_dir=str(output_dir),
    )


def resolve_profile(environ: Mapping[str, str] | None = None) -> str:
    """Perfil de `THEHOUSE_ML_TRAIN_PROFILE`; `full` si no está o está vacía."""
    env = os.environ if environ is None else environ
    profile = env.get(PROFILE_ENV_VAR, "").strip() or DEFAULT_PROFILE
    if profile not in PROFILES:
        raise TrainParamsError(f"{PROFILE_ENV_VAR}={profile!r}: perfiles válidos {list(PROFILES)}")
    return profile


def parse_train_params(data: object) -> TrainParams:
    """Valida el JSON de parámetros; `TrainParamsError` con todos los problemas."""
    if not isinstance(data, dict) or not isinstance(data.get("metadata"), dict):
        raise TrainParamsError(f"{TRAIN_PARAMS_RESOURCE}: se esperaba {{metadata, parameters}}")
    parameters = data.get("parameters")
    if not isinstance(parameters, dict):
        raise TrainParamsError(f"{TRAIN_PARAMS_RESOURCE}: falta parameters")
    if data["metadata"].get("synthetic") is not True:
        raise TrainParamsError(f"{TRAIN_PARAMS_RESOURCE}: metadata.synthetic debe ser true")
    problems: list[str] = []
    values: dict[str, object] = {}
    bases: dict[str, str] = {}
    for name in EXPECTED:
        entry = parameters.get(name)
        if not isinstance(entry, dict) or set(entry) != {"value", "basis"}:
            problems.append(f"{name}: se esperaba {{value, basis}}")
            continue
        basis = entry["basis"]
        if not isinstance(basis, str) or not basis.startswith(BASIS_PREFIXES):
            problems.append(f"{name}.basis: debe empezar con «calibrado:» o «SUPUESTO documentado»")
        else:
            bases[name] = basis
        values[name] = entry["value"]
    extra = sorted(set(parameters) - set(EXPECTED))
    if extra:
        problems.append(f"parámetros desconocidos: {extra}")
    if problems:
        raise TrainParamsError(f"{TRAIN_PARAMS_RESOURCE} inválido:\n- " + "\n- ".join(problems))

    cv_splits = _str_list(values["folds.splits"], "folds.splits", problems, KNOWN_SPLITS)
    if "test" in cv_splits:
        problems.append("folds.splits: la partición test nunca entra en los pliegues (spec ML-05 §6.2)")
    stratify_by = values["folds.stratify_by"]
    if stratify_by != "university_code":
        problems.append(f"folds.stratify_by: se esperaba university_code, llegó {stratify_by!r}")
    metric = values["selection.metric"]
    if metric not in SELECTION_METRICS:
        problems.append(f"selection.metric: se esperaba {list(SELECTION_METRICS)}, llegó {metric!r}")
    solver = values["logreg.solver"]
    if solver != "lbfgs":
        problems.append(f"logreg.solver: se esperaba lbfgs, llegó {solver!r}")
    clip = values["logreg.clip"]
    if clip != "training_range":
        problems.append(f"logreg.clip: se esperaba training_range, llegó {clip!r}")
    tol = values["logreg.tol"]
    if not _is_number(tol) or float(str(tol)) <= 0:
        problems.append(f"logreg.tol: se esperaba un número > 0, llegó {tol!r}")
        tol = 1e-4
    label_gain = values["lightgbm.label_gain"]
    if (
        not isinstance(label_gain, list)
        or not all(_is_number(item) for item in label_gain)
        or [float(item) for item in label_gain] != [float(2**grade - 1) for grade in range(len(label_gain))]
    ):
        problems.append(f"lightgbm.label_gain: se esperaba 2^rel - 1 (0, 1, 3, 7), llegó {label_gain!r}")
        label_gain = [0, 1, 3, 7]
    deterministic = values["lightgbm.deterministic"]
    if deterministic is not True:
        problems.append("lightgbm.deterministic: debe ser true (spec ML-05 §6.3)")
    num_threads = _int(values["lightgbm.num_threads"], "lightgbm.num_threads", problems, 1)
    if num_threads != 1:
        problems.append("lightgbm.num_threads: debe ser 1 (spec ML-05 §6.3: un hilo)")

    params = TrainParams(
        version=str(data["metadata"].get("params_version", "")),
        decision=str(data["metadata"].get("decision", "")),
        seed=_int(values["seed"], "seed", problems, 0),
        n_splits=_int(values["folds.n_splits"], "folds.n_splits", problems, 2),
        cv_splits=cv_splits,
        stratify_by=str(stratify_by),
        selection_metric=str(metric),
        logreg_c=_positive_list(values["logreg.C"], "logreg.C", problems, integer=False),
        logreg_transforms=_str_list(values["logreg.transforms"], "logreg.transforms", problems, LOGREG_TRANSFORMS),
        n_bins=_int(values["logreg.quantile_bins.n_bins"], "logreg.quantile_bins.n_bins", problems, 2),
        solver=str(solver),
        max_iter=_int(values["logreg.max_iter"], "logreg.max_iter", problems, 1),
        tol=float(str(tol)),
        clip=str(clip),
        num_leaves=tuple(
            int(item)
            for item in _positive_list(values["lightgbm.num_leaves"], "lightgbm.num_leaves", problems, integer=True)
        ),
        learning_rate=_positive_list(
            values["lightgbm.learning_rate"], "lightgbm.learning_rate", problems, integer=False
        ),
        n_estimators=tuple(
            int(item)
            for item in _positive_list(values["lightgbm.n_estimators"], "lightgbm.n_estimators", problems, integer=True)
        ),
        min_data_in_leaf=_int(values["lightgbm.min_data_in_leaf"], "lightgbm.min_data_in_leaf", problems, 1),
        label_gain=tuple(float(item) for item in label_gain),
        monotone_decreasing=_str_list(values["lightgbm.monotone_decreasing"], "lightgbm.monotone_decreasing", problems),
        num_threads=num_threads,
        deterministic=deterministic is True,
        ci_profile=_ci_profile(values["profiles.ci"], problems),
        bases=bases,
    )
    if params.ci_profile.logreg not in params.logreg_grid():
        problems.append(f"profiles.ci.logreg: {params.ci_profile.logreg.id} no está en la rejilla")
    if params.ci_profile.lightgbm not in params.lightgbm_grid():
        problems.append(f"profiles.ci.lightgbm: {params.ci_profile.lightgbm.id} no está en la rejilla")
    if params.ci_profile.n_splits > params.n_splits:
        problems.append("profiles.ci.n_splits: no puede superar folds.n_splits")
    if problems:
        raise TrainParamsError(f"{TRAIN_PARAMS_RESOURCE} inválido:\n- " + "\n- ".join(problems))
    return params


def load_train_params(path: Path | None = None) -> TrainParams:
    """Carga `train_params.json` (empaquetado) o el archivo indicado."""
    if path is None:
        text = resources.files(PARAMS_PACKAGE).joinpath(TRAIN_PARAMS_RESOURCE).read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    return parse_train_params(json.loads(text))


__all__ = [
    "DEFAULT_PROFILE",
    "PROFILES",
    "PROFILE_ENV_VAR",
    "TRAIN_PARAMS_RESOURCE",
    "CIProfile",
    "LightGBMConfig",
    "LogRegConfig",
    "TrainParams",
    "TrainParamsError",
    "load_train_params",
    "parse_train_params",
    "resolve_profile",
]
