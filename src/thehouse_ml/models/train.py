"""Etapa `train` de ML-05: validación cruzada y entrenamiento reproducible (ML-05 T02, ML-05-R02).

Pasos (spec ML-05 §6.2 y §6.3, decisiones 2, 3 y 5 de M-ML-036):

1. `load_training_data` (T01) entrega X (las 10 features de `feature_order`),
   la relevancia 0-3, el estudiante, el cuarto y la partición; aquí se toman
   **solo** los estudiantes de train + validación. La partición test no entra en
   los pliegues, la búsqueda ni el modelo final (se mide una vez en T03).
2. `assign_folds`: 5 pliegues con `StratifiedGroupKFold(shuffle=True)` sobre
   una fila por estudiante (grupo = `student_id`, estrato = campus).
3. Para cada configuración de la rejilla de `train_params.json` y cada pliegue,
   se ajusta con los otros 4 pliegues y se mide el NDCG@10 medio de los
   estudiantes retenidos con `thehouse_ml.evaluation.ranking` (puntaje
   descendente, desempate `room_id` ascendente, como el baseline).
4. Se elige por el NDCG@10 medio entre pliegues (desempate: menor desviación y
   después la primera de la rejilla) y se reentrena cada familia con todo
   train + validación.
5. El baseline (`score_features` y `rank` de ML-04, sin entrenar) se mide en los
   mismos estudiantes retenidos de cada pliegue para la comparación de T03.

Modelos:

- **Regresión logística** pointwise, `y = 1` si `relevance ≥ 2` (umbral de
  `baseline_eval_params.json`), con recorte al `training_range` y un pipeline de
  scikit-learn con solo `StandardScaler` o `KBinsDiscretizer` (convertidores en
  skl2onnx) y `LogisticRegression`. Se guarda **como JSON** (recorte,
  transformación, coeficientes e intercepto) y `LogRegModel` lo reconstruye.
- **LightGBM LambdaRank** con las 4 restricciones monotónicas decrecientes, un
  hilo y `deterministic = true`, guardado en su **texto nativo**
  (`Booster.save_model`).

Sin pickle (constitución §13.1). Nunca se leen la utilidad ni los pesos ocultos:
las únicas lecturas son las de `training_data.TRAINING_INPUTS`.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from thehouse_ml.evaluation.ranking import ndcg_at_k
from thehouse_ml.features import default_schema
from thehouse_ml.features.loader import FeatureSchema
from thehouse_ml.models.baseline_config import BaselineConfig, default_config
from thehouse_ml.models.baseline_report import load_eval_params
from thehouse_ml.models.baseline_score import ScoredRoom, ScoreResult, rank, score_features
from thehouse_ml.models.train_config import (
    DEFAULT_PROFILE,
    LightGBMConfig,
    LogRegConfig,
    TrainParams,
    load_train_params,
    resolve_profile,
)
from thehouse_ml.models.training_data import TRAINING_INPUTS, TrainingData, load_training_data
from thehouse_ml.paths import ProjectPaths

LOGREG_MODEL = "logreg.json"
LIGHTGBM_MODEL = "lightgbm_lambdarank.txt"
TRAIN_MANIFEST = "train_manifest.json"
TRAINING_REPORT = "training_report.md"
TRAIN_CV_METRICS = "train_cv_metrics.json"
LOGREG_FORMAT = "thehouse-ml/logreg-json/1"
NDCG_K = 10
MONOTONE_SAMPLE_ROWS = 200
MONOTONE_STEPS = 21
TRAIN_SKIP_MESSAGE = (
    "train: omitido; faltan " + ", ".join(f"data/processed/{name}" for name in TRAINING_INPUTS) + " (ejecute `prepare`)"
)
# Avisos esperados de KBinsDiscretizer con features binarias o constantes (los intervalos vacíos se quitan).
_EXPECTED_WARNINGS = ("Bins whose width are too small", "is constant and will be replaced with 0")


class TrainError(RuntimeError):
    """Datos o resultados del entrenamiento inválidos (pliegues, grupos o modelo)."""


# ---------------------------------------------------------------------------
# Datos de validación cruzada y pliegues
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CVData:
    """Pares de train + validación con su pliegue; arreglos alineados por posición (orden `student_id`, `room_id`)."""

    feature_order: tuple[str, ...]
    features: np.ndarray
    relevance: np.ndarray
    group: np.ndarray
    room_id: np.ndarray
    fold: np.ndarray
    students: pd.DataFrame  # una fila por estudiante: metadatos de T01 y `fold`

    @property
    def n_pairs(self) -> int:
        return len(self.relevance)


def assign_folds(students: pd.DataFrame, params: TrainParams) -> pd.Series:
    """Pliegue (0..n_splits-1) de cada estudiante de `params.cv_splits`, agrupado y estratificado por campus."""
    chosen = students[students["split"].isin(params.cv_splits)].sort_values("student_id", kind="mergesort")
    if chosen.empty:
        raise TrainError(f"no hay estudiantes en {list(params.cv_splits)}")
    ids = chosen["student_id"].astype(str).to_numpy()
    strata = chosen[params.stratify_by].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=params.n_splits, shuffle=True, random_state=params.seed)
    fold = np.full(len(ids), -1, dtype=np.int64)
    with warnings.catch_warnings():
        # Un campus con menos estudiantes que pliegues no impide la partición; scikit-learn solo avisa.
        warnings.filterwarnings("ignore", message="The least populated class", category=UserWarning)
        for number, (_fit, held) in enumerate(splitter.split(np.zeros((len(ids), 1)), strata, groups=ids)):
            fold[held] = number
    if (fold < 0).any():
        raise TrainError("hay estudiantes sin pliegue")
    return pd.Series(fold, index=pd.Index(ids, name="student_id"), name="fold")


def build_cv_data(data: TrainingData, params: TrainParams) -> CVData:
    """Pares y estudiantes de train + validación con su pliegue; la partición test queda fuera."""
    folds = assign_folds(data.students, params)
    mask = data.split.isin(params.cv_splits).to_numpy()
    group = data.group[mask].astype(str).to_numpy()
    students = data.students[data.students["split"].isin(params.cv_splits)].copy()
    students["fold"] = students["student_id"].astype(str).map(folds).astype("int64")
    return CVData(
        feature_order=data.feature_order,
        features=data.features[mask].to_numpy(dtype=np.float64, copy=True),
        relevance=data.relevance[mask].to_numpy(dtype=np.int64, copy=True),
        group=group,
        room_id=data.room_id[mask].astype(str).to_numpy(),
        fold=folds.loc[group].to_numpy(dtype=np.int64),
        students=students.reset_index(drop=True),
    )


def fold_counts(cv: CVData) -> pd.DataFrame:
    """Estudiantes por pliegue y campus (filas: pliegue; columnas: campus y total) y pares por pliegue."""
    table = pd.crosstab(cv.students["fold"], cv.students["university_code"]).sort_index(axis=1)
    table["students"] = table.sum(axis=1)
    pairs = pd.Series(cv.fold).value_counts().sort_index()
    table["pairs"] = pairs.reindex(table.index).astype("int64")
    return table


def group_sizes(group: np.ndarray) -> list[int]:
    """Tamaño de cada bloque consecutivo de `group` (LightGBM pide los grupos contiguos)."""
    if len(group) == 0:
        return []
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    sizes = np.diff(np.r_[starts, len(group)])
    if len(starts) != len(set(group.tolist())):
        raise TrainError("los pares de un estudiante no están contiguos")
    return [int(size) for size in sizes]


# ---------------------------------------------------------------------------
# Métricas por estudiante y por pliegue (mismas definiciones que el baseline)
# ---------------------------------------------------------------------------


def ndcg_by_student(
    scores: np.ndarray, relevance: np.ndarray, group: np.ndarray, room_id: np.ndarray
) -> dict[str, float | None]:
    """NDCG@10 por estudiante ordenando por puntaje descendente y `room_id` ascendente (M-ML-029/M-ML-034)."""
    frame = pd.DataFrame({"group": group, "score": scores, "room_id": room_id, "relevance": relevance})
    frame = frame.sort_values(["group", "score", "room_id"], ascending=[True, False, True], kind="mergesort")
    return {
        str(student): ndcg_at_k([int(value) for value in values], NDCG_K)
        for student, values in frame.groupby("group", sort=True)["relevance"]
    }


@dataclass(frozen=True)
class FoldScore:
    """NDCG@10 medio de los estudiantes retenidos de un pliegue."""

    fold: int
    students: int
    undefined: int  # estudiantes sin ganancia (NDCG no definido), excluidos de la media
    ndcg10: float

    def as_dict(self) -> dict[str, object]:
        return {"fold": self.fold, "students": self.students, "undefined": self.undefined, "ndcg@10": self.ndcg10}


def fold_score(fold: int, per_student: Mapping[str, float | None]) -> FoldScore:
    defined = [value for value in per_student.values() if value is not None]
    if not defined:
        raise TrainError(f"pliegue {fold}: ningún estudiante con NDCG@10 definido")
    return FoldScore(fold, len(defined), len(per_student) - len(defined), math.fsum(defined) / len(defined))


@dataclass(frozen=True)
class GridResult:
    """Resultado de una configuración de la rejilla en los pliegues."""

    config_id: str
    config: Mapping[str, object]
    folds: tuple[FoldScore, ...]
    convergence_warnings: int = 0

    @property
    def values(self) -> list[float]:
        return [item.ndcg10 for item in self.folds]

    @property
    def mean(self) -> float:
        return math.fsum(self.values) / len(self.values)

    @property
    def std(self) -> float:
        """Desviación estándar muestral entre pliegues (0 con un solo pliegue)."""
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "config_id": self.config_id,
            "config": dict(self.config),
            "ndcg@10_by_fold": self.values,
            "ndcg@10_mean": self.mean,
            "ndcg@10_std": self.std,
            "convergence_warnings": self.convergence_warnings,
        }


def select_best(results: Sequence[GridResult]) -> GridResult:
    """Mayor NDCG@10 medio; desempate: menor desviación y después la primera de la rejilla."""
    if not results:
        raise TrainError("rejilla vacía")
    best_index = max(range(len(results)), key=lambda i: (results[i].mean, -results[i].std, -i))
    return results[best_index]


def cross_validate(
    cv: CVData, n_splits: int, fit_and_score: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> tuple[FoldScore, ...]:
    """Para cada pliegue, ajusta con los demás y puntúa solo los pares retenidos (`fit_and_score(fit, held)`)."""
    scores: list[FoldScore] = []
    for fold in range(n_splits):
        held = cv.fold == fold
        fit = ~held
        predicted = fit_and_score(fit, held)
        if predicted.shape != (int(held.sum()),):
            raise TrainError(f"pliegue {fold}: el modelo devolvió {predicted.shape} puntajes")
        per_student = ndcg_by_student(predicted, cv.relevance[held], cv.group[held], cv.room_id[held])
        scores.append(fold_score(fold, per_student))
    return tuple(scores)


# ---------------------------------------------------------------------------
# Regresión logística (pointwise, aditiva, JSON sin pickle)
# ---------------------------------------------------------------------------


def clip_bounds(schema: FeatureSchema) -> tuple[np.ndarray, np.ndarray]:
    """Límites de recorte por feature: `training_range` del esquema o, si no tiene, `range` (spec §6.6)."""
    lower: list[float] = []
    upper: list[float] = []
    features = {item.name: item for item in schema.pair_features}
    for name in schema.feature_order:
        feature = features[name]
        bounds = feature.training_range or feature.range
        if bounds.min is None or bounds.max is None:
            raise TrainError(f"{name}: sin training_range ni range cerrado para el recorte")
        lower.append(float(bounds.min))
        upper.append(float(bounds.max))
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


@dataclass(frozen=True)
class LogRegModel:
    """Regresión logística reconstruible desde JSON: recorte → transformación → logit lineal.

    `standard`: z = (clip(x) - mean) / scale. `quantile_bins`: one-hot del
    intervalo `searchsorted(inner_edges, clip(x), side="right")` de cada feature
    (igual que `KBinsDiscretizer.transform`). El logit es `diseño @ coef + intercept`.
    """

    feature_order: tuple[str, ...]
    config: LogRegConfig
    clip_lower: tuple[float, ...]
    clip_upper: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float
    mean: tuple[float, ...] | None = None
    scale: tuple[float, ...] | None = None
    inner_edges: tuple[tuple[float, ...], ...] | None = None

    def design(self, features: np.ndarray) -> np.ndarray:
        """Matriz que recibe `LogisticRegression` (después del recorte y la transformación)."""
        values = np.clip(np.asarray(features, dtype=np.float64), self.clip_lower, self.clip_upper)
        if self.config.transform == "standard":
            if self.mean is None or self.scale is None:
                raise TrainError("modelo standard sin mean o scale")
            scaled: np.ndarray = (values - np.asarray(self.mean)) / np.asarray(self.scale)
            return scaled
        if self.inner_edges is None:
            raise TrainError("modelo quantile_bins sin inner_edges")
        blocks: list[np.ndarray] = []
        for column, edges in enumerate(self.inner_edges):
            index = np.searchsorted(np.asarray(edges, dtype=np.float64), values[:, column], side="right")
            block = np.zeros((len(values), len(edges) + 1), dtype=np.float64)
            block[np.arange(len(values)), index] = 1.0
            blocks.append(block)
        return np.hstack(blocks)

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """Logit de cada fila (mismo cálculo que `LogisticRegression.decision_function`)."""
        coef = np.asarray(self.coef, dtype=np.float64).reshape(-1, 1)
        result: np.ndarray = (self.design(features) @ coef + self.intercept).ravel()
        return result

    def to_dict(self) -> dict[str, Any]:
        transform: dict[str, Any] = {"type": self.config.transform}
        if self.config.transform == "standard":
            transform.update(mean=list(self.mean or ()), scale=list(self.scale or ()))
        else:
            transform["inner_edges"] = [list(edges) for edges in self.inner_edges or ()]
        return {
            "format": LOGREG_FORMAT,
            "feature_order": list(self.feature_order),
            "config": self.config.as_dict(),
            "clip": {"lower": list(self.clip_lower), "upper": list(self.clip_upper)},
            "transform": transform,
            "coef": list(self.coef),
            "intercept": self.intercept,
        }

    @staticmethod
    def _floats(values: Sequence[Any]) -> tuple[float, ...]:
        return tuple(float(value) for value in values)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LogRegModel:
        if data.get("format") != LOGREG_FORMAT:
            raise TrainError(f"formato de regresión logística desconocido: {data.get('format')!r}")
        config = LogRegConfig(str(data["config"]["transform"]), float(data["config"]["C"]))
        transform = data["transform"]
        if transform["type"] != config.transform:
            raise TrainError("transform.type no coincide con config.transform")
        floats = cls._floats
        return cls(
            feature_order=tuple(str(name) for name in data["feature_order"]),
            config=config,
            clip_lower=floats(data["clip"]["lower"]),
            clip_upper=floats(data["clip"]["upper"]),
            coef=floats(data["coef"]),
            intercept=float(data["intercept"]),
            mean=floats(transform["mean"]) if config.transform == "standard" else None,
            scale=floats(transform["scale"]) if config.transform == "standard" else None,
            inner_edges=tuple(floats(edges) for edges in transform["inner_edges"])
            if config.transform == "quantile_bins"
            else None,
        )


def logreg_pipeline(config: LogRegConfig, params: TrainParams) -> Pipeline:
    """Pipeline con solo transformadores con convertidor en skl2onnx (spec §6.3; lista verificada en T05)."""
    transform: StandardScaler | KBinsDiscretizer
    if config.transform == "standard":
        transform = StandardScaler()
    else:
        transform = KBinsDiscretizer(
            n_bins=params.n_bins,
            encode="onehot-dense",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",
            subsample=None,
        )
    classifier = LogisticRegression(
        C=config.c, solver=params.solver, max_iter=params.max_iter, tol=params.tol, random_state=params.seed
    )
    return Pipeline([("transform", transform), ("logreg", classifier)])


def _fit_quietly(fit: Callable[[], object]) -> int:
    """Ejecuta `fit` contando los avisos de no convergencia; los avisos esperados de los intervalos se omiten."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit()
    convergence = 0
    for item in caught:
        if issubclass(item.category, ConvergenceWarning):
            convergence += 1
        elif not any(text in str(item.message) for text in _EXPECTED_WARNINGS):
            warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
    return convergence


def fit_logreg(
    features: np.ndarray,
    target: np.ndarray,
    config: LogRegConfig,
    params: TrainParams,
    bounds: tuple[np.ndarray, np.ndarray],
    feature_order: tuple[str, ...],
) -> tuple[LogRegModel, Pipeline, int]:
    """Ajusta el pipeline sobre las features recortadas y extrae sus parámetros a `LogRegModel`."""
    lower, upper = bounds
    clipped = np.clip(features, lower, upper)
    pipeline = logreg_pipeline(config, params)
    convergence = _fit_quietly(lambda: pipeline.fit(clipped, target))
    transform = pipeline.named_steps["transform"]
    classifier = pipeline.named_steps["logreg"]
    common: dict[str, Any] = {
        "feature_order": feature_order,
        "config": config,
        "clip_lower": tuple(float(value) for value in lower),
        "clip_upper": tuple(float(value) for value in upper),
        "coef": tuple(float(value) for value in classifier.coef_.ravel()),
        "intercept": float(classifier.intercept_[0]),
    }
    if config.transform == "standard":
        model = LogRegModel(
            **common,
            mean=tuple(float(value) for value in transform.mean_),
            scale=tuple(float(value) for value in transform.scale_),
        )
    else:
        model = LogRegModel(
            **common,
            inner_edges=tuple(tuple(float(edge) for edge in edges[1:-1]) for edges in transform.bin_edges_),
        )
    return model, pipeline, convergence


def logreg_target(relevance: np.ndarray, threshold: int) -> np.ndarray:
    """Objetivo pointwise: 1 si `relevance ≥ threshold` (2, M-ML-034)."""
    return (relevance >= threshold).astype(np.int64)


# ---------------------------------------------------------------------------
# LightGBM LambdaRank (texto nativo, sin pickle)
# ---------------------------------------------------------------------------


def lightgbm_params(config: LightGBMConfig, params: TrainParams, feature_order: tuple[str, ...]) -> dict[str, Any]:
    """Parámetros de `lightgbm.train` para una configuración de la rejilla."""
    return {
        "objective": "lambdarank",
        "label_gain": list(params.label_gain),
        "num_leaves": config.num_leaves,
        "learning_rate": config.learning_rate,
        "min_data_in_leaf": params.min_data_in_leaf,
        "monotone_constraints": params.monotone_constraints(feature_order),
        "num_threads": params.num_threads,
        "deterministic": params.deterministic,
        "force_row_wise": True,
        "seed": params.seed,
        "verbosity": -1,
    }


def fit_lightgbm(
    features: np.ndarray,
    relevance: np.ndarray,
    group: np.ndarray,
    config: LightGBMConfig,
    params: TrainParams,
    feature_order: tuple[str, ...],
) -> lightgbm.Booster:
    """Entrena LambdaRank con los grupos por estudiante (contiguos) y `n_estimators` árboles."""
    settings = lightgbm_params(config, params, feature_order)
    dataset = lightgbm.Dataset(
        features, label=relevance, group=group_sizes(group), feature_name=list(feature_order), params=settings
    )
    return lightgbm.train(settings, dataset, num_boost_round=config.n_estimators)


def lightgbm_scores(booster: lightgbm.Booster, features: np.ndarray, num_threads: int = 1) -> np.ndarray:
    return np.asarray(booster.predict(features, num_threads=num_threads), dtype=np.float64)


def monotone_violations(
    predict: Callable[[np.ndarray], np.ndarray],
    sample: np.ndarray,
    feature_order: tuple[str, ...],
    constrained: Sequence[str],
    bounds: tuple[np.ndarray, np.ndarray],
    steps: int = MONOTONE_STEPS,
) -> dict[str, int]:
    """Filas de `sample` en las que el puntaje **sube** al subir cada feature restringida (lo demás fijo)."""
    lower, upper = bounds
    result: dict[str, int] = {}
    for name in constrained:
        column = feature_order.index(name)
        grid = np.linspace(lower[column], upper[column], steps)
        matrix = np.repeat(sample, steps, axis=0)
        matrix[:, column] = np.tile(grid, len(sample))
        scores = predict(matrix).reshape(len(sample), steps)
        result[name] = int((np.diff(scores, axis=1) > 0).any(axis=1).sum())
    return result


def monotone_sample(features: np.ndarray, rows: int = MONOTONE_SAMPLE_ROWS) -> np.ndarray:
    """Filas equiespaciadas (deterministas) para la verificación monotónica."""
    positions = np.unique(np.linspace(0, len(features) - 1, min(rows, len(features))).astype(np.int64))
    sample: np.ndarray = features[positions].copy()
    return sample


# ---------------------------------------------------------------------------
# Baseline por pliegue (no se entrena: score_features y rank de ML-04)
# ---------------------------------------------------------------------------


def baseline_ndcg_by_student(cv: CVData, schema: FeatureSchema, config: BaselineConfig) -> dict[str, float | None]:
    """NDCG@10 del baseline por estudiante, con el orden de `rank` (suma sin redondear desc, `room_id` asc)."""
    results: list[ScoreResult] = [
        score_features([float(value) for value in row], config, schema) for row in cv.features
    ]
    by_student: dict[str, list[int]] = {}
    for position, student in enumerate(cv.group.tolist()):
        by_student.setdefault(str(student), []).append(position)
    output: dict[str, float | None] = {}
    for student in sorted(by_student):
        rows = by_student[student]
        relevance = {str(cv.room_id[i]): int(cv.relevance[i]) for i in rows}
        ordered = rank(ScoredRoom(str(cv.room_id[i]), results[i]) for i in rows)
        output[student] = ndcg_at_k([relevance[item.room_id] for item in ordered], NDCG_K)
    return output


def baseline_fold_scores(cv: CVData, per_student: Mapping[str, float | None], n_splits: int) -> tuple[FoldScore, ...]:
    folds = cv.students.set_index(cv.students["student_id"].astype(str))["fold"]
    return tuple(
        fold_score(fold, {student: per_student[student] for student in folds.index[folds == fold]})
        for fold in range(n_splits)
    )


# ---------------------------------------------------------------------------
# Entrenamiento completo (puro: sin I/O)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainResult:
    """Todo lo que produce `train`: pliegues, rejillas, baseline por pliegue y modelos finales."""

    params: TrainParams
    relevance_threshold: int
    schema_version: str
    labels_params_version: str
    summary: Mapping[str, Mapping[str, int]]
    excluded: tuple[tuple[str, str, int], ...]
    cv: CVData
    logreg_results: tuple[GridResult, ...]
    lightgbm_results: tuple[GridResult, ...]
    baseline_folds: tuple[FoldScore, ...]
    logreg_selected: GridResult
    lightgbm_selected: GridResult
    logreg_model: LogRegModel
    logreg_final_convergence_warnings: int
    lightgbm_booster: lightgbm.Booster
    monotone_check: Mapping[str, int]
    monotone_sample_rows: int


def train_models(
    data: TrainingData,
    params: TrainParams,
    relevance_threshold: int,
    schema: FeatureSchema,
    config: BaselineConfig,
) -> TrainResult:
    """Pliegues, búsqueda en rejilla, baseline por pliegue y modelos finales en train + validación."""
    params.check_schema(schema)
    if tuple(schema.feature_order) != data.feature_order:
        raise TrainError("feature_order de los datos distinto del esquema")
    cv = build_cv_data(data, params)
    bounds = clip_bounds(schema)
    target = logreg_target(cv.relevance, relevance_threshold)

    logreg_results: list[GridResult] = []
    for lr_config in params.logreg_grid():
        warned: list[int] = []

        def fit_score_lr(
            fit: np.ndarray, held: np.ndarray, lr_config: LogRegConfig = lr_config, warned: list[int] = warned
        ) -> np.ndarray:
            model, _pipeline, convergence = fit_logreg(
                cv.features[fit], target[fit], lr_config, params, bounds, cv.feature_order
            )
            warned.append(convergence)
            return model.decision_function(cv.features[held])

        folds = cross_validate(cv, params.n_splits, fit_score_lr)
        logreg_results.append(GridResult(lr_config.id, lr_config.as_dict(), folds, sum(warned)))

    lightgbm_results: list[GridResult] = []
    for gbm_config in params.lightgbm_grid():

        def fit_score_gbm(fit: np.ndarray, held: np.ndarray, gbm_config: LightGBMConfig = gbm_config) -> np.ndarray:
            booster = fit_lightgbm(
                cv.features[fit], cv.relevance[fit], cv.group[fit], gbm_config, params, cv.feature_order
            )
            return lightgbm_scores(booster, cv.features[held], params.num_threads)

        lightgbm_results.append(
            GridResult(gbm_config.id, gbm_config.as_dict(), cross_validate(cv, params.n_splits, fit_score_gbm))
        )

    baseline = baseline_fold_scores(cv, baseline_ndcg_by_student(cv, schema, config), params.n_splits)

    logreg_selected = select_best(logreg_results)
    lightgbm_selected = select_best(lightgbm_results)
    lr_best = next(item for item in params.logreg_grid() if item.id == logreg_selected.config_id)
    gbm_best = next(item for item in params.lightgbm_grid() if item.id == lightgbm_selected.config_id)
    logreg_model, _pipeline, final_warnings = fit_logreg(cv.features, target, lr_best, params, bounds, cv.feature_order)
    booster = fit_lightgbm(cv.features, cv.relevance, cv.group, gbm_best, params, cv.feature_order)
    sample = monotone_sample(cv.features)
    check = monotone_violations(
        lambda matrix: lightgbm_scores(booster, matrix, params.num_threads),
        sample,
        cv.feature_order,
        params.monotone_decreasing,
        bounds,
    )
    return TrainResult(
        params=params,
        relevance_threshold=relevance_threshold,
        schema_version=data.schema_version,
        labels_params_version=data.labels_params_version,
        summary=data.summary(),
        excluded=tuple((item.student_id, item.split, item.candidates) for item in data.excluded),
        cv=cv,
        logreg_results=tuple(logreg_results),
        lightgbm_results=tuple(lightgbm_results),
        baseline_folds=baseline,
        logreg_selected=logreg_selected,
        lightgbm_selected=lightgbm_selected,
        logreg_model=logreg_model,
        logreg_final_convergence_warnings=final_warnings,
        lightgbm_booster=booster,
        monotone_check=check,
        monotone_sample_rows=len(sample),
    )


# ---------------------------------------------------------------------------
# Escritura (única I/O de salida) y etapa
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def logreg_document(result: TrainResult) -> dict[str, Any]:
    """JSON de la regresión logística final (sin pickle), con su procedencia."""
    cv = result.cv
    return {
        "synthetic": True,
        "model": "logistic_regression",
        "objective": f"pointwise: y = 1 si relevance >= {result.relevance_threshold}",
        "feature_schema_version": result.schema_version,
        "labels_params_version": result.labels_params_version,
        "train_params_version": result.params.version,
        "trained_on": {
            "splits": list(result.params.cv_splits),
            "students": len(cv.students),
            "pairs": cv.n_pairs,
        },
        **result.logreg_model.to_dict(),
    }


def write_outputs(models_dir: Path, result: TrainResult) -> dict[str, dict[str, Any]]:
    """Escribe los modelos en `models_dir`; devuelve nombre → {bytes, sha256} de los modelos."""
    models_dir.mkdir(parents=True, exist_ok=True)
    logreg_path = models_dir / LOGREG_MODEL
    _write_json(logreg_path, logreg_document(result))
    lightgbm_path = models_dir / LIGHTGBM_MODEL
    result.lightgbm_booster.save_model(str(lightgbm_path))
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in (logreg_path, lightgbm_path)
    }


def output_dirs(paths: ProjectPaths, params: TrainParams, profile: str) -> tuple[Path, Path]:
    """(modelos, reportes): `full` → `models/` y `reports/`; `ci` → `profiles.ci.output_dir` (fuera de git)."""
    if profile == DEFAULT_PROFILE:
        return paths.models, paths.reports
    target = paths.root / params.ci_profile.output_dir
    return target, target


def run_train(paths: ProjectPaths, params: TrainParams | None = None, profile: str | None = None) -> str:
    """Etapa `train`: entrena, escribe los modelos (fuera de git) y los reportes de validación cruzada.

    `profile` (o `THEHOUSE_ML_TRAIN_PROFILE`): `full` por defecto, la corrida oficial que escribe
    `reports/`; `ci` reduce la rejilla y los pliegues y escribe todo en `profiles.ci.output_dir`.
    """
    from thehouse_ml.models.train_report import metrics_document, render_report  # evita un ciclo de importes

    if not all((paths.data_processed / name).is_file() for name in TRAINING_INPUTS):
        return TRAIN_SKIP_MESSAGE
    profile = profile or resolve_profile()
    base = params or load_train_params()
    params = base.for_profile(profile)
    models_dir, reports_dir = output_dirs(paths, base, profile)
    eval_params = load_eval_params()
    schema = default_schema()
    data = load_training_data(paths, schema)
    if eval_params.min_candidates != 2:
        raise TrainError("baseline_eval_params.min_candidates distinto del mínimo del cargador (2)")
    inputs = {name: sha256_file(paths.data_processed / name) for name in TRAINING_INPUTS}
    result = train_models(data, params, eval_params.relevance_threshold, schema, default_config())
    models = write_outputs(models_dir, result)
    folds = result.cv.students.sort_values("student_id", kind="mergesort")
    manifest = {
        "synthetic": True,
        "spec": "ML-05 1.0.0, T02 (ML-05-R02)",
        "profile": profile,
        "train_params_version": params.version,
        "feature_schema_version": result.schema_version,
        "labels_params_version": result.labels_params_version,
        "inputs": {name: {"sha256": digest} for name, digest in inputs.items()},
        "models": models,
        "selected": {
            "logreg": result.logreg_selected.config_id,
            "lightgbm": result.lightgbm_selected.config_id,
        },
        "folds": {str(row.student_id): int(row.fold) for row in folds.itertuples(index=False)},
    }
    _write_json(models_dir / TRAIN_MANIFEST, manifest)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / TRAIN_CV_METRICS
    _write_json(metrics_path, metrics_document(result, inputs, models, profile))
    report_path = reports_dir / TRAINING_REPORT
    report_path.write_text(render_report(result, inputs, models, profile), encoding="utf-8", newline="\n")
    return (
        f"train (perfil {profile}): {len(result.cv.students)} estudiantes de {'+'.join(params.cv_splits)} "
        f"en {params.n_splits} pliegues; "
        f"{result.logreg_selected.config_id} NDCG@10 {result.logreg_selected.mean:.3f}, "
        f"{result.lightgbm_selected.config_id} NDCG@10 {result.lightgbm_selected.mean:.3f} "
        f"(DATOS SINTÉTICOS) -> {(models_dir / LOGREG_MODEL).as_posix()}; "
        f"{(models_dir / LIGHTGBM_MODEL).as_posix()}; {report_path.as_posix()}"
    )


__all__ = [
    "LIGHTGBM_MODEL",
    "LOGREG_MODEL",
    "TRAINING_REPORT",
    "TRAIN_CV_METRICS",
    "TRAIN_MANIFEST",
    "TRAIN_SKIP_MESSAGE",
    "CVData",
    "FoldScore",
    "GridResult",
    "LogRegModel",
    "TrainError",
    "TrainResult",
    "assign_folds",
    "build_cv_data",
    "clip_bounds",
    "cross_validate",
    "fit_lightgbm",
    "fit_logreg",
    "fold_counts",
    "group_sizes",
    "lightgbm_scores",
    "logreg_pipeline",
    "monotone_violations",
    "ndcg_by_student",
    "output_dirs",
    "run_train",
    "select_best",
    "train_models",
]
