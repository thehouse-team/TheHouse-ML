"""Ablación por grupo de features, métricas por pliegue, segmentos en test e hipótesis > $3,500 (ML-05 T04, R04).

Validación cruzada (**solo** train + validación; test nunca entra):

- `cv_from_manifest` reconstruye los pares de los pliegues con la **misma asignación**
  estudiante → pliegue de `train_manifest.json` (T02).
- `cv_scores` reajusta cada familia en cada pliegue con los hiperparámetros **elegidos** en
  T02 (de `train_cv_metrics.json`, sin búsqueda nueva) y puntúa solo los retenidos.
- `summarize_folds` da NDCG@10, NDCG@5, Precision@5 y MAP por pliegue con las funciones de
  `thehouse_ml.evaluation.ranking` (traspaso T03-H2); el NDCG@10 por pliegue debe
  reproducir el de `train_cv_metrics.json` (`check_against_train`).
- Ablación: el mismo reajuste quitando las features de cada uno de los 6 grupos de
  `factor_groups.FACTOR_GROUPS`; ΔNDCG@10 por pliegue contra el modelo completo.

Segmentos (spec §6.7, constitución §11.3): se calculan sobre las métricas por estudiante
de la medición única de T03 (sin reentrenar, sin volver a puntuar test ni ajustar nada):
total, por campus, sin `seed_sin_verificar` y por rango de `budget.max`, con bootstrap
pareado y la semilla de cada segmento (`bootstrap:<seed>:<segment_id>`, como el baseline).
Los intervalos del baseline deben reproducir los de `baseline_metrics.json` y las medias
de cada partición (campus, presupuesto) deben agregarse a las del total.

Hipótesis del segmento > $3,500 (spec §6.7, regla fijada en M-ML-040 antes de mirar los
segmentos): ver `hypothesis`.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from thehouse_ml.evaluation.ranking import METRIC_NAMES, Interval, StudentMetrics
from thehouse_ml.features.loader import FeatureSchema
from thehouse_ml.models.baseline_config import BaselineConfig
from thehouse_ml.models.baseline_report import EvalParams
from thehouse_ml.models.factor_groups import FACTOR_GROUPS, FactorGroup, check_groups, kept_columns
from thehouse_ml.models.train import (
    CVData,
    clip_bounds,
    fit_lightgbm,
    fit_logreg,
    lightgbm_scores,
    logreg_target,
)
from thehouse_ml.models.train_config import DEFAULT_PROFILE, LightGBMConfig, LogRegConfig, TrainParams
from thehouse_ml.models.trained_eval import (
    CANDIDATES,
    PRIMARY_METRIC,
    EvalSplit,
    PairedResult,
    TrainedEvalError,
    baseline_metrics,
    bootstrap_rng,
    check_baseline_intervals,
    paired_intervals,
    ranked_metrics,
)
from thehouse_ml.models.training_data import TrainingData

ABLATION_METRICS = "ablation_metrics.json"
ABLATION_REPORT = "ablation_report.md"
MODELS = ("baseline", *CANDIDATES)
# Perfil `ci` (prueba de humo, M-ML-040): la ablación de LightGBM (6 grupos x pliegues, ~7 s por ajuste) se omite.
CI_ABLATION_MODELS = ("logreg",)
HYPOTHESIS_SEGMENT = "budget:budget_gt_3500"
HYPOTHESIS_GROUP = "budget"
HYPOTHESIS_MODEL = "logreg"
AGGREGATION_TOLERANCE = 1e-9
REPRODUCTION_TOLERANCE = 1e-12

VERDICT_CONFIRMED = "confirmed"
VERDICT_REFUTED = "refuted"
VERDICT_INCONCLUSIVE = "inconclusive"


# ---------------------------------------------------------------------------
# Pliegues de T02 y reajuste por pliegue (solo train + validación)
# ---------------------------------------------------------------------------


def cv_from_manifest(data: TrainingData, folds: Mapping[str, int], splits: Sequence[str]) -> CVData:
    """Pares de `splits` (train + validación) con el pliegue de `train_manifest.json`; test queda fuera."""
    if "test" in splits:
        raise TrainedEvalError("la partición test nunca entra en los pliegues")
    mask = data.split.isin(list(splits)).to_numpy()
    students = data.students[data.students["split"].isin(list(splits))].copy()
    ids = set(students["student_id"].astype(str))
    if ids != set(folds):
        raise TrainedEvalError(
            f"la asignación de pliegues de train_manifest.json no cubre los estudiantes de {list(splits)} "
            f"({len(ids ^ set(folds))} distintos)"
        )
    students["fold"] = students["student_id"].astype(str).map(folds).astype("int64")
    group = data.group[mask].astype(str).to_numpy()
    return CVData(
        feature_order=data.feature_order,
        features=data.features[mask].to_numpy(dtype=np.float64, copy=True),
        relevance=data.relevance[mask].to_numpy(dtype=np.int64, copy=True),
        group=group,
        room_id=data.room_id[mask].astype(str).to_numpy(),
        fold=np.asarray([folds[student] for student in group], dtype=np.int64),
        students=students.reset_index(drop=True),
    )


def selected_configs(cv_metrics: Mapping[str, Any]) -> tuple[LogRegConfig, LightGBMConfig]:
    """Hiperparámetros elegidos en T02 (sin búsqueda nueva), de `train_cv_metrics.json`."""
    selected = cv_metrics["selected"]
    lr = selected["logreg"]["config"]
    gbm = selected["lightgbm"]["config"]
    logreg = LogRegConfig(str(lr["transform"]), float(lr["C"]))
    lightgbm = LightGBMConfig(int(gbm["num_leaves"]), float(gbm["learning_rate"]), int(gbm["n_estimators"]))
    for config, name in ((logreg, "logreg"), (lightgbm, "lightgbm")):
        if config.id != selected[name]["config_id"]:
            raise TrainedEvalError(f"{name}: config {config.id} distinta de config_id {selected[name]['config_id']}")
    return logreg, lightgbm


FitScore = Callable[[np.ndarray, np.ndarray, tuple[int, ...]], np.ndarray]


def cv_scores(cv: CVData, n_folds: int, fit_score: FitScore, columns: tuple[int, ...]) -> np.ndarray:
    """Puntaje de cada par cuando su estudiante está retenido: ajuste con los otros pliegues y `columns`."""
    scores = np.full(cv.n_pairs, np.nan, dtype=np.float64)
    for fold in range(n_folds):
        held = cv.fold == fold
        if not held.any():
            raise TrainedEvalError(f"pliegue {fold} sin estudiantes")
        predicted = fit_score(~held, held, columns)
        if predicted.shape != (int(held.sum()),):
            raise TrainedEvalError(f"pliegue {fold}: el modelo devolvió {predicted.shape} puntajes")
        scores[held] = predicted
    if not np.isfinite(scores).all():
        raise TrainedEvalError("hay pares sin puntaje de validación cruzada")
    return scores


def logreg_fit_score(
    cv: CVData, config: LogRegConfig, params: TrainParams, schema: FeatureSchema, threshold: int
) -> FitScore:
    lower, upper = clip_bounds(schema)
    target = logreg_target(cv.relevance, threshold)

    def fit_score(fit: np.ndarray, held: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
        cols = list(columns)
        order = tuple(cv.feature_order[i] for i in columns)
        model, _pipeline, _warnings = fit_logreg(
            cv.features[fit][:, cols], target[fit], config, params, (lower[cols], upper[cols]), order
        )
        return model.decision_function(cv.features[held][:, cols])

    return fit_score


def lightgbm_fit_score(cv: CVData, config: LightGBMConfig, params: TrainParams) -> FitScore:
    def fit_score(fit: np.ndarray, held: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
        cols = list(columns)
        order = tuple(cv.feature_order[i] for i in columns)
        booster = fit_lightgbm(cv.features[fit][:, cols], cv.relevance[fit], cv.group[fit], config, params, order)
        return lightgbm_scores(booster, cv.features[held][:, cols], params.num_threads)

    return fit_score


# ---------------------------------------------------------------------------
# Métricas por pliegue (traspaso T03-H2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldMetrics:
    """Medias por pliegue de las 4 métricas de un modelo (estudiantes retenidos con métricas definidas)."""

    name: str
    by_fold: tuple[Mapping[str, float], ...]
    students: tuple[int, ...]
    undefined: tuple[int, ...]

    def values(self, metric: str) -> list[float]:
        return [item[metric] for item in self.by_fold]

    def mean(self, metric: str) -> float:
        return math.fsum(self.values(metric)) / len(self.by_fold)

    def std(self, metric: str) -> float:
        """Desviación estándar muestral entre pliegues (0 con un solo pliegue), como T02."""
        values = self.values(metric)
        return statistics.stdev(values) if len(values) > 1 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "by_fold": [
                {"fold": fold, "students": self.students[fold], "undefined": self.undefined[fold], **dict(item)}
                for fold, item in enumerate(self.by_fold)
            ],
            "mean": {metric: self.mean(metric) for metric in METRIC_NAMES},
            "std": {metric: self.std(metric) for metric in METRIC_NAMES},
        }


def summarize_folds(
    name: str, per_student: Mapping[str, StudentMetrics | None], fold_of: Mapping[str, int], n_folds: int
) -> FoldMetrics:
    """Media de cada métrica por pliegue; los estudiantes con métricas no definidas se cuentan aparte."""
    by_fold: list[dict[str, float]] = []
    counts: list[int] = []
    undefined: list[int] = []
    for fold in range(n_folds):
        members = [student for student in sorted(per_student) if fold_of[student] == fold]
        values = [per_student[student] for student in members]
        present = [value for value in values if value is not None]
        if not present:
            raise TrainedEvalError(f"{name}: pliegue {fold} sin estudiantes con métricas definidas")
        by_fold.append(
            {
                metric: math.fsum(item.values()[index] for item in present) / len(present)
                for index, metric in enumerate(METRIC_NAMES)
            }
        )
        counts.append(len(present))
        undefined.append(len(values) - len(present))
    return FoldMetrics(name, tuple(by_fold), tuple(counts), tuple(undefined))


def check_against_train(folds: Mapping[str, FoldMetrics], cv_metrics: Mapping[str, Any]) -> None:
    """El NDCG@10 por pliegue recalculado debe ser el de `train_cv_metrics.json` (mismos pliegues y modelos)."""
    expected = {
        "baseline": cv_metrics["baseline"]["ndcg@10_by_fold"],
        "logreg": cv_metrics["selected"]["logreg"]["ndcg@10_by_fold"],
        "lightgbm": cv_metrics["selected"]["lightgbm"]["ndcg@10_by_fold"],
    }
    for name, values in expected.items():
        actual = folds[name].values(PRIMARY_METRIC)
        if len(actual) != len(values) or any(
            not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=REPRODUCTION_TOLERANCE)
            for a, b in zip(actual, values, strict=True)
        ):
            raise TrainedEvalError(
                f"{name}: el NDCG@10 por pliegue recalculado {actual} no reproduce train_cv_metrics.json {values}; "
                "pliegues, hiperparámetros o datos distintos de T02"
            )


# ---------------------------------------------------------------------------
# Ablación por grupo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationResult:
    """NDCG@10 por pliegue de un modelo sin las features de un grupo, frente al modelo completo."""

    model: str
    group: FactorGroup
    kept: tuple[str, ...]
    ndcg10_by_fold: tuple[float, ...]
    full_by_fold: tuple[float, ...]

    @property
    def deltas(self) -> list[float]:
        return [value - full for value, full in zip(self.ndcg10_by_fold, self.full_by_fold, strict=True)]

    @property
    def delta_mean(self) -> float:
        return math.fsum(self.deltas) / len(self.deltas)

    @property
    def delta_std(self) -> float:
        return statistics.stdev(self.deltas) if len(self.deltas) > 1 else 0.0

    @property
    def folds_below_full(self) -> int:
        return sum(1 for delta in self.deltas if delta < 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": self.group.id,
            "features_removed": list(self.group.features),
            "features_kept": list(self.kept),
            "ndcg@10_by_fold": list(self.ndcg10_by_fold),
            "ndcg@10_mean": math.fsum(self.ndcg10_by_fold) / len(self.ndcg10_by_fold),
            "delta_ndcg@10_by_fold": self.deltas,
            "delta_ndcg@10_mean": self.delta_mean,
            "delta_ndcg@10_std": self.delta_std,
            "folds_below_full": self.folds_below_full,
        }


def ablate(
    cv: CVData,
    model: str,
    fit_score: FitScore,
    full: FoldMetrics,
    n_folds: int,
    threshold: int,
    groups: Sequence[FactorGroup] = FACTOR_GROUPS,
) -> list[AblationResult]:
    """Reajusta `model` sin cada grupo (mismos pliegues e hiperparámetros) y mide NDCG@10 por pliegue."""
    fold_of = _fold_of(cv)
    results: list[AblationResult] = []
    for group in groups:
        columns = kept_columns(cv.feature_order, group.features)
        scores = cv_scores(cv, n_folds, fit_score, columns)
        per_student = ranked_metrics(scores, cv.relevance, cv.group, cv.room_id, threshold)
        summary = summarize_folds(f"{model} sin {group.id}", per_student, fold_of, n_folds)
        results.append(
            AblationResult(
                model=model,
                group=group,
                kept=tuple(cv.feature_order[i] for i in columns),
                ndcg10_by_fold=tuple(summary.values(PRIMARY_METRIC)),
                full_by_fold=tuple(full.values(PRIMARY_METRIC)),
            )
        )
    return results


def _fold_of(cv: CVData) -> dict[str, int]:
    return {str(row.student_id): int(row.fold) for row in cv.students.itertuples(index=False)}


@dataclass(frozen=True)
class CrossValidation:
    """Métricas por pliegue de los 3 modelos y ablación (solo train + validación)."""

    n_folds: int
    students: int
    pairs: int
    folds: Mapping[str, FoldMetrics]
    ablation: Mapping[str, tuple[AblationResult, ...]]
    ablated_models: tuple[str, ...]


def cross_validation(
    data: TrainingData,
    manifest: Mapping[str, Any],
    cv_metrics: Mapping[str, Any],
    params: TrainParams,
    schema: FeatureSchema,
    baseline_config: BaselineConfig,
    threshold: int,
    ablated_models: Sequence[str] = CANDIDATES,
) -> CrossValidation:
    """Traspaso T03-H2 (4 métricas por pliegue) y ablación de los 6 grupos para `ablated_models`."""
    check_groups(data.feature_order)
    folds_map = {str(key): int(value) for key, value in manifest["folds"].items()}
    splits = [str(item) for item in cv_metrics["folds"]["splits"]]
    n_folds = int(cv_metrics["folds"]["n_splits"])
    if sorted(set(folds_map.values())) != list(range(n_folds)):
        raise TrainedEvalError(
            f"train_manifest.json: pliegues {sorted(set(folds_map.values()))}, se esperaban {n_folds}"
        )
    cv = cv_from_manifest(data, folds_map, splits)
    fold_of = _fold_of(cv)
    logreg_config, lightgbm_config = selected_configs(cv_metrics)
    fit_scores = {
        "logreg": logreg_fit_score(cv, logreg_config, params, schema, threshold),
        "lightgbm": lightgbm_fit_score(cv, lightgbm_config, params),
    }
    everything = tuple(range(len(cv.feature_order)))
    split = EvalSplit("+".join(splits), cv.feature_order, cv.features, cv.relevance, cv.group, cv.room_id)
    folds = {
        "baseline": summarize_folds(
            "baseline", baseline_metrics(split, schema, baseline_config, threshold), fold_of, n_folds
        )
    }
    for name in CANDIDATES:
        scores = cv_scores(cv, n_folds, fit_scores[name], everything)
        per_student = ranked_metrics(scores, cv.relevance, cv.group, cv.room_id, threshold)
        folds[name] = summarize_folds(name, per_student, fold_of, n_folds)
    check_against_train(folds, cv_metrics)
    ablation = {
        name: tuple(ablate(cv, name, fit_scores[name], folds[name], n_folds, threshold)) for name in ablated_models
    }
    return CrossValidation(n_folds, len(cv.students), cv.n_pairs, folds, ablation, tuple(ablated_models))


def ablated_models_for(profile: str) -> tuple[str, ...]:
    """`full`: los dos modelos; `ci`: solo la regresión logística (M-ML-040)."""
    return CANDIDATES if profile == DEFAULT_PROFILE else CI_ABLATION_MODELS


# ---------------------------------------------------------------------------
# Segmentos en test (de la medición única de T03)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    id: str
    kind: str
    label: str
    students: tuple[str, ...]


def build_segments(
    reference_rows: Sequence[Mapping[str, Any]], schema: FeatureSchema, params: EvalParams
) -> list[Segment]:
    """Mismos segmentos, ids y orden de estudiantes que `baseline_report.build_segments` (M-ML-034)."""
    order = [str(row["student_id"]) for row in reference_rows]
    rows = {str(row["student_id"]): row for row in reference_rows}
    segments = [Segment("total", "total", "Total", tuple(order))]
    for code in schema.catalog("university_code"):
        group = tuple(student for student in order if rows[student]["university_code"] == code)
        if group:
            suffix = "" if rows[group[0]]["seed_verified"] else " (sin verificar)"
            segments.append(Segment(f"campus:{code}", "campus", f"{code}{suffix}", group))
    segments.append(
        Segment(
            "without_seed_sin_verificar",
            "verified",
            "Sin `seed_sin_verificar`",
            tuple(student for student in order if rows[student]["seed_verified"]),
        )
    )
    for band in params.budget_ranges:
        group = tuple(student for student in order if band.contains(int(rows[student]["budget_max"])))
        segments.append(Segment(f"budget:{band.id}", "budget", f"`budget.max` {band.label}", group))
    return segments


@dataclass(frozen=True)
class SegmentResult:
    segment: Segment
    paired: PairedResult


def segment_results(
    segments: Sequence[Segment],
    baseline: Mapping[str, StudentMetrics],
    models: Mapping[str, Mapping[str, StudentMetrics]],
    params: EvalParams,
) -> list[SegmentResult | None]:
    """Bootstrap pareado por segmento con su semilla `bootstrap:<seed>:<segment_id>`; `None` si está vacío."""
    results: list[SegmentResult | None] = []
    for segment in segments:
        if not segment.students:
            results.append(None)
            continue
        paired = paired_intervals(
            segment.students,
            baseline,
            models,
            resamples=params.resamples,
            confidence=params.confidence,
            rng=bootstrap_rng(params, segment.id),
        )
        results.append(SegmentResult(segment, paired))
    return results


def check_segments(results: Sequence[SegmentResult | None], reference: Mapping[str, Any]) -> None:
    """Baseline por segmento = `baseline_metrics.json`; las medias de campus y presupuesto se agregan al total."""
    expected = {item["id"]: item for item in reference["segments"]}
    present = [item for item in results if item is not None]
    for item in present:
        entry = expected.get(item.segment.id)
        if entry is None or "baseline" not in entry:
            raise TrainedEvalError(f"segmento {item.segment.id} ausente de baseline_metrics.json")
        if entry["students_evaluated"] != len(item.segment.students):
            raise TrainedEvalError(f"segmento {item.segment.id}: número de estudiantes distinto del baseline")
        check_baseline_intervals(entry["baseline"], item.paired.baseline)
    total = next(item for item in present if item.segment.id == "total")
    n_total = len(total.segment.students)
    for kind in ("campus", "budget"):
        parts = [item for item in present if item.segment.kind == kind]
        if sum(len(item.segment.students) for item in parts) != n_total:
            raise TrainedEvalError(f"los segmentos {kind} no cubren a todos los estudiantes")
        for metric in METRIC_NAMES:
            series = [("baseline", total.paired.baseline[metric], [p.paired.baseline[metric] for p in parts])]
            for name in total.paired.models:
                series.append((name, total.paired.models[name][metric], [p.paired.models[name][metric] for p in parts]))
                series.append(
                    (
                        f"diff {name}",
                        total.paired.differences[name][metric],
                        [p.paired.differences[name][metric] for p in parts],
                    )
                )
            for label, whole, pieces in series:
                weighted = math.fsum(
                    piece.mean * len(part.segment.students) for piece, part in zip(pieces, parts, strict=True)
                )
                if not math.isclose(weighted / n_total, whole.mean, rel_tol=0.0, abs_tol=AGGREGATION_TOLERANCE):
                    raise TrainedEvalError(f"{kind}/{label}/{metric}: los segmentos no se agregan al total")


# ---------------------------------------------------------------------------
# Hipótesis del segmento > $3,500 (spec §6.7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hypothesis:
    """Veredicto y condiciones: 1 mayor mejora de los 3 rangos; 2 IC excluye 0; 3 ablación de presupuesto empeora."""

    verdict: str
    largest_improvement: bool
    interval_excludes_zero: bool | None  # None: el IC incluye 0; False: el IC está por debajo de 0
    budget_ablation_reduces: bool
    reasons: tuple[str, ...]
    details: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "conditions": {
                "1_largest_improvement_of_the_three_budget_ranges": self.largest_improvement,
                "2_ci_excludes_zero_above": self.interval_excludes_zero,
                "3_budget_ablation_reduces_ndcg@10": self.budget_ablation_reduces,
            },
            "reasons": list(self.reasons),
            **dict(self.details),
        }


def hypothesis(budget_deltas: Mapping[str, Interval], target: str, budget_ablation: AblationResult) -> Hypothesis:
    """Regla de M-ML-040 (fijada antes de calcular los segmentos).

    - **confirmada** si (1) la ΔNDCG@10 media del candidato en `target` es estrictamente la mayor de los
      rangos de presupuesto, (2) su IC 95 % está por encima de 0 y (3) quitar el grupo presupuesto baja el
      NDCG@10 medio entre pliegues (ΔNDCG@10 media < 0);
    - **refutada** si falla (1) o (3), o si el IC de (2) está entero por debajo de 0 (la evidencia puntual va
      en contra);
    - **no concluyente** si se cumplen (1) y (3) pero el IC de (2) incluye el 0.
    """
    if target not in budget_deltas or len(budget_deltas) < 2:
        raise TrainedEvalError(f"hipótesis: faltan los rangos de presupuesto ({list(budget_deltas)})")
    mine = budget_deltas[target]
    others = {key: value for key, value in budget_deltas.items() if key != target}
    best_other = max(others, key=lambda key: others[key].mean)
    largest = mine.mean > others[best_other].mean
    excludes: bool | None = True if mine.low > 0 else (False if mine.high < 0 else None)
    reduces = budget_ablation.delta_mean < 0
    reasons: list[str] = []
    reasons.append(
        f"(1) Δ NDCG@10 en {target} = {mine.mean:+.3f} "
        + ("es la mayor" if largest else f"no es la mayor ({best_other}: {others[best_other].mean:+.3f})")
        + " de los rangos de presupuesto"
    )
    reasons.append(
        f"(2) IC 95 % [{mine.low:+.3f}, {mine.high:+.3f}] "
        + {True: "excluye el 0 (por encima)", False: "está por debajo de 0", None: "incluye el 0"}[excludes]
    )
    reasons.append(
        f"(3) quitar el grupo presupuesto cambia el NDCG@10 en {budget_ablation.delta_mean:+.4f} de media "
        f"({budget_ablation.folds_below_full}/{len(budget_ablation.deltas)} pliegues por debajo): "
        + ("lo empeora" if reduces else "no lo empeora")
    )
    if largest and excludes is True and reduces:
        verdict = VERDICT_CONFIRMED
    elif not largest or not reduces or excludes is False:
        verdict = VERDICT_REFUTED
    else:
        verdict = VERDICT_INCONCLUSIVE
    details = {
        "model": budget_ablation.model,
        "segment": target,
        "delta_ndcg@10_by_budget_range": {key: value.as_dict() for key, value in budget_deltas.items()},
        "budget_ablation_delta_ndcg@10_mean": budget_ablation.delta_mean,
        "budget_ablation_folds_below_full": budget_ablation.folds_below_full,
    }
    return Hypothesis(verdict, largest, excludes, reduces, tuple(reasons), details)


def budget_deltas(results: Sequence[SegmentResult | None], model: str = HYPOTHESIS_MODEL) -> dict[str, Interval]:
    """Δ NDCG@10 pareada del candidato por rango de presupuesto, en el orden de `budget_ranges`."""
    return {
        item.segment.id: item.paired.differences[model][PRIMARY_METRIC]
        for item in results
        if item is not None and item.segment.kind == "budget"
    }


# ---------------------------------------------------------------------------
# Evaluación completa de T04 (pura: sin escritura)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationEvaluation:
    """Pliegues con 4 métricas, ablación, segmentos de test e hipótesis > $3,500."""

    profile: str
    cv_source: str
    cv: CrossValidation
    segments: tuple[SegmentResult | None, ...]
    hypothesis: Hypothesis
    eval_params: EvalParams


def reference_metrics(reference_rows: Sequence[Mapping[str, Any]]) -> dict[str, StudentMetrics]:
    """`per_student_baseline` (ya verificado por T03 contra el baseline recalculado) como `StudentMetrics`."""
    return {
        str(row["student_id"]): StudentMetrics(*(float(row[metric]) for metric in METRIC_NAMES))
        for row in reference_rows
    }


def evaluate_ablation(
    per_student: Mapping[str, Mapping[str, StudentMetrics]],
    data: TrainingData,
    manifest: Mapping[str, Any],
    cv_metrics: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    params: TrainParams,
    schema: FeatureSchema,
    baseline_config: BaselineConfig,
    eval_params: EvalParams,
    profile: str,
    cv_source: str,
) -> AblationEvaluation:
    """T04 completo: `per_student` son las métricas de test de T03 (no se vuelve a puntuar test)."""
    cv = cross_validation(
        data,
        manifest,
        cv_metrics,
        params,
        schema,
        baseline_config,
        eval_params.relevance_threshold,
        ablated_models_for(profile),
    )
    rows = reference["per_student_baseline"]
    segments = build_segments(rows, schema, eval_params)
    results = segment_results(segments, reference_metrics(rows), per_student, eval_params)
    check_segments(results, reference)
    budget_ablation = next(item for item in cv.ablation[HYPOTHESIS_MODEL] if item.group.id == HYPOTHESIS_GROUP)
    verdict = hypothesis(budget_deltas(results), HYPOTHESIS_SEGMENT, budget_ablation)
    return AblationEvaluation(profile, cv_source, cv, tuple(results), verdict, eval_params)


__all__ = [
    "ABLATION_METRICS",
    "ABLATION_REPORT",
    "CI_ABLATION_MODELS",
    "HYPOTHESIS_GROUP",
    "HYPOTHESIS_MODEL",
    "HYPOTHESIS_SEGMENT",
    "MODELS",
    "VERDICT_CONFIRMED",
    "VERDICT_INCONCLUSIVE",
    "VERDICT_REFUTED",
    "AblationEvaluation",
    "AblationResult",
    "CrossValidation",
    "FoldMetrics",
    "Hypothesis",
    "Segment",
    "SegmentResult",
    "ablate",
    "ablated_models_for",
    "budget_deltas",
    "build_segments",
    "check_against_train",
    "check_segments",
    "cross_validation",
    "cv_from_manifest",
    "cv_scores",
    "evaluate_ablation",
    "hypothesis",
    "lightgbm_fit_score",
    "logreg_fit_score",
    "reference_metrics",
    "segment_results",
    "selected_configs",
    "summarize_folds",
]
