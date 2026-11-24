"""Etapa `evaluate` de ML-05: medición única en test y decisión de promoción (ML-05 T03, ML-05-R03).

Pasos (spec ML-05 §6.2, §6.4 y §6.5; decisiones 2, 4 y 5 de M-ML-036):

1. `verify_models` comprueba el SHA-256 de los modelos finales de T02
   (`logreg.json` y `lightgbm_lambdarank.txt`) contra `train_manifest.json` y que
   se entrenaron con los mismos datos (SHA-256 de `TRAINING_INPUTS`); **no** se
   reentrena nada. Un SHA-256 distinto es un error.
2. `select_split` toma **solo** la partición test de ML-03 (300 estudiantes con 2 o
   más candidatos) y cada modelo puntúa sus pares; el ranking es puntaje
   descendente con desempate por `room_id` ascendente y las métricas son las de
   `thehouse_ml.evaluation.ranking` (mismas definiciones que el baseline).
3. Se recalcula el baseline (`score_features` y `rank` de ML-04) en los mismos
   estudiantes y se exige que reproduzca `per_student_baseline` de
   `reports/baseline_metrics.json` (mismos 300 estudiantes y mismas métricas).
4. `paired_intervals`: bootstrap pareado por estudiante con los parámetros de
   `baseline_eval_params.json` (1,000 remuestreos, percentil 95 %, semilla
   `bootstrap:<seed>:<segment_id>` con el mismo `segment_id` que el baseline:
   `total`), del que salen los intervalos de cada modelo y de la diferencia
   modelo - baseline. Con la misma semilla y el mismo orden de estudiantes, el
   intervalo del baseline coincide con el de `baseline_metrics.json`.
5. `promotion` evalúa las 3 condiciones de la spec §6.5 por candidato con la
   tabla por pliegue de `train_cv_metrics.json` (T02) y la diferencia en test.

**Test se evalúa una sola vez** (spec §6.2 y §7 regla 2): esta etapa no elige
hiperparámetros, features, calibración ni umbrales; solo lee los modelos
finales y mide. Volver a correrla produce los mismos bytes (sin fechas ni rutas
absolutas). Nunca lee la utilidad ni los pesos ocultos: las únicas lecturas de
Parquet son las de `training_data.TRAINING_INPUTS`.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd

from thehouse_ml.evaluation.ranking import METRIC_NAMES, Interval, StudentMetrics, bootstrap_means, student_metrics
from thehouse_ml.features import default_schema
from thehouse_ml.features.loader import FeatureSchema
from thehouse_ml.models.baseline_config import BaselineConfig, default_config
from thehouse_ml.models.baseline_report import BASELINE_METRICS, EvalParams, EvaluationError, load_eval_params
from thehouse_ml.models.baseline_score import ScoredRoom, rank, score_features
from thehouse_ml.models.train import (
    LIGHTGBM_MODEL,
    LOGREG_MODEL,
    TRAIN_CV_METRICS,
    TRAIN_MANIFEST,
    LogRegModel,
    lightgbm_scores,
    output_dirs,
    sha256_file,
)
from thehouse_ml.models.train_config import TrainParams, load_train_params, resolve_profile
from thehouse_ml.models.training_data import TRAINING_INPUTS, TrainingData, load_training_data
from thehouse_ml.paths import ProjectPaths

TRAINED_METRICS = "trained_metrics.json"
TRAINED_REPORT = "trained_report.md"
CANDIDATES = ("logreg", "lightgbm")
MODEL_FILES = {"logreg": LOGREG_MODEL, "lightgbm": LIGHTGBM_MODEL}
PRIMARY_METRIC = "ndcg@10"
SEGMENT_ID = "total"  # mismo segment_id que el total de baseline_metrics.json: misma semilla de bootstrap
BASELINE_TOLERANCE = 1e-12
TRAINED_SKIP_MESSAGE = "evaluate (ML-05): omitido; faltan los modelos de `train` o sus métricas de validación cruzada"

# Resultados de cada candidato y de la decisión (claves estables del JSON).
OUTCOME_CANDIDATE = "candidate_subject_to_t05"
OUTCOME_NOT_PROMOTED = "not_promoted"
OUTCOME_REFERENCE = "reference_not_promoted_in_v1"
DECISION_CANDIDATE = "logreg_candidate_subject_to_t05"
DECISION_BASELINE = "baseline_stays"


class TrainedEvalError(EvaluationError):
    """Modelos, métricas de referencia o resultados de la evaluación de ML-05 inválidos."""


# ---------------------------------------------------------------------------
# Modelos finales verificados (sin reentrenar)
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TrainedEvalError(f"{path.name}: se esperaba un objeto JSON")
    return data


def verify_models(models_dir: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    """SHA-256 de cada modelo final contra `train_manifest.json`; `TrainedEvalError` si falta o no coincide."""
    recorded = manifest.get("models")
    if not isinstance(recorded, dict):
        raise TrainedEvalError(f"{TRAIN_MANIFEST}: falta models")
    digests: dict[str, str] = {}
    for name in CANDIDATES:
        filename = MODEL_FILES[name]
        entry = recorded.get(filename)
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise TrainedEvalError(f"{TRAIN_MANIFEST}: falta el SHA-256 de {filename}")
        path = models_dir / filename
        if not path.is_file():
            raise TrainedEvalError(f"falta el modelo {filename} (ejecute `train`)")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise TrainedEvalError(
                f"{filename}: SHA-256 {actual} distinto del de {TRAIN_MANIFEST} ({entry['sha256']}); "
                "el modelo no es el que produjo `train`"
            )
        digests[filename] = actual
    return digests


def verify_inputs(paths: ProjectPaths, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Las entradas actuales deben ser las mismas con las que se entrenaron los modelos (SHA-256)."""
    recorded = manifest.get("inputs")
    if not isinstance(recorded, dict):
        raise TrainedEvalError(f"{TRAIN_MANIFEST}: falta inputs")
    digests: dict[str, str] = {}
    for name in TRAINING_INPUTS:
        expected = recorded.get(name, {}).get("sha256") if isinstance(recorded.get(name), dict) else None
        actual = sha256_file(paths.data_processed / name)
        if actual != expected:
            raise TrainedEvalError(
                f"data/processed/{name}: SHA-256 distinto del registrado en {TRAIN_MANIFEST}; "
                "los modelos se entrenaron con otros datos (ejecute `train`)"
            )
        digests[name] = actual
    return digests


def load_models(models_dir: Path) -> tuple[LogRegModel, lightgbm.Booster]:
    """Reconstruye la regresión logística desde JSON y LightGBM desde su texto nativo (sin pickle)."""
    logreg = LogRegModel.from_dict(_read_json(models_dir / LOGREG_MODEL))
    booster = lightgbm.Booster(model_file=str(models_dir / LIGHTGBM_MODEL))
    return logreg, booster


# ---------------------------------------------------------------------------
# Partición test y métricas por estudiante
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSplit:
    """Pares de la partición evaluada, alineados por posición (orden `student_id`, `room_id`)."""

    split: str
    feature_order: tuple[str, ...]
    features: np.ndarray
    relevance: np.ndarray
    group: np.ndarray
    room_id: np.ndarray

    @property
    def n_pairs(self) -> int:
        return len(self.relevance)

    @property
    def students(self) -> list[str]:
        return sorted(set(self.group.tolist()))


def select_split(data: TrainingData, split: str) -> EvalSplit:
    """Solo los pares de `split` (test); los estudiantes con < 2 candidatos ya los excluyó el cargador."""
    mask = (data.split == split).to_numpy()
    if not mask.any():
        raise TrainedEvalError(f"no hay pares en la partición {split}")
    return EvalSplit(
        split=split,
        feature_order=data.feature_order,
        features=data.features[mask].to_numpy(dtype=np.float64, copy=True),
        relevance=data.relevance[mask].to_numpy(dtype=np.int64, copy=True),
        group=data.group[mask].astype(str).to_numpy(),
        room_id=data.room_id[mask].astype(str).to_numpy(),
    )


def ranked_metrics(
    scores: np.ndarray, relevance: np.ndarray, group: np.ndarray, room_id: np.ndarray, threshold: int
) -> dict[str, StudentMetrics | None]:
    """Métricas por estudiante ordenando por puntaje descendente y `room_id` ascendente (M-ML-029/M-ML-034)."""
    if not (len(scores) == len(relevance) == len(group) == len(room_id)):
        raise TrainedEvalError("puntajes, relevancias, grupos y cuartos de distinta longitud")
    if not np.isfinite(scores).all():
        raise TrainedEvalError("hay puntajes no finitos")
    frame = pd.DataFrame({"group": group, "score": scores, "room_id": room_id, "relevance": relevance})
    frame = frame.sort_values(["group", "score", "room_id"], ascending=[True, False, True], kind="mergesort")
    return {
        str(student): student_metrics([int(value) for value in values], threshold)
        for student, values in frame.groupby("group", sort=True)["relevance"]
    }


def baseline_metrics(
    test: EvalSplit, schema: FeatureSchema, config: BaselineConfig, threshold: int
) -> dict[str, StudentMetrics | None]:
    """Baseline recalculado con `score_features` y `rank` (suma sin redondear desc, `room_id` asc), como ML-04."""
    results = [score_features([float(value) for value in row], config, schema) for row in test.features]
    rows: dict[str, list[int]] = {}
    for position, student in enumerate(test.group.tolist()):
        rows.setdefault(str(student), []).append(position)
    output: dict[str, StudentMetrics | None] = {}
    for student in sorted(rows):
        relevance = {str(test.room_id[i]): int(test.relevance[i]) for i in rows[student]}
        ordered = rank(ScoredRoom(str(test.room_id[i]), results[i]) for i in rows[student])
        output[student] = student_metrics([relevance[item.room_id] for item in ordered], threshold)
    return output


def defined(metrics: Mapping[str, StudentMetrics | None], name: str) -> dict[str, StudentMetrics]:
    """Estudiantes con todas las métricas definidas; con las etiquetas de M-ML-032 son todos."""
    missing = sorted(student for student, value in metrics.items() if value is None)
    if missing:
        raise TrainedEvalError(f"{name}: estudiantes sin cuarto relevante (métricas no definidas): {missing[:5]}")
    return {student: value for student, value in metrics.items() if value is not None}


def check_baseline_reference(
    reference: Sequence[Mapping[str, Any]], recomputed: Mapping[str, StudentMetrics]
) -> list[str]:
    """`per_student_baseline` debe tener exactamente los estudiantes evaluados y las mismas métricas.

    Devuelve los estudiantes en el orden de `per_student_baseline` (el del bootstrap del baseline).
    """
    order = [str(row["student_id"]) for row in reference]
    if len(set(order)) != len(order):
        raise TrainedEvalError(f"{BASELINE_METRICS}: per_student_baseline con estudiantes repetidos")
    if set(order) != set(recomputed):
        only_reference = sorted(set(order) - set(recomputed))[:5]
        only_test = sorted(set(recomputed) - set(order))[:5]
        raise TrainedEvalError(
            f"los estudiantes evaluados de test no son los de per_student_baseline "
            f"(solo en la referencia: {only_reference}; solo en test: {only_test})"
        )
    for row in reference:
        values = recomputed[str(row["student_id"])].as_dict()
        for metric in METRIC_NAMES:
            if not math.isclose(float(row[metric]), values[metric], rel_tol=0.0, abs_tol=BASELINE_TOLERANCE):
                raise TrainedEvalError(
                    f"{row['student_id']}: el baseline recalculado ({metric} {values[metric]!r}) no reproduce "
                    f"per_student_baseline ({row[metric]!r})"
                )
    return order


# ---------------------------------------------------------------------------
# Comparación pareada por estudiante (bootstrap)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedResult:
    """Intervalos del baseline, de cada modelo y de la diferencia modelo - baseline (por métrica)."""

    students: tuple[str, ...]
    baseline: Mapping[str, Interval]
    models: Mapping[str, Mapping[str, Interval]]
    differences: Mapping[str, Mapping[str, Interval]]


def paired_intervals(
    students: Sequence[str],
    baseline: Mapping[str, StudentMetrics],
    models: Mapping[str, Mapping[str, StudentMetrics]],
    *,
    resamples: int,
    confidence: float,
    rng: random.Random,
) -> PairedResult:
    """Bootstrap pareado: los mismos índices de estudiante para el baseline, cada modelo y sus diferencias."""
    series: dict[str, list[float]] = {}
    for index, metric in enumerate(METRIC_NAMES):
        base = [baseline[student].values()[index] for student in students]
        series[f"baseline:{metric}"] = base
        for name, values in models.items():
            model = [values[student].values()[index] for student in students]
            series[f"model:{name}:{metric}"] = model
            series[f"diff:{name}:{metric}"] = [m - b for m, b in zip(model, base, strict=True)]
    flat = bootstrap_means(series, resamples=resamples, confidence=confidence, rng=rng)
    return PairedResult(
        students=tuple(students),
        baseline={metric: flat[f"baseline:{metric}"] for metric in METRIC_NAMES},
        models={name: {metric: flat[f"model:{name}:{metric}"] for metric in METRIC_NAMES} for name in models},
        differences={name: {metric: flat[f"diff:{name}:{metric}"] for metric in METRIC_NAMES} for name in models},
    )


def bootstrap_rng(params: EvalParams, segment_id: str = SEGMENT_ID) -> random.Random:
    """Mismo generador que el baseline (`baseline_report.summarize_segment`) para el segmento."""
    return random.Random(f"bootstrap:{params.bootstrap_seed}:{segment_id}")  # noqa: S311 - bootstrap reproducible


def check_baseline_intervals(reference: Mapping[str, Any], recomputed: Mapping[str, Interval]) -> None:
    """El intervalo del baseline recalculado debe ser el de `baseline_metrics.json` (misma semilla y orden)."""
    for metric in METRIC_NAMES:
        expected = reference[metric]
        actual = recomputed[metric].as_dict()
        for key in ("mean", "ci_low", "ci_high"):
            if not math.isclose(float(expected[key]), actual[key], rel_tol=0.0, abs_tol=BASELINE_TOLERANCE):
                raise TrainedEvalError(
                    f"el bootstrap del baseline ({metric} {key} {actual[key]!r}) no reproduce "
                    f"{BASELINE_METRICS} ({expected[key]!r}): semilla u orden de estudiantes distintos"
                )


# ---------------------------------------------------------------------------
# Tabla por pliegue (T02) y criterio de promoción (spec §6.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldSeries:
    """NDCG@10 por pliegue de un modelo (de `train_cv_metrics.json`)."""

    name: str
    config_id: str
    values: tuple[float, ...]
    mean: float
    std: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "ndcg@10_by_fold": list(self.values),
            "ndcg@10_mean": self.mean,
            "ndcg@10_std": self.std,
        }


def fold_series(cv_metrics: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, FoldSeries]:
    """Baseline y modelos elegidos por pliegue; comprueba que vienen de la misma corrida que los modelos."""
    selected = cv_metrics.get("selected", {})
    chosen = manifest.get("selected", {})
    files = cv_metrics.get("final_models", {}).get("files", {})
    for filename in MODEL_FILES.values():
        if files.get(filename, {}).get("sha256") != manifest.get("models", {}).get(filename, {}).get("sha256"):
            raise TrainedEvalError(f"{TRAIN_CV_METRICS} y {TRAIN_MANIFEST} no son de la misma corrida ({filename})")
    output: dict[str, FoldSeries] = {}
    base = cv_metrics["baseline"]
    output["baseline"] = FoldSeries(
        "baseline",
        str(base["model_version"]),
        tuple(float(value) for value in base["ndcg@10_by_fold"]),
        float(base["ndcg@10_mean"]),
        float(base["ndcg@10_std"]),
    )
    for name in CANDIDATES:
        entry = selected[name]
        if entry["config_id"] != chosen.get(name):
            raise TrainedEvalError(f"{name}: la configuración de {TRAIN_CV_METRICS} no es la de {TRAIN_MANIFEST}")
        output[name] = FoldSeries(
            name,
            str(entry["config_id"]),
            tuple(float(value) for value in entry["ndcg@10_by_fold"]),
            float(entry["ndcg@10_mean"]),
            float(entry["ndcg@10_std"]),
        )
    lengths = {len(item.values) for item in output.values()}
    if len(lengths) != 1:
        raise TrainedEvalError(f"{TRAIN_CV_METRICS}: distinto número de pliegues entre modelos")
    return output


@dataclass(frozen=True)
class Condition:
    """Una condición de la spec §6.5: `passed` es `None` si se verifica en otra task (T05)."""

    passed: bool | None
    detail: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, **self.detail}


def cv_condition(model: FoldSeries, baseline: FoldSeries, required_folds: int) -> Condition:
    """Condición 1: NDCG@10 mayor que el del baseline en **cada** pliegue, con los `required_folds` pliegues."""
    deltas = [value - base for value, base in zip(model.values, baseline.values, strict=True)]
    above = sum(1 for delta in deltas if delta > 0)
    return Condition(
        passed=len(deltas) == required_folds and above == required_folds,
        detail={
            "folds": len(deltas),
            "required_folds": required_folds,
            "folds_above_baseline": above,
            "delta_ndcg@10_by_fold": deltas,
        },
    )


def holdout_condition(difference: Interval) -> Condition:
    """Condición 2: límite inferior del IC 95 % de la diferencia pareada NDCG@10 > 0."""
    return Condition(passed=difference.low > 0, detail={"delta_ndcg@10": difference.as_dict()})


def breakdown_condition(name: str) -> Condition:
    """Condición 3: desglose aditivo que suma el total; solo la LR puede cumplirla y se verifica en T05."""
    if name == "logreg":
        return Condition(
            passed=None,
            detail={
                "status": "pending_t05",
                "note": "modelo aditivo por construcción; el desglose que suma el total se verifica en T05",
            },
        )
    return Condition(
        passed=False,
        detail={"status": "not_additive", "note": "LightGBM no ofrece desglose aditivo exportable (decisión 5)"},
    )


@dataclass(frozen=True)
class CandidateResult:
    name: str
    conditions: tuple[Condition, Condition, Condition]
    outcome: str

    @property
    def meets_cv_and_test(self) -> bool:
        return self.conditions[0].passed is True and self.conditions[1].passed is True

    def as_dict(self) -> dict[str, Any]:
        return {
            "conditions": {
                "1_cv_all_folds_above_baseline": self.conditions[0].as_dict(),
                "2_test_paired_ci_above_zero": self.conditions[1].as_dict(),
                "3_additive_breakdown": self.conditions[2].as_dict(),
            },
            "meets_conditions_1_and_2": self.meets_cv_and_test,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class Decision:
    outcome: str
    promoted_model: str | None  # None: se queda el baseline; "logreg" sujeto a T05
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {"outcome": self.outcome, "candidate": self.promoted_model, "text": self.text}


def promotion(
    folds: Mapping[str, FoldSeries],
    differences: Mapping[str, Mapping[str, Interval]],
    required_folds: int,
) -> tuple[dict[str, CandidateResult], Decision]:
    """Condiciones 1-3 por candidato y decisión (spec §6.5 y §6.6, decisiones 4 y 5)."""
    candidates: dict[str, CandidateResult] = {}
    for name in CANDIDATES:
        conditions = (
            cv_condition(folds[name], folds["baseline"], required_folds),
            holdout_condition(differences[name][PRIMARY_METRIC]),
            breakdown_condition(name),
        )
        meets = conditions[0].passed is True and conditions[1].passed is True
        promotable = OUTCOME_CANDIDATE if meets else OUTCOME_NOT_PROMOTED
        # Solo la LR (aditiva) es promovible; LightGBM es referencia aunque cumpla 1 y 2 (decisión 5).
        candidates[name] = CandidateResult(name, conditions, promotable if name == "logreg" else OUTCOME_REFERENCE)
    if candidates["logreg"].meets_cv_and_test:
        decision = Decision(
            DECISION_CANDIDATE,
            "logreg",
            "La regresión logística cumple las condiciones 1 y 2: queda candidata a promoción, sujeta a T05 "
            "(condición 3, desglose aditivo que suma el total). Hasta entonces `baseline-1.0.0` sigue vigente.",
        )
    else:
        tail = (
            " LightGBM cumple las condiciones 1 y 2, pero no se promueve en v1 (decisión 5): queda como trabajo futuro."
            if candidates["lightgbm"].meets_cv_and_test
            else ""
        )
        decision = Decision(
            DECISION_BASELINE,
            None,
            "Ningún candidato promovible cumple las condiciones 1 y 2: se queda el baseline `baseline-1.0.0` "
            "(constitución §10.1)." + tail,
        )
    return candidates, decision


# ---------------------------------------------------------------------------
# Evaluación completa (pura: sin escritura)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainedEvaluation:
    """Todo lo que produce la medición en test de ML-05."""

    profile: str
    train_params: TrainParams
    eval_params: EvalParams
    schema_version: str
    labels_params_version: str
    baseline_version: str
    inputs: Mapping[str, str]
    models: Mapping[str, str]  # archivo → SHA-256 verificado
    configs: Mapping[str, str]
    test: EvalSplit
    per_student: Mapping[str, Mapping[str, StudentMetrics]]  # modelo → estudiante → métricas
    paired: PairedResult
    folds: Mapping[str, FoldSeries]
    candidates: Mapping[str, CandidateResult]
    decision: Decision
    cv_source: str
    required_folds: int


def evaluate_trained(
    data: TrainingData,
    logreg: LogRegModel,
    booster: lightgbm.Booster,
    baseline_reference: Mapping[str, Any],
    cv_metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    eval_params: EvalParams,
    train_params: TrainParams,
    schema: FeatureSchema,
    config: BaselineConfig,
    profile: str,
    models: Mapping[str, str],
    inputs: Mapping[str, str],
    cv_source: str,
) -> TrainedEvaluation:
    """Puntúa solo test con cada modelo, compara contra el baseline y decide (sin I/O)."""
    if logreg.feature_order != data.feature_order:
        raise TrainedEvalError("feature_order de la regresión logística distinto del de los datos")
    if tuple(booster.feature_name()) != data.feature_order:
        raise TrainedEvalError("feature_order de LightGBM distinto del de los datos")
    test = select_split(data, eval_params.split)
    threshold = eval_params.relevance_threshold
    scores = {
        "logreg": logreg.decision_function(test.features),
        "lightgbm": lightgbm_scores(booster, test.features, train_params.num_threads),
    }
    per_student = {
        name: defined(ranked_metrics(values, test.relevance, test.group, test.room_id, threshold), name)
        for name, values in scores.items()
    }
    baseline = defined(baseline_metrics(test, schema, config, threshold), "baseline")
    order = check_baseline_reference(baseline_reference["per_student_baseline"], baseline)
    paired = paired_intervals(
        order,
        baseline,
        per_student,
        resamples=eval_params.resamples,
        confidence=eval_params.confidence,
        rng=bootstrap_rng(eval_params),
    )
    total = next(item for item in baseline_reference["segments"] if item["id"] == SEGMENT_ID)
    check_baseline_intervals(total["baseline"], paired.baseline)
    folds = fold_series(cv_metrics, manifest)
    candidates, decision = promotion(folds, paired.differences, train_params.n_splits)
    return TrainedEvaluation(
        profile=profile,
        train_params=train_params,
        eval_params=eval_params,
        schema_version=data.schema_version,
        labels_params_version=data.labels_params_version,
        baseline_version=config.model_version,
        inputs=inputs,
        models=models,
        configs={name: folds[name].config_id for name in CANDIDATES},
        test=test,
        per_student=per_student,
        paired=paired,
        folds=folds,
        candidates=candidates,
        decision=decision,
        cv_source=cv_source,
        required_folds=train_params.n_splits,
    )


def _check_reference(reference: Mapping[str, Any], params: EvalParams, schema: FeatureSchema, version: str) -> None:
    checks = {
        "synthetic": (reference.get("synthetic"), True),
        "model_version": (reference.get("model_version"), version),
        "eval_params_version": (reference.get("eval_params_version"), params.version),
        "feature_schema_version": (reference.get("feature_schema_version"), schema.version),
        "evaluation.split": (reference.get("evaluation", {}).get("split"), params.split),
    }
    wrong = {key: actual for key, (actual, expected) in checks.items() if actual != expected}
    if wrong:
        raise TrainedEvalError(f"{BASELINE_METRICS}: valores inesperados {wrong}")


# ---------------------------------------------------------------------------
# Etapa (única I/O)
# ---------------------------------------------------------------------------


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def run_trained_evaluate(paths: ProjectPaths, params: TrainParams | None = None, profile: str | None = None) -> str:
    """Medición única en test de los modelos finales de `train` y decisión de promoción.

    `profile` (o `THEHOUSE_ML_TRAIN_PROFILE`): `full` lee `models/` y `reports/train_cv_metrics.json` y
    escribe `reports/trained_metrics.json` y `reports/trained_report.md`; `ci` lee y escribe en
    `profiles.ci.output_dir` (fuera de git) y nunca toca los reportes versionados. La condición 1 exige
    siempre los pliegues del perfil completo (5): el perfil `ci` (2 pliegues) no puede cumplirla.
    """
    # Importes diferidos: evitan ciclos (los reportes y la ablación importan este módulo).
    from thehouse_ml.models.ablation import ABLATION_METRICS, ABLATION_REPORT, evaluate_ablation
    from thehouse_ml.models.ablation_report import VERDICT_ASCII, ablation_document, fold_table, render_ablation_report
    from thehouse_ml.models.trained_report import metrics_document, render_report

    profile = profile or resolve_profile()
    base = params or load_train_params()
    models_dir, reports_dir = output_dirs(paths, base, profile)
    needed = [
        *(paths.data_processed / name for name in TRAINING_INPUTS),
        models_dir / TRAIN_MANIFEST,
        *(models_dir / name for name in MODEL_FILES.values()),
        reports_dir / TRAIN_CV_METRICS,
        paths.reports / BASELINE_METRICS,
    ]
    if not all(path.is_file() for path in needed):
        return TRAINED_SKIP_MESSAGE
    manifest = _read_json(models_dir / TRAIN_MANIFEST)
    cv_metrics = _read_json(reports_dir / TRAIN_CV_METRICS)
    for name, document in ((TRAIN_MANIFEST, manifest), (TRAIN_CV_METRICS, cv_metrics)):
        if document.get("profile") != profile:
            raise TrainedEvalError(f"{name}: perfil {document.get('profile')!r}, se esperaba {profile!r}")
        if document.get("synthetic") is not True:
            raise TrainedEvalError(f"{name}: synthetic debe ser true")
    models = verify_models(models_dir, manifest)
    inputs = verify_inputs(paths, manifest)
    logreg, booster = load_models(models_dir)

    eval_params = load_eval_params()
    schema = default_schema()
    config = default_config()
    reference = _read_json(paths.reports / BASELINE_METRICS)
    _check_reference(reference, eval_params, schema, config.model_version)
    data = load_training_data(paths, schema)
    cv_source = (reports_dir / TRAIN_CV_METRICS).relative_to(paths.root).as_posix()
    result = evaluate_trained(
        data,
        logreg,
        booster,
        reference,
        cv_metrics,
        manifest,
        eval_params=eval_params,
        train_params=base,
        schema=schema,
        config=config,
        profile=profile,
        models=models,
        inputs=inputs,
        cv_source=cv_source,
    )
    # ML-05 T04: pliegues con 4 métricas y ablación (solo train + validación) y segmentos de esta misma medición.
    ablation = evaluate_ablation(
        result.per_student,
        data,
        manifest,
        cv_metrics,
        reference,
        params=base.for_profile(profile),
        schema=schema,
        baseline_config=config,
        eval_params=eval_params,
        profile=profile,
        cv_source=cv_source,
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / TRAINED_METRICS
    _write_json(metrics_path, metrics_document(result))
    report_path = reports_dir / TRAINED_REPORT
    report_path.write_text(render_report(result, fold_table(ablation.cv)), encoding="utf-8", newline="\n")
    ablation_path = reports_dir / ABLATION_METRICS
    _write_json(ablation_path, ablation_document(ablation))
    ablation_report_path = reports_dir / ABLATION_REPORT
    ablation_report_path.write_text(render_ablation_report(ablation), encoding="utf-8", newline="\n")
    lr = result.paired.differences["logreg"][PRIMARY_METRIC]
    gbm = result.paired.differences["lightgbm"][PRIMARY_METRIC]
    return (
        f"evaluate (ML-05, perfil {profile}): test con {len(result.paired.students)} estudiantes; "
        f"dif. NDCG@10 vs baseline: logreg {lr.mean:+.3f} [{lr.low:+.3f}, {lr.high:+.3f}], "
        f"lightgbm {gbm.mean:+.3f} [{gbm.low:+.3f}, {gbm.high:+.3f}]; decisión: {result.decision.outcome} "
        f"(DATOS SINTÉTICOS) -> {report_path.relative_to(paths.root).as_posix()}; "
        f"{metrics_path.relative_to(paths.root).as_posix()}\n"
        f"evaluate (ML-05 T04, perfil {profile}): {ablation.cv.n_folds} pliegues con 4 métricas, ablación de "
        f"{'+'.join(ablation.cv.ablated_models)}, segmentos en test; hipótesis > $3,500: "
        f"{VERDICT_ASCII[ablation.hypothesis.verdict]} (DATOS SINTÉTICOS) -> "
        f"{ablation_report_path.relative_to(paths.root).as_posix()}; {ablation_path.relative_to(paths.root).as_posix()}"
    )


__all__ = [
    "CANDIDATES",
    "DECISION_BASELINE",
    "DECISION_CANDIDATE",
    "OUTCOME_CANDIDATE",
    "OUTCOME_NOT_PROMOTED",
    "OUTCOME_REFERENCE",
    "TRAINED_METRICS",
    "TRAINED_REPORT",
    "TRAINED_SKIP_MESSAGE",
    "CandidateResult",
    "Condition",
    "Decision",
    "EvalSplit",
    "FoldSeries",
    "PairedResult",
    "TrainedEvalError",
    "TrainedEvaluation",
    "baseline_metrics",
    "bootstrap_rng",
    "check_baseline_intervals",
    "check_baseline_reference",
    "cv_condition",
    "evaluate_trained",
    "fold_series",
    "holdout_condition",
    "load_models",
    "paired_intervals",
    "promotion",
    "ranked_metrics",
    "run_trained_evaluate",
    "select_split",
    "verify_inputs",
    "verify_models",
]
