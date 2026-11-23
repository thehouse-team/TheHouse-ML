"""Etapa ``evaluate`` del modelo base: métricas de ranking y reporte (ML-04 T06, ML-04-R06).

Lee (todo fuera de git, en `data/processed/`):

- `pairs.parquet` (ML-03 T06, esquema 2.0.0): pares candidatos, partición y features;
- `labels.parquet` (ML-04 T04): relevancia sintética 0-3 por par;
- `students.parquet` y `rooms.parquet` (ML-03 T04/T05): solo para los segmentos
  de presupuesto y para explicar el ejemplo.

**No** lee `labels_hidden_utility.parquet` ni `labels_hidden_weights.parquet`
(M-ML-032): la evaluación solo ve la relevancia, como verá ML-05. Una prueba
lo verifica.

El modelo base puntúa cada par con `score_features` (T03) y ordena con `rank`
(suma sin redondear descendente, desempate por `room_id`). Las métricas
(`thehouse_ml.evaluation.ranking`) se calculan por estudiante en la partición
test, con estudiantes de 2 o más candidatos, y se resumen con bootstrap por
estudiante. Como referencias se calculan el mismo ranking al azar (piso, con
semilla) y el orden ideal según las etiquetas (techo, NDCG = 1).

Escribe `reports/baseline_report.md` y `reports/baseline_metrics.json` (en
git), sin fechas ni rutas absolutas: correr `evaluate` dos veces no cambia nada.
Parámetros: `baseline_eval_params.json` (M-ML-034).
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd

from thehouse_ml.evaluation.ranking import (
    METRIC_NAMES,
    Interval,
    StudentMetrics,
    bootstrap_means,
    student_metrics,
)
from thehouse_ml.features import default_schema
from thehouse_ml.features.loader import FeatureSchema
from thehouse_ml.models.baseline_config import BaselineConfig, default_config
from thehouse_ml.models.baseline_score import ScoredRoom, ScoreResult, rank, score_features
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.synthetic.students_stage import _pct, _row

EVAL_PARAMS_RESOURCE = "baseline_eval_params.json"
PARAMS_PACKAGE = "thehouse_ml.models"
BASELINE_REPORT = "baseline_report.md"
BASELINE_METRICS = "baseline_metrics.json"
PAIRS_INPUT = "pairs.parquet"
LABELS_INPUT = "labels.parquet"
STUDENTS_INPUT = "students.parquet"
ROOMS_INPUT = "rooms.parquet"
INPUTS = (PAIRS_INPUT, LABELS_INPUT, STUDENTS_INPUT, ROOMS_INPUT)
EVALUATE_SKIP_MESSAGE = (
    "evaluate: omitido; faltan " + ", ".join(f"data/processed/{name}" for name in INPUTS) + " (ejecute `prepare`)"
)
SYNTHETIC_BANNER = "DATOS SINTÉTICOS — ETIQUETAS SINTÉTICAS"
RANKERS = ("baseline", "random", "ideal")
BASIS_PREFIXES = ("calibrado:", "SUPUESTO documentado")
METRIC_LABELS = {"ndcg@10": "NDCG@10", "ndcg@5": "NDCG@5", "precision@5": "Precision@5", "map": "MAP"}
HISTOGRAM_WIDTH = 40
SCORE_PERCENTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


class EvaluationError(RuntimeError):
    """Entradas o parámetros de la evaluación inválidos."""


# ---------------------------------------------------------------------------
# Parámetros (M-ML-034)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetRange:
    """Rango de `budget.max` (MXN, inclusivo; `None` = sin límite)."""

    id: str
    label: str
    min: int | None
    max: int | None

    def contains(self, value: int) -> bool:
        return (self.min is None or value >= self.min) and (self.max is None or value <= self.max)


@dataclass(frozen=True)
class EvalParams:
    """Parámetros de `baseline_eval_params.json`."""

    version: str
    decision: str
    split: str
    min_candidates: int
    relevance_threshold: int
    resamples: int
    confidence: float
    bootstrap_seed: int
    random_seed: int
    budget_ranges: tuple[BudgetRange, ...]
    example_min_candidates: int
    example_top_n: int
    bases: Mapping[str, str]


def _int(value: object, name: str, problems: list[str], minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        problems.append(f"{name}: se esperaba un entero ≥ {minimum}, llegó {value!r}")
        return minimum
    return value


def _budget_ranges(value: object, problems: list[str]) -> tuple[BudgetRange, ...]:
    if not isinstance(value, list) or not value:
        problems.append("budget_ranges: se esperaba una lista no vacía")
        return ()
    ranges: list[BudgetRange] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"id", "label", "min", "max"}:
            problems.append(f"budget_ranges[{index}]: se esperaba {{id, label, min, max}}")
            continue
        low, high = item["min"], item["max"]
        bounds_ok = all(b is None or (isinstance(b, int) and not isinstance(b, bool)) for b in (low, high))
        if not bounds_ok or not isinstance(item["id"], str) or not isinstance(item["label"], str):
            problems.append(f"budget_ranges[{index}]: tipos inválidos")
            continue
        ranges.append(BudgetRange(item["id"], item["label"], low, high))
    if ranges:
        if ranges[0].min is not None or ranges[-1].max is not None:
            problems.append("budget_ranges: el primer rango no debe tener mínimo ni el último máximo")
        for before, after in itertools.pairwise(ranges):
            if before.max is None or after.min != before.max + 1:
                problems.append(f"budget_ranges: {before.id} y {after.id} no son contiguos")
    return tuple(ranges)


def parse_eval_params(data: object) -> EvalParams:
    """Valida el JSON de parámetros; `EvaluationError` con todos los problemas."""
    if not isinstance(data, dict) or not isinstance(data.get("metadata"), dict):
        raise EvaluationError(f"{EVAL_PARAMS_RESOURCE}: se esperaba {{metadata, parameters}}")
    parameters = data.get("parameters")
    if not isinstance(parameters, dict):
        raise EvaluationError(f"{EVAL_PARAMS_RESOURCE}: falta parameters")
    problems: list[str] = []
    values: dict[str, object] = {}
    bases: dict[str, str] = {}
    expected = (
        "split",
        "min_candidates",
        "relevance_threshold",
        "bootstrap.resamples",
        "bootstrap.confidence",
        "bootstrap.seed",
        "random_order.seed",
        "budget_ranges",
        "example.min_candidates",
        "example.top_n",
    )
    for name in expected:
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
    extra = sorted(set(parameters) - set(expected))
    if extra:
        problems.append(f"parámetros desconocidos: {extra}")
    if problems:
        raise EvaluationError(f"{EVAL_PARAMS_RESOURCE} inválido:\n- " + "\n- ".join(problems))
    split = values["split"]
    if split not in ("train", "validation", "test"):
        problems.append(f"split: {split!r} no es train, validation ni test")
    confidence = values["bootstrap.confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, float) or not 0.0 < confidence < 1.0:
        problems.append(f"bootstrap.confidence: se esperaba un número en (0, 1), llegó {confidence!r}")
        confidence = 0.95
    params = EvalParams(
        version=str(data["metadata"].get("params_version", "")),
        decision=str(data["metadata"].get("decision", "")),
        split=str(split),
        min_candidates=_int(values["min_candidates"], "min_candidates", problems, 2),
        relevance_threshold=_int(values["relevance_threshold"], "relevance_threshold", problems, 1),
        resamples=_int(values["bootstrap.resamples"], "bootstrap.resamples", problems, 1),
        confidence=confidence,
        bootstrap_seed=_int(values["bootstrap.seed"], "bootstrap.seed", problems, 0),
        random_seed=_int(values["random_order.seed"], "random_order.seed", problems, 0),
        budget_ranges=_budget_ranges(values["budget_ranges"], problems),
        example_min_candidates=_int(values["example.min_candidates"], "example.min_candidates", problems, 2),
        example_top_n=_int(values["example.top_n"], "example.top_n", problems, 1),
        bases=bases,
    )
    if problems:
        raise EvaluationError(f"{EVAL_PARAMS_RESOURCE} inválido:\n- " + "\n- ".join(problems))
    return params


def load_eval_params(path: Path | None = None) -> EvalParams:
    """Carga `baseline_eval_params.json` (empaquetado) o el archivo indicado."""
    if path is None:
        text = resources.files(PARAMS_PACKAGE).joinpath(EVAL_PARAMS_RESOURCE).read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    return parse_eval_params(json.loads(text))


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def _read(paths: ProjectPaths, name: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Única lectura de Parquet de la etapa: solo nombres de `INPUTS` (nunca los archivos ocultos)."""
    if name not in INPUTS:
        raise EvaluationError(f"{name} no es una entrada de la evaluación")
    return pd.read_parquet(paths.data_processed / name, columns=None if columns is None else list(columns))


def join_labels(pairs: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Une pares y etiquetas por (`student_id`, `room_id`); cada par debe tener exactamente una etiqueta."""
    keys = ["student_id", "room_id"]
    for name, frame in ((PAIRS_INPUT, pairs), (LABELS_INPUT, labels)):
        if frame.duplicated(keys).any():
            raise EvaluationError(f"{name}: pares (student_id, room_id) duplicados")
    if not bool(labels["synthetic"].all()):
        raise EvaluationError(f"{LABELS_INPUT}: hay etiquetas sin synthetic = true")
    merged = pairs.merge(
        labels[[*keys, "split", "relevance"]].rename(columns={"split": "label_split"}),
        on=keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        raise EvaluationError(
            f"{unmatched} pares sin etiqueta o etiquetas sin par entre {PAIRS_INPUT} y {LABELS_INPUT}"
        )
    if not (merged["split"] == merged["label_split"]).all():
        raise EvaluationError(f"la partición de {LABELS_INPUT} no coincide con la de {PAIRS_INPUT}")
    merged = merged.drop(columns=["_merge", "label_split"])
    merged["relevance"] = merged["relevance"].astype("int64")
    return merged.sort_values(keys, kind="mergesort").reset_index(drop=True)


def score_frame(frame: pd.DataFrame, schema: FeatureSchema, config: BaselineConfig) -> list[ScoreResult]:
    """`score_features` (T03) para cada par, en el orden del DataFrame."""
    order = list(schema.feature_order)
    return [
        score_features([float(value) for value in row], config, schema)
        for row in frame[order].itertuples(index=False, name=None)
    ]


# ---------------------------------------------------------------------------
# Rankings por estudiante
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentEval:
    """Un estudiante de la partición evaluada con sus rankings y métricas."""

    student_id: str
    university_code: str
    seed_verified: bool
    budget_max: int
    candidates: int
    relevant: int
    baseline_order: tuple[str, ...]
    metrics: Mapping[str, StudentMetrics] | None  # None = excluido

    @property
    def included(self) -> bool:
        return self.metrics is not None


def random_order(room_ids: Sequence[str], student_id: str, seed: int) -> list[str]:
    """Permutación al azar de los candidatos (en orden de `room_id`) con semilla por estudiante."""
    order = sorted(room_ids)
    random.Random(f"random_order:{seed}:{student_id}").shuffle(order)  # noqa: S311 - referencia reproducible
    return order


def ideal_order(relevance: Mapping[str, int]) -> list[str]:
    """Orden ideal: relevancia descendente, desempate por `room_id`."""
    return sorted(relevance, key=lambda room_id: (-relevance[room_id], room_id))


def evaluate_students(
    frame: pd.DataFrame,
    results: Sequence[ScoreResult],
    budgets: Mapping[str, int],
    params: EvalParams,
) -> list[StudentEval]:
    """Rankings del baseline, al azar e ideal y sus métricas para cada estudiante de `params.split`."""
    evaluated: list[StudentEval] = []
    split = frame["split"] == params.split
    subset = frame[split]
    split_results = [result for result, keep in zip(results, split.tolist(), strict=True) if keep]
    groups: dict[str, list[int]] = {}
    for position, student_id in enumerate(subset["student_id"].tolist()):
        groups.setdefault(str(student_id), []).append(position)
    room_ids = subset["room_id"].astype(str).tolist()
    relevances = subset["relevance"].astype(int).tolist()
    codes = subset["university_code"].astype(str).tolist()
    verified = subset["seed_verified"].astype(bool).tolist()
    for student_id in sorted(groups):
        rows = groups[student_id]
        relevance = {room_ids[i]: relevances[i] for i in rows}
        baseline = [item.room_id for item in rank(ScoredRoom(room_ids[i], split_results[i]) for i in rows)]
        relevant = sum(1 for value in relevance.values() if value >= params.relevance_threshold)
        metrics: dict[str, StudentMetrics] | None = None
        if len(rows) >= params.min_candidates:
            orders = {
                "baseline": baseline,
                "random": random_order(list(relevance), student_id, params.random_seed),
                "ideal": ideal_order(relevance),
            }
            computed = {
                name: student_metrics([relevance[r] for r in order], params.relevance_threshold)
                for name, order in orders.items()
            }
            if all(value is not None for value in computed.values()):
                metrics = {name: value for name, value in computed.items() if value is not None}
        if student_id not in budgets:
            raise EvaluationError(f"{STUDENTS_INPUT}: falta el estudiante {student_id}")
        evaluated.append(
            StudentEval(
                student_id=student_id,
                university_code=codes[rows[0]],
                seed_verified=verified[rows[0]],
                budget_max=budgets[student_id],
                candidates=len(rows),
                relevant=relevant,
                baseline_order=tuple(baseline),
                metrics=metrics,
            )
        )
    return evaluated


# ---------------------------------------------------------------------------
# Segmentos y bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """Grupo de estudiantes de la partición evaluada."""

    id: str
    kind: str
    label: str
    students: tuple[StudentEval, ...]

    @property
    def included(self) -> tuple[StudentEval, ...]:
        return tuple(item for item in self.students if item.included)

    @property
    def excluded_few_candidates(self) -> int:
        return sum(1 for item in self.students if not item.included and item.candidates < 2)

    @property
    def excluded_no_relevant(self) -> int:
        return sum(1 for item in self.students if not item.included and item.candidates >= 2)


def build_segments(students: Sequence[StudentEval], schema: FeatureSchema, params: EvalParams) -> list[Segment]:
    """Total, por campus, sin `seed_sin_verificar` y por rango de `budget.max` (constitución §11.3)."""
    segments = [Segment("total", "total", "Total", tuple(students))]
    for code in schema.catalog("university_code"):
        group = tuple(item for item in students if item.university_code == code)
        if group:
            suffix = "" if group[0].seed_verified else " (sin verificar)"
            segments.append(Segment(f"campus:{code}", "campus", f"{code}{suffix}", group))
    segments.append(
        Segment(
            "without_seed_sin_verificar",
            "verified",
            "Sin `seed_sin_verificar`",
            tuple(item for item in students if item.seed_verified),
        )
    )
    for band in params.budget_ranges:
        group = tuple(item for item in students if band.contains(item.budget_max))
        segments.append(Segment(f"budget:{band.id}", "budget", f"`budget.max` {band.label}", group))
    return segments


@dataclass(frozen=True)
class SegmentResult:
    """Medias con intervalo por ranker, techo ideal y diferencia baseline - azar."""

    segment: Segment
    intervals: Mapping[str, Mapping[str, Interval]]  # ranker o "lift" → métrica → intervalo
    ideal: Mapping[str, float]


def summarize_segment(segment: Segment, params: EvalParams) -> SegmentResult | None:
    """Bootstrap pareado por estudiante del segmento; `None` si no queda ningún estudiante evaluado."""
    included = segment.included
    if not included:
        return None
    series: dict[str, list[float]] = {}
    for ranker in ("baseline", "random"):
        for index, metric in enumerate(METRIC_NAMES):
            series[f"{ranker}:{metric}"] = [_metrics(item)[ranker].values()[index] for item in included]
    for index, metric in enumerate(METRIC_NAMES):
        series[f"lift:{metric}"] = [
            _metrics(item)["baseline"].values()[index] - _metrics(item)["random"].values()[index] for item in included
        ]
    rng = random.Random(f"bootstrap:{params.bootstrap_seed}:{segment.id}")  # noqa: S311 - bootstrap reproducible
    flat = bootstrap_means(series, resamples=params.resamples, confidence=params.confidence, rng=rng)
    intervals: dict[str, dict[str, Interval]] = {}
    for key, interval in flat.items():
        ranker, metric = key.split(":", 1)
        intervals.setdefault(ranker, {})[metric] = interval
    ideal = {
        metric: math.fsum(_metrics(item)["ideal"].values()[index] for item in included) / len(included)
        for index, metric in enumerate(METRIC_NAMES)
    }
    return SegmentResult(segment, intervals, ideal)


def _metrics(item: StudentEval) -> Mapping[str, StudentMetrics]:
    if item.metrics is None:
        raise EvaluationError(f"{item.student_id} no tiene métricas")
    return item.metrics


# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------


def metrics_document(
    results: Sequence[SegmentResult | None],
    segments: Sequence[Segment],
    students: Sequence[StudentEval],
    params: EvalParams,
    schema: FeatureSchema,
    config: BaselineConfig,
    labels_version: str,
) -> dict[str, Any]:
    """Contenido de `reports/baseline_metrics.json` (referencia que ML-05 debe superar)."""
    total = segments[0]
    segment_entries: list[dict[str, Any]] = []
    for segment, result in zip(segments, results, strict=True):
        entry: dict[str, Any] = {
            "id": segment.id,
            "kind": segment.kind,
            "label": segment.label,
            "students_in_split": len(segment.students),
            "students_evaluated": len(segment.included),
            "excluded_fewer_than_min_candidates": segment.excluded_few_candidates,
            "excluded_no_relevant_room": segment.excluded_no_relevant,
        }
        if result is not None:
            for ranker in ("baseline", "random", "lift"):
                key = "lift_baseline_minus_random" if ranker == "lift" else ranker
                entry[key] = {metric: result.intervals[ranker][metric].as_dict() for metric in METRIC_NAMES}
            entry["ideal"] = {metric: {"mean": result.ideal[metric]} for metric in METRIC_NAMES}
        segment_entries.append(entry)
    return {
        "title": f"{SYNTHETIC_BANNER} — métricas del modelo base {config.model_version}",
        "synthetic": True,
        "warning": (
            "Métricas calculadas con etiquetas SINTÉTICAS (utilidad oculta documentada más ruido, M-ML-032): "
            "no son desempeño real ni preferencias reales (constitución §8.2 y §20.3); se reemplazan con "
            "señales reales en ML-07."
        ),
        "model_version": config.model_version,
        "feature_schema_version": schema.version,
        "labels_params_version": labels_version,
        "eval_params_version": params.version,
        "decision": params.decision,
        "reference_for": "ML-05 (constitución §10.1: sin mejora demostrada sobre el modelo base no se promueve)",
        "primary_metric": "ndcg@10",
        "evaluation": {
            "split": params.split,
            "min_candidates": params.min_candidates,
            "students_in_split": len(total.students),
            "students_evaluated": len(total.included),
            "excluded_fewer_than_min_candidates": total.excluded_few_candidates,
            "excluded_no_relevant_room": total.excluded_no_relevant,
            "ranking": "suma sin redondear descendente, desempate por room_id ascendente (rank, T03)",
        },
        "metrics": {
            "names": list(METRIC_NAMES),
            "relevance_threshold": params.relevance_threshold,
            "ndcg": "DCG@k = sum_{i<=k} (2^rel_i - 1) / log2(i + 1); NDCG@k = DCG@k / IDCG@k",
            "precision@5": "relevantes (relevance >= umbral) en las 5 primeras / 5 (denominador fijo)",
            "map": "media de AP; AP = (1/R) sum_{i relevante} Precision@i sobre la lista completa",
            "no_relevant_room": "NDCG o AP no definidos: el estudiante se excluye y se cuenta",
        },
        "bootstrap": {
            "unit": "student",
            "resamples": params.resamples,
            "confidence": params.confidence,
            "interval": "percentil con interpolación lineal",
            "paired": True,
            "seed": params.bootstrap_seed,
            "rng": 'random.Random("bootstrap:<seed>:<segment_id>").choices',
        },
        "random_reference": {
            "seed": params.random_seed,
            "rng": 'random.Random("random_order:<seed>:<student_id>").shuffle sobre room_id ordenados',
        },
        "budget_ranges": [
            {"id": band.id, "label": band.label, "min": band.min, "max": band.max} for band in params.budget_ranges
        ],
        "segments": segment_entries,
        "per_student_baseline": [
            {
                "student_id": item.student_id,
                "university_code": item.university_code,
                "seed_verified": item.seed_verified,
                "budget_max": item.budget_max,
                "candidates": item.candidates,
                **_metrics(item)["baseline"].as_dict(),
            }
            for item in students
            if item.included
        ],
    }


def _f3(value: float) -> str:
    return f"{value:.3f}"


def _ci(interval: Interval) -> str:
    return f"{_f3(interval.mean)} [{_f3(interval.low)}, {_f3(interval.high)}]"


def _signed_ci(interval: Interval) -> str:
    return f"{interval.mean:+.3f} [{interval.low:+.3f}, {interval.high:+.3f}]"


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _metrics_table(rows: Sequence[SegmentResult | None], segments: Sequence[Segment]) -> list[str]:
    header = ["Segmento", "Estudiantes", "Fuera", *(f"{METRIC_LABELS[m]} [IC 95 %]" for m in METRIC_NAMES)]
    lines = [_row(header), _row(["---", "---:", "---:", *(["---:"] * len(METRIC_NAMES))])]
    for segment, result in zip(segments, rows, strict=True):
        fuera = segment.excluded_few_candidates + segment.excluded_no_relevant
        if result is None:
            cells = ["—"] * len(METRIC_NAMES)
        else:
            cells = [_ci(result.intervals["baseline"][metric]) for metric in METRIC_NAMES]
        lines.append(_row([segment.label, f"{len(segment.included):,}", f"{fuera}", *cells]))
    return lines


def _comparison_table(rows: Sequence[SegmentResult | None], segments: Sequence[Segment], metric: str) -> list[str]:
    header = ["Segmento", "Azar (piso)", "Baseline", "Ideal (techo)", "Baseline - azar [IC 95 %]", "Brecha cerrada"]
    lines = [_row(header), _row(["---", "---:", "---:", "---:", "---:", "---:"])]
    for segment, result in zip(segments, rows, strict=True):
        if result is None:
            lines.append(_row([segment.label, "—", "—", "—", "—", "—"]))
            continue
        base = result.intervals["baseline"][metric].mean
        rand = result.intervals["random"][metric].mean
        ideal = result.ideal[metric]
        gap = (base - rand) / (ideal - rand) if ideal > rand else math.nan
        lines.append(
            _row(
                [
                    segment.label,
                    _ci(result.intervals["random"][metric]),
                    _ci(result.intervals["baseline"][metric]),
                    _f3(ideal),
                    _signed_ci(result.intervals["lift"][metric]),
                    "—" if math.isnan(gap) else f"{100 * gap:.0f} %",
                ]
            )
        )
    return lines


def _histogram(scores: pd.Series) -> list[str]:
    bins = [(low, low + 9) for low in range(0, 90, 10)] + [(90, 100)]
    counts = [int(((scores >= low) & (scores <= high)).sum()) for low, high in bins]
    peak = max(counts) or 1
    lines = ["```text"]
    for (low, high), count in zip(bins, counts, strict=True):
        bar = "#" * round(HISTOGRAM_WIDTH * count / peak)
        lines.append(f"{low:>3}-{high:<3} | {bar:<{HISTOGRAM_WIDTH}} {count:>7,} ({_pct(count, len(scores))})")
    lines.append("```")
    return lines


def _distribution_section(results: Sequence[ScoreResult], config: BaselineConfig) -> list[str]:
    scores = pd.Series([item.match_score for item in results], dtype="int64")
    quantiles = scores.quantile(list(SCORE_PERCENTILES), interpolation="linear")
    lines = [
        "## 2. Distribución de `matchScore`",
        "",
        f"Todos los pares candidatos de `{PAIRS_INPUT}` (las tres particiones; **{len(scores):,}** pares que pasan"
        " HC01-HC03). Histograma por decenas (la última barra incluye 100):",
        "",
        *_histogram(scores),
        "",
        _row(["Mín.", *(f"p{round(q * 100)}" for q in SCORE_PERCENTILES), "Máx.", "Media"]),
        _row(["---:"] * (len(SCORE_PERCENTILES) + 3)),
        _row(
            [
                f"{int(scores.min())}",
                *(f"{float(quantiles[q]):g}" for q in SCORE_PERCENTILES),
                f"{int(scores.max())}",
                f"{float(scores.mean()):.1f}",
            ]
        ),
        "",
        "Percentiles con interpolación lineal. Puntos por factor (sin redondear) en los mismos pares:",
        "",
        _row(["Factor", "Máximo", "Media", "p25", "Mediana", "p75", "Con puntos completos", "Con 0 puntos"]),
        _row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for index, factor in enumerate(config.factors):
        points = pd.Series([item.breakdown[index].raw_points for item in results], dtype="float64")
        p25, p50, p75 = (float(points.quantile(q, interpolation="linear")) for q in (0.25, 0.5, 0.75))
        full = int((points >= factor.weight - 1e-9).sum())
        zero = int((points <= 1e-9).sum())
        lines.append(
            _row(
                [
                    f"{factor.label} (`{factor.id}`)",
                    f"{factor.weight:g}",
                    f"{float(points.mean()):.1f}",
                    f"{p25:.1f}",
                    f"{p50:.1f}",
                    f"{p75:.1f}",
                    _pct(full, len(points)),
                    _pct(zero, len(points)),
                ]
            )
        )
    return lines


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    return [str(item) for item in value]  # type: ignore[attr-defined]


def _optional(value: object) -> object | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def explain_factor(factor_id: str, value: float, student: Mapping[str, Any], room: Mapping[str, Any]) -> str:
    """Frase corta que explica los puntos de un factor con los datos del par (sin datos personales)."""
    price = int(room["price_mxn"])
    if factor_id == "budget":
        low = _optional(student["budget.min"])
        if value == 0.0:
            low_text = f"{_money(float(low))}" if low is not None else "sin mínimo"  # type: ignore[arg-type]
            return f"renta {_money(price)} dentro de su rango ({low_text} a {_money(int(student['budget.max']))})"
        return f"renta {_money(price)}, {100 * value:.0f} % por debajo de su mínimo {_money(float(low))}"  # type: ignore[arg-type]
    if factor_id == "distance":
        km = float(room["_distance_km"])
        if value <= 0.5:
            zone = "≤ 0.5 de su máximo: puntos completos"
        elif value <= 1.0:
            zone = "entre 0.5 y 1 de su máximo: decae"
        else:
            zone = "más lejos que su máximo: 0 puntos"
        return f"{km:.1f} km del campus, `distance_ratio` {value:.2f} ({zone})"
    if factor_id == "essential_services":
        wanted = _as_list(student["essentialServices"])
        if not wanted:
            return "no pidió servicios esenciales"
        offered = set(_as_list(room["services"]))
        return f"cubre {sum(1 for item in wanted if item in offered)} de {len(wanted)} servicios pedidos"
    if factor_id == "place_type":
        same = "coincide con" if value == 1.0 else "no coincide con"
        return f"`{room['placeType']}` {same} lo pedido (`{student['placeType']}`)"
    if factor_id == "property_type":
        wanted = _as_list(student["propertyTypes"])
        if not wanted:
            return f"`{room['propertyType']}`; sin preferencia de inmueble"
        same = "está en" if value == 1.0 else "no está en"
        return f"`{room['propertyType']}` {same} lo pedido ({', '.join(f'`{item}`' for item in wanted)})"
    wanted_set = set(_as_list(student["essentialServices"]))
    extras = [item for item in _as_list(room["services"]) if item not in wanted_set]
    noun = "servicio extra no pedido" if len(extras) == 1 else "servicios extra no pedidos"
    return f"{len(extras)} {noun} (`extra_services_ratio` {value:.2f})"


def pick_example(students: Sequence[StudentEval], params: EvalParams) -> StudentEval | None:
    """Primer estudiante evaluado (por id) de un campus verificado con suficientes candidatos."""
    for item in students:
        if item.included and item.seed_verified and item.candidates >= params.example_min_candidates:
            return item
    return None


def _example_section(
    example: StudentEval | None,
    frame: pd.DataFrame,
    results: Sequence[ScoreResult],
    students_table: pd.DataFrame,
    rooms_table: pd.DataFrame,
    params: EvalParams,
    config: BaselineConfig,
) -> list[str]:
    lines = ["## 3. Ejemplo explicado: un estudiante con su top " + str(params.example_top_n), ""]
    if example is None:
        return [*lines, "Ningún estudiante cumple la regla del ejemplo.", ""]
    student = students_table.loc[example.student_id].to_dict()
    positions = {
        str(room_id): index
        for index, (room_id, student_id) in enumerate(
            zip(frame["room_id"].tolist(), frame["student_id"].tolist(), strict=True)
        )
        if student_id == example.student_id
    }
    relevance = {room_id: int(frame["relevance"].iloc[index]) for room_id, index in positions.items()}
    max_km = _optional(student["maxDistanceKm"])
    low = _optional(student["budget.min"])
    lines += [
        f"Regla fija (M-ML-034): el primer estudiante de `{params.split}` por `student_id`, de un campus verificado,"
        f" con al menos {params.example_min_candidates} candidatos. Estudiante y cuartos **sintéticos**; se omiten"
        " `livingWith` y mascotas (restricciones duras que ya filtraron a los candidatos).",
        "",
        _row(["Campo", "Valor"]),
        _row(["---", "---"]),
        _row(["Estudiante", f"`{example.student_id}` (campus `{example.university_code}`)"]),
        _row(
            [
                "Presupuesto",
                f"{_money(float(low)) if low is not None else 'sin mínimo'} a {_money(example.budget_max)}",  # type: ignore[arg-type]
            ]
        ),
        _row(
            [
                "Distancia máxima",
                f"{float(max_km):g} km" if max_km is not None else "sin dato (se usa 30 km, M-ML-022)",  # type: ignore[arg-type]
            ]
        ),
        _row(["Transporte", f"`{student['transport']}`"]),
        _row(["Tipo de lugar", f"`{student['placeType']}`"]),
        _row(
            [
                "Tipo de inmueble",
                ", ".join(f"`{item}`" for item in _as_list(student["propertyTypes"])) or "sin preferencia",
            ]
        ),
        _row(
            [
                "Servicios esenciales",
                ", ".join(f"`{item}`" for item in _as_list(student["essentialServices"])) or "ninguno",
            ]
        ),
        _row(["Candidatos", f"{example.candidates} (relevantes con relevancia ≥ 2: {example.relevant})"]),
        "",
        "Top " + str(params.example_top_n) + " del modelo base (orden por la suma sin redondear; puntos del desglose"
        " con 1 decimal) y la relevancia sintética de cada cuarto:",
        "",
        _row(
            [
                "#",
                "Cuarto",
                "Renta",
                "`matchScore`",
                *(f"{factor.id} ({factor.weight:g})" for factor in config.factors),
                "Suma sin redondear",
                "Relevancia",
            ]
        ),
        _row(["---:", "---", "---:", "---:", *(["---:"] * len(config.factors)), "---:", "---:"]),
    ]
    top = example.baseline_order[: params.example_top_n]
    for rank_position, room_id in enumerate(top, start=1):
        result = results[positions[room_id]]
        room = rooms_table.loc[room_id]
        lines.append(
            _row(
                [
                    f"{rank_position}",
                    f"`{room_id}`",
                    _money(int(room["price_mxn"])),
                    f"**{result.match_score}**",
                    *(f"{item.points:.1f}" for item in result.breakdown),
                    f"{result.raw_total:.6f}",
                    f"{relevance[room_id]}",
                ]
            )
        )
    lines += ["", "Por qué recibe esos puntos cada cuarto:", ""]
    for rank_position, room_id in enumerate(top, start=1):
        index = positions[room_id]
        result = results[index]
        room = {**rooms_table.loc[room_id].to_dict(), "_distance_km": float(frame["distance_km"].iloc[index])}
        parts = [
            f"{factor.label} {item.points:.1f}/{factor.weight:g}: "
            + explain_factor(factor.id, item.value, student, room)
            for factor, item in zip(config.factors, result.breakdown, strict=True)
        ]
        lines.append(
            f"{rank_position}. **`{room_id}` — `matchScore` {result.match_score}** (relevancia sintética"
            f" {relevance[room_id]}). " + "; ".join(parts) + "."
        )
    best = [room_id for room_id in example.baseline_order if relevance[room_id] == 3]
    where = [example.baseline_order.index(room_id) + 1 for room_id in best]
    metrics = _metrics(example)["baseline"]
    lines += [
        "",
        f"Sus cuartos con relevancia 3 quedan en las posiciones {', '.join(str(p) for p in where)} del ranking"
        f" del modelo base. Métricas de este estudiante: NDCG@10 {_f3(metrics.ndcg_10)}, NDCG@5"
        f" {_f3(metrics.ndcg_5)}, Precision@5 {_f3(metrics.precision_5)}, AP {_f3(metrics.average_precision)}.",
        "",
        "Salida del primer cuarto en el formato de la spec §6.4 (`matchScore` + `breakdown`):",
        "",
        "```json",
        json.dumps(results[positions[top[0]]].to_dict(), ensure_ascii=False, indent=2),
        "```",
        "",
        "El desglose suma el total antes de redondear (constitución §12): la suma de los puntos con 1 decimal puede"
        " diferir de `matchScore` hasta en 0.8 (spec §6.3).",
        "",
    ]
    return lines


def render_report(
    frame: pd.DataFrame,
    results: Sequence[ScoreResult],
    students: Sequence[StudentEval],
    segments: Sequence[Segment],
    summaries: Sequence[SegmentResult | None],
    students_table: pd.DataFrame,
    rooms_table: pd.DataFrame,
    params: EvalParams,
    schema: FeatureSchema,
    config: BaselineConfig,
    labels_version: str,
) -> str:
    """Reporte Markdown determinista del modelo base."""
    total = segments[0]
    main = summaries[0]
    weights = " · ".join(f"{factor.id} {factor.weight:g}" for factor in config.factors)
    lines = [
        f"# {SYNTHETIC_BANNER} — Reporte del modelo base {config.model_version} (esquema {schema.version})",
        "",
        "> **DATOS SINTÉTICOS — ETIQUETAS SINTÉTICAS.** Estudiantes, cuartos y etiquetas son **sintéticos**. Las",
        "> etiquetas de relevancia **no son preferencias reales**: salen de una utilidad oculta documentada más ruido",
        "> (constitución §8.2, M-ML-032), con riesgo de **circularidad** (ML-04-RK02). Ninguna métrica de este reporte",
        "> es desempeño real (constitución §20.3); se reemplazan con señales reales en ML-07.",
        "",
        "Generado por `uv run python -m thehouse_ml evaluate` (`thehouse_ml.models.baseline_report`). Modelo"
        f" `{config.model_version}` ({weights}); esquema de features {schema.version}; etiquetas"
        f" `labels_params.json` {labels_version}; parámetros de evaluación `{EVAL_PARAMS_RESOURCE}` {params.version}"
        f" (decisión {params.decision}). Métricas exactas en `reports/{BASELINE_METRICS}`.",
        "",
        "## 1. Resumen",
        "",
        f"- Evaluación en la partición **`{params.split}`**: {len(total.students):,} estudiantes; evaluados"
        f" **{len(total.included):,}**; fuera por tener menos de {params.min_candidates} candidatos:"
        f" **{total.excluded_few_candidates}**; fuera por no tener ningún cuarto relevante:"
        f" **{total.excluded_no_relevant}**.",
    ]
    if main is not None:
        lift = main.intervals["lift"]["ndcg@10"]
        lines += [
            f"- **NDCG@10 (principal): {_ci(main.intervals['baseline']['ndcg@10'])}** frente a"
            f" {_ci(main.intervals['random']['ndcg@10'])} con un orden al azar; diferencia {_signed_ci(lift)}"
            " (bootstrap pareado).",
            f"- Estas cifras son la **referencia que ML-05 debe superar** (constitución §10.1), sobre las mismas"
            f" etiquetas sintéticas y los mismos {len(total.included):,} estudiantes.",
        ]
    lines += [
        "- Entradas: "
        + ", ".join(f"`data/processed/{name}`" for name in INPUTS)
        + ". **No** se leen `labels_hidden_utility.parquet` ni `labels_hidden_weights.parquet`.",
        "",
        *_distribution_section(results, config),
        "",
        *_example_section(pick_example(students, params), frame, results, students_table, rooms_table, params, config),
        "## 4. Métricas de ranking por segmento",
        "",
        f"Media por estudiante e **intervalo percentil del {params.confidence * 100:g} %** por bootstrap: se"
        f" remuestrean **estudiantes** con reemplazo ({params.resamples:,} remuestreos por segmento, semilla"
        f' {params.bootstrap_seed}, `random.Random("bootstrap:<seed>:<segment_id>")`). NDCG con ganancia'
        " `2^rel - 1` y descuento `log2(posición + 1)`; Precision@5 y MAP con **relevante = relevancia ≥"
        f" {params.relevance_threshold}**. «Fuera» = estudiantes del segmento excluidos (menos de"
        f" {params.min_candidates} candidatos o sin cuarto relevante).",
        "",
        *_metrics_table(summaries, segments),
        "",
        "Los segmentos de un solo campus tienen pocos estudiantes (unos 30): sus intervalos son anchos y no permiten"
        " concluir diferencias entre campus. Los rangos de presupuesto usan `budget.max` del estudiante.",
        "",
        "## 5. Comparación contra el azar y el techo",
        "",
        "Piso: el mismo cálculo con los candidatos en un orden al azar (semilla"
        f' {params.random_seed}, `random.Random("random_order:<seed>:<student_id>")`). Techo: el orden ideal según'
        " las etiquetas (NDCG = 1). «Brecha cerrada» = (baseline - azar) / (ideal - azar). La diferencia usa el"
        " mismo remuestreo que las medias (pareado).",
        "",
        "### 5.1 NDCG@10 (principal)",
        "",
        *_comparison_table(summaries, segments, "ndcg@10"),
        "",
    ]
    for metric in METRIC_NAMES[1:]:
        lines += [f"### 5.{METRIC_NAMES.index(metric) + 1} {METRIC_LABELS[metric]}", ""]
        lines += [*_comparison_table(summaries, segments, metric), ""]
    lines += [
        "Precision@5 divide siempre entre 5: con menos de 5 relevantes el techo es menor que 1 (con los cortes de"
        " M-ML-032 hay ⌈0.2·n⌉ relevantes por estudiante).",
        "",
        "## 6. Cómo leer estas cifras",
        "",
        "- El modelo base **no** se ajustó con estas etiquetas (spec §7 regla 3): sus pesos se fijaron antes de"
        " generarlas.",
        "- Las etiquetas premian lo mismo que las reglas en lo general, pero con otros pesos y otras curvas"
        " (correlación de Spearman en `reports/labels_synthetic.md` §6): el margen entre el baseline y el"
        " techo es lo que un modelo entrenado (ML-05) podría recuperar **de esta función sintética**, no de"
        " preferencias reales.",
        "- Sin atributos sensibles: los segmentos son campus y presupuesto (constitución §11.3); ningún factor ni"
        " segmento usa género, edad u origen.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Etapa
# ---------------------------------------------------------------------------


def run_evaluate(paths: ProjectPaths, params: EvalParams | None = None) -> str:
    """Evalúa el modelo base y escribe `reports/baseline_report.md` y `reports/baseline_metrics.json`."""
    if not all((paths.data_processed / name).is_file() for name in INPUTS):
        return EVALUATE_SKIP_MESSAGE
    params = params or load_eval_params()
    schema = default_schema()
    config = default_config()
    pairs = _read(paths, PAIRS_INPUT)
    labels = _read(paths, LABELS_INPUT)
    versions = sorted(set(labels["labels_params_version"].astype(str)))
    if len(versions) != 1:
        raise EvaluationError(f"{LABELS_INPUT}: se esperaba una sola versión de labels_params, hay {versions}")
    frame = join_labels(pairs, labels)
    students_table = _read(paths, STUDENTS_INPUT).set_index("student_id")
    rooms_table = _read(paths, ROOMS_INPUT).set_index("room_id")
    budgets = {str(key): int(value) for key, value in students_table["budget.max"].items()}

    results = score_frame(frame, schema, config)
    students = evaluate_students(frame, results, budgets, params)
    if not students:
        raise EvaluationError(f"no hay estudiantes en la partición {params.split}")
    segments = build_segments(students, schema, params)
    summaries = [summarize_segment(segment, params) for segment in segments]

    paths.reports.mkdir(parents=True, exist_ok=True)
    document = metrics_document(summaries, segments, students, params, schema, config, versions[0])
    metrics_path = paths.reports / BASELINE_METRICS
    metrics_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    report = render_report(
        frame, results, students, segments, summaries, students_table, rooms_table, params, schema, config, versions[0]
    )
    report_path = paths.reports / BASELINE_REPORT
    report_path.write_text(report, encoding="utf-8", newline="\n")
    main = summaries[0]
    headline = "" if main is None else f"; NDCG@10 {_ci(main.intervals['baseline']['ndcg@10'])}"
    return (
        f"evaluate: {config.model_version} en {params.split} con {len(segments[0].included)} estudiantes"
        f"{headline} (DATOS SINTÉTICOS) -> {report_path.as_posix()}; {metrics_path.as_posix()}"
    )


__all__ = [
    "BASELINE_METRICS",
    "BASELINE_REPORT",
    "EVALUATE_SKIP_MESSAGE",
    "EVAL_PARAMS_RESOURCE",
    "INPUTS",
    "SYNTHETIC_BANNER",
    "BudgetRange",
    "EvalParams",
    "EvaluationError",
    "Segment",
    "SegmentResult",
    "StudentEval",
    "build_segments",
    "evaluate_students",
    "explain_factor",
    "ideal_order",
    "join_labels",
    "load_eval_params",
    "metrics_document",
    "parse_eval_params",
    "pick_example",
    "random_order",
    "render_report",
    "run_evaluate",
    "score_frame",
    "summarize_segment",
]
