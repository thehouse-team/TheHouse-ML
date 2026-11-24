"""Reporte de la medición en test de ML-05 T03 (`reports/trained_metrics.json` y `reports/trained_report.md`).

Determinista (sin fechas, tiempos ni rutas absolutas): dos corridas con los
mismos modelos, datos y parámetros escriben los mismos bytes. Todo es **sintético**
(etiquetas de M-ML-032): no es desempeño real (ML-07). Los segmentos son de T04.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from thehouse_ml.evaluation.ranking import METRIC_NAMES, Interval
from thehouse_ml.models.baseline_report import BASELINE_METRICS, METRIC_LABELS, SYNTHETIC_BANNER
from thehouse_ml.models.train import TRAIN_MANIFEST
from thehouse_ml.models.train_config import DEFAULT_PROFILE
from thehouse_ml.models.trained_eval import (
    CANDIDATES,
    MODEL_FILES,
    PRIMARY_METRIC,
    SEGMENT_ID,
    TrainedEvaluation,
)

SPEC = "ML-05 1.0.0, T03 (ML-05-R03)"
MODEL_LABELS = {"baseline": "Baseline `baseline-1.0.0`", "logreg": "Regresión logística", "lightgbm": "LightGBM"}
CONDITION_LABELS = (
    "1. Validación cruzada: NDCG@10 > baseline en los 5 pliegues",
    "2. Test: IC 95 % de Δ NDCG@10 pareada con límite inferior > 0",
    "3. Desglose aditivo que suma el total (constitución §12)",
)
SINGLE_MEASUREMENT = (
    "La partición test se evaluó una sola vez con los modelos finales de T02 (SHA-256 verificados); "
    "ningún hiperparámetro, feature, calibración, umbral ni decisión se ajustó con estos resultados "
    "(spec ML-05 §6.2 y §7 regla 2). Cambiar algo del modelo después invalida esta medición."
)
WARNING = (
    "Métricas con etiquetas SINTÉTICAS (utilidad oculta documentada más ruido, M-ML-032): el modelo aprende la "
    "función del simulador; no son desempeño real ni preferencias reales (constitución §8.2 y §20.3) y se "
    "reemplazan con señales reales en ML-07."
)


def metrics_document(result: TrainedEvaluation) -> dict[str, Any]:
    """Contenido de `reports/trained_metrics.json`."""
    params = result.eval_params
    paired = result.paired
    return {
        "title": f"{SYNTHETIC_BANNER} — medición única en test de los modelos de ML-05",
        "synthetic": True,
        "banner": SYNTHETIC_BANNER,
        "warning": WARNING,
        "spec": SPEC,
        "profile": result.profile,
        "official": result.profile == DEFAULT_PROFILE,
        "single_test_measurement": SINGLE_MEASUREMENT,
        "feature_schema_version": result.schema_version,
        "labels_params_version": result.labels_params_version,
        "train_params_version": result.train_params.version,
        "eval_params_version": params.version,
        "baseline_model_version": result.baseline_version,
        "inputs": {name: {"sha256": digest} for name, digest in result.inputs.items()},
        "models": {
            name: {
                "file": MODEL_FILES[name],
                "sha256": result.models[MODEL_FILES[name]],
                "sha256_verified_against": TRAIN_MANIFEST,
                "config_id": result.configs[name],
            }
            for name in CANDIDATES
        },
        "evaluation": {
            "split": result.test.split,
            "students_evaluated": len(paired.students),
            "pairs": result.test.n_pairs,
            "min_candidates": params.min_candidates,
            "ranking": "puntaje descendente, desempate por room_id ascendente (igual que el baseline)",
            "metrics": list(METRIC_NAMES),
            "relevance_threshold": params.relevance_threshold,
            "metric_functions": "thehouse_ml.evaluation.ranking (M-ML-034)",
            "baseline_reference": f"reports/{BASELINE_METRICS} (per_student_baseline)",
            "baseline_reproduced": True,
        },
        "bootstrap": {
            "unit": "student",
            "paired": True,
            "resamples": params.resamples,
            "confidence": params.confidence,
            "seed": params.bootstrap_seed,
            "segment_id": SEGMENT_ID,
            "rng": f'random.Random("bootstrap:{params.bootstrap_seed}:{SEGMENT_ID}").choices',
            "interval": "percentil con interpolación lineal",
            "baseline_interval_matches_reference": True,
        },
        "test": {
            "baseline": {metric: paired.baseline[metric].as_dict() for metric in METRIC_NAMES},
            **{name: {metric: paired.models[name][metric].as_dict() for metric in METRIC_NAMES} for name in CANDIDATES},
        },
        "paired_difference_vs_baseline": {
            name: {metric: paired.differences[name][metric].as_dict() for metric in METRIC_NAMES} for name in CANDIDATES
        },
        "cross_validation": {
            "source": result.cv_source,
            "metric": PRIMARY_METRIC,
            "n_splits": len(result.folds["baseline"].values),
            "std": "desviación estándar muestral entre pliegues",
            "baseline": result.folds["baseline"].as_dict(),
            **{name: result.folds[name].as_dict() for name in CANDIDATES},
        },
        "promotion": {
            "criterion": "spec ML-05 §6.5 (las 3 condiciones) y §6.6 (solo un modelo aditivo explicable)",
            "required_folds": result.required_folds,
            "candidates": {name: result.candidates[name].as_dict() for name in CANDIDATES},
            "decision": result.decision.as_dict(),
        },
        "per_student_test": [
            {
                "student_id": student,
                **{name: result.per_student[name][student].as_dict() for name in CANDIDATES},
            }
            for student in paired.students
        ],
    }


def _f3(value: float) -> str:
    return f"{value:.3f}"


def _f4(value: float) -> str:
    return f"{value:.4f}"


def _ci(interval: Interval) -> str:
    return f"{_f3(interval.mean)} [{_f3(interval.low)}, {_f3(interval.high)}]"


def _signed_ci(interval: Interval) -> str:
    return f"{interval.mean:+.3f} [{interval.low:+.3f}, {interval.high:+.3f}]"


def _verdict(passed: bool | None) -> str:
    if passed is None:
        return "pendiente (T05)"
    return "**sí**" if passed else "no"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render_report(result: TrainedEvaluation, fold_metrics: Sequence[str] = ()) -> str:
    """`reports/trained_report.md` (en git con el perfil completo).

    `fold_metrics`: tabla de las 4 métricas por pliegue de T04 (traspaso T03-H2), si se calculó.
    """
    params = result.eval_params
    paired = result.paired
    folds = result.folds
    n_folds = len(folds["baseline"].values)
    labels = [METRIC_LABELS[metric] for metric in METRIC_NAMES]
    lines: list[str] = [
        "# ML-05 T03 — Evaluación en test y decisión de promoción",
        "",
        f"> **{SYNTHETIC_BANNER}.** Las etiquetas son sintéticas (utilidad oculta documentada más ruido, M-ML-032): "
        "los modelos aprenden la función del simulador y estas métricas **no** son desempeño real ni preferencias "
        "reales (constitución §8.2 y §20.3). El desempeño real solo se afirma con señales reales en **ML-07**.",
        "",
        f"Generado por `python -m thehouse_ml evaluate` ({SPEC}). Métricas completas, por estudiante y con las "
        "condiciones en `reports/trained_metrics.json`; ablación, segmentos e hipótesis > $3,500 (T04) en "
        "`reports/ablation_report.md`.",
        "",
        f"Perfil: **`{result.profile}`**"
        + (
            " (corrida oficial)."
            if result.profile == DEFAULT_PROFILE
            else " (prueba de humo del CI, **no oficial**: pliegues reducidos; no decide la promoción)."
        ),
        "",
        "## Medición única en test",
        "",
        SINGLE_MEASUREMENT,
        "",
        "| Modelo | Archivo (fuera de git) | Configuración | SHA-256 (verificado contra `train_manifest.json`) |",
        "| --- | --- | --- | --- |",
    ]
    for name in CANDIDATES:
        filename = MODEL_FILES[name]
        lines.append(
            f"| {MODEL_LABELS[name]} | `{filename}` | `{result.configs[name]}` | `{result.models[filename]}` |"
        )
    lines += [
        "",
        f"Partición **{result.test.split}** de ML-03: **{len(paired.students)} estudiantes** con "
        f"{params.min_candidates} o más candidatos y **{result.test.n_pairs:,} pares**. Cada modelo ordena los "
        "candidatos de cada estudiante por puntaje descendente con desempate por `room_id` ascendente; NDCG@10, "
        "NDCG@5, Precision@5 y MAP con `thehouse_ml.evaluation.ranking` (M-ML-034, relevante si `relevance ≥ "
        f"{params.relevance_threshold}`). El baseline recalculado reproduce exactamente `per_student_baseline` de "
        f"`reports/{BASELINE_METRICS}` (mismos {len(paired.students)} estudiantes y mismas métricas).",
        "",
        "## Test: baseline, regresión logística y LightGBM",
        "",
        f"Media por estudiante con IC 95 % por bootstrap pareado ({params.resamples:,} remuestreos de estudiantes, "
        f"percentil, semilla `bootstrap:{params.bootstrap_seed}:{SEGMENT_ID}`, la misma que el total del baseline: "
        "su intervalo coincide con el de `baseline_metrics.json`).",
        "",
        _row(["Modelo", *(f"{label} [IC 95 %]" for label in labels)]),
        _row(["---", *(["---:"] * len(METRIC_NAMES))]),
        _row([MODEL_LABELS["baseline"], *(_ci(paired.baseline[metric]) for metric in METRIC_NAMES)]),
    ]
    for name in CANDIDATES:
        lines.append(_row([MODEL_LABELS[name], *(_ci(paired.models[name][metric]) for metric in METRIC_NAMES)]))
    lines += [
        "",
        "## Diferencia pareada contra el baseline (test)",
        "",
        "Modelo - baseline por estudiante (mismos estudiantes y mismos índices de remuestreo), media con IC 95 %.",
        "",
        _row(["Modelo", *(f"Δ {label} [IC 95 %]" for label in labels)]),
        _row(["---", *(["---:"] * len(METRIC_NAMES))]),
    ]
    for name in CANDIDATES:
        lines.append(
            _row([MODEL_LABELS[name], *(_signed_ci(paired.differences[name][metric]) for metric in METRIC_NAMES)])
        )
    lines += [
        "",
        "## Validación cruzada por pliegue (NDCG@10)",
        "",
        f"De `{result.cv_source}` (T02, train + validación; test no participa): mismos estudiantes retenidos en "
        "cada pliegue para los tres modelos. Media ± desviación estándar muestral entre pliegues.",
        "",
        "| Pliegue | Baseline | Regresión logística | Δ | LightGBM | Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    base = folds["baseline"]
    for fold in range(n_folds):
        cells = [f"P{fold}", _f4(base.values[fold])]
        for name in CANDIDATES:
            value = folds[name].values[fold]
            cells += [_f4(value), f"{value - base.values[fold]:+.4f}"]
        lines.append(_row(cells))
    summary = ["**Media ± desv.**", f"{_f4(base.mean)} ± {_f4(base.std)}"]
    for name in CANDIDATES:
        summary += [f"{_f4(folds[name].mean)} ± {_f4(folds[name].std)}", f"{folds[name].mean - base.mean:+.4f}"]
    lines += [_row(summary), ""]
    if fold_metrics:
        lines += [
            "### Las 4 métricas por pliegue (T04, traspaso T03-H2)",
            "",
            "Recalculadas en T04 con la misma asignación de pliegues de `train_manifest.json` y los hiperparámetros "
            "elegidos en T02 (sin búsqueda nueva); el NDCG@10 reproduce exactamente la tabla anterior. Media ± "
            "desviación estándar muestral entre pliegues.",
            "",
            *fold_metrics,
            "",
        ]
    lines += [
        "## Criterio de promoción (spec §6.5)",
        "",
        f"Condición 1 con los {result.required_folds} pliegues del perfil completo. Solo la regresión logística "
        "(aditiva) puede cumplir la condición 3, que se verifica en T05; LightGBM es referencia del potencial y no "
        "se promueve en v1 aunque cumpla 1 y 2 (decisión 5).",
        "",
        "| Condición | Regresión logística | LightGBM |",
        "| --- | --- | --- |",
    ]
    for index, label in enumerate(CONDITION_LABELS):
        cells = [label]
        for name in CANDIDATES:
            condition = result.candidates[name].conditions[index]
            detail = ""
            if index == 0:
                detail = f" ({condition.detail['folds_above_baseline']}/{condition.detail['folds']} pliegues)"
            elif index == 1:
                diff = paired.differences[name][PRIMARY_METRIC]
                detail = f" ({_signed_ci(diff)})"
            elif name == "lightgbm":
                detail = " (no aditivo)"
            cells.append(_verdict(condition.passed) + detail)
        lines.append(_row(cells))
    outcomes = {
        "candidate_subject_to_t05": "candidata a promoción, sujeta a T05",
        "not_promoted": "no se promueve",
        "reference_not_promoted_in_v1": "referencia (no se promueve en v1)",
    }
    lines.append(_row(["**Resultado**", *(f"**{outcomes[result.candidates[name].outcome]}**" for name in CANDIDATES)]))
    lines += [
        "",
        f"**Decisión:** {result.decision.text}",
        "",
        "## Advertencias",
        "",
        "- **DATOS SINTÉTICOS:** etiquetas generadas por un simulador (M-ML-032), con circularidad (ML-05-RK01): "
        "superar al baseline aquí significa aprender mejor la utilidad oculta del simulador, no recomendar mejor "
        "a estudiantes reales. No es desempeño real; eso es de ML-07.",
        "- La promoción no es definitiva hasta T05 (condición 3) y la model card (T06); ML-06 exporta el modelo "
        "que resulte.",
        "- Las métricas por segmento (campus, sin `seed_sin_verificar`, rango de presupuesto) y la hipótesis del "
        "segmento > $3,500 son de T04, con esta misma medición (sin volver a medir en test).",
        "",
    ]
    return "\n".join(lines)


__all__ = ["metrics_document", "render_report"]
