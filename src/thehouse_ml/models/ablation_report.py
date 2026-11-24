"""Salidas de ML-05 T04: `reports/ablation_metrics.json`, `reports/ablation_report.md` y la tabla por pliegue.

Deterministas (sin fechas, tiempos ni rutas absolutas). Todo es **sintético** (M-ML-032):
no es desempeño real (ML-07).
"""

from __future__ import annotations

from typing import Any

from thehouse_ml.evaluation.ranking import METRIC_NAMES, Interval
from thehouse_ml.models.ablation import (
    CI_ABLATION_MODELS,
    HYPOTHESIS_GROUP,
    HYPOTHESIS_MODEL,
    HYPOTHESIS_SEGMENT,
    MODELS,
    VERDICT_CONFIRMED,
    VERDICT_INCONCLUSIVE,
    VERDICT_REFUTED,
    AblationEvaluation,
    CrossValidation,
)
from thehouse_ml.models.baseline_report import METRIC_LABELS, SYNTHETIC_BANNER
from thehouse_ml.models.factor_groups import FACTOR_GROUPS, OUTSIDE_GROUPS
from thehouse_ml.models.train import TRAIN_MANIFEST
from thehouse_ml.models.train_config import DEFAULT_PROFILE
from thehouse_ml.models.trained_eval import CANDIDATES, PRIMARY_METRIC

SPEC = "ML-05 1.0.0, T04 (ML-05-R04)"
MODEL_LABELS = {"baseline": "Baseline", "logreg": "Regresión logística", "lightgbm": "LightGBM"}
VERDICT_LABELS = {
    VERDICT_CONFIRMED: "CONFIRMADA",
    VERDICT_REFUTED: "REFUTADA",
    VERDICT_INCONCLUSIVE: "NO CONCLUYENTE",
}
VERDICT_ASCII = {VERDICT_CONFIRMED: "confirmada", VERDICT_REFUTED: "refutada", VERDICT_INCONCLUSIVE: "no concluyente"}
WARNING = (
    "Métricas con etiquetas SINTÉTICAS (utilidad oculta documentada más ruido, M-ML-032): no son desempeño real ni "
    "preferencias reales (constitución §8.2 y §20.3); se reemplazan con señales reales en ML-07."
)
HYPOTHESIS_RULE = (
    "Confirmada si (1) la Δ NDCG@10 media de la regresión logística contra el baseline en `budget.max` > $3,500 es "
    "estrictamente la mayor de los tres rangos, (2) su IC 95 % está por encima de 0 y (3) quitar el grupo "
    "presupuesto baja el NDCG@10 medio entre pliegues; refutada si falla (1) o (3) o si el IC está entero por "
    "debajo de 0; no concluyente si se cumplen (1) y (3) pero el IC incluye el 0 (M-ML-040, fijada antes de "
    "calcular los segmentos). Los rangos se comparan por su media puntual: sus intervalos pueden solaparse."
)


def _ablation_entry(evaluation: AblationEvaluation, name: str) -> dict[str, Any]:
    if name not in evaluation.cv.ablation:
        return {"omitted": True, "reason": "perfil ci: solo se ablaciona la regresión logística (M-ML-040)"}
    return {"omitted": False, "groups": [item.as_dict() for item in evaluation.cv.ablation[name]]}


def ablation_document(evaluation: AblationEvaluation) -> dict[str, Any]:
    """Contenido de `reports/ablation_metrics.json`."""
    cv = evaluation.cv
    params = evaluation.eval_params
    segments: list[dict[str, Any]] = []
    for item in evaluation.segments:
        if item is None:
            continue
        paired = item.paired
        entry: dict[str, Any] = {
            "id": item.segment.id,
            "kind": item.segment.kind,
            "label": item.segment.label,
            "students": len(item.segment.students),
            "baseline": {metric: paired.baseline[metric].as_dict() for metric in METRIC_NAMES},
        }
        for name in CANDIDATES:
            entry[name] = {metric: paired.models[name][metric].as_dict() for metric in METRIC_NAMES}
            entry[f"diff_{name}_minus_baseline"] = {
                metric: paired.differences[name][metric].as_dict() for metric in METRIC_NAMES
            }
        segments.append(entry)
    return {
        "title": f"{SYNTHETIC_BANNER} — ablación, métricas por pliegue y segmentos de ML-05",
        "synthetic": True,
        "banner": SYNTHETIC_BANNER,
        "warning": WARNING,
        "spec": SPEC,
        "profile": evaluation.profile,
        "official": evaluation.profile == DEFAULT_PROFILE,
        "feature_groups": [
            {"id": group.id, "label": group.label, "features": list(group.features)} for group in FACTOR_GROUPS
        ],
        "outside_groups": dict(OUTSIDE_GROUPS),
        "cross_validation": {
            "note": "solo train + validación; test nunca entra (spec §6.7)",
            "fold_assignment": f"{TRAIN_MANIFEST} (T02)",
            "hyperparameters": f"los elegidos en T02 ({evaluation.cv_source}); sin búsqueda nueva",
            "n_splits": cv.n_folds,
            "students": cv.students,
            "pairs": cv.pairs,
            "reproduces_train_cv_ndcg@10": True,
            "std": "desviación estándar muestral entre pliegues",
            "metrics": {name: cv.folds[name].as_dict() for name in MODELS},
        },
        "ablation": {
            "metric": PRIMARY_METRIC,
            "delta": "NDCG@10 sin el grupo - NDCG@10 del modelo completo, por pliegue",
            "models_ablated": list(cv.ablated_models),
            **{name: _ablation_entry(evaluation, name) for name in CANDIDATES},
        },
        "segments": {
            "source": "métricas por estudiante de la medición única en test de T03 (sin volver a puntuar test)",
            "bootstrap": {
                "unit": "student",
                "paired": True,
                "resamples": params.resamples,
                "confidence": params.confidence,
                "seed": params.bootstrap_seed,
                "rng": 'random.Random("bootstrap:<seed>:<segment_id>").choices',
            },
            "baseline_matches_reference": True,
            "partitions_aggregate_to_total": True,
            "items": segments,
        },
        "hypothesis_budget_gt_3500": {
            "rule": HYPOTHESIS_RULE,
            "model": HYPOTHESIS_MODEL,
            "segment": HYPOTHESIS_SEGMENT,
            "ablation_group": HYPOTHESIS_GROUP,
            **evaluation.hypothesis.as_dict(),
        },
    }


def _f4(value: float) -> str:
    return f"{value:.4f}"


def _ci(interval: Interval) -> str:
    return f"{interval.mean:.3f} [{interval.low:.3f}, {interval.high:.3f}]"


def _signed_ci(interval: Interval) -> str:
    return f"{interval.mean:+.3f} [{interval.low:+.3f}, {interval.high:+.3f}]"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def fold_table(cv: CrossValidation) -> list[str]:
    """Tabla de las 4 métricas por pliegue (baseline, LR, LightGBM) con media ± desviación (traspaso T03-H2)."""
    folds = [f"P{fold}" for fold in range(cv.n_folds)]
    lines = [
        _row(["Métrica", "Modelo", *folds, "Media ± desv."]),
        _row(["---", "---", *(["---:"] * cv.n_folds), "---:"]),
    ]
    for metric in METRIC_NAMES:
        for name in MODELS:
            summary = cv.folds[name]
            lines.append(
                _row(
                    [
                        METRIC_LABELS[metric],
                        MODEL_LABELS[name],
                        *(_f4(value) for value in summary.values(metric)),
                        f"{_f4(summary.mean(metric))} ± {_f4(summary.std(metric))}",
                    ]
                )
            )
    return lines


def render_ablation_report(evaluation: AblationEvaluation) -> str:
    """`reports/ablation_report.md`."""
    cv = evaluation.cv
    lines: list[str] = [
        "# ML-05 T04 — Ablación, segmentos e hipótesis del segmento > $3,500",
        "",
        f"> **{SYNTHETIC_BANNER}.** Las etiquetas son sintéticas (utilidad oculta documentada más ruido, M-ML-032): "
        "estas métricas **no** son desempeño real ni preferencias reales (constitución §8.2 y §20.3). El desempeño "
        "real solo se afirma con señales reales en **ML-07**.",
        "",
        f"Generado por `python -m thehouse_ml evaluate` ({SPEC}). Métricas completas en "
        "`reports/ablation_metrics.json`; la tabla por pliegue con las 4 métricas está también en "
        "`reports/trained_report.md`.",
        "",
        f"Perfil: **`{evaluation.profile}`**"
        + (
            " (corrida oficial)."
            if evaluation.profile == DEFAULT_PROFILE
            else " (prueba de humo del CI, **no oficial**: pliegues reducidos y sin ablación de LightGBM)."
        ),
        "",
        "## Grupos de features",
        "",
        "Los 6 factores del desglose (spec §6.6, mismos `id` que `baseline-1.0.0`). Cada feature de `feature_order` "
        "está en un solo grupo.",
        "",
        "| Grupo | Factor | Features |",
        "| --- | --- | --- |",
    ]
    for group in FACTOR_GROUPS:
        lines.append(f"| `{group.id}` | {group.label} | " + ", ".join(f"`{name}`" for name in group.features) + " |")
    for name, reason in OUTSIDE_GROUPS.items():
        lines.append(f"| — | (fuera de los grupos) | `{name}`: {reason} |")
    lines += [
        "",
        "## Métricas por pliegue (traspaso T03-H2)",
        "",
        f"{cv.n_folds} pliegues de T02 (misma asignación de `models/{TRAIN_MANIFEST}`), {cv.students:,} estudiantes y "
        f"{cv.pairs:,} pares de train + validación; hiperparámetros elegidos en T02 (`{evaluation.cv_source}`), sin "
        "búsqueda nueva. El NDCG@10 por pliegue reproduce exactamente el de T02. Test no participa.",
        "",
        *fold_table(cv),
        "",
        "## Ablación por grupo (solo pliegues)",
        "",
        "Cada modelo se reajusta en cada pliegue sin las features del grupo, con los mismos hiperparámetros. "
        "Δ = NDCG@10 sin el grupo - NDCG@10 del modelo completo; media ± desviación muestral entre pliegues y "
        "pliegues en los que empeora.",
        "",
    ]
    header = ["Grupo quitado"]
    for name in CANDIDATES:
        header += [f"{MODEL_LABELS[name]}: NDCG@10 sin el grupo", "Δ NDCG@10 media ± desv.", "Pliegues peor"]
    lines += [_row(header), _row(["---", *(["---:"] * (3 * len(CANDIDATES)))])]
    full = ["(modelo completo)"]
    for name in CANDIDATES:
        full += [f"{_f4(cv.folds[name].mean(PRIMARY_METRIC))}", "—", "—"]
    lines.append(_row(full))
    for index, group in enumerate(FACTOR_GROUPS):
        cells = [f"{group.label} (`{group.id}`)"]
        for name in CANDIDATES:
            if name not in cv.ablation:
                cells += ["omitido (ci)", "—", "—"]
                continue
            ablated = cv.ablation[name][index]
            mean_value = sum(ablated.ndcg10_by_fold) / len(ablated.ndcg10_by_fold)
            cells += [
                _f4(mean_value),
                f"{ablated.delta_mean:+.4f} ± {ablated.delta_std:.4f}",
                f"{ablated.folds_below_full}/{len(ablated.deltas)}",
            ]
        lines.append(_row(cells))
    if evaluation.profile != DEFAULT_PROFILE:
        lines += ["", f"Perfil `ci`: solo se ablaciona {', '.join(CI_ABLATION_MODELS)} (M-ML-040)."]
    lines += [
        "",
        "## Segmentos en test (medición única de T03)",
        "",
        "Métricas por estudiante de la medición única de T03 (modelos oficiales, sin reentrenar ni volver a puntuar "
        f"test). Bootstrap pareado por estudiante, {evaluation.eval_params.resamples:,} remuestreos, IC 95 %, "
        f"semilla `bootstrap:{evaluation.eval_params.bootstrap_seed}:<segment_id>` (la del baseline en cada "
        "segmento: sus intervalos coinciden con `baseline_metrics.json`). Las medias por campus y por rango de "
        "presupuesto se agregan exactamente a las del total.",
        "",
        "| Segmento | Estudiantes | Baseline NDCG@10 | Regresión logística | Δ LR [IC 95 %] | LightGBM | "
        "Δ LightGBM [IC 95 %] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in evaluation.segments:
        if result is None:
            continue
        paired = result.paired
        lines.append(
            _row(
                [
                    result.segment.label,
                    f"{len(result.segment.students):,}",
                    _ci(paired.baseline[PRIMARY_METRIC]),
                    _ci(paired.models["logreg"][PRIMARY_METRIC]),
                    _signed_ci(paired.differences["logreg"][PRIMARY_METRIC]),
                    _ci(paired.models["lightgbm"][PRIMARY_METRIC]),
                    _signed_ci(paired.differences["lightgbm"][PRIMARY_METRIC]),
                ]
            )
        )
    verdict = evaluation.hypothesis
    lines += [
        "",
        "NDCG@5, Precision@5 y MAP por segmento, con sus diferencias pareadas, en `reports/ablation_metrics.json`.",
        "",
        "## Hipótesis del segmento `budget.max` > $3,500 (M-ML-035, herencia 4)",
        "",
        "Hipótesis: el baseline es más débil en > $3,500 porque puntúa igual cualquier renta dentro del presupuesto, "
        "mientras la utilidad oculta prefiere lo más barato.",
        "",
        f"Regla: {HYPOTHESIS_RULE}",
        "",
        *(f"- {reason}" for reason in verdict.reasons),
        "",
        f"**Veredicto: {VERDICT_LABELS[verdict.verdict]}.**",
        "",
        "## Advertencias",
        "",
        "- **DATOS SINTÉTICOS:** con etiquetas del simulador, la ablación mide qué features necesita el modelo "
        "para imitar la utilidad oculta (circularidad, ML-05-RK01), no qué valoran los estudiantes reales (ML-07).",
        "- Los segmentos reutilizan la medición única de T03: no se ajustó nada con ellos (spec §6.2).",
        "- Los segmentos pequeños (p. ej. un campus, ~30 estudiantes) tienen intervalos anchos.",
        "",
    ]
    return "\n".join(lines)


__all__ = ["VERDICT_ASCII", "ablation_document", "fold_table", "render_ablation_report"]
