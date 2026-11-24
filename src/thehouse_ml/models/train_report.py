"""Reporte de la validación cruzada de ML-05 T02 (`reports/training_report.md` y `reports/train_cv_metrics.json`).

Todo es determinista (sin fechas, tiempos ni rutas absolutas): dos corridas con
los mismos datos y parámetros escriben los mismos bytes. Las métricas son de
etiquetas **sintéticas** y solo de los pliegues de train + validación; la
partición test no se evalúa aquí (T03).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from thehouse_ml.models.baseline_report import SYNTHETIC_BANNER
from thehouse_ml.models.train import (
    LIGHTGBM_MODEL,
    LOGREG_MODEL,
    TRAIN_CV_METRICS,
    TRAIN_MANIFEST,
    FoldScore,
    GridResult,
    TrainResult,
    fold_counts,
)
from thehouse_ml.models.train_config import DEFAULT_PROFILE, TRAIN_PARAMS_RESOURCE


def _deltas(model: GridResult, baseline: Sequence[FoldScore]) -> list[float]:
    return [item.ndcg10 - base.ndcg10 for item, base in zip(model.folds, baseline, strict=True)]


def metrics_document(
    result: TrainResult,
    inputs: Mapping[str, str],
    models: Mapping[str, Mapping[str, Any]],
    profile: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """`reports/train_cv_metrics.json`: rejillas, NDCG@10 por pliegue, selección y baseline por pliegue."""
    params = result.params
    counts = fold_counts(result.cv)
    baseline = list(result.baseline_folds)
    baseline_values = [item.ndcg10 for item in baseline]
    comparison = {
        name: {
            "config_id": selected.config_id,
            "delta_ndcg@10_by_fold": _deltas(selected, baseline),
            "folds_above_baseline": sum(1 for value in _deltas(selected, baseline) if value > 0),
        }
        for name, selected in (("logreg", result.logreg_selected), ("lightgbm", result.lightgbm_selected))
    }
    return {
        "synthetic": True,
        "banner": SYNTHETIC_BANNER,
        "spec": "ML-05 1.0.0, T02 (ML-05-R02)",
        "profile": profile,
        "note": (
            "Métricas de validación cruzada con etiquetas SINTÉTICAS (M-ML-032) sobre train + validación; "
            "no son desempeño real y la partición test no se evalúa aquí (T03)."
        ),
        "train_params_version": params.version,
        "train_params_decision": params.decision,
        "feature_schema_version": result.schema_version,
        "labels_params_version": result.labels_params_version,
        "relevance_threshold": result.relevance_threshold,
        "inputs": {name: {"sha256": digest} for name, digest in inputs.items()},
        "data": {
            "summary": {name: dict(values) for name, values in result.summary.items()},
            "excluded_students": [
                {"student_id": student, "split": split, "candidates": candidates}
                for student, split, candidates in result.excluded
            ],
        },
        "folds": {
            "n_splits": params.n_splits,
            "splits": list(params.cv_splits),
            "seed": params.seed,
            "students": len(result.cv.students),
            "pairs": result.cv.n_pairs,
            "by_fold": [
                {
                    "fold": int(fold),
                    "students": int(row["students"]),
                    "pairs": int(row["pairs"]),
                    "by_campus": {
                        str(code): int(row[code]) for code in counts.columns if code not in ("students", "pairs")
                    },
                }
                for fold, row in counts.iterrows()
            ],
        },
        "logreg_grid": [item.as_dict() for item in result.logreg_results],
        "lightgbm_grid": [item.as_dict() for item in result.lightgbm_results],
        "baseline": {
            "model_version": "baseline-1.0.0",
            "ndcg@10_by_fold": baseline_values,
            "ndcg@10_mean": GridResult("baseline", {}, tuple(baseline)).mean,
            "ndcg@10_std": GridResult("baseline", {}, tuple(baseline)).std,
            "students_by_fold": [item.students for item in baseline],
        },
        "selected": {
            "logreg": result.logreg_selected.as_dict(),
            "lightgbm": result.lightgbm_selected.as_dict(),
        },
        "comparison_with_baseline": comparison,
        "final_models": {
            "trained_on": {
                "splits": list(params.cv_splits),
                "students": len(result.cv.students),
                "pairs": result.cv.n_pairs,
            },
            "files": {name: dict(values) for name, values in models.items()},
            "logreg_convergence_warnings": result.logreg_final_convergence_warnings,
            "lightgbm_monotone_check": {
                "sample_rows": result.monotone_sample_rows,
                "violations": dict(result.monotone_check),
            },
        },
    }


def _f4(value: float) -> str:
    return f"{value:.4f}"


def _signed(value: float) -> str:
    return f"{value:+.4f}"


def _grid_table(results: Sequence[GridResult], selected: GridResult, n_splits: int) -> list[str]:
    folds = " | ".join(f"P{fold}" for fold in range(n_splits))
    lines = [
        f"| Configuración | {folds} | Media | Desv. |",
        "| --- | " + " | ".join("---:" for _ in range(n_splits)) + " | ---: | ---: |",
    ]
    for item in results:
        mark = " **(elegida)**" if item.config_id == selected.config_id else ""
        values = " | ".join(_f4(value) for value in item.values)
        lines.append(f"| `{item.config_id}`{mark} | {values} | {_f4(item.mean)} | {_f4(item.std)} |")
    return lines


def render_report(
    result: TrainResult,
    inputs: Mapping[str, str],
    models: Mapping[str, Mapping[str, Any]],
    profile: str = DEFAULT_PROFILE,
) -> str:
    """`reports/training_report.md` (en git)."""
    params = result.params
    n = params.n_splits
    counts = fold_counts(result.cv)
    campuses = [str(code) for code in counts.columns if code not in ("students", "pairs")]
    baseline = list(result.baseline_folds)
    base_grid = GridResult("baseline-1.0.0", {}, tuple(baseline))
    lines: list[str] = [
        "# ML-05 T02 — Validación cruzada y entrenamiento",
        "",
        f"> **{SYNTHETIC_BANNER}.** Métricas de validación cruzada sobre **train + validación** con las etiquetas "
        "sintéticas de ML-04 (M-ML-032): el modelo aprende la utilidad oculta del simulador más su ruido; **no** es "
        "desempeño real (constitución §8.2 y §20.3). La partición **test no se evalúa aquí**: se mide una sola vez "
        "en T03.",
        "",
        "Generado por `python -m thehouse_ml train` (spec ML-05 1.0.0, T02). Parámetros: "
        f"`src/thehouse_ml/models/{TRAIN_PARAMS_RESOURCE}` {params.version} ({params.decision}). Métricas en "
        f"`reports/{TRAIN_CV_METRICS}`; modelos y asignación de pliegues en `models/` (fuera de git).",
        "",
        f"Perfil: **`{profile}`**"
        + (
            " (corrida oficial: rejilla completa)."
            if profile == DEFAULT_PROFILE
            else " (prueba de humo del CI, no oficial: 1 combinación por modelo y pocos pliegues)."
        ),
        "",
        "## Datos",
        "",
        f"- Esquema de features {result.schema_version}; `labels_params` {result.labels_params_version}; entradas "
        "(solo `training_data.TRAINING_INPUTS`; nunca la utilidad ni los pesos ocultos):",
        "",
        "| Archivo | SHA-256 |",
        "| --- | --- |",
    ]
    lines += [f"| `data/processed/{name}` | `{digest}` |" for name, digest in inputs.items()]
    lines += [
        "",
        "| Partición | Pares | Estudiantes | Excluidos (< 2 candidatos) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in result.summary.items():
        label = f"**{name}**" if name == "total" else f"`{name}`"
        lines.append(f"| {label} | {values['pairs']:,} | {values['students']:,} | {values['excluded_students']} |")
    lines += [
        "",
        f"Pliegues y búsqueda: **{len(result.cv.students):,} estudiantes** y **{result.cv.n_pairs:,} pares** de "
        f"{' + '.join(params.cv_splits)}. Objetivo de la regresión logística: `relevance ≥ "
        f"{result.relevance_threshold}` (`baseline_eval_params.json`, M-ML-034).",
        "",
        "## Pliegues",
        "",
        f"`StratifiedGroupKFold(n_splits={n}, shuffle=True, random_state={params.seed})` con una fila por estudiante "
        "(grupo = `student_id`, estrato = campus). Ningún estudiante está en dos pliegues; test no participa.",
        "",
        "| Pliegue | " + " | ".join(f"`{code}`" for code in campuses) + " | Estudiantes | Pares |",
        "| --- | " + " | ".join("---:" for _ in campuses) + " | ---: | ---: |",
    ]
    for fold, row in counts.iterrows():
        cells = " | ".join(str(int(row[code])) for code in campuses)
        lines.append(f"| P{int(fold)} | {cells} | {int(row['students'])} | {int(row['pairs']):,} |")
    lines += [
        "",
        "## Rejilla y NDCG@10 por pliegue",
        "",
        "NDCG@10 medio de los estudiantes retenidos de cada pliegue (`thehouse_ml.evaluation.ranking`, puntaje "
        "descendente y desempate por `room_id`, igual que el baseline). Se elige el mayor NDCG@10 medio; desempate: "
        "menor desviación estándar muestral y después la primera de la rejilla (ordenada de la más simple a la más "
        "compleja).",
        "",
        f"### Regresión logística ({len(result.logreg_results)} configuraciones)",
        "",
        "Pointwise, recorte de cada feature a su `training_range` (o `range`), `StandardScaler` (`standard`) o "
        f"`KBinsDiscretizer` por cuantiles con {params.n_bins} intervalos y one-hot (`quantile_bins`), "
        f"`LogisticRegression(solver={params.solver}, max_iter={params.max_iter}, tol={params.tol:g})`.",
        "",
        *_grid_table(result.logreg_results, result.logreg_selected, n),
        "",
    ]
    warned = sum(item.convergence_warnings for item in result.logreg_results)
    lines += [
        f"Avisos de no convergencia en la búsqueda: {warned}; en el modelo final: "
        f"{result.logreg_final_convergence_warnings}.",
        "",
        f"### LightGBM LambdaRank ({len(result.lightgbm_results)} configuraciones)",
        "",
        f"`label_gain` {list(int(value) for value in params.label_gain)}, `min_data_in_leaf` "
        f"{params.min_data_in_leaf}, restricciones monotónicas -1 en "
        + ", ".join(f"`{name}`" for name in params.monotone_decreasing)
        + f", `num_threads` {params.num_threads}, `deterministic` = true, `force_row_wise` = true, `seed` "
        f"{params.seed}; sin parada temprana.",
        "",
        *_grid_table(result.lightgbm_results, result.lightgbm_selected, n),
        "",
        "## Hiperparámetros elegidos",
        "",
        "| Familia | Configuración | NDCG@10 medio ± desv. |",
        "| --- | --- | ---: |",
        f"| Regresión logística | `{result.logreg_selected.config_id}` | {_f4(result.logreg_selected.mean)} ± "
        f"{_f4(result.logreg_selected.std)} |",
        f"| LightGBM LambdaRank | `{result.lightgbm_selected.config_id}` | {_f4(result.lightgbm_selected.mean)} ± "
        f"{_f4(result.lightgbm_selected.std)} |",
        "",
        "## Comparación por pliegue con el baseline",
        "",
        "El baseline `baseline-1.0.0` (`score_features` y `rank` de ML-04) no se entrena: se mide en los mismos "
        "estudiantes retenidos de cada pliegue. Es la entrada de la condición 1 de la spec §6.5, que se evalúa "
        "en T03 junto con la medición en test.",
        "",
        "| Pliegue | Estudiantes | Baseline | Regresión logística | Δ | LightGBM | Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lr_deltas = _deltas(result.logreg_selected, baseline)
    gbm_deltas = _deltas(result.lightgbm_selected, baseline)
    for fold in range(n):
        lines.append(
            f"| P{fold} | {baseline[fold].students} | {_f4(baseline[fold].ndcg10)} | "
            f"{_f4(result.logreg_selected.values[fold])} | {_signed(lr_deltas[fold])} | "
            f"{_f4(result.lightgbm_selected.values[fold])} | {_signed(gbm_deltas[fold])} |"
        )
    lines += [
        f"| **Media ± desv.** | | {_f4(base_grid.mean)} ± {_f4(base_grid.std)} | "
        f"{_f4(result.logreg_selected.mean)} ± {_f4(result.logreg_selected.std)} | "
        f"{_signed(result.logreg_selected.mean - base_grid.mean)} | "
        f"{_f4(result.lightgbm_selected.mean)} ± {_f4(result.lightgbm_selected.std)} | "
        f"{_signed(result.lightgbm_selected.mean - base_grid.mean)} |",
        "",
        f"Pliegues por encima del baseline: regresión logística {sum(1 for v in lr_deltas if v > 0)}/{n}, "
        f"LightGBM {sum(1 for v in gbm_deltas if v > 0)}/{n} (informativo; la decisión de promoción es de T03).",
        "",
        "## Modelos finales",
        "",
        f"Reentrenados con los hiperparámetros elegidos sobre todo {' + '.join(params.cv_splits)} "
        f"({len(result.cv.students):,} estudiantes, {result.cv.n_pairs:,} pares). Sin pickle: la regresión logística "
        "en JSON (recorte, transformación, coeficientes e intercepto) y LightGBM en su texto nativo "
        f"(`Booster.save_model`). Manifiesto con la asignación de pliegues: `models/{TRAIN_MANIFEST}`.",
        "",
        "| Archivo (fuera de git) | Bytes | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    for name in (LOGREG_MODEL, LIGHTGBM_MODEL):
        lines.append(f"| `models/{name}` | {models[name]['bytes']:,} | `{models[name]['sha256']}` |")
    violations = dict(result.monotone_check)
    lines += [
        "",
        f"Verificación monotónica de LightGBM: en {result.monotone_sample_rows} pares equiespaciados, cada feature "
        "restringida se recorre en 21 valores de su `training_range` con lo demás fijo; filas en las que el puntaje "
        "sube: " + ", ".join(f"`{name}` {count}" for name, count in violations.items()) + ".",
        "",
        "### Coeficientes de la regresión logística final",
        "",
    ]
    lines += _coefficients(result)
    lines.append("")
    return "\n".join(lines)


def _coefficients(result: TrainResult) -> list[str]:
    model = result.logreg_model
    lines = [f"Intercepto: {model.intercept:.6f}.", ""]
    if model.config.transform == "standard":
        lines += ["| Feature | Recorte | Media | Escala | Coeficiente |", "| --- | --- | ---: | ---: | ---: |"]
        for index, name in enumerate(model.feature_order):
            lines.append(
                f"| `{name}` | [{model.clip_lower[index]:g}, {model.clip_upper[index]:g}] | "
                f"{(model.mean or ())[index]:.6f} | {(model.scale or ())[index]:.6f} | {model.coef[index]:.6f} |"
            )
        return lines
    lines += ["| Feature | Recorte | Cortes internos | Coeficiente por intervalo |", "| --- | --- | --- | --- |"]
    position = 0
    for index, name in enumerate(model.feature_order):
        edges = (model.inner_edges or ())[index]
        width = len(edges) + 1
        coefs = ", ".join(f"{value:.4f}" for value in model.coef[position : position + width])
        position += width
        cuts = ", ".join(f"{edge:.4g}" for edge in edges) or "—"
        lines.append(f"| `{name}` | [{model.clip_lower[index]:g}, {model.clip_upper[index]:g}] | {cuts} | {coefs} |")
    return lines


__all__ = ["metrics_document", "render_report"]
