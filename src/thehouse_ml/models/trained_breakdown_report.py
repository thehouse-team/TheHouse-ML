"""Reporte del puntaje 0-100 y el desglose de la regresión logística (ML-05 T05, `reports/trained_breakdown.md`).

Determinista (sin fechas, tiempos ni rutas absolutas): dos corridas con los mismos modelos,
datos y parámetros escriben los mismos bytes. Todo es **sintético** (etiquetas de M-ML-032).
Los ejemplos se eligen con una regla fija (no por sus resultados): para cada rango de
`budget.max` de `baseline_eval_params.json`, el primer estudiante de test por `student_id`
de un campus verificado con al menos `example_min_candidates` candidatos.

ML-05 T07 (T06-H1): la distribución de `matchScore` y los ejemplos explicados se calculan una
sola vez (`distribution_document` y `examples_document`), se guardan en el bloque `breakdown`
de `reports/trained_metrics.json` y el Markdown se renderiza desde esos mismos diccionarios, así
que la model card los lee del JSON sin volver a puntuar test.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from thehouse_ml.models.baseline_config import BaselineConfig, default_config
from thehouse_ml.models.baseline_report import SCORE_PERCENTILES, SYNTHETIC_BANNER, BudgetRange, _histogram
from thehouse_ml.models.baseline_score import score_features
from thehouse_ml.models.train import LOGREG_MODEL, TRAIN_MANIFEST
from thehouse_ml.models.train_config import DEFAULT_PROFILE
from thehouse_ml.models.trained_breakdown import (
    AFFINE_TOLERANCE,
    MONOTONE_TOLERANCE,
    BreakdownEvaluation,
    MonotonicityCheck,
    score_features_logreg,
)
from thehouse_ml.models.trained_eval import TrainedEvaluation

TRAINED_BREAKDOWN = "trained_breakdown.md"
SPEC = "ML-05 1.0.0, T05 (ML-05-R05)"
FACTOR_SHORT = {
    "budget": "Presup.",
    "distance": "Dist.",
    "essential_services": "Serv.",
    "place_type": "Lugar",
    "property_type": "Inmueble",
    "extras": "Extras",
}
DIRECTION_LABELS = {1: "↑ suma", -1: "↑ no suma", None: "—"}
# Mismas decenas que el histograma de `baseline_report._histogram` (la última incluye 100).
HISTOGRAM_BINS: tuple[tuple[int, int], ...] = (*((low, low + 9) for low in range(0, 90, 10)), (90, 100))


@dataclass(frozen=True)
class Example:
    """Un estudiante de test con su top N (posiciones de la partición evaluada, ya ordenadas)."""

    budget_range: BudgetRange
    student_id: str
    university_code: str
    budget_max: int
    candidates: int
    positions: tuple[int, ...]


def pick_examples(result: TrainedEvaluation) -> list[Example]:
    """Regla fija: por rango de `budget.max`, el primer estudiante de test (por id) verificado con ≥ N candidatos."""
    params = result.eval_params
    students = result.students
    test = students[students["split"] == result.test.split].sort_values("student_id", kind="mergesort")
    raw_total = result.breakdown.scored.raw_total
    examples: list[Example] = []
    for budget_range in params.budget_ranges:
        for row in test.itertuples(index=False):
            if not (
                bool(row.seed_verified)
                and int(row.candidates) >= params.example_min_candidates
                and budget_range.contains(int(row.budget_max))
            ):
                continue
            positions = np.flatnonzero(result.test.group == row.student_id)
            ordered = sorted(positions.tolist(), key=lambda i: (-float(raw_total[i]), str(result.test.room_id[i])))
            examples.append(
                Example(
                    budget_range=budget_range,
                    student_id=str(row.student_id),
                    university_code=str(row.university_code),
                    budget_max=int(row.budget_max),
                    candidates=int(row.candidates),
                    positions=tuple(ordered[: params.example_top_n]),
                )
            )
            break
    return examples


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _yes(value: bool | None) -> str:
    if value is None:
        return "—"
    return "**sí**" if value else "**no**"


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _coefficients_section(evaluation: BreakdownEvaluation) -> list[str]:
    lines = [
        "## 1. Coeficientes de la regresión logística",
        "",
        f"Modelo final de T02 (`models/{LOGREG_MODEL}`, transformación `standard`, C = "
        f"{evaluation.model.logreg.config.c:g}); se lee con su SHA-256 verificado contra `{TRAIN_MANIFEST}` y "
        "**no se reentrena**. Cada feature se recorta a su `training_range` (o `range`) y entra estandarizada: "
        "el término del logit es `coef · (clip(x) - media) / escala`, así que «logit por unidad» = `coef / escala`. "
        f"Intercepto `b0` = {evaluation.model.intercept:.6f}.",
        "",
        _row(
            [
                "Feature",
                "Factor",
                "Recorte",
                "coef (estandarizado)",
                "Escala",
                "Logit por unidad",
                "Esperado",
                "Signo coherente",
            ]
        ),
        _row(["---", "---", "---", "---:", "---:", "---:", "---", "---"]),
    ]
    terms = {term.name: term for factor in evaluation.model.factors for term in factor.terms}
    terms.update({term.name: term for term in evaluation.model.fixed_terms})
    for row in evaluation.monotonicity.coefficients:
        term = terms[row.feature]
        lines.append(
            _row(
                [
                    f"`{row.feature}`",
                    f"`{row.factor}`" if row.factor else "término fijo",
                    f"[{term.lower:g}, {term.upper:g}]",
                    f"{row.coef:+.6f}",
                    f"{row.scale:.6f}",
                    f"{row.slope:+.6f}",
                    DIRECTION_LABELS[row.expected],
                    _yes(row.consistent),
                ]
            )
        )
    lines += [
        "",
        "«Esperado»: «↑ suma» = más valor no debe restar puntos; «↑ no suma» = más valor no debe sumar puntos "
        "(más renta o más distancia). `price_below_min` crece cuando la renta **baja** por debajo de `budget.min`, "
        "así que su signo «esperado» respecto a la renta es positivo. `pets_compatible` es constante (1) en los "
        "candidatos (HC03): su coeficiente es 0 y su término va al término fijo con `b0`, no a un factor.",
        "",
    ]
    return lines


def _calibration_section(evaluation: BreakdownEvaluation, config: BaselineConfig) -> list[str]:
    model = evaluation.model
    weights = config.weights
    lines = [
        "## 2. Calibración 0-100 (spec §6.6)",
        "",
        "Transformación afín por factor sobre el recorte: con `C_k` la suma de los términos del factor `k` y "
        "`[min_k, max_k]` su rango alcanzable dentro de los `training_range`,",
        "",
        "```text",
        "puntos_k   = 100 · (C_k - min_k) / S        S = Σ_k (max_k - min_k)",
        "maxPoints_k = 100 · (max_k - min_k) / S      Σ_k maxPoints_k = 100",
        "matchScore = floor(round6(Σ_k puntos_k) + 0.5)   (M-ML-029/031)",
        "logit      = b0 + término fijo + Σ_k min_k + S · Σ_k puntos_k / 100",
        "```",
        "",
        f"S = {model.total_span:.6f} unidades de logit; Σ_k min_k = {model.min_logit_sum:.6f}. El total es una "
        "función afín **creciente** del logit, así que el orden coincide con el de `decision_function` para "
        "entradas dentro del rango (fuera del rango se recortan). `maxPoints` se emite con el redondeo por "
        "factor (1 decimal); los puntos sin redondear de un factor nunca superan su máximo sin redondear.",
        "",
        _row(
            [
                "Factor",
                "Features",
                "min_k (logit)",
                "max_k (logit)",
                "maxPoints (sin redondear)",
                "maxPoints",
                "Peso del baseline",
            ]
        ),
        _row(["---", "---", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for factor in model.factors:
        lines.append(
            _row(
                [
                    f"{factor.label} (`{factor.id}`)",
                    ", ".join(f"`{name}`" for name in factor.features),
                    f"{factor.min_logit:+.4f}",
                    f"{factor.max_logit:+.4f}",
                    f"{factor.raw_max_points:.4f}",
                    f"**{factor.max_points:g}**",
                    f"{weights.get(factor.id, 0):g}",
                ]
            )
        )
    raw_sum = sum(factor.raw_max_points for factor in model.factors)
    rounded_sum = sum(factor.max_points for factor in model.factors)
    lines += [
        _row(["**Total**", "", "", "", f"{raw_sum:.4f}", f"**{rounded_sum:g}**", f"{config.total_points:g}"]),
        "",
        "**Lectura:** `maxPoints` mide cuánto puede mover el logit cada factor **dentro de todo su `training_range`**, "
        "no su importancia en los candidatos típicos. La distancia domina porque su rango llega a 30 km "
        "(`distance_km`, `effective_distance_km`) y 60 (`distance_ratio`), mientras la mayoría de los candidatos "
        "están a pocos kilómetros: casi todos reciben una parte grande de sus puntos de distancia y las "
        "diferencias entre cuartos cercanos se ven en décimas. Comparado con el baseline (30/25/20/10/10/5), "
        "presupuesto y servicios pesan mucho menos. Ver el hallazgo T05-H2 de la evidencia.",
        "",
    ]
    return lines


def _verification_section(evaluation: BreakdownEvaluation, result: TrainedEvaluation) -> list[str]:
    sums = evaluation.sums
    order = evaluation.order
    return [
        "## 3. Verificación del desglose en test",
        "",
        f"Todos los pares de la partición **{result.test.split}** que puntuó T03 (**{sums.pairs:,}** pares de "
        f"**{order.students}** estudiantes; misma medición, sin volver a evaluar el modelo):",
        "",
        _row(["Comprobación", "Resultado", "Criterio"]),
        _row(["---", "---", "---"]),
        _row(
            [
                "Total contra la suma sin redondear",
                f"máx. {sums.max_total_gap:.6f}; fuera de tolerancia: {sums.total_violations}",
                f"`|Σ sin redondear - matchScore| ≤ {sums.total_tolerance:g}`",
            ]
        ),
        _row(
            [
                "Suma reportada contra la suma sin redondear",
                f"máx. {sums.max_reported_gap:.6f}; fuera de tolerancia: {sums.reported_violations}",
                f"`|Σ reportada - Σ sin redondear| ≤ {sums.breakdown_tolerance:g}`",
            ]
        ),
        _row(
            [
                "Cada factor en [0, `maxPoints`]",
                f"pares fuera: {sums.factor_out_of_range}",
                "sin redondear y redondeado",
            ]
        ),
        _row(["`matchScore` en [0, 100]", f"pares fuera: {sums.score_out_of_range}", "entero"]),
        _row(
            [
                "Logit reconstruido desde el puntaje",
                f"máx. |diferencia| {sums.max_affine_gap:.2e}",
                f"≤ {AFFINE_TOLERANCE:g} (transformación afín exacta salvo binary64)",
            ]
        ),
        _row(
            [
                "Orden por `raw_total` = orden del logit",
                f"{order.identical}/{order.students} estudiantes idénticos",
                "desempate por `room_id` ascendente",
            ]
        ),
        _row(
            [
                "Métricas con el puntaje 0-100",
                "idénticas a las de T03" if evaluation.metrics_unchanged else "**distintas**",
                "NDCG@10, NDCG@5, Precision@5 y MAP por estudiante",
            ]
        ),
        "",
        f"**El desglose suma el total:** {_yes(sums.passed and order.passed and evaluation.metrics_unchanged)}.",
        "",
    ]


def distribution_document(evaluation: BreakdownEvaluation) -> dict[str, Any]:
    """Distribución de `matchScore` y de los puntos por factor en los pares de test (para `trained_metrics.json`)."""
    scores = pd.Series(evaluation.scored.match_score, dtype="int64")
    quantiles = scores.quantile(list(SCORE_PERCENTILES), interpolation="linear")
    factor_points: list[dict[str, Any]] = []
    for index, factor in enumerate(evaluation.model.factors):
        points = pd.Series(evaluation.scored.raw_points[:, index], dtype="float64")
        mean = float(points.mean())
        factor_points.append(
            {
                "factor": factor.id,
                "label": factor.label,
                "maxPoints": factor.max_points,
                "raw_max_points": factor.raw_max_points,
                "mean": mean,
                "p25": float(points.quantile(0.25, interpolation="linear")),
                "p50": float(points.quantile(0.5, interpolation="linear")),
                "p75": float(points.quantile(0.75, interpolation="linear")),
                "mean_share_of_max": mean / factor.raw_max_points,
            }
        )
    return {
        "scope": "pares candidatos de test de la medición única de T03 (DATOS SINTÉTICOS)",
        "pairs": len(scores),
        "matchScore": {
            "min": int(scores.min()),
            "max": int(scores.max()),
            "mean": float(scores.mean()),
            "interpolation": "linear",
            "percentiles": {f"p{round(q * 100)}": float(quantiles[q]) for q in SCORE_PERCENTILES},
            "histogram": [
                {"low": low, "high": high, "count": int(((scores >= low) & (scores <= high)).sum())}
                for low, high in HISTOGRAM_BINS
            ],
        },
        "factor_points_basis": "puntos sin redondear por par",
        "factor_points": factor_points,
    }


def _distribution_section(evaluation: BreakdownEvaluation, document: dict[str, Any]) -> list[str]:
    scores = pd.Series(evaluation.scored.match_score, dtype="int64")
    summary = document["matchScore"]
    lines = [
        "## 4. Distribución de `matchScore` en test",
        "",
        f"**{document['pairs']:,}** pares candidatos de test. Histograma por decenas (la última barra incluye 100):",
        "",
        *_histogram(scores),
        "",
        _row(["Mín.", *summary["percentiles"], "Máx.", "Media"]),
        _row(["---:"] * (len(summary["percentiles"]) + 3)),
        _row(
            [
                f"{summary['min']}",
                *(f"{value:g}" for value in summary["percentiles"].values()),
                f"{summary['max']}",
                f"{summary['mean']:.1f}",
            ]
        ),
        "",
        "Puntos por factor (sin redondear) en los mismos pares:",
        "",
        _row(["Factor", "maxPoints", "Media", "p25", "Mediana", "p75", "Media / maxPoints"]),
        _row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for item in document["factor_points"]:
        lines.append(
            _row(
                [
                    f"{item['label']} (`{item['factor']}`)",
                    f"{item['maxPoints']:g}",
                    f"{item['mean']:.2f}",
                    f"{item['p25']:.2f}",
                    f"{item['p50']:.2f}",
                    f"{item['p75']:.2f}",
                    f"{100 * item['mean'] / item['raw_max_points']:.0f} %",
                ]
            )
        )
    lines.append("")
    return lines


def _check_row(check: MonotonicityCheck, label: str) -> str:
    worst = "—"
    if check.worst_case:
        case = {key: value for key, value in check.worst_case.items() if not key.startswith("points_")}
        worst = (
            f"+{check.worst_increase:.4f} pts ({', '.join(f'{k} = {v}' for k, v in case.items())}; "
            f"{check.worst_case['points_from']:.4f} → {check.worst_case['points_to']:.4f})"
        )
    return _row([label, f"{check.curves}", f"{check.steps:,}", f"{check.violations}", worst, _yes(check.passed)])


def _monotonicity_section(evaluation: BreakdownEvaluation) -> list[str]:
    result = evaluation.monotonicity
    inconsistent = [row.feature for row in result.coefficients if row.consistent is False]
    threshold = result.budget_ratio_threshold
    below_min = next(row for row in result.coefficients if row.feature == "price_below_min")
    max_gain = 100 * abs(below_min.slope) / evaluation.model.total_span
    lines = [
        "## 5. Monotonía por factor (hallazgo obligatorio)",
        "",
        "Coeficientes con signo «incorrecto» respecto a su variable natural: "
        + (", ".join(f"`{name}`" for name in inconsistent) if inconsistent else "ninguno")
        + ". Como pueden compensarse con otra feature del mismo factor (colinealidad), la comprobación decisiva es "
        "**a nivel de factor**: una rejilla que construye pares reales con `compute_pair_features` (ML-03) y "
        "recorre la variable natural; el puntaje del factor no puede subir más de "
        f"{MONOTONE_TOLERANCE:g} puntos entre dos pasos consecutivos.",
        "",
        _row(["Factor y variable", "Curvas", "Pasos", "Pasos en que sube", "Mayor subida", "Monótono"]),
        _row(["---", "---:", "---:", "---:", "---", "---"]),
        _check_row(result.budget, "Presupuesto al subir la renta"),
        _check_row(result.distance, "Distancia al aumentar la distancia"),
        "",
        f"- **Rejilla de presupuesto:** {result.budget.domain}.",
        f"- **Rejilla de distancia:** {result.distance.domain}.",
    ]
    if threshold is not None:
        observed = result.observed_ratio
        lines += [
            "- **Zona exacta:** con renta < `budget.min`, al subir la renta `price_to_budget` sube "
            "(resta puntos) y `price_below_min` baja (con su coeficiente negativo, **suma** puntos); el factor sube "
            f"si y solo si `budget.max / budget.min > {threshold:.2f}` (cociente de los logits por unidad de "
            "`price_to_budget` y `price_below_min`). La subida máxima posible es el término completo de "
            f"`price_below_min`: {max_gain:.4f} puntos sobre 100.",
            f"- **Datos sintéticos:** {result.observed_pairs_below_min:,} pares con renta < `budget.min`; el mayor "
            "`budget.max / budget.min` implícito (`(1 - price_below_min) / price_to_budget`) es "
            + (f"{observed:.2f}" if observed is not None else "—")
            + (
                f", **por debajo** de {threshold:.2f}: ningún par observado cae en la zona."
                if result.observed_violations is False
                else f", **por encima** de {threshold:.2f}: hay pares observados en la zona."
            ),
        ]
    lines += [
        "",
        f"**Monotonía a nivel de factor:** {_yes(result.passed)}.",
        "",
    ]
    if not result.budget.passed:
        observed_in_zone = "0" if result.observed_violations is False else "algunos"
        accepted = "monotonicity_budget" in evaluation.accepted_deviations
        lines += [
            "**Hallazgo T05-H1.** El coeficiente de "
            f"`price_below_min` es negativo ({below_min.coef:+.4f} estandarizado): el modelo «penaliza» quedar "
            "por debajo del mínimo como el baseline, pero eso significa que, por debajo de `budget.min`, más "
            "renta suma puntos por esa feature. A nivel de factor lo compensa `price_to_budget` salvo cuando "
            f"`budget.max` es más de {threshold or 0:.1f} veces `budget.min` "
            "(perfiles válidos en NV-11, ausentes de los datos sintéticos; efecto ≤ "
            f"{max_gain:.3f} puntos sobre 100; {observed_in_zone} pares observados en la zona).",
            "",
        ]
        if accepted:
            lines += [
                "**Desviación aceptada (M-ML-041, APROBADO por el responsable el 2026-11-24).** La condición 3 de "
                "la spec §6.5 es «desglose aditivo que suma el total», y se cumple; la verificación de monotonía "
                "del presupuesto fue una protección adicional del «EJECUTA» de T05, no parte del criterio. La "
                "no monotonía se acepta como desviación conocida **después de ver el resultado de esa "
                "verificación**, y no se corrige en este modelo. Trabajo futuro: `logreg-1.1.0` sin "
                "`price_below_min` o con su coeficiente ≥ 0, evaluado con datos nuevos en ML-07 (test ya se usó, "
                "spec §6.2). La model card (T06) y NV-23 deben mencionarla.",
                "",
            ]
    return lines


def examples_document(result: TrainedEvaluation, config: BaselineConfig) -> list[dict[str, Any]]:
    """Ejemplos explicados (regla fija de `pick_examples`): top N y salida JSON del primer lugar."""
    evaluation = result.breakdown
    model = evaluation.model
    documents: list[dict[str, Any]] = []
    for example in pick_examples(result):
        top: list[dict[str, Any]] = []
        for rank_index, position in enumerate(example.positions, start=1):
            baseline = score_features([float(value) for value in result.test.features[position]], config)
            top.append(
                {
                    "rank": rank_index,
                    "room_id": str(result.test.room_id[position]),
                    "matchScore": int(evaluation.scored.match_score[position]),
                    "points": {
                        factor.id: float(evaluation.scored.points[position, k])
                        for k, factor in enumerate(model.factors)
                    },
                    "relevance": int(result.test.relevance[position]),
                    "baseline_matchScore": int(baseline.match_score),
                }
            )
        first = example.positions[0]
        values = {name: float(result.test.features[first][i]) for i, name in enumerate(result.test.feature_order)}
        scored = score_features_logreg(list(values.values()), model)
        documents.append(
            {
                "budget_range": example.budget_range.id,
                "budget_range_label": example.budget_range.label,
                "student_id": example.student_id,
                "university_code": example.university_code,
                "budget_max": example.budget_max,
                "candidates": example.candidates,
                "top": top,
                "first_place": {
                    "room_id": str(result.test.room_id[first]),
                    "inputs": values,
                    "raw_total": scored.raw_total,
                    "probability": scored.probability,
                    "output": scored.to_dict(),
                },
            }
        )
    return documents


def _example_rows(evaluation: BreakdownEvaluation, example: dict[str, Any]) -> list[str]:
    model = evaluation.model
    header = [
        "#",
        "Cuarto",
        "matchScore",
        *(f"{FACTOR_SHORT[f.id]} (/{f.max_points:g})" for f in model.factors),
        "Relevancia sintética",
        "Baseline",
    ]
    lines = [_row(header), _row(["---:", "---", "---:", *(["---:"] * len(model.factors)), "---:", "---:"])]
    for item in example["top"]:
        lines.append(
            _row(
                [
                    str(item["rank"]),
                    f"`{item['room_id']}`",
                    f"**{item['matchScore']}**",
                    *(f"{item['points'][factor.id]:.1f}" for factor in model.factors),
                    str(item["relevance"]),
                    str(item["baseline_matchScore"]),
                ]
            )
        )
    return lines


def _examples_section(result: TrainedEvaluation, examples: list[dict[str, Any]]) -> list[str]:
    params = result.eval_params
    lines = [
        "## 6. Ejemplos explicados",
        "",
        f"Regla fija (definida antes de ver los resultados): para cada rango de `budget.max` "
        f"({', '.join(item.label for item in params.budget_ranges)}), el primer estudiante de test por `student_id` "
        f"de un campus verificado con al menos {params.example_min_candidates} candidatos; su top "
        f"{params.example_top_n} por `raw_total` (desempate `room_id`). Ids sintéticos; sin datos personales. "
        "«Relevancia sintética» es la etiqueta 0-3 del simulador (M-ML-032) y «Baseline» el `matchScore` de "
        "`baseline-1.0.0` para el mismo par, como referencia.",
        "",
    ]
    if not examples:
        return [*lines, "Ningún estudiante cumple la regla.", ""]
    for number, example in enumerate(examples, start=1):
        lines += [
            f"### 6.{number} `{example['student_id']}` — {example['university_code']}, `budget.max` "
            f"{_money(example['budget_max'])} ({example['budget_range_label']}), {example['candidates']} candidatos",
            "",
            *_example_rows(result.breakdown, example),
            "",
        ]
        first = example["first_place"]
        output = first["output"]
        values = first["inputs"]
        explained = [
            f"{FACTOR_SHORT[item['factor']]} {item['points']:g}/{item['maxPoints']:g}" for item in output["breakdown"]
        ]
        lines += [
            f"Primer lugar `{first['room_id']}`: {', '.join(explained)} → `matchScore` "
            f"**{output['matchScore']}** (suma sin redondear {first['raw_total']:.6f}; probabilidad "
            f"{first['probability']:.3f}, dato informativo). Renta = {values['price_to_budget']:.0%} de `budget.max`, "
            f"{values['distance_km']:.2f} km (efectiva {values['effective_distance_km']:.2f} km, "
            f"{values['distance_ratio']:.2f} de su máximo), cobertura de servicios {values['services_coverage']:.0%}, "
            f"extras {values['extra_services_ratio']:.2f}.",
            "",
        ]
        if number == 1:
            lines += [
                "Salida JSON de ese cuarto (formato del baseline con `maxPoints` de la calibración y lista "
                "`features` en los factores de varias features; `value` = valor de entrada):",
                "",
                "```json",
                json.dumps(output, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
    return lines


def _decision_section(result: TrainedEvaluation) -> list[str]:
    condition = result.candidates["logreg"].conditions[2]
    return [
        "## 7. Condición 3 de promoción y decisión",
        "",
        _row(["Condición 3 (regresión logística)", "Resultado"]),
        _row(["---", "---"]),
        _row(
            [
                "El desglose suma el total (tolerancias, rangos, orden y métricas)",
                _yes(
                    result.breakdown.sums.passed
                    and result.breakdown.order.passed
                    and result.breakdown.metrics_unchanged
                ),
            ]
        ),
        _row(["Monotonía a nivel de factor (presupuesto y distancia)", _yes(result.breakdown.monotonicity.passed)]),
        _row(
            [
                "Desviaciones aceptadas (no bloquean)",
                ", ".join(f"`{name}` (M-ML-041)" for name in result.breakdown.deviations_applied) or "ninguna",
            ]
        ),
        _row(["**Condición 3** (spec §6.5: desglose aditivo que suma el total)", _yes(condition.passed)]),
        "",
        f"**Decisión:** {result.decision.text}",
        "",
    ]


def render_breakdown_report(result: TrainedEvaluation, config: BaselineConfig) -> str:
    """`reports/trained_breakdown.md` (en git con el perfil completo)."""
    evaluation = result.breakdown
    lines = [
        "# ML-05 T05 — Puntaje 0-100 y desglose de la regresión logística",
        "",
        f"> **{SYNTHETIC_BANNER}.** Los coeficientes se aprendieron de etiquetas sintéticas (M-ML-032) y la "
        "distribución y los ejemplos son de pares sintéticos: no describen preferencias ni desempeño reales "
        "(constitución §8.2 y §20.3; ML-07).",
        "",
        f"Generado por `python -m thehouse_ml evaluate` ({SPEC}). Modelo `{evaluation.model.model_version}`, esquema "
        f"{evaluation.model.feature_schema_version}. Perfil: **`{result.profile}`**"
        + (" (corrida oficial)." if result.profile == DEFAULT_PROFILE else " (prueba de humo del CI, **no oficial**)."),
        "",
        *_coefficients_section(evaluation),
        *_calibration_section(evaluation, config),
        *_verification_section(evaluation, result),
        *_distribution_section(evaluation, distribution_document(evaluation)),
        *_monotonicity_section(evaluation),
        *_examples_section(result, examples_document(result, config)),
        *_decision_section(result),
        "## Advertencias",
        "",
        "- **DATOS SINTÉTICOS:** etiquetas de un simulador con circularidad (ML-05-RK01); no es desempeño real.",
        "- Test se midió una sola vez en T03; T05 reutiliza esos pares y no ajusta nada con sus resultados.",
        "- El formato de `breakdown` (lista `features`, `maxPoints` variables) y el recorte al `training_range` se "
        "coordinan con `TheHouse-GenAI` y NV-23 (responsable).",
        "",
    ]
    return "\n".join(lines)


def breakdown_document(result: TrainedEvaluation, config: BaselineConfig | None = None) -> dict[str, Any]:
    """Bloque `breakdown` de `reports/trained_metrics.json`, con la distribución y los ejemplos (T07, T06-H1)."""
    evaluation = result.breakdown
    return {
        "spec": SPEC,
        "report": f"reports/{TRAINED_BREAKDOWN}",
        "calibration": evaluation.model.as_dict(),
        "sums": evaluation.sums.as_dict(),
        "order": evaluation.order.as_dict(),
        "metrics_unchanged_by_calibration": evaluation.metrics_unchanged,
        "monotonicity": evaluation.monotonicity.as_dict(),
        "sums_total": evaluation.sums_total,
        "failed_checks": evaluation.failed_checks,
        "accepted_deviations": {name: evaluation.accepted_deviations[name] for name in evaluation.deviations_applied},
        "passed": evaluation.passed,
        "distribution": distribution_document(evaluation),
        "examples": examples_document(result, config or default_config()),
    }


__all__ = [
    "HISTOGRAM_BINS",
    "TRAINED_BREAKDOWN",
    "Example",
    "breakdown_document",
    "distribution_document",
    "examples_document",
    "pick_examples",
    "render_breakdown_report",
]
