"""Model card de ML-05 (T06, `reports/model_card.md`), generada automáticamente desde los JSON de `reports/`.

Todas las cifras salen de `reports/trained_metrics.json`, `reports/train_cv_metrics.json`,
`reports/ablation_metrics.json` y `reports/baseline_metrics.json`; los pesos del baseline, de
`baseline_params.json` (`default_config`), las versiones de scikit-learn y LightGBM, de `uv.lock`,
y los meses de las encuestas, de `thehouse_ml.inpc`. No hay métricas escritas a mano:
`render_model_card` es la única fuente del texto, igual que `render_data_card` en ML-02.

Las únicas constantes de trazabilidad (fecha de la promoción, PR y decisiones M-ML) están en
este módulo porque no forman parte de ningún JSON (M-ML-042). La distribución de `matchScore` y
los ejemplos explicados solo existen en `reports/trained_breakdown.md` (no en un JSON), así que
la model card los enlaza sin copiar sus cifras: volver a puntuar test está prohibido (spec §6.2).

Determinista (sin fechas de ejecución, tiempos ni rutas absolutas): dos generaciones con los
mismos JSON escriben los mismos bytes. Todo es **sintético** (etiquetas de M-ML-032).
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thehouse_ml.inpc import ENIGH_SURVEY_MONTHS, ENVI_SURVEY_MONTHS, REFERENCE_MONTH
from thehouse_ml.models.baseline_config import default_config
from thehouse_ml.models.baseline_report import EvaluationError
from thehouse_ml.paths import ProjectPaths

MODEL_CARD = "model_card.md"
TRAINED_METRICS_JSON = "trained_metrics.json"
TRAIN_CV_METRICS_JSON = "train_cv_metrics.json"
ABLATION_METRICS_JSON = "ablation_metrics.json"
BASELINE_METRICS_JSON = "baseline_metrics.json"
INPUT_REPORTS: tuple[str, ...] = (
    TRAINED_METRICS_JSON,
    TRAIN_CV_METRICS_JSON,
    ABLATION_METRICS_JSON,
    BASELINE_METRICS_JSON,
)
UV_LOCK = "uv.lock"
LIBRARIES: tuple[str, ...] = ("scikit-learn", "lightgbm", "numpy", "pandas", "scipy")
SPEC = "ML-05 1.0.0, T06 (ML-05-R06)"
TITLE_BANNER = "DATOS SINTÉTICOS"
MODEL_CARD_SKIP_MESSAGE = "model card: omitida (faltan reportes de evaluate o uv.lock; ejecute `evaluate`)"

# Trazabilidad que no está en ningún JSON (M-ML-042).
PROMOTION_DATE = "2026-11-24"
PROMOTION_DECISION = "M-ML-041"
PULL_REQUESTS: tuple[tuple[str, int, str], ...] = (
    ("spec 1.0.0 (M-ML-036)", 30, "docs/ml-05-spec"),
    ("T01 dependencias y carga de datos (M-ML-037)", 31, "feat/ml-05-deps-data"),
    ("T02 validación cruzada y entrenamiento (M-ML-038)", 32, "feat/ml-05-train"),
    ("T03 medición única en test (M-ML-039)", 33, "feat/ml-05-eval"),
    ("T04 ablación y segmentos (M-ML-040)", 34, "feat/ml-05-ablation"),
    ("T05 puntaje 0-100 y desglose (M-ML-041)", 35, "feat/ml-05-breakdown"),
)
MODEL_CARD_BRANCH = "docs/ml-05-model-card"
DECISIONS: tuple[tuple[str, str], ...] = (
    ("M-ML-029", "ML-04 aprobada: modelo base, esquema 2.0.0 y etiquetas sintéticas"),
    ("M-ML-031", "puntaje 0-100 con desglose y redondeo del baseline"),
    ("M-ML-032", "simulador de etiquetas sintéticas (`labels_params.json` 1.0.0)"),
    ("M-ML-034", "evaluación del baseline (`baseline_eval_params.json` 1.0.0)"),
    ("M-ML-035", "cierre de ML-04 y herencia para ML-05"),
    ("M-ML-036", "ML-05 1.0.0 aprobada: criterio de promoción"),
    ("M-ML-037", "T01: dependencias y cargador"),
    ("M-ML-038", "T02: pliegues, rejilla, semilla y perfil `ci`"),
    ("M-ML-039", "T03: medición única en test y comparación pareada"),
    ("M-ML-040", "T04: ablación, segmentos e hipótesis > $3,500"),
    ("M-ML-041", "T05: calibración, desglose, desviación T05-H1 y promoción"),
    ("M-ML-042", "T06: generación de esta model card (APROBADO)"),
)
METRICS: tuple[str, ...] = ("ndcg@10", "ndcg@5", "precision@5", "map")
METRIC_LABELS: dict[str, str] = {"ndcg@10": "NDCG@10", "ndcg@5": "NDCG@5", "precision@5": "Precision@5", "map": "MAP"}
MODELS: tuple[str, ...] = ("baseline", "logreg", "lightgbm")
MODEL_LABELS: dict[str, str] = {
    "baseline": "Baseline (`baseline-1.0.0`)",
    "logreg": "Regresión logística",
    "lightgbm": "LightGBM LambdaRank",
}
SECTION_TITLES: tuple[str, ...] = (
    "## 1. Resumen",
    "## 2. Uso previsto y fuera de alcance",
    "## 3. Datos",
    "## 4. Entrenamiento",
    "## 5. Métricas",
    "## 6. Ablación y segmentos",
    "## 7. Desglose del puntaje",
    "## 8. Limitaciones y riesgos",
    "## 9. Decisión de promoción y trabajo futuro",
    "## 10. Versiones y trazabilidad",
)


class ModelCardError(EvaluationError):
    """Un reporte de entrada falta o no es la corrida oficial sintética que exige la model card."""


@dataclass(frozen=True)
class ModelCardInputs:
    """Entradas de la model card, ya leídas (la renderización es pura)."""

    trained: Mapping[str, Any]
    cv: Mapping[str, Any]
    ablation: Mapping[str, Any]
    baseline: Mapping[str, Any]
    baseline_weights: Mapping[str, float]
    libraries: Mapping[str, str]
    package_version: str


# ---------------------------------------------------------------------------
# Lectura (única I/O)
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ModelCardError(f"{path.name}: se esperaba un objeto JSON")
    return document


def library_versions(lock_path: Path, names: Sequence[str] = LIBRARIES) -> dict[str, str]:
    """Versiones exactas de `uv.lock` (las instaladas por `uv sync --frozen`)."""
    with lock_path.open("rb") as handle:
        lock = tomllib.load(handle)
    found = {
        str(package["name"]): str(package["version"])
        for package in lock.get("package", [])
        if package.get("name") in names and "version" in package
    }
    missing = [name for name in names if name not in found]
    if missing:
        raise ModelCardError(f"{lock_path.name}: faltan {missing}")
    return {name: found[name] for name in names}


def collect_inputs(paths: ProjectPaths, reports_dir: Path | None = None) -> ModelCardInputs:
    """Lee los 4 JSON de `reports/`, los pesos del baseline y las versiones de `uv.lock`."""
    from thehouse_ml import __version__

    folder = reports_dir or paths.reports
    return ModelCardInputs(
        trained=_read_json(folder / TRAINED_METRICS_JSON),
        cv=_read_json(folder / TRAIN_CV_METRICS_JSON),
        ablation=_read_json(folder / ABLATION_METRICS_JSON),
        baseline=_read_json(paths.reports / BASELINE_METRICS_JSON),
        baseline_weights=default_config().weights,
        libraries=library_versions(paths.root / UV_LOCK),
        package_version=__version__,
    )


def check_inputs(inputs: ModelCardInputs) -> None:
    """La model card solo describe la corrida oficial (`full`), sintética y de un modelo promovido o del baseline."""
    problems: list[str] = []
    for name, document in (
        (TRAINED_METRICS_JSON, inputs.trained),
        (TRAIN_CV_METRICS_JSON, inputs.cv),
        (ABLATION_METRICS_JSON, inputs.ablation),
        (BASELINE_METRICS_JSON, inputs.baseline),
    ):
        if document.get("synthetic") is not True:
            problems.append(f"{name}: synthetic debe ser true")
    for name, document in ((TRAINED_METRICS_JSON, inputs.trained), (ABLATION_METRICS_JSON, inputs.ablation)):
        if document.get("profile") != "full" or document.get("official") is not True:
            problems.append(f"{name}: debe ser la corrida oficial (perfil full)")
    if inputs.cv.get("profile") != "full":
        problems.append(f"{TRAIN_CV_METRICS_JSON}: debe ser el perfil full")
    trained_sha = {name: model["sha256"] for name, model in inputs.trained.get("models", {}).items()}
    cv_sha = {name: entry["sha256"] for name, entry in inputs.cv.get("final_models", {}).get("files", {}).items()}
    for model in inputs.trained.get("models", {}).values():
        if cv_sha.get(model["file"]) != model["sha256"]:
            problems.append(
                f"SHA-256 de {model['file']} distinto entre {TRAINED_METRICS_JSON} y {TRAIN_CV_METRICS_JSON}"
            )
    if not trained_sha:
        problems.append(f"{TRAINED_METRICS_JSON}: sin modelos")
    if problems:
        raise ModelCardError("; ".join(problems))


# ---------------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------------


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _s(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def _ci(stat: Mapping[str, float], digits: int = 3, signed: bool = False) -> str:
    fmt = _s if signed else _f
    return f"{fmt(stat['mean'], digits)} [{fmt(stat['ci_low'], digits)}, {fmt(stat['ci_high'], digits)}]"


def _int(value: int) -> str:
    return f"{value:,}"


def _months(months: Sequence[str]) -> str:
    years = sorted({month[:4] for month in months})
    return "/".join(years)


def _segment(inputs: ModelCardInputs, segment_id: str) -> Mapping[str, Any]:
    for item in inputs.ablation["segments"]["items"]:
        if item["id"] == segment_id:
            return item  # type: ignore[no-any-return]
    raise ModelCardError(f"{ABLATION_METRICS_JSON}: falta el segmento {segment_id!r}")


def _campuses(inputs: ModelCardInputs) -> list[Mapping[str, Any]]:
    return [item for item in inputs.ablation["segments"]["items"] if item["kind"] == "campus"]


def _unverified(inputs: ModelCardInputs) -> list[str]:
    return [item["id"].split(":", 1)[1] for item in _campuses(inputs) if "sin verificar" in item["label"]]


def _model_version(inputs: ModelCardInputs) -> str:
    return str(inputs.trained["breakdown"]["calibration"]["model_version"])


def _promoted(inputs: ModelCardInputs) -> bool:
    return bool(inputs.trained["promotion"]["decision"]["outcome"] == "logreg_promoted")


def _factors(inputs: ModelCardInputs) -> list[Mapping[str, Any]]:
    return list(inputs.trained["breakdown"]["calibration"]["factors"])


def _ablation_groups(inputs: ModelCardInputs, model: str) -> list[Mapping[str, Any]]:
    return list(inputs.ablation["ablation"][model]["groups"])


def _rank(values: Mapping[str, float], *, reverse: bool) -> dict[str, int]:
    ordered = sorted(values, key=lambda key: (-values[key] if reverse else values[key], key))
    return {key: index + 1 for index, key in enumerate(ordered)}


# ---------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------


def _summary(inputs: ModelCardInputs) -> list[str]:
    trained = inputs.trained
    lr = trained["test"]["logreg"]["ndcg@10"]
    base = trained["test"]["baseline"]["ndcg@10"]
    diff = trained["paired_difference_vs_baseline"]["logreg"]["ndcg@10"]
    status = "**PROMOVIDO**" if _promoted(inputs) else "**NO PROMOVIDO** (se queda el baseline)"
    return [
        SECTION_TITLES[0],
        "",
        _row(("Campo", "Valor")),
        _row(("---", "---")),
        _row(("Modelo", f"`{_model_version(inputs)}` — regresión logística aditiva con puntaje 0-100 y desglose")),
        _row(("Estado", f"{status} como modelo de ML-05 el {PROMOTION_DATE} ({PROMOTION_DECISION})")),
        _row(("Respaldo", f"`{trained['baseline_model_version']}` (modelo base por reglas de ML-04)")),
        _row(("Esquema de features", f"`feature_schema.json` {trained['feature_schema_version']}")),
        _row(("Decisión (`trained_metrics.json`)", f"`{trained['promotion']['decision']['outcome']}`")),
        _row(
            (
                "NDCG@10 en test (DATOS SINTÉTICOS)",
                f"{_ci(lr)} contra {_ci(base)} del baseline; diferencia pareada {_ci(diff, signed=True)}",
            )
        ),
        _row(("Spec", "ML-05 1.0.0 (M-ML-036)")),
        "",
        trained["promotion"]["decision"]["text"],
        "",
    ]


def _intended_use(inputs: ModelCardInputs) -> list[str]:
    campuses = _campuses(inputs)
    names = ", ".join(f"`{item['id'].split(':', 1)[1]}`" for item in campuses)
    return [
        SECTION_TITLES[1],
        "",
        "**Uso previsto.**",
        "",
        f"- Ordenar los cuartos candidatos de un estudiante de las **{len(campuses)} universidades** de la región "
        f"Tula/Tepeji ({names}) y darle a cada uno un `matchScore` de 0 a 100 con su desglose por los "
        f"{len(_factors(inputs))} factores (constitución §1 y §12).",
        "- Lo consume el backend dentro del monolito (NV-23, puerto `ICompatibilityModel`) después de la "
        "exportación a ONNX de ML-06; `TheHouse-GenAI` explica el puntaje y su desglose, nunca lo calcula.",
        "",
        "**Fuera de alcance.**",
        "",
        "- **No decide** la aceptación de un estudiante ni el precio del cuarto: solo ordena y explica.",
        "- **No usa género ni atributos sensibles** (edad, origen) como señal de puntaje: las entradas son las "
        f"{len(inputs.trained['breakdown']['monotonicity']['coefficients'])} features de `feature_order` del "
        "esquema; el género solo interviene en la restricción dura `livingWith` que declara el propio estudiante "
        "(constitución §11.2).",
        "- **Los pares excluidos por HC01-HC03 no reciben puntaje** (renta sobre `budget.max`, `livingWith` y "
        "mascotas; constitución §11.1): el modelo solo puntúa candidatos y devuelve el par excluido con el "
        "formato de ML-04.",
        "- No se afirma desempeño real: todas las métricas son sobre **DATOS SINTÉTICOS** (sección 3).",
        "",
    ]


def _data(inputs: ModelCardInputs) -> list[str]:
    summary = inputs.cv["data"]["summary"]
    excluded = inputs.cv["data"]["excluded_students"]
    excluded_text = ", ".join(
        f"`{item['student_id']}` ({item['split']}, {item['candidates']} candidato)" for item in excluded
    )
    lines = [
        SECTION_TITLES[2],
        "",
        "> **DATOS SINTÉTICOS.** Estudiantes, cuartos, pares y etiquetas son sintéticos.",
        "",
        "- **Estudiantes, cuartos y pares sintéticos** de ML-03, calibrados con fuentes de INEGI (ENVI "
        f"{_months(ENVI_SURVEY_MONTHS)}, ENIGH {_months(ENIGH_SURVEY_MONTHS)} deflactadas con el INPC a "
        f"{REFERENCE_MONTH}, Censo y DENUE): ver la [data card](data_card.md).",
        f"- **Etiquetas sintéticas** de relevancia 0-3 de ML-04 (`labels.parquet`, `labels_params.json` "
        f"{inputs.trained['labels_params_version']}, M-ML-032): una utilidad oculta documentada más ruido. ML-05 "
        "no genera etiquetas y nunca lee la utilidad ni los pesos ocultos (prueba obligatoria de T01).",
        f"- Objetivo de la regresión logística: `relevance ≥ {inputs.cv['relevance_threshold']}`.",
        "",
        _row(("Partición", "Pares", "Estudiantes", "Excluidos (< 2 candidatos)")),
        _row(("---", "---:", "---:", "---:")),
    ]
    for split in ("train", "validation", "test", "total"):
        item = summary[split]
        lines.append(_row((split, _int(item["pairs"]), _int(item["students"]), _int(item["excluded_students"]))))
    lines += [
        "",
        f"Excluidos: {excluded_text or 'ninguno'}.",
        "",
        "**Advertencia de circularidad (ML-05-RK01).** Las etiquetas salen de un simulador cuya utilidad usa las "
        "mismas features que ve el modelo: el modelo aprende esa función más el ruido. Las métricas de esta "
        "ficha **validan el pipeline** (partición, entrenamiento, comparación y desglose), **no el desempeño "
        "real**, que solo se medirá con señales reales en ML-07 (constitución §8.2 y §8.3).",
        "",
    ]
    return lines


def _training(inputs: ModelCardInputs) -> list[str]:
    cv = inputs.cv
    folds = cv["folds"]
    selected = cv["selected"]
    final = cv["final_models"]
    lines = [
        SECTION_TITLES[3],
        "",
        f"- **Validación cruzada:** {folds['n_splits']} pliegues `StratifiedGroupKFold` agrupados por estudiante y "
        f"estratificados por campus sobre {' + '.join(folds['splits'])} ({_int(folds['students'])} estudiantes, "
        f"{_int(folds['pairs'])} pares), semilla {folds['seed']}; ningún estudiante en dos pliegues. La partición "
        "test solo se usó una vez, en T03.",
        "",
        _row(("Pliegue", "Estudiantes", "Pares")),
        _row(("---", "---:", "---:")),
    ]
    for fold in folds["by_fold"]:
        lines.append(_row((f"P{fold['fold']}", _int(fold["students"]), _int(fold["pairs"]))))
    lines += [
        "",
        f"- **Rejilla** (`train_params.json` {cv['train_params_version']}, {cv['train_params_decision']}), elegida "
        "por el mayor NDCG@10 medio entre pliegues (desempate: menor desviación y después la más simple):",
        "",
        _row(("Configuración", "NDCG@10 media", "Desv.", "Elegida")),
        _row(("---", "---:", "---:", "---")),
    ]
    chosen = {selected[model]["config_id"] for model in ("logreg", "lightgbm")}
    for grid in ("logreg_grid", "lightgbm_grid"):
        for item in cv[grid]:
            mark = "**sí**" if item["config_id"] in chosen else ""
            lines.append(
                _row((f"`{item['config_id']}`", _f(item["ndcg@10_mean"], 4), _f(item["ndcg@10_std"], 4), mark))
            )
    violations = final["lightgbm_monotone_check"]["violations"]
    constrained = ", ".join(f"`{name}`" for name in violations)
    lines += [
        "",
        f"- **Hiperparámetros elegidos:** regresión logística `{selected['logreg']['config_id']}`; LightGBM "
        f"`{selected['lightgbm']['config_id']}` con restricciones monotónicas en {constrained} "
        f"({sum(violations.values())} violaciones en la comprobación del modelo final).",
        f"- **Modelos finales:** reentrenados en {' + '.join(final['trained_on']['splits'])} "
        f"({_int(final['trained_on']['students'])} estudiantes, {_int(final['trained_on']['pairs'])} pares); "
        "guardados sin pickle (JSON y texto nativo de LightGBM, constitución §13.1).",
        "- **Reproducibilidad:** misma semilla, `uv.lock` y datos con SHA-256 → mismos pliegues, coeficientes, "
        "modelos y métricas; dos corridas de `train` dieron SHA-256 idénticos (M-ML-038).",
        "",
        _row(("Archivo", "Bytes", "SHA-256")),
        _row(("---", "---:", "---")),
    ]
    for name, entry in final["files"].items():
        lines.append(_row((f"`models/{name}`", _int(entry["bytes"]), f"`{entry['sha256']}`")))
    lines.append("")
    return lines


def _metrics(inputs: ModelCardInputs) -> list[str]:
    trained = inputs.trained
    cv_metrics = inputs.ablation["cross_validation"]["metrics"]
    n_folds = inputs.ablation["cross_validation"]["n_splits"]
    lines = [
        SECTION_TITLES[4],
        "",
        "> **DATOS SINTÉTICOS.** Métricas por estudiante de M-ML-034; ranking por puntaje descendente con desempate "
        "por `room_id`.",
        "",
        f"**Validación cruzada** ({n_folds} pliegues; media ± desviación estándar muestral):",
        "",
        _row(("Modelo", "Métrica", *(f"P{i}" for i in range(n_folds)), "Media ± desv.")),
        _row(("---", "---", *("---:" for _ in range(n_folds)), "---:")),
    ]
    for model in MODELS:
        block = cv_metrics[model]
        for metric in METRICS:
            values = [_f(fold[metric], 4) for fold in block["by_fold"]]
            mean_std = f"{_f(block['mean'][metric], 4)} ± {_f(block['std'][metric], 4)}"
            lines.append(_row((MODEL_LABELS[model], METRIC_LABELS[metric], *values, mean_std)))
    evaluation = trained["evaluation"]
    lines += [
        "",
        f"**Test** (medición única: {_int(evaluation['students_evaluated'])} estudiantes, "
        f"{_int(evaluation['pairs'])} pares; IC 95 % por bootstrap pareado por estudiante, "
        f"{_int(trained['bootstrap']['resamples'])} remuestreos, semilla {trained['bootstrap']['seed']}):",
        "",
        _row(("Modelo", *(METRIC_LABELS[m] for m in METRICS))),
        _row(("---", *("---:" for _ in METRICS))),
    ]
    for model in MODELS:
        lines.append(_row((MODEL_LABELS[model], *(_ci(trained["test"][model][m]) for m in METRICS))))
    lines += [
        "",
        "**Diferencia pareada contra el baseline** (test):",
        "",
        _row(("Modelo - baseline", *(METRIC_LABELS[m] for m in METRICS))),
        _row(("---", *("---:" for _ in METRICS))),
    ]
    for model in ("logreg", "lightgbm"):
        diff = trained["paired_difference_vs_baseline"][model]
        lines.append(_row((MODEL_LABELS[model], *(_ci(diff[m], signed=True) for m in METRICS))))
    lines += ["", "**Criterio de promoción** (spec §6.5):", ""]
    lines += [_row(("Condición", *(MODEL_LABELS[m] for m in ("logreg", "lightgbm")))), _row(("---", "---", "---"))]
    candidates = trained["promotion"]["candidates"]
    labels = {
        "1_cv_all_folds_above_baseline": "1. NDCG@10 mayor que el baseline en cada pliegue",
        "2_test_paired_ci_above_zero": "2. IC 95 % de Δ NDCG@10 en test con límite inferior > 0",
        "3_additive_breakdown": "3. Desglose aditivo que suma el total",
    }
    for condition_id, label in labels.items():
        cells = []
        for model in ("logreg", "lightgbm"):
            condition = candidates[model]["conditions"][condition_id]
            detail = ""
            if condition_id == "1_cv_all_folds_above_baseline":
                detail = f" ({condition['folds_above_baseline']}/{condition['required_folds']})"
            elif condition_id == "3_additive_breakdown":
                detail = f" (`{condition['status']}`)"
            cells.append(("**sí**" if condition["passed"] else "no") + detail)
        lines.append(_row((label, *cells)))
    lines.append(_row(("Resultado", *(f"`{candidates[m]['outcome']}`" for m in ("logreg", "lightgbm")))))
    lr_test = trained["test"]["logreg"]["ndcg@10"]
    gbm_test = trained["test"]["lightgbm"]["ndcg@10"]
    lr_cv = cv_metrics["logreg"]["mean"]["ndcg@10"]
    gbm_cv = cv_metrics["lightgbm"]["mean"]["ndcg@10"]
    overlap = lr_test["ci_low"] <= gbm_test["ci_high"] and gbm_test["ci_low"] <= lr_test["ci_high"]
    lines += [
        "",
        "**Casi empate entre la regresión logística y LightGBM.** LightGBM - regresión logística en NDCG@10: "
        f"{_s(gbm_cv - lr_cv, 4)} en la media de los pliegues y {_s(gbm_test['mean'] - lr_test['mean'], 4)} en test; "
        f"sus intervalos de test {'se solapan' if overlap else 'no se solapan'} ({_ci(lr_test)} y {_ci(gbm_test)}). "
        "No se calculó un intervalo pareado LightGBM - regresión logística (la spec compara cada modelo contra el "
        "baseline). Se eligió la regresión logística porque es la única que cumple la condición 3: ofrece un "
        "desglose aditivo por factor que suma el total y es exportable con su explicación (constitución §12); "
        f"LightGBM: «{candidates['lightgbm']['conditions']['3_additive_breakdown']['note']}».",
        "",
    ]
    return lines


def _ablation_and_segments(inputs: ModelCardInputs) -> list[str]:
    labels = {group["id"]: group["label"] for group in inputs.ablation["feature_groups"]}
    lr_groups = {group["group"]: group for group in _ablation_groups(inputs, "logreg")}
    gbm_groups = {group["group"]: group for group in _ablation_groups(inputs, "lightgbm")}
    n_folds = inputs.ablation["cross_validation"]["n_splits"]
    order = sorted(lr_groups, key=lambda key: (lr_groups[key]["delta_ndcg@10_mean"], key))
    lines = [
        SECTION_TITLES[5],
        "",
        "> **DATOS SINTÉTICOS.**",
        "",
        "**Ablación** (importancia real de cada factor): se reentrena cada modelo sin las features del grupo, con "
        f"los mismos hiperparámetros y en los {n_folds} pliegues (nunca en test); Δ = NDCG@10 sin el grupo - "
        "completo. Ordenada de mayor a menor importancia para la regresión logística.",
        "",
        _row(("Grupo", "Features", "Δ NDCG@10 LR (media ± desv.)", "Pliegues peores LR", "Δ NDCG@10 LightGBM")),
        _row(("---", "---", "---:", "---:", "---:")),
    ]
    for key in order:
        lr = lr_groups[key]
        gbm = gbm_groups[key]
        lines.append(
            _row(
                (
                    f"{labels[key]} (`{key}`)",
                    ", ".join(f"`{f}`" for f in lr["features_removed"]),
                    f"{_s(lr['delta_ndcg@10_mean'], 4)} ± {_f(lr['delta_ndcg@10_std'], 4)}",
                    f"{lr['folds_below_full']}/{n_folds}",
                    f"{_s(gbm['delta_ndcg@10_mean'], 4)} ± {_f(gbm['delta_ndcg@10_std'], 4)}",
                )
            )
        )
    outside = ", ".join(f"`{name}`" for name in inputs.ablation["outside_groups"])
    lines += [
        "",
        f"{outside} no pertenece a ningún grupo: es constante (1) en los candidatos y se queda en todos los modelos.",
        "",
        "**Segmentos** (constitución §11.3; de la medición única de test, NDCG@10 con IC 95 %; Δ = pareada contra "
        "el baseline):",
        "",
        _row(("Segmento", "Estudiantes", "Baseline", "Regresión logística", "Δ LR", "LightGBM", "Δ LightGBM")),
        _row(("---", "---:", "---:", "---:", "---:", "---:", "---:")),
    ]
    for item in inputs.ablation["segments"]["items"]:
        lines.append(
            _row(
                (
                    item["label"],
                    _int(item["students"]),
                    _ci(item["baseline"]["ndcg@10"]),
                    _ci(item["logreg"]["ndcg@10"]),
                    _ci(item["diff_logreg_minus_baseline"]["ndcg@10"], signed=True),
                    _ci(item["lightgbm"]["ndcg@10"]),
                    _ci(item["diff_lightgbm_minus_baseline"]["ndcg@10"], signed=True),
                )
            )
        )
    hyp = inputs.ablation["hypothesis_budget_gt_3500"]
    ranges = hyp["delta_ndcg@10_by_budget_range"]
    top = ranges[hyp["segment"]]
    overlapping = [
        _segment(inputs, key)["label"]
        for key, stat in ranges.items()
        if key != hyp["segment"] and stat["ci_high"] >= top["ci_low"]
    ]
    verdict = {"confirmed": "CONFIRMADA", "refuted": "REFUTADA", "inconclusive": "NO CONCLUYENTE"}[hyp["verdict"]]
    lines += [
        "",
        f"**Hipótesis del segmento `budget.max` > $3,500: {verdict}** (regla de M-ML-040, fijada antes de calcular "
        "los segmentos).",
        "",
        *(f"- {reason}" for reason in hyp["reasons"]),
        "",
        "Salvedades (aceptadas al aprobar M-ML-040):",
        "",
        f"- **T04-H2:** la condición (1) compara medias puntuales; el IC de > $3,500 ({_ci(top, signed=True)}) se "
        f"solapa con el de {', '.join(overlapping) or 'ningún otro rango'}. La regla no exige que las diferencias "
        "entre rangos sean significativas.",
        f"- **T04-H3:** la condición (3) usa la ablación **global** del grupo presupuesto "
        f"({_s(hyp['budget_ablation_delta_ndcg@10_mean'], 4)}, {hyp['budget_ablation_folds_below_full']}/{n_folds} "
        "pliegues), no una ablación por rango de presupuesto como sugiere la spec §6.7.",
        "",
    ]
    return lines


def _breakdown(inputs: ModelCardInputs) -> list[str]:
    breakdown = inputs.trained["breakdown"]
    calibration = breakdown["calibration"]
    span = calibration["total_span_logit"]
    sums = breakdown["sums"]
    coefficients = breakdown["monotonicity"]["coefficients"]
    lr_groups = {group["group"]: group["delta_ndcg@10_mean"] for group in _ablation_groups(inputs, "logreg")}
    max_points = {factor["factor"]: factor["raw_max_points"] for factor in _factors(inputs)}
    top_factor = max(_factors(inputs), key=lambda factor: factor["raw_max_points"])
    rank_points = _rank(max_points, reverse=True)
    rank_ablation = _rank(lr_groups, reverse=False)
    lines = [
        SECTION_TITLES[6],
        "",
        f"**Calibración** ({calibration['method']}; M-ML-041): cada feature se recorta a su `training_range` antes de "
        f"puntuar; S = {_f(span, 6)} unidades de logit, intercepto b0 = {_f(calibration['intercept'], 6)}. "
        "`matchScore` es una función afín creciente del logit, así que el orden del ranking es el del modelo.",
        "",
        "**Coeficientes** (entrada estandarizada; «puntos por unidad» = 100 · |logit por unidad| / S):",
        "",
        _row(("Feature", "Factor", "coef (estandarizado)", "Logit por unidad", "Puntos por unidad", "Signo coherente")),
        _row(("---", "---", "---:", "---:", "---:", "---")),
    ]
    for item in coefficients:
        coherent = {True: "sí", False: "**no**", None: "—"}[item["sign_consistent"]]
        lines.append(
            _row(
                (
                    f"`{item['feature']}`",
                    f"`{item['factor']}`" if item["factor"] else "término fijo",
                    _s(item["coef_standardized"], 6),
                    _s(item["logit_per_unit"], 6),
                    _f(100 * abs(item["logit_per_unit"]) / span, 4),
                    coherent,
                )
            )
        )
    lines += [
        "",
        "**`maxPoints` frente a los pesos del baseline y a la ablación:**",
        "",
        _row(
            (
                "Factor",
                "maxPoints",
                "Peso del baseline",
                "Δ NDCG@10 ablación LR",
                "Orden por maxPoints",
                "Orden por ablación",
            )
        ),
        _row(("---", "---:", "---:", "---:", "---:", "---:")),
    ]
    for factor in _factors(inputs):
        key = factor["factor"]
        lines.append(
            _row(
                (
                    f"{factor['label']} (`{key}`)",
                    _f(factor["maxPoints"], 1),
                    f"{inputs.baseline_weights[key]:g}",
                    _s(lr_groups[key], 4),
                    str(rank_points[key]),
                    str(rank_ablation[key]),
                )
            )
        )
    lines += [
        _row(
            (
                "**Total**",
                _f(calibration["max_points_sum"], 1),
                f"{sum(inputs.baseline_weights.values()):g}",
                "",
                "",
                "",
            )
        ),
        "",
        f"Verificación en {_int(sums['pairs'])} pares de test: máx. |Σ sin redondear - `matchScore`| "
        f"{_f(sums['max_abs_raw_total_minus_matchScore'], 5)} (≤ {sums['total_tolerance']}); máx. |Σ reportada - Σ| "
        f"{_f(sums['max_abs_reported_sum_minus_raw_total'], 4)} (≤ {sums['breakdown_tolerance']}); "
        f"{sums['factors_outside_0_maxPoints']} factores fuera de [0, `maxPoints`]; "
        f"{sums['matchScore_outside_0_100']} totales fuera de [0, 100]; orden idéntico al del logit en "
        f"{breakdown['order']['identical_order']}/{breakdown['order']['students']} estudiantes.",
        "",
        "**T05-H2 — `maxPoints` no es importancia.** `maxPoints` mide cuánto puede mover el logit cada factor dentro "
        "de **todo** su `training_range`, no cuánto importa en los candidatos típicos; la medida de importancia es "
        "la ablación (sección 6). La distancia recibe el mayor `maxPoints` porque su rango llega hasta los límites "
        "del esquema mientras la mayoría de los candidatos están cerca: casi todos obtienen una parte grande de sus "
        "puntos de distancia y **los cuartos cercanos tienden a puntajes altos**. La distribución de `matchScore` y "
        "de los puntos por factor en test, y los ejemplos explicados con su JSON, están en "
        "[`trained_breakdown.md`](trained_breakdown.md) §4 y §6 (no se copian aquí porque no forman parte de un JSON).",
        "",
        "**Nota para `TheHouse-GenAI` (cómo interpretar y explicar el `breakdown`):**",
        "",
        "1. Explicar cada factor como **«obtuvo `points` de `maxPoints` posibles»** (proporción lograda), nunca "
        "presentar `maxPoints` como el peso o la importancia del factor ni comparar `maxPoints` entre factores "
        f"(«{top_factor['label'].lower()} vale el {_f(top_factor['maxPoints'], 1)} % de la decisión» es "
        "**incorrecto**).",
        "2. Para decir qué distingue a un cuarto de otro, usar la **diferencia de `points`** entre cuartos del mismo "
        "estudiante: entre cuartos cercanos la distancia difiere en pocas décimas y los factores con `maxPoints` "
        "pequeño (tipo de lugar, servicios) pueden decidir el orden.",
        "3. Un `matchScore` alto es sobre todo «cuarto cercano»; no debe leerse como garantía de que todo coincide. "
        "Mencionar los factores con proporción baja.",
        "4. Los factores de varias features (`budget`, `distance`) traen la lista `features` con su `value` de "
        "entrada **sin recortar**; el puntaje usa el valor recortado al `training_range`.",
        "5. No recalcular ni modificar `matchScore` ni `points` (constitución §2.5); no mencionar `probability` ni "
        "`logit`, que no forman parte del contrato.",
        "6. Todo lo de esta ficha es **sintético**: no afirmar que el puntaje refleja preferencias reales.",
        "",
    ]
    return lines


def _limitations(inputs: ModelCardInputs) -> list[str]:
    budget = inputs.trained["breakdown"]["monotonicity"]["budget"]
    monotonicity = inputs.trained["breakdown"]["monotonicity"]
    worst = budget["worst_case"]
    unverified = _unverified(inputs)
    verified = _segment(inputs, "without_seed_sin_verificar")
    excluded = inputs.cv["data"]["excluded_students"]
    zone = "ningún par observado" if not monotonicity["observed_pairs_in_violation_zone"] else "hay pares observados"
    verified_delta = _ci(verified["diff_logreg_minus_baseline"]["ndcg@10"], signed=True)
    return [
        SECTION_TITLES[7],
        "",
        f"- **Desviación T05-H1 (aceptada, {PROMOTION_DECISION}):** el coeficiente de `price_below_min` es negativo; "
        "por debajo de `budget.min` el factor presupuesto **sube con la renta** si `budget.max / budget.min` > "
        f"{_f(monotonicity['budget_ratio_threshold'], 1)} ({budget['violations']} de {_int(budget['steps'])} pasos de "
        f"la rejilla; subida máxima {_s(budget['worst_increase_points'], 3)} puntos sobre 100 con `budget.min` "
        f"{worst['budget.min']} y `budget.max` {_int(worst['budget.max'])}). En los datos sintéticos el mayor cociente "
        f"es {_f(monotonicity['observed_max_budget_ratio'], 2)}: {zone} en la zona. NV-23 debe conocerla; se "
        "corrige en `logreg-1.1.0`.",
        f"- **Datos de encuestas antiguos:** la calibración usa ENVI {_months(ENVI_SURVEY_MONTHS)} y ENIGH "
        f"{_months(ENIGH_SURVEY_MONTHS)} deflactadas con el INPC a {REFERENCE_MONTH}; no observan el mercado actual.",
        "- **Renta estatal:** la renta no se estima a nivel municipal (no cumple la regla de suficiencia); se usa la "
        "estimación estatal de ENVI ([data card](data_card.md) §5 y §10).",
        f"- **{len(unverified)} universidades sin verificar** (`seed_sin_verificar`: "
        f"{', '.join(f'`{code}`' for code in unverified)}): sus coordenadas no se confirmaron; el segmento sin ellas "
        f"({_int(verified['students'])} estudiantes) da Δ NDCG@10 LR {verified_delta}.",
        f"- **Estudiantes con 1 candidato:** {len(excluded)} excluido(s) de entrenamiento y evaluación "
        f"({', '.join(f'`{item["student_id"]}`' for item in excluded)}); en producción un estudiante con un solo "
        "candidato recibe su puntaje, pero no hay ranking que evaluar.",
        "- **Circularidad y etiquetas sintéticas** (ML-05-RK01): sección 3.",
        "- **Paridad pendiente:** la exportación a ONNX, la paridad Python ↔ onnxruntime y la de .NET se verifican "
        "en ML-06 y NV-23 (ML-05-RK03); el recorte al `training_range` debe ser la misma regla en el backend.",
        "- **Campos de Property pendientes (NV-12, RK03 de ML-03):** el cuarto del backend aún no tiene coordenadas, "
        "`placeType`, `propertyType`, mascotas, `livingWith` ni servicios como catálogo; sin ellos el modelo no se "
        "puede usar en producción.",
        "- **Timeout del CI (T05-H3):** el job `quality` necesitó un tiempo límite mayor porque las pruebas casi lo "
        "agotaban (`.github/workflows/ci.yml`); queda pendiente optimizar las pruebas lentas.",
        "",
    ]


def _decision_and_future(inputs: ModelCardInputs) -> list[str]:
    decision = inputs.trained["promotion"]["decision"]
    deviations = inputs.trained["breakdown"]["accepted_deviations"]
    return [
        SECTION_TITLES[8],
        "",
        f"**Decisión ({PROMOTION_DECISION}, {PROMOTION_DATE}):** `{decision['outcome']}`. {decision['text']}",
        "",
        *(f"- Desviación aceptada `{name}`: {text}." for name, text in deviations.items()),
        "",
        "**Trabajo futuro:**",
        "",
        "1. `logreg-1.1.0` sin `price_below_min` o con su coeficiente ≥ 0 (corrige T05-H1), evaluado con datos "
        "nuevos de ML-07: la partición test ya se usó y no puede medir un modelo reentrenado (spec §6.2).",
        "2. Revisar la calibración de `maxPoints` (T05-H2) para que refleje mejor el rango observado en los "
        "candidatos y no todo el `training_range`; requiere una versión nueva del modelo y coordinarlo con "
        "`TheHouse-GenAI` y NV-23.",
        "3. Optimizar las pruebas lentas (fixtures de sesión, sin regenerar los datos completos) para volver a un "
        "margen holgado en el CI (T05-H3, T07).",
        "4. LightGBM como candidato si se resuelve un desglose aditivo explicable y exportable (p. ej., "
        "contribuciones por factor); hoy queda como referencia del potencial.",
        "5. Medir el desempeño real con señales reales de TheHouse (ML-07).",
        "",
    ]


def _traceability(inputs: ModelCardInputs) -> list[str]:
    trained = inputs.trained
    lines = [
        SECTION_TITLES[9],
        "",
        _row(("Elemento", "Versión")),
        _row(("---", "---")),
        _row(("Modelo promovido", f"`{_model_version(inputs)}`")),
        _row(("Respaldo", f"`{trained['baseline_model_version']}`")),
        _row(("Esquema de features", trained["feature_schema_version"])),
        _row(("`train_params.json`", trained["train_params_version"])),
        _row(("`labels_params.json`", trained["labels_params_version"])),
        _row(("`baseline_eval_params.json`", trained["eval_params_version"])),
        _row(("Paquete `thehouse-ml`", inputs.package_version)),
        *(_row((f"`{name}` (`uv.lock`)", version)) for name, version in inputs.libraries.items()),
        "",
        _row(("Archivo", "SHA-256", "Verificado contra")),
        _row(("---", "---", "---")),
    ]
    for model in trained["models"].values():
        lines.append(
            _row((f"`models/{model['file']}`", f"`{model['sha256']}`", f"`{model['sha256_verified_against']}`"))
        )
    for name, entry in trained["inputs"].items():
        lines.append(_row((f"`data/processed/{name}`", f"`{entry['sha256']}`", "`train_manifest.json`")))
    lines += [
        "",
        "Pull requests de ML-05 (`thehouse-team/TheHouse-ML`):",
        "",
        _row(("Trabajo", "PR", "Rama")),
        _row(("---", "---:", "---")),
        *(_row((work, f"#{number}", f"`{branch}`")) for work, number, branch in PULL_REQUESTS),
        _row(("T06 model card (M-ML-042)", "pendiente", f"`{MODEL_CARD_BRANCH}`")),
        "",
        "Decisiones:",
        "",
        *(f"- **{decision}** — {text}" for decision, text in DECISIONS),
        "",
        "Reportes de origen: `reports/trained_metrics.json`, `reports/train_cv_metrics.json`, "
        "`reports/ablation_metrics.json` y `reports/baseline_metrics.json`; detalle en "
        "[`trained_report.md`](trained_report.md), [`ablation_report.md`](ablation_report.md), "
        "[`trained_breakdown.md`](trained_breakdown.md), [`training_report.md`](training_report.md) y "
        "[`baseline_report.md`](baseline_report.md).",
        "",
    ]
    return lines


def render_model_card(inputs: ModelCardInputs) -> str:
    """Renderiza las 10 secciones de la model card a partir de los JSON ya leídos."""
    check_inputs(inputs)
    header = [
        f"# Model card — `{_model_version(inputs)}` (ML-05) — {TITLE_BANNER}",
        "",
        f"> **{inputs.trained['banner']}.** {inputs.trained['warning']}",
        "",
        f"Generada automáticamente (`thehouse_ml.models.model_card.render_model_card`, {SPEC}) desde los JSON de "
        "`reports/`, `baseline_params.json` y `uv.lock`; ninguna cifra está escrita a mano. La regenera "
        "`python -m thehouse_ml evaluate` (perfil `full`).",
        "",
    ]
    sections = (
        _summary(inputs),
        _intended_use(inputs),
        _data(inputs),
        _training(inputs),
        _metrics(inputs),
        _ablation_and_segments(inputs),
        _breakdown(inputs),
        _limitations(inputs),
        _decision_and_future(inputs),
        _traceability(inputs),
    )
    return "\n".join([*header, *("\n".join(section) for section in sections)]).rstrip("\n") + "\n"


def run_model_card(paths: ProjectPaths, reports_dir: Path | None = None) -> str:
    """Genera `reports/model_card.md` y devuelve el mensaje para la consola."""
    folder = reports_dir or paths.reports
    needed = [
        *(folder / name for name in INPUT_REPORTS[:3]),
        paths.reports / BASELINE_METRICS_JSON,
        paths.root / UV_LOCK,
    ]
    if not all(path.is_file() for path in needed):
        return MODEL_CARD_SKIP_MESSAGE
    text = render_model_card(collect_inputs(paths, folder))
    target = folder / MODEL_CARD
    target.write_text(text, encoding="utf-8", newline="\n")
    return f"model card (ML-05 T06): {target.relative_to(paths.root).as_posix()} (DATOS SINTÉTICOS)"


__all__ = [
    "MODEL_CARD",
    "MODEL_CARD_SKIP_MESSAGE",
    "SECTION_TITLES",
    "ModelCardError",
    "ModelCardInputs",
    "check_inputs",
    "collect_inputs",
    "library_versions",
    "render_model_card",
    "run_model_card",
]
