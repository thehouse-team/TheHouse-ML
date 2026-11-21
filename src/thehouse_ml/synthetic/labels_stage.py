"""Etapa ``prepare`` (etiquetas sintéticas): Parquet y reporte (ML-04 T04, ML-04-R04).

Lee `data/processed/pairs.parquet` (ML-03 T06, esquema 2.0.0), simula la
relevancia con `thehouse_ml.synthetic.labels` y escribe:

- `data/processed/labels.parquet` (fuera de git): `student_id`, `room_id`,
  `split`, `relevance` (0-3), `synthetic = true` y `labels_params_version`;
  una fila por par de `pairs.parquet`, en el mismo orden;
- `data/processed/labels_hidden_utility.parquet` y
  `data/processed/labels_hidden_weights.parquet` (fuera de git): la utilidad
  oculta por par y los pesos ocultos por estudiante, aparte para que ML-05 no
  los use como entrada por accidente;
- `reports/labels_synthetic.md` (en git), titulado «DATOS SINTÉTICOS — NO SON
  PREFERENCIAS REALES».

La correlación de Spearman entre la suma sin redondear del modelo base
(`score_features`, T03) y la utilidad oculta se calcula aquí, no en el
simulador, y debe quedar en `check.spearman_range` (ML-04-RK02); si no,
`LabelsError` después de escribir el reporte. El reporte no lleva fechas ni
rutas absolutas: correr `prepare` dos veces no cambia ningún archivo.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import pandas as pd

from thehouse_ml.features import default_schema
from thehouse_ml.features.loader import FeatureSchema
from thehouse_ml.models import default_config, score_features
from thehouse_ml.models.baseline_config import BaselineConfig
from thehouse_ml.paths import ProjectPaths
from thehouse_ml.synthetic.labels import (
    FACTORS,
    GRADES,
    LABELS_PARAMS_RESOURCE,
    LabelInput,
    LabelParams,
    LabelSet,
    load_params,
    relevance_by_position,
    simulate_labels,
)
from thehouse_ml.synthetic.pairs import SPLITS
from thehouse_ml.synthetic.pairs_stage import PAIRS_PARQUET
from thehouse_ml.synthetic.students_stage import _pct, _row, quartiles

LABELS_PARQUET = "labels.parquet"
HIDDEN_UTILITY_PARQUET = "labels_hidden_utility.parquet"
HIDDEN_WEIGHTS_PARQUET = "labels_hidden_weights.parquet"
LABELS_REPORT = "labels_synthetic.md"
LABELS_TITLE = "# DATOS SINTÉTICOS — NO SON PREFERENCIAS REALES"
LABELS_SKIP_MESSAGE = (
    f"labels: omitido; falta data/processed/{PAIRS_PARQUET} (ejecute `prepare` con los insumos de ML-02 y ML-03)"
)
LABEL_COLUMNS = ("student_id", "room_id", "split", "relevance", "synthetic", "labels_params_version")
RELEVANCES = (3, 2, 1, 0)
TOTAL = "**Total**"
TOTAL_VERIFIED = "**Total sin `seed_sin_verificar`**"
POSITION_EXAMPLES = (1, 2, 3, 4, 5, 10, 200)


class LabelsError(RuntimeError):
    """`pairs.parquet` no sirve para etiquetar o la correlación con el baseline sale del rango documentado."""


# ---------------------------------------------------------------------------
# Parquet ↔ objetos
# ---------------------------------------------------------------------------


def label_inputs(frame: pd.DataFrame, params: LabelParams) -> list[LabelInput]:
    """Pares de `pairs.parquet` como entradas del simulador (solo ids y las features de `utility.curves`)."""
    features = sorted({curve.feature for curve in params.curves.values()})
    missing = [name for name in ("student_id", "room_id", "split", *features) if name not in frame.columns]
    if missing:
        raise LabelsError(f"{PAIRS_PARQUET}: faltan las columnas {missing}")
    columns = ["student_id", "room_id", *features]
    return [
        LabelInput(str(row[0]), str(row[1]), dict(zip(features, (float(value) for value in row[2:]), strict=True)))
        for row in frame[columns].itertuples(index=False, name=None)
    ]


def labels_frame(label_set: LabelSet, splits: dict[str, str], params: LabelParams) -> pd.DataFrame:
    """Tabla de etiquetas (una fila por par, en orden de `student_id` y `room_id`)."""
    rows = [
        {
            "student_id": label.student_id,
            "room_id": label.room_id,
            "split": splits[label.student_id],
            "relevance": label.relevance,
            "synthetic": True,
            "labels_params_version": params.version,
        }
        for label in label_set.labels
    ]
    frame = pd.DataFrame(rows, columns=list(LABEL_COLUMNS))
    return frame.astype(
        {
            "student_id": "string",
            "room_id": "string",
            "split": "string",
            "relevance": "int64",
            "synthetic": "bool",
            "labels_params_version": "string",
        }
    )


def hidden_utility_frame(label_set: LabelSet) -> pd.DataFrame:
    """Utilidad oculta (con ruido) por par; fuera de git y aparte de las etiquetas."""
    rows = [
        {"student_id": item.student_id, "room_id": item.room_id, "utility": item.utility, "synthetic": True}
        for item in label_set.labels
    ]
    frame = pd.DataFrame(rows, columns=["student_id", "room_id", "utility", "synthetic"])
    return frame.astype({"student_id": "string", "room_id": "string", "utility": "float64", "synthetic": "bool"})


def hidden_weights_frame(label_set: LabelSet) -> pd.DataFrame:
    """Pesos ocultos por estudiante (suman 100); fuera de git."""
    columns = ["student_id", "candidates", *(f"weight_{factor}" for factor in FACTORS), "synthetic"]
    rows = [
        {
            "student_id": item.student_id,
            "candidates": item.candidates,
            **{f"weight_{factor}": item.weights[factor] for factor in FACTORS},
            "synthetic": True,
        }
        for item in label_set.students
    ]
    frame = pd.DataFrame(rows, columns=columns)
    dtypes = {"student_id": "string", "candidates": "int64", "synthetic": "bool"}
    dtypes.update({f"weight_{factor}": "float64" for factor in FACTORS})
    return frame.astype(dtypes)


# ---------------------------------------------------------------------------
# Correlación con el modelo base (ML-04-RK02)
# ---------------------------------------------------------------------------


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Correlación de Spearman (Pearson de los rangos promedio); NaN si alguna serie es constante."""
    if len(xs) != len(ys):
        raise ValueError("spearman: series de distinta longitud")
    if len(xs) < 2:
        return math.nan
    x = pd.Series(xs, dtype="float64").rank(method="average")
    y = pd.Series(ys, dtype="float64").rank(method="average")
    if x.nunique() < 2 or y.nunique() < 2:
        return math.nan
    return float(x.corr(y, method="pearson"))


def baseline_totals(frame: pd.DataFrame, schema: FeatureSchema, config: BaselineConfig) -> list[float]:
    """Suma sin redondear del modelo base (`score_features`, T03) para cada par, en el orden del Parquet."""
    order = list(schema.feature_order)
    return [
        score_features([float(value) for value in row], config, schema).raw_total
        for row in frame[order].itertuples(index=False, name=None)
    ]


def per_student_spearman(
    student_ids: Sequence[str], baseline: Sequence[float], utilities: Sequence[float]
) -> list[float]:
    """Spearman dentro de cada estudiante con 2 o más candidatos y series no constantes."""
    groups: dict[str, tuple[list[float], list[float]]] = {}
    for student_id, base, value in zip(student_ids, baseline, utilities, strict=True):
        group = groups.setdefault(student_id, ([], []))
        group[0].append(base)
        group[1].append(value)
    values = [spearman(base, value) for base, value in groups.values()]
    return [value for value in values if not math.isnan(value)]


# ---------------------------------------------------------------------------
# Etapa
# ---------------------------------------------------------------------------


def run_labels(paths: ProjectPaths) -> str:
    """Genera las etiquetas sintéticas, los archivos ocultos y `reports/labels_synthetic.md`."""
    pairs_path = paths.data_processed / PAIRS_PARQUET
    if not pairs_path.is_file():
        return LABELS_SKIP_MESSAGE
    schema = default_schema()
    params = load_params(schema)
    config = default_config()
    pairs = pd.read_parquet(pairs_path)
    if pairs.duplicated(["student_id", "room_id"]).any():
        raise LabelsError(f"{PAIRS_PARQUET}: pares duplicados")
    pairs = pairs.sort_values(["student_id", "room_id"], kind="mergesort").reset_index(drop=True)
    splits = {str(key): str(value) for key, value in pairs.groupby("student_id")["split"].first().items()}
    label_set = simulate_labels(label_inputs(pairs, params), params)

    labels = labels_frame(label_set, splits, params)
    same_order = labels["student_id"].tolist() == pairs["student_id"].tolist() and (
        labels["room_id"].tolist() == pairs["room_id"].tolist()
    )
    if not same_order:
        raise LabelsError("las etiquetas no corresponden uno a uno con los pares de pairs.parquet")
    labels.to_parquet(paths.data_processed / LABELS_PARQUET, index=False)
    hidden_utility_frame(label_set).to_parquet(paths.data_processed / HIDDEN_UTILITY_PARQUET, index=False)
    hidden_weights_frame(label_set).to_parquet(paths.data_processed / HIDDEN_WEIGHTS_PARQUET, index=False)

    baseline = baseline_totals(pairs, schema, config)
    utilities = [item.utility for item in label_set.labels]
    correlation = Correlation(
        pooled=spearman(baseline, utilities),
        per_student=tuple(per_student_spearman(labels["student_id"].tolist(), baseline, utilities)),
        expected=params.spearman_range,
    )
    report_path = paths.reports / LABELS_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = render_report(pairs, labels, label_set, params, schema, config, correlation)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    if not correlation.meets:
        low, high = correlation.expected
        raise LabelsError(
            f"Spearman baseline-utilidad oculta = {correlation.pooled:.3f}, fuera de [{low:g}, {high:g}] "
            f"(ML-04-RK02); ver {report_path.as_posix()}"
        )
    return (
        f"labels: {len(labels)} etiquetas sintéticas de {len(label_set.students)} estudiantes "
        f"(Spearman con el baseline {correlation.pooled:.3f}) -> "
        f"{(paths.data_processed / LABELS_PARQUET).as_posix()}; reporte {report_path.as_posix()}"
    )


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Correlation:
    """Spearman baseline-utilidad: sobre todos los pares (la que se comprueba) y por estudiante."""

    pooled: float
    per_student: tuple[float, ...]
    expected: tuple[float, float]

    @property
    def meets(self) -> bool:
        low, high = self.expected
        return not math.isnan(self.pooled) and low <= self.pooled <= high


def _curve_text(feature: str, shape: str, scale: float | None) -> str:
    if shape == "identity":
        return f"`{feature}`"
    if shape == "one_minus":
        return f"`1 - {feature}`"
    return f"`exp(-{feature} / {scale:g})`"


def _baseline_curve_text(config: BaselineConfig, factor_id: str) -> str:
    factor = config.factor(factor_id)
    points = " · ".join(f"({x:g}, {y:g})" for x, y in factor.curve.points)
    return f"`{factor.feature}` por tramos {points}"


def _relevance_cells(group: pd.DataFrame) -> list[object]:
    total = len(group)
    counts = group["relevance"].value_counts()
    cells: list[object] = [f"{group['student_id'].nunique():,}", f"{total:,}"]
    for grade in RELEVANCES:
        count = int(counts.get(grade, 0))
        cells.append(f"{count:,} ({_pct(count, total)})")
    return cells


def _relevance_table(groups: Sequence[tuple[str, pd.DataFrame]], label: str) -> list[str]:
    header = [label, "Estudiantes", "Pares", *(f"Relevancia {grade}" for grade in RELEVANCES)]
    lines = [_row(header), _row(["---", *(["---:"] * (len(header) - 1))])]
    lines += [_row([name, *_relevance_cells(group)]) for name, group in groups]
    return lines


def _f3(value: float) -> str:
    return f"{value:.3f}"


def render_report(
    pairs: pd.DataFrame,
    labels: pd.DataFrame,
    label_set: LabelSet,
    params: LabelParams,
    schema: FeatureSchema,
    config: BaselineConfig,
    correlation: Correlation,
) -> str:
    """Reporte Markdown determinista de las etiquetas sintéticas."""
    merged = labels.assign(
        university_code=pairs["university_code"].to_numpy(), seed_verified=pairs["seed_verified"].to_numpy()
    )
    curves = params.curves
    base = params.base_weights
    low, high = correlation.expected
    lines = [
        LABELS_TITLE,
        "",
        "> **DATOS SINTÉTICOS — ETIQUETAS SINTÉTICAS.** Estas etiquetas **no son preferencias reales** de ningún",
        "> estudiante: salen de una **utilidad oculta documentada más ruido** (constitución §8.2), calculada sobre",
        "> estudiantes y cuartos sintéticos. Un modelo entrenado con ellas aprende esa función y ese ruido, no el",
        "> comportamiento de personas reales; ninguna métrica calculada con ellas es desempeño real. Hay riesgo de",
        "> **circularidad** (ML-04-RK02): la utilidad usa las mismas features que el modelo base. Se **reemplazan con",
        "> señales reales en ML-07**.",
        "",
        "Generado por `uv run python -m thehouse_ml prepare` (`thehouse_ml.synthetic.labels_stage`). Parámetros:",
        f"`src/thehouse_ml/synthetic/{LABELS_PARAMS_RESOURCE}` versión {params.version}"
        f" (decisión {params.metadata.get('decision', '—')}). Entrada: `data/processed/{PAIRS_PARQUET}`"
        f" (esquema de features {schema.version}).",
        "",
        "## 1. Resumen",
        "",
        f"- Pares etiquetados: **{len(labels):,}** de {len(label_set.students):,} estudiantes (semilla"
        f" {params.seed}); una etiqueta por par de `{PAIRS_PARQUET}`, ninguna fuera de los candidatos.",
        f"- `data/processed/{LABELS_PARQUET}` (fuera de git): "
        + ", ".join(f"`{column}`" for column in LABEL_COLUMNS)
        + ".",
        "- Utilidad oculta y pesos ocultos por estudiante, aparte y fuera de git:"
        f" `data/processed/{HIDDEN_UTILITY_PARQUET}` y `data/processed/{HIDDEN_WEIGHTS_PARQUET}`."
        " No son entradas de ningún modelo.",
        f"- Spearman entre el puntaje del modelo base ({config.model_version}, suma sin redondear de"
        f" `score_features`) y la utilidad oculta: **{_f3(correlation.pooled)}**"
        f" (rango documentado [{low:g}, {high:g}]:" + (" **cumple**)." if correlation.meets else " **NO CUMPLE**)."),
        "",
        "## 2. Utilidad oculta",
        "",
        "Utilidad = Σ peso oculto * curva(feature) + ruido gaussiano. Los pesos base y dos curvas difieren del",
        "modelo base para que las etiquetas no sean una copia de las reglas.",
        "",
        _row(["Factor", f"Peso {config.model_version}", "Peso base oculto", "Curva del modelo base", "Curva oculta"]),
        _row(["---", "---:", "---:", "---", "---"]),
    ]
    for factor in FACTORS:
        curve = curves[factor]
        lines.append(
            _row(
                [
                    f"`{factor}`",
                    f"{config.factor(factor).weight:g}",
                    f"{base[factor]:g}",
                    _baseline_curve_text(config, factor),
                    _curve_text(curve.feature, curve.shape, curve.scale_km),
                ]
            )
        )
    lines += [
        "",
        f"- **Variación por estudiante:** cada peso base * `exp({params.multiplier_sigma:g} · z)`, `z` normal estándar,"
        " renormalizado para sumar 100.",
        f"- **Ruido:** normal de media 0 y desviación {params.noise_sigma:g} puntos por par.",
        f'- **Aleatoriedad:** `random.Random("labels:<seed>:<student_id>")`: {len(FACTORS)} normales para los'
        " multiplicadores y después una por candidato en orden de `room_id`.",
        "- El simulador **no** usa el modelo base ni ningún atributo sensible o de género (constitución §11).",
        "",
        "## 3. Pesos ocultos por estudiante",
        "",
        "Distribución entre estudiantes (percentiles con interpolación lineal); en cada estudiante suman 100.",
        "",
        _row(["Factor", "Peso base", "Mín.", "p25", "Mediana", "p75", "Máx."]),
        _row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for factor in FACTORS:
        values = [item.weights[factor] for item in label_set.students]
        if values:
            p25, median, p75 = quartiles(values)
            cells = [f"{min(values):.1f}", f"{p25:.1f}", f"{median:.1f}", f"{p75:.1f}", f"{max(values):.1f}"]
        else:
            cells = ["—"] * 5
        lines.append(_row([f"`{factor}`", f"{base[factor]:g}", *cells]))
    cumulative: list[Fraction] = []
    for grade in GRADES:
        cumulative.append((cumulative[-1] if cumulative else Fraction(0)) + params.bands[grade])
    lines += [
        "",
        "## 4. Relevancia graduada 0-3",
        "",
        "Dentro de los `n` candidatos de cada estudiante, ordenados por utilidad descendente (empates por `room_id`",
        f"ascendente), la posición `r` recibe 3 si `r ≤ ⌈{float(cumulative[0]):g}·n⌉`, 2 si"
        f" `r ≤ ⌈{float(cumulative[1]):g}·n⌉`, 1 si `r ≤ ⌈{float(cumulative[2]):g}·n⌉` y 0 en otro caso.",
        "Todo estudiante con al menos un candidato tiene al menos un 3. Ejemplos:",
        "",
        _row(["Candidatos `n`", "Relevancia 3", "Relevancia 2", "Relevancia 1", "Relevancia 0"]),
        _row(["---:", "---:", "---:", "---:", "---:"]),
    ]
    for n in POSITION_EXAMPLES:
        grades = relevance_by_position(n, params)
        lines.append(_row([f"{n}", *(f"{grades.count(grade)}" for grade in RELEVANCES)]))
    counts = [item.candidates for item in label_set.students]
    lines += [
        "",
        "Estudiantes por número de candidatos: "
        + ", ".join(
            [
                f"1 → **{sum(1 for n in counts if n == 1):,}**",
                f"2 → **{sum(1 for n in counts if n == 2):,}**",
                f"3-9 → **{sum(1 for n in counts if 3 <= n <= 9):,}**",
                f"≥ 10 → **{sum(1 for n in counts if n >= 10):,}**",
            ]
        )
        + ".",
        "",
        "## 5. Distribución de relevancia",
        "",
        "### 5.1 Total y por partición",
        "",
    ]
    split_groups = [(f"`{split}`", merged[merged["split"] == split]) for split in SPLITS]
    lines += _relevance_table([*split_groups, (TOTAL, merged)], "Partición")
    lines += ["", "### 5.2 Por campus", ""]
    campus_groups = [
        (f"`{code}`" + ("" if bool(group["seed_verified"].iloc[0]) else " (sin verificar)"), group)
        for code in schema.catalog("university_code")
        if len(group := merged[merged["university_code"] == code])
    ]
    lines += _relevance_table(
        [*campus_groups, (TOTAL, merged), (TOTAL_VERIFIED, merged[merged["seed_verified"]])], "Campus"
    )
    per_student = list(correlation.per_student)
    lines += [
        "",
        "## 6. No circularidad: correlación con el modelo base",
        "",
        "Spearman (Pearson de rangos promedio) entre la suma sin redondear del modelo base y la utilidad oculta con",
        "ruido. Positiva: las etiquetas premian lo mismo que las reglas en lo general. Claramente menor que 1: el",
        "orden de las etiquetas no es el del modelo base (ML-04-RK02).",
        "",
        _row(["Medida", "Valor"]),
        _row(["---", "---:"]),
        _row(["Sobre todos los pares (la que se comprueba)", f"**{_f3(correlation.pooled)}**"]),
        _row(["Rango documentado (`check.spearman_range`)", f"[{low:g}, {high:g}]"]),
        _row(
            [
                f"Mediana por estudiante ({len(per_student):,} estudiantes con series no constantes)",
                _f3(statistics.median(per_student)) if per_student else "—",
            ]
        ),
        _row(
            [
                "p25 - p75 por estudiante",
                f"{_f3(quartiles(per_student)[0])} - {_f3(quartiles(per_student)[2])}" if per_student else "—",
            ]
        ),
        "",
        "## 7. Privacidad y limitaciones",
        "",
        "- Solo ids sintéticos (`S…`, `R…`), la partición y la relevancia; sin datos personales (constitución §7) ni",
        "  género, edad u origen (constitución §11).",
        "- Pesos, curvas, ruido y cortes son SUPUESTOS documentados (M-ML-032), no calibrados con conducta real.",
        "- Las métricas de ranking que se calculen con estas etiquetas (T06, ML-05) son sobre **datos sintéticos** y",
        "  se reportan como tales; se reemplazan con señales reales en ML-07.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "HIDDEN_UTILITY_PARQUET",
    "HIDDEN_WEIGHTS_PARQUET",
    "LABELS_PARQUET",
    "LABELS_REPORT",
    "LABELS_SKIP_MESSAGE",
    "LABELS_TITLE",
    "LABEL_COLUMNS",
    "Correlation",
    "LabelsError",
    "baseline_totals",
    "hidden_utility_frame",
    "hidden_weights_frame",
    "label_inputs",
    "labels_frame",
    "per_student_spearman",
    "render_report",
    "run_labels",
    "spearman",
]
