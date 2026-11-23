# ML-05 — Modelo entrenado y evaluación — Plan

| Campo | Valor |
| --- | --- |
| ID | ML-05 |
| Documento | `plan.md` |
| Versión | 1.0.0 |
| Fecha | 2026-11-23 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | César Yael López Trujillo — 2026-11-23 (M-ML-036) |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |

## Enfoque

Primero las dependencias y la carga de los datos que ya entregó ML-04 (sin generar etiquetas); después los pliegues, el entrenamiento y la comparación contra el baseline con el criterio de promoción; al final ablación, segmentos, calibración y desglose del modelo explicable y la model card. La partición test se toca una sola vez, en T03, con los modelos finales.

Ubicación propuesta (`specs/ARQUITECTURA.md` §4): `src/thehouse_ml/models/train.py` y `train_params.json` (entrenamiento y rejilla), un módulo de reporte de ML-05 en `src/thehouse_ml/models/`, y las métricas de `src/thehouse_ml/evaluation/ranking.py` reutilizadas sin cambios. Etapas: `train` (pliegues, búsqueda, modelos finales en `models/`) y `evaluate` (medición en test y reportes).

## F1 — Diseño y contrato

- **T01 · Dependencias y carga de datos** (R01): `scikit-learn` y `lightgbm` en `pyproject.toml` y `uv.lock` (pip-audit limpio; si hace falta, `[[tool.mypy.overrides]]` para sus importes, como pandas); cargador de `pairs.parquet` + `labels.parquet` con validaciones; exclusión y recuento de estudiantes con < 2 candidatos; prueba obligatoria de que no se leen los archivos ocultos.

## F2 — Implementación

- **T02 · Validación cruzada y entrenamiento reproducible** (R02): `StratifiedGroupKFold` (5, por estudiante, estratificado por campus) sobre train + validación; rejilla de `train_params.json` 1.0.0 para regresión logística y LightGBM (con restricciones monotónicas) solo con los pliegues; modelos finales reentrenados en train + validación y guardados sin pickle; doble corrida con SHA-256 iguales. Decisión M-ML de T02: rejilla y semilla.
- **T03 · Evaluación contra el baseline y decisión de promoción** (R03): métricas por pliegue (baseline, regresión logística, LightGBM); medición única en test con bootstrap pareado contra `per_student_baseline`; condiciones 1–2 de la §6.5 y condición 3 por construcción (pendiente de verificar en T05); `reports/trained_metrics.json` y `reports/trained_report.md`.

## F3 — Calidad, seguridad y datos

- **T04 · Ablación y segmentos** (R04): ablación de los 6 grupos solo con los pliegues; segmentos de test (campus, presupuesto, sin `seed_sin_verificar`); veredicto de la hipótesis del segmento > $3,500.
- **T05 · Calibración 0–100 y desglose** (R05): recorte de cada feature a su `training_range` (misma regla que el backend), calibración afín del logit y desglose por los 6 factores para la regresión logística, con el redondeo de M-ML-029/031; verificación en todos los pares de test (condición 3 de la §6.5); prueba de que el pipeline solo usa transformadores con convertidor en `skl2onnx`. Se ejecuta aunque la regresión logística no se promueva, porque es parte del criterio. Decisión M-ML de T05: método de calibración y formato de `breakdown`.
- **T06 · Model card** (R06): `reports/model_card.md` con «DATOS SINTÉTICOS» y la decisión final de promoción (condiciones 1–3).

## F4 — Evidencia

- **T07 · Pruebas y evidencia** (R07): suite completa con cobertura ≥ 80 % del código nuevo; ruff, mypy y pip-audit; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Verificación

Cada fase termina con pruebas en verde, ruff y mypy sin errores y revisión de coherencia y de licencias antes de pasar a la siguiente. Ninguna task posterior a T03 vuelve a medir en test ni ajusta nada con sus resultados.
