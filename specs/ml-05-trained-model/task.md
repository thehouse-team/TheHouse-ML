# ML-05 — Modelo entrenado y evaluación — Task list

| Campo | Valor |
| --- | --- |
| ID | ML-05 |
| Documento | `task.md` |
| Versión | 1.0.0 |
| Fecha | 2026-11-23 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | César Yael López Trujillo — 2026-11-23 (M-ML-036) |
| Estado de ejecución | **🟡 EN EJECUCIÓN — T01 completa (1/7); ver `evidence.md`** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Tasks totales | 7 |
| Tasks completadas | 1 |

## Reglas de esta lista

1. T01 completa (2026-11-23), con evidencia en `evidence.md`; T02–T07 abiertas.
2. Una task solo se completa con aceptación, evidencia reproducible y revisión.
3. Ninguna task se ejecuta sin «EJECUTA».
4. Un control no ejecutado no se reporta como PASS.
5. La partición test solo se usa en T03 (medición única); T04–T06 reutilizan esa medición sin ajustar nada con ella.

## F1 — Diseño y contrato

- [x] **T01 · Dependencias y carga de datos** — Requisito: ML-05-R01 · Aceptación: `scikit-learn` y `lightgbm` en `pyproject.toml` y `uv.lock` con `uv lock --check` y pip-audit limpio; cargador de `pairs.parquet` + `labels.parquet` que valida `synthetic = true`, `labels_params_version` 1.0.0 y las 10 columnas de `feature_order`, y excluye y cuenta estudiantes con < 2 candidatos; prueba obligatoria de que no se leen `labels_hidden_utility.parquet` ni `labels_hidden_weights.parquet`; sin generación de etiquetas.

## F2 — Implementación

- [ ] **T02 · Validación cruzada y entrenamiento reproducible** — Requisito: ML-05-R02 · Aceptación: 5 pliegues agrupados por estudiante y estratificados por campus sobre train + validación, sin estudiantes compartidos; rejilla de `train_params.json` solo con los pliegues; modelos finales en train + validación; LightGBM con las 4 restricciones monotónicas; dos corridas de `train` con SHA-256 iguales; sin pickle.
- [ ] **T03 · Evaluación contra el baseline y decisión de promoción** — Requisito: ML-05-R03 · Aceptación: métricas por pliegue y media ± desviación de baseline, regresión logística y LightGBM; medición única en test con intervalos y diferencia pareada contra `per_student_baseline`; condiciones de la §6.5 por candidato y decisión en `reports/trained_metrics.json` y `reports/trained_report.md`, marcados como sintéticos.

## F3 — Calidad, seguridad y datos

- [ ] **T04 · Ablación y segmentos** — Requisito: ML-05-R04 · Aceptación: ΔNDCG@10 por cada uno de los 6 grupos de features (solo pliegues); métricas y diferencias pareadas por campus, sin `seed_sin_verificar` y por rango de presupuesto en test; hipótesis del segmento > $3,500 confirmada, refutada o no concluyente según la spec §6.7.
- [ ] **T05 · Calibración 0–100 y desglose** — Requisito: ML-05-R05 · Aceptación: cada feature recortada a su `training_range` antes de puntuar (misma regla en Python y en el backend); puntaje 0–100 de la regresión logística y desglose por los 6 factores que suma el total con las tolerancias de M-ML-029/031 en todos los pares de test; ranking idéntico al del logit para entradas dentro del rango; pruebas de propiedades; pipeline solo con transformadores con convertidor en `skl2onnx`.
- [ ] **T06 · Model card** — Requisito: ML-05-R06 · Aceptación: `reports/model_card.md` con «DATOS SINTÉTICOS», uso previsto, datos, modelos, métricas contra el baseline, ablación, segmentos, desglose de ejemplo, limitaciones, decisión de promoción y trabajo futuro.

## F4 — Evidencia

- [ ] **T07 · Pruebas y evidencia** — Requisito: ML-05-R07 · Aceptación: pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff, mypy y pip-audit sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Matriz task → requisito

| Task | Requisito | Fase | Depende de | Estado |
| --- | --- | --- | --- | --- |
| T01 | R01 | F1 | — | COMPLETA — PASS WITH WARNINGS (M-ML-037; T01-H1 se confirma en el CI) |
| T02 | R02 | F2 | T01 | ABIERTA |
| T03 | R03 | F2 | T02 | ABIERTA |
| T04 | R04 | F3 | T02, T03 | ABIERTA |
| T05 | R05 | F3 | T02, T03 | ABIERTA |
| T06 | R06 | F3 | T03, T04, T05 | ABIERTA |
| T07 | R07 | F4 | T01–T06 | ABIERTA |
