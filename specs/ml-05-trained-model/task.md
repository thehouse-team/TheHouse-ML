# ML-05 — Modelo entrenado y evaluación — Task list

| Campo | Valor |
| --- | --- |
| ID | ML-05 |
| Documento | `task.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Tasks totales | 6 |
| Tasks completadas | 0 |

## Reglas de esta lista

1. Todas las tasks están abiertas; escribir este documento no completa ninguna.
2. Una task solo se completa con aceptación, evidencia reproducible y revisión.
3. Ninguna task se ejecuta sin «EJECUTA».
4. Un control no ejecutado no se reporta como PASS.

## F1 — Diseño y contrato

- [ ] **T01 · Etiquetas documentadas** — Requisito: ML-05-R01 · Aceptación: Generador de etiquetas reproducible con la utilidad oculta descrita en el reporte.

## F2 — Implementación

- [ ] **T02 · Entrenamiento reproducible** — Requisito: ML-05-R02 · Aceptación: `train` produce el mismo modelo y las mismas métricas con la misma semilla y los mismos datos.
- [ ] **T03 · Evaluación contra el modelo base** — Requisito: ML-05-R03 · Aceptación: Tabla con NDCG@10, Precision@5 y MAP (media ± desviación) del modelo base, la regresión logística y LightGBM.

## F3 — Calidad, seguridad y datos

- [ ] **T04 · Ablación y segmentos** — Requisito: ML-05-R04 · Aceptación: Impacto de quitar cada grupo de features y métricas por universidad y rango de presupuesto.
- [ ] **T05 · Model card** — Requisito: ML-05-R05 · Aceptación: `reports/model_card.md` con uso previsto, datos, métricas, limitaciones (datos sintéticos) y decisión de promoción.

## F4 — Evidencia

- [ ] **T06 · Pruebas y evidencia** — Requisito: ML-05-R06 · Aceptación: Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Matriz task → requisito

| Task | Requisito | Fase | Estado |
| --- | --- | --- | --- |
| T01 | R01 | F1 | ABIERTA |
| T02 | R02 | F2 | ABIERTA |
| T03 | R03 | F2 | ABIERTA |
| T04 | R04 | F3 | ABIERTA |
| T05 | R05 | F3 | ABIERTA |
| T06 | R06 | F4 | ABIERTA |
