# ML-05 — Modelo entrenado y evaluación — Plan

| Campo | Valor |
| --- | --- |
| ID | ML-05 |
| Documento | `plan.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |

## Enfoque

Primero las etiquetas y la partición; después los modelos y la tabla contra el modelo base; al final ablación, segmentos y la model card con la decisión de promoción.

## F1 — Diseño y contrato

- **T01 · Etiquetas documentadas** (R01): Generador de etiquetas reproducible con la utilidad oculta descrita en el reporte.

## F2 — Implementación

- **T02 · Entrenamiento reproducible** (R02): `train` produce el mismo modelo y las mismas métricas con la misma semilla y los mismos datos.
- **T03 · Evaluación contra el modelo base** (R03): Tabla con NDCG@10, Precision@5 y MAP (media ± desviación) del modelo base, la regresión logística y LightGBM.

## F3 — Calidad, seguridad y datos

- **T04 · Ablación y segmentos** (R04): Impacto de quitar cada grupo de features y métricas por universidad y rango de presupuesto.
- **T05 · Model card** (R05): `reports/model_card.md` con uso previsto, datos, métricas, limitaciones (datos sintéticos) y decisión de promoción.

## F4 — Evidencia

- **T06 · Pruebas y evidencia** (R06): Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Verificación

Cada fase termina con pruebas en verde, ruff y mypy sin errores y revisión de coherencia y de licencias de datos antes de pasar a la siguiente.
