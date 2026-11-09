# ML-04 — Modelo base por reglas — Task list

| Campo | Valor |
| --- | --- |
| ID | ML-04 |
| Documento | `task.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Tasks totales | 5 |
| Tasks completadas | 0 |

## Reglas de esta lista

1. Todas las tasks están abiertas; escribir este documento no completa ninguna.
2. Una task solo se completa con aceptación, evidencia reproducible y revisión.
3. Ninguna task se ejecuta sin «EJECUTA».
4. Un control no ejecutado no se reporta como PASS.

## F1 — Diseño y contrato

- [ ] **T02 · Configuración versionada** — Requisito: ML-04-R02 · Aceptación: Pesos y curvas en un archivo con versión `baseline-x.y.z`; el reporte muestra la versión usada.

## F2 — Implementación

- [ ] **T01 · Puntaje y desglose** — Requisito: ML-04-R01 · Aceptación: Puntaje 0–100 con desglose por factor que suma el total; pruebas de propiedades (monotonía en precio y distancia).
- [ ] **T03 · Casos dorados** — Requisito: ML-04-R03 · Aceptación: Al menos 50 casos (entradas, features, desglose, total) exportados para la prueba de paridad del backend.

## F3 — Calidad, seguridad y datos

- [ ] **T04 · Reporte del modelo base** — Requisito: ML-04-R04 · Aceptación: Distribución de puntajes, ejemplos explicados y métricas de ranking sobre la partición de prueba de ML-03 (marcadas como sintéticas).

## F4 — Evidencia

- [ ] **T05 · Pruebas y evidencia** — Requisito: ML-04-R05 · Aceptación: Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Matriz task → requisito

| Task | Requisito | Fase | Estado |
| --- | --- | --- | --- |
| T01 | R01 | F2 | ABIERTA |
| T02 | R02 | F1 | ABIERTA |
| T03 | R03 | F2 | ABIERTA |
| T04 | R04 | F3 | ABIERTA |
| T05 | R05 | F4 | ABIERTA |
