# ML-07 — Señales reales de TheHouse — Task list

| Campo | Valor |
| --- | --- |
| ID | ML-07 |
| Documento | `task.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Tasks totales | 4 |
| Tasks completadas | 0 |

## Reglas de esta lista

1. Todas las tasks están abiertas; escribir este documento no completa ninguna.
2. Una task solo se completa con aceptación, evidencia reproducible y revisión.
3. Ninguna task se ejecuta sin «EJECUTA».
4. Un control no ejecutado no se reporta como PASS.

## F2 — Implementación

- [ ] **T01 · Ingesta anonimizada** — Requisito: ML-07-R01 · Aceptación: Lector de la exportación con validación de esquema y prueba de ausencia de datos personales.
- [ ] **T02 · Evaluación real** — Requisito: ML-07-R02 · Aceptación: Métricas del modelo base y del vigente sobre datos reales con partición temporal.

## F3 — Calidad, seguridad y datos

- [ ] **T03 · Reentrenamiento** — Requisito: ML-07-R03 · Aceptación: Nuevo modelo solo si mejora sobre datos reales; nueva versión por ML-06.

## F4 — Evidencia

- [ ] **T04 · Pruebas y evidencia** — Requisito: ML-07-R04 · Aceptación: Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Matriz task → requisito

| Task | Requisito | Fase | Estado |
| --- | --- | --- | --- |
| T01 | R01 | F2 | ABIERTA |
| T02 | R02 | F2 | ABIERTA |
| T03 | R03 | F3 | ABIERTA |
| T04 | R04 | F4 | ABIERTA |
