# ML-06 — Artefacto ONNX y contrato con el backend — Task list

| Campo | Valor |
| --- | --- |
| ID | ML-06 |
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

## F2 — Implementación

- [ ] **T01 · Exportación verificada** — Requisito: ML-06-R01 · Aceptación: `export` produce `model.onnx` que onnxruntime carga y ejecuta sobre la partición de prueba.
- [ ] **T02 · Paridad** — Requisito: ML-06-R02 · Aceptación: Puntajes de Python y ONNX dentro de la tolerancia en todos los casos dorados.

## F3 — Calidad, seguridad y datos

- [ ] **T03 · Rendimiento** — Requisito: ML-06-R03 · Aceptación: 200 cuartos en menos de 20 ms en CPU (mediana de 100 corridas).
- [ ] **T04 · Paquete y release** — Requisito: ML-06-R04 · Aceptación: Release `model-vX.Y.Z` con los seis archivos y sus hashes, aprobado por el responsable.
- [ ] **T05 · Guía de integración** — Requisito: ML-06-R05 · Aceptación: `docs/backend-integration.md` con el contrato de entrada y salida, el cálculo de features y cómo correr los casos dorados en .NET.

## F4 — Evidencia

- [ ] **T06 · Pruebas y evidencia** — Requisito: ML-06-R06 · Aceptación: Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Matriz task → requisito

| Task | Requisito | Fase | Estado |
| --- | --- | --- | --- |
| T01 | R01 | F2 | ABIERTA |
| T02 | R02 | F2 | ABIERTA |
| T03 | R03 | F3 | ABIERTA |
| T04 | R04 | F3 | ABIERTA |
| T05 | R05 | F3 | ABIERTA |
| T06 | R06 | F4 | ABIERTA |
