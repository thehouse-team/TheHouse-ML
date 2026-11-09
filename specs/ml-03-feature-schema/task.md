# ML-03 — Esquema de features y estudiantes sintéticos — Task list

| Campo | Valor |
| --- | --- |
| ID | ML-03 |
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

- [ ] **T01 · Esquema de features** — Requisito: ML-03-R01 · Aceptación: `feature_schema.json` con nombres, tipos, rangos, catálogos y versión; validado contra NV-11 por una prueba.

## F2 — Implementación

- [ ] **T02 · Cálculo de features de referencia** — Requisito: ML-03-R02 · Aceptación: Función pura que, dado estudiante y cuarto, produce las features del par; pruebas con casos límite (presupuesto exacto, distancia 0, sin servicios).
- [ ] **T03 · Restricciones duras** — Requisito: ML-03-R03 · Aceptación: Los pares que violan presupuesto, `livingWith` o mascotas se excluyen; pruebas por cada regla.
- [ ] **T04 · Estudiantes sintéticos** — Requisito: ML-03-R04 · Aceptación: Generador reproducible (misma semilla → mismos perfiles) que respeta rangos y catálogos; reporte de distribuciones.

## F3 — Calidad, seguridad y datos

- [ ] **T05 · Conjunto de pares** — Requisito: ML-03-R05 · Aceptación: Pares candidatos por estudiante con partición por estudiante para entrenamiento, validación y prueba, sin fuga.

## F4 — Evidencia

- [ ] **T06 · Pruebas y evidencia** — Requisito: ML-03-R06 · Aceptación: Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Matriz task → requisito

| Task | Requisito | Fase | Estado |
| --- | --- | --- | --- |
| T01 | R01 | F1 | ABIERTA |
| T02 | R02 | F2 | ABIERTA |
| T03 | R03 | F2 | ABIERTA |
| T04 | R04 | F2 | ABIERTA |
| T05 | R05 | F3 | ABIERTA |
| T06 | R06 | F4 | ABIERTA |
