# ML-03 — Esquema de features, estudiantes sintéticos y catálogo sintético de cuartos — Task list

| Campo | Valor |
| --- | --- |
| ID | ML-03 |
| Documento | `task.md` |
| Versión | 1.0.0 |
| Fecha | 2026-11-07 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | César Yael López Trujillo — 2026-11-19 |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Tasks totales | 7 |
| Tasks completadas | 1 |

## Reglas de esta lista

1. Todas las tasks están abiertas; escribir este documento no completa ninguna.
2. Una task solo se completa con aceptación, evidencia reproducible y revisión.
3. Ninguna task se ejecuta sin «EJECUTA».
4. Un control no ejecutado no se reporta como PASS.

## F1 — Diseño y contrato

- [x] **T01 · Esquema de features** — Requisito: ML-03-R01 · Aceptación: `feature_schema.json` con nombres, tipos, rangos, catálogos y versión; validado contra NV-11 por una prueba.

## F2 — Implementación

- [ ] **T02 · Cálculo de features de referencia** — Requisito: ML-03-R02 · Aceptación: Función pura que, dado estudiante y cuarto, produce las features del par; pruebas con casos límite (presupuesto exacto, distancia 0, sin servicios).
- [ ] **T03 · Restricciones duras** — Requisito: ML-03-R03 · Aceptación: Los pares que violan presupuesto, `livingWith` o mascotas se excluyen; pruebas por cada regla.
- [ ] **T04 · Estudiantes sintéticos** — Requisito: ML-03-R04 · Aceptación: Generador reproducible (misma semilla → mismos perfiles) que respeta rangos y catálogos; reporte de distribuciones; presupuesto calibrado con la renta regional (`regional_calibration.parquet`): prueba de que la mediana de `budget.max` por universidad es coherente con la renta p50 regional.
- [ ] **T05 · Catálogo sintético de cuartos** — Requisito: ML-03-R05 · Aceptación: Generador reproducible (misma semilla → mismos cuartos) que produce ~3,000 cuartos entre los 10 campuses del seed, todos a ≤30 km del campus; rangos y catálogos de NV-11; renta coherente con la calibración por municipio — prueba que compara la mediana generada contra la calibrada; reporte de distribuciones del catálogo (servicios, `placeType`, `propertyTypes`, mascotas, `livingWith`, distancia a campus) etiquetado como sintético.

## F3 — Calidad, seguridad y datos

- [ ] **T06 · Conjunto de pares** — Requisito: ML-03-R06 · Aceptación: Pares candidatos por estudiante con partición por estudiante para entrenamiento, validación y prueba, sin fuga; prueba de que la mayoría de los estudiantes tiene al menos un cuarto que pasa las restricciones duras (≥ 90 % de estudiantes con al menos un cuarto compatible; necesario el catálogo de cuartos T05).

## F4 — Evidencia

- [ ] **T07 · Pruebas y evidencia** — Requisito: ML-03-R07 · Aceptación: Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Matriz task → requisito

| Task | Requisito | Fase | Estado |
| --- | --- | --- | --- |
| T01 | R01 | F1 | COMPLETA (M-ML-022; `evidence.md`) |
| T02 | R02 | F2 | ABIERTA |
| T03 | R03 | F2 | ABIERTA |
| T04 | R04 | F2 | ABIERTA |
| T05 | R05 | F2 | ABIERTA |
| T06 | R06 | F3 | ABIERTA |
| T07 | R07 | F4 | ABIERTA |
