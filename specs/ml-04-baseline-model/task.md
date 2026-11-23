# ML-04 — Modelo base por reglas — Task list

| Campo | Valor |
| --- | --- |
| ID | ML-04 |
| Documento | `task.md` |
| Versión | 1.0.0 |
| Fecha | 2026-11-20 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | César Yael López Trujillo — 2026-11-20 (M-ML-029) |
| Estado de ejecución | **🟡 EN EJECUCIÓN — T01, T02, T03, T04 y T05 completas (5/7); ver `evidence.md`** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Tasks totales | 7 |
| Tasks completadas | 5 |

## Reglas de esta lista

1. T01, T02, T03 y T04 completas (2026-11-20) y T05 completa (2026-11-23), con evidencia en `evidence.md`; T06 y T07 abiertas.
2. Una task solo se completa con aceptación, evidencia reproducible y revisión.
3. Ninguna task se ejecuta sin «EJECUTA».
4. Un control no ejecutado no se reporta como PASS.
5. Las decisiones 1–5 de la spec §10 están APROBADO (M-ML-029). Los valores SUPUESTO de la spec §6.5 quedaron aprobados con M-ML-032 (T04); T06 no se cierra hasta que se aprueben, con su M-ML, los de la §6.6.

## F1 — Diseño y contrato

- [x] **T01 · Configuración versionada** — Requisito: ML-04-R01 · Decisiones: 2, 3 · Aceptación: factores, pesos (total 100), curvas y redondeo de la spec §6.2–6.3 en un archivo con versión `baseline-1.0.0`, validado al cargarlo (pesos suman 100, cada feature existe en el esquema); la salida y el reporte muestran la versión.
- [x] **T02 · Esquema 2.0.0 y features nuevas** — Requisito: ML-04-R02 · Decisión: 1 · Aceptación: `feature_schema.json` **2.0.0** (versión mayor) con `distance_ratio`, `price_below_min` y `extra_services_ratio` al final de `feature_order`, en `changelog` y con `metadata.versioning` actualizado; `compute_pair_features` las calcula; las 7 features anteriores no cambian; `pairs.parquet` regenerado (dos corridas, mismo SHA-256); `pair_feature_cases.json` ampliado con valores calculados a mano.

## F2 — Implementación

- [x] **T03 · Puntaje y desglose** — Requisito: ML-04-R03 · Decisiones: 2, 3 · Aceptación: `matchScore` 0–100 (nombre de `TheHouse-GenAI`) con `breakdown` por factor (campo nuevo propuesto) en el formato de la spec §6.4; pares excluidos sin puntaje; redondeo de la §6.3; pruebas de propiedades: monotonía en precio y en distancia, total entre 0 y 100 y desglose coherente.
- [x] **T04 · Simulador de etiquetas sintéticas** — Requisito: ML-04-R04 · Decisión: 4 · Aceptación: semilla fija y `labels_params.json` 1.0.0; pesos ocultos distintos de los del modelo base más ruido; relevancia 0–3 dentro de los candidatos de cada estudiante; `data/processed/labels.parquet` fuera de git, reproducible y con `synthetic = true`; reporte con la advertencia de que no son preferencias reales y se reemplazan en ML-07.
- [x] **T05 · Casos dorados** — Requisito: ML-04-R05 · Decisiones: 2, 3 · Aceptación: al menos 50 casos (entradas, features, desglose y total) calculados a mano o con un recuento independiente, con los casos límite del plan, exportados para la prueba de paridad del backend.

## F3 — Calidad, seguridad y datos

- [ ] **T06 · Reporte del modelo base** — Requisito: ML-04-R06 · Decisión: 5 · Aceptación: distribución de puntajes, ejemplos explicados con desglose y NDCG@10 (principal), NDCG@5, Precision@5 y MAP en la partición test, con estudiantes de 2 o más candidatos (cuántos quedan fuera); intervalos por bootstrap por estudiante (semilla y número de remuestreos documentados); en total, por campus, sin las 3 `seed_sin_verificar` y por rango de presupuesto; todo marcado como sintético; referencia de ML-05.

## F4 — Evidencia

- [ ] **T07 · Pruebas y evidencia** — Requisito: ML-04-R07 · Aceptación: pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Matriz task → requisito

| Task | Requisito | Fase | Depende de | Estado |
| --- | --- | --- | --- | --- |
| T01 | R01 | F1 | Decisiones 2 y 3 | COMPLETA — PASS (M-ML-030) |
| T02 | R02 | F1 | Decisión 1 | COMPLETA — PASS (M-ML-030) |
| T03 | R03 | F2 | T01, T02 | COMPLETA — PASS (M-ML-031) |
| T04 | R04 | F2 | T02, decisión 4 | COMPLETA — PASS (M-ML-032) |
| T05 | R05 | F2 | T03 | COMPLETA — PASS (M-ML-033) |
| T06 | R06 | F3 | T03, T04, decisión 5 | ABIERTA |
| T07 | R07 | F4 | T01–T06 | ABIERTA |
