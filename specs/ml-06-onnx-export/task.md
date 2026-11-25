# ML-06 — Artefacto ONNX y contrato con el backend — Task list

| Campo | Valor |
| --- | --- |
| ID | ML-06 |
| Documento | `task.md` |
| Versión | 1.0.0 |
| Fecha | 2026-11-24 |
| Estado documental | **✅ APROBADA — 1.0.0** |
| Aprobación humana | César Yael López Trujillo — 2026-11-24 (M-ML-044) |
| Estado de ejecución | **🟡 EN EJECUCIÓN — T01, T02, T03, T04, T05 y T06 completas (6/7), 2026-11-24 y 2026-11-25 (M-ML-045, M-ML-046, M-ML-047, M-ML-048, M-ML-049, M-ML-050); T07 ⛔ NO EJECUTAR HASTA «EJECUTA»; ver `evidence.md`** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Tasks totales | 7 |
| Tasks completadas | 6 |

## Reglas de esta lista

1. T01 (M-ML-045) y T02 (M-ML-046) completas (2026-11-24) y T03 (M-ML-047), T04 (M-ML-048), T05 (M-ML-049) y T06 (M-ML-050) completas (2026-11-25) con evidencia en `evidence.md`; T07 abierta. Escribir este documento no completa ninguna.
2. Una task solo se completa con aceptación, evidencia reproducible y revisión.
3. Ninguna task se ejecuta sin «EJECUTA». Las decisiones 1–5 de la spec §10 están APROBADAS (M-ML-044).
4. Un control no ejecutado no se reporta como PASS (en particular, las pruebas que necesitan el modelo oficial y se omiten en CI).
5. Ninguna task reentrena, recalibra ni modifica `logreg.json`, `feature_schema.json` ni ningún params.

## F1 — Diseño y contrato

- [x] **T01 · Dependencias y exportación verificada** — Requisito: ML-06-R01 · Decisiones: 1, 2, 5 · Aceptación: `onnx` y `onnxruntime` en `pyproject.toml` y `uv.lock` (`uv lock --check`, pip-audit limpio, decisión M-ML con versiones, licencias y opset); `export` verifica el SHA-256 de `logreg.json` y construye el grafo de la spec §6.2 en double con sus metadatos; `onnx.checker` sin errores; onnxruntime CPU lo ejecuta sobre los pares de test; `model.onnx` idéntico en dos ejecuciones.

## F2 — Implementación

- [x] **T02 · Paridad Python ↔ ONNX** — Requisito: ML-06-R02 · Decisión: 2 · Aceptación: en los pares de test, la rejilla fuera del `training_range` y los casos dorados, \|Δ\| ≤ 1e-4 en `factor_points` y `raw_total`, 0 diferencias en puntos redondeados y `matchScore`, máximo \|Δ\| reportado; prueba de que entradas, constantes y salidas son double.
- [x] **T03 · Casos dorados de `logreg-1.0.0`** — Requisito: ML-06-R03 · Decisiones: 1, 3 · Aceptación: `tests/fixtures/logreg_golden_cases.json` generado con recuento independiente (entradas de los 81 casos del baseline + recorte ≥ 4, extremos ≥ 2, bordes de redondeo ≥ 6, zona T05-H1 ≥ 3, término fijo ≥ 1), idéntico a la salida del generador, con la regla de margen; paquete y ONNX coinciden con todos dentro de las tolerancias.

## F3 — Calidad, seguridad y datos

- [x] **T04 · Rendimiento** — Requisito: ML-06-R04 · Aceptación: 200 cuartos en onnxruntime CPU, mediana < 20 ms en 100 corridas tras 10 de calentamiento (sesión ya cargada), con p95, máximo, hilos y CPU/SO en la evidencia.
- [x] **T05 · Paquete de entrega** — Requisito: ML-06-R05 · Decisión: 4 · Aceptación: `dist/model-v1.0.0/` con los 7 archivos de la spec §6.6, `sha256sum -c SHA256SUMS` correcto, copias byte a byte, sección «Artefacto ONNX» en `model_card.md`, `/dist/` ignorado con prueba de higiene, mismos SHA-256 en dos ejecuciones y lista de verificación para que el responsable publique `model-v1.0.0`.
- [x] **T06 · Guía de integración** — Requisito: ML-06-R06 · Decisiones: 2, 3, 4 · Aceptación: `docs/backend-integration.md` con contrato de entrada y salida, cálculo de features, orden de pasos (faltantes → incompleto → restricciones → features → ONNX → redondeo → JSON → ranking), redondeo en .NET, casos dorados con `Microsoft.ML.OnnxRuntime`, descarga y verificación del release y desviación T05-H1; prueba de coherencia con el ONNX y el esquema.

## F4 — Evidencia

- [ ] **T07 · Pruebas y evidencia** — Requisito: ML-06-R07 · Aceptación: cobertura ≥ 80 % del código nuevo; ruff, mypy y pip-audit sin errores; `evidence.md` con commit, comandos, resultados, hallazgos, revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`, y verificación del release publicado (o pendiente registrado).

## Matriz task → requisito

| Task | Requisito | Decisiones | Fase | Estado |
| --- | --- | --- | --- | --- |
| T01 | R01 | 1, 2, 5 | F1 | COMPLETA — PASS WITH WARNINGS (M-ML-045; `/dist/` en `.gitignore` y paquete publicable en T05; gitleaks y suite completa en el CI del PR) |
| T02 | R02 | 2 | F2 | COMPLETA — PASS WITH WARNINGS (M-ML-046; casos dorados de logreg en T03; `export` solo reporta y T05 debe marcar no publicable si la paridad falla; regla `-0` → `0` en la guía de T06; errata de la §6.4 en T07) |
| T03 | R03 | 1, 3 | F2 | COMPLETA — PASS WITH WARNINGS (M-ML-047; 109 casos, 1 entrada del baseline reemplazada por la regla de margen; copia como `golden_cases.json` en T05; errata 9 → 6 excluidos de la §6.4 y 2 errores de mypy preexistentes en `tests/` en T07; gitleaks y suite completa en el CI del PR) |
| T04 | R04 | — | F3 | COMPLETA — PASS WITH WARNINGS (M-ML-048; mediana 0.0477 ms con `logreg-1.0.0`; `export` debe medir y reportar el rendimiento y la sección «Artefacto ONNX» de la model card incluirlo en T05; gitleaks y suite completa en el CI del PR) |
| T05 | R05 | 4 | F3 | COMPLETA — PASS WITH WARNINGS (M-ML-049; paquete de 7 archivos, reproducible, no publicable con árbol sucio, paridad o casos dorados en falla; el rendimiento se reporta y no bloquea; la publicación del release `model-v1.0.0` la hace el responsable a mano después del merge, desde `main` limpio, y se registra en T07) |
| T06 | R06 | 2, 3, 4 | F3 | COMPLETA — PASS WITH WARNINGS (M-ML-050; `docs/backend-integration.md` y `tests/unit/test_backend_integration_doc.py` en verde; incidencia del release aceptada como publicación inicial; `baseline_params.json` fijado por commit y SHA-256, en el paquete solo en `model-v1.0.1`; `Property.Price` en la spec de NV-23; suite completa y gitleaks en el CI del PR) |
| T07 | R07 | — | F4 | ABIERTA |
