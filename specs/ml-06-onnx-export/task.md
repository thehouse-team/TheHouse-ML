# ML-06 — Artefacto ONNX y contrato con el backend — Task list

| Campo | Valor |
| --- | --- |
| ID | ML-06 |
| Documento | `task.md` |
| Versión | 1.0.0 |
| Fecha | 2026-11-24 |
| Estado documental | **✅ APROBADA — 1.0.0** |
| Aprobación humana | César Yael López Trujillo — 2026-11-24 (M-ML-044) |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Tasks totales | 7 |
| Tasks completadas | 0 |

## Reglas de esta lista

1. Todas las tasks están abiertas; escribir este documento no completa ninguna.
2. Una task solo se completa con aceptación, evidencia reproducible y revisión.
3. Ninguna task se ejecuta sin «EJECUTA». Las decisiones 1–5 de la spec §10 están APROBADAS (M-ML-044).
4. Un control no ejecutado no se reporta como PASS (en particular, las pruebas que necesitan el modelo oficial y se omiten en CI).
5. Ninguna task reentrena, recalibra ni modifica `logreg.json`, `feature_schema.json` ni ningún params.

## F1 — Diseño y contrato

- [ ] **T01 · Dependencias y exportación verificada** — Requisito: ML-06-R01 · Decisiones: 1, 2, 5 · Aceptación: `onnx` y `onnxruntime` en `pyproject.toml` y `uv.lock` (`uv lock --check`, pip-audit limpio, decisión M-ML con versiones, licencias y opset); `export` verifica el SHA-256 de `logreg.json` y construye el grafo de la spec §6.2 en double con sus metadatos; `onnx.checker` sin errores; onnxruntime CPU lo ejecuta sobre los pares de test; `model.onnx` idéntico en dos ejecuciones.

## F2 — Implementación

- [ ] **T02 · Paridad Python ↔ ONNX** — Requisito: ML-06-R02 · Decisión: 2 · Aceptación: en los pares de test, la rejilla fuera del `training_range` y los casos dorados, \|Δ\| ≤ 1e-4 en `factor_points` y `raw_total`, 0 diferencias en puntos redondeados y `matchScore`, máximo \|Δ\| reportado; prueba de que entradas, constantes y salidas son double.
- [ ] **T03 · Casos dorados de `logreg-1.0.0`** — Requisito: ML-06-R03 · Decisiones: 1, 3 · Aceptación: `tests/fixtures/logreg_golden_cases.json` generado con recuento independiente (entradas de los 81 casos del baseline + recorte ≥ 4, extremos ≥ 2, bordes de redondeo ≥ 6, zona T05-H1 ≥ 3, término fijo ≥ 1), idéntico a la salida del generador, con la regla de margen; paquete y ONNX coinciden con todos dentro de las tolerancias.

## F3 — Calidad, seguridad y datos

- [ ] **T04 · Rendimiento** — Requisito: ML-06-R04 · Aceptación: 200 cuartos en onnxruntime CPU, mediana < 20 ms en 100 corridas tras 10 de calentamiento (sesión ya cargada), con p95, máximo, hilos y CPU/SO en la evidencia.
- [ ] **T05 · Paquete de entrega** — Requisito: ML-06-R05 · Decisión: 4 · Aceptación: `dist/model-v1.0.0/` con los 7 archivos de la spec §6.6, `sha256sum -c SHA256SUMS` correcto, copias byte a byte, sección «Artefacto ONNX» en `model_card.md`, `/dist/` ignorado con prueba de higiene, mismos SHA-256 en dos ejecuciones y lista de verificación para que el responsable publique `model-v1.0.0`.
- [ ] **T06 · Guía de integración** — Requisito: ML-06-R06 · Decisiones: 2, 3, 4 · Aceptación: `docs/backend-integration.md` con contrato de entrada y salida, cálculo de features, orden de pasos (faltantes → incompleto → restricciones → features → ONNX → redondeo → JSON → ranking), redondeo en .NET, casos dorados con `Microsoft.ML.OnnxRuntime`, descarga y verificación del release y desviación T05-H1; prueba de coherencia con el ONNX y el esquema.

## F4 — Evidencia

- [ ] **T07 · Pruebas y evidencia** — Requisito: ML-06-R07 · Aceptación: cobertura ≥ 80 % del código nuevo; ruff, mypy y pip-audit sin errores; `evidence.md` con commit, comandos, resultados, hallazgos, revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`, y verificación del release publicado (o pendiente registrado).

## Matriz task → requisito

| Task | Requisito | Decisiones | Fase | Estado |
| --- | --- | --- | --- | --- |
| T01 | R01 | 1, 2, 5 | F1 | ABIERTA |
| T02 | R02 | 2 | F2 | ABIERTA |
| T03 | R03 | 1, 3 | F2 | ABIERTA |
| T04 | R04 | — | F3 | ABIERTA |
| T05 | R05 | 4 | F3 | ABIERTA |
| T06 | R06 | 2, 3, 4 | F3 | ABIERTA |
| T07 | R07 | — | F4 | ABIERTA |
