# ML-06 — Artefacto ONNX y contrato con el backend — Plan

| Campo | Valor |
| --- | --- |
| ID | ML-06 |
| Documento | `plan.md` |
| Versión | 1.0.0 |
| Fecha | 2026-11-24 |
| Estado documental | **✅ APROBADA — 1.0.0** |
| Aprobación humana | César Yael López Trujillo — 2026-11-24 (M-ML-044) |
| Estado de ejecución | **✅ COMPLETA — T01–T07 (7/7), 2026-11-24 a 2026-11-25; cierre M-ML-051, ver `evidence.md`** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |

## Enfoque

Primero las dependencias y el grafo ONNX de `logreg-1.0.0` construido con `onnx` desde `logreg.json` (opción a de la spec §6.2); después la paridad contra `score_features_logreg` y los casos dorados propios del modelo con recuento independiente; al final el rendimiento, el paquete en `dist/`, la guía para NV-23 y la evidencia. No se reentrena ni se recalibra nada; la partición test de ML-05 solo se usa para comparar puntajes Python ↔ ONNX, no para medir métricas.

Ubicación propuesta (`specs/ARQUITECTURA.md` §4): `src/thehouse_ml/export/` con `onnx_graph.py` (construcción del grafo y metadatos), `parity.py` (comparación Python ↔ onnxruntime), `benchmark.py` (rendimiento), `package.py` (paquete y `SHA256SUMS`) y la etapa `export` (`stages.py` solo para encadenarla); el generador de casos dorados en `tests/` (`build_logreg_golden_cases.py`, reutilizando `tests/golden_recount.py`); la guía en `docs/backend-integration.md`. Perfil `ci` (`THEHOUSE_ML_TRAIN_PROFILE=ci`): misma etapa con `models/ci/logreg.json` y salida en `dist/ci/` (no publicable).

## F1 — Diseño y contrato

- **T01 · Dependencias y exportación verificada** (R01): `onnx` y `onnxruntime` en `pyproject.toml` y `uv.lock` (pip-audit limpio; override de mypy si alguna no publica tipos) con su decisión M-ML (versiones, licencias, opset; registra `onnx` como dependencia fuera del texto literal de la §5); verificación del SHA-256 de `logreg.json`; grafo en double (`Max`/`Min`, `Sub`, `Div`, `Mul`, `MatMul` por factor, calibración, `Add` encadenado) con `metadata_props`; `onnx.checker`; carga y ejecución en onnxruntime CPU sobre los pares de test; `model.onnx` idéntico en dos ejecuciones.

## F2 — Implementación

- **T02 · Paridad Python ↔ ONNX** (R02): comparación con `score_features_logreg` en los 30,755 pares de test, en una rejilla fuera del `training_range` y en los casos dorados; tolerancia 1e-4 sin redondear y 0 en redondeados y `matchScore`; prueba de que todo el grafo es double.
- **T03 · Casos dorados de `logreg-1.0.0`** (R03): generador con recuento independiente (aritmética exacta, sin importar `thehouse_ml`); las entradas de los 81 casos del baseline más recorte, extremos, bordes de redondeo, zona T05-H1 y término fijo; regla de margen; prueba de identidad del archivo; omisión con motivo sin el modelo oficial.

## F3 — Calidad, seguridad y datos

- **T04 · Rendimiento** (R04): 200 cuartos, 10 corridas de calentamiento y 100 medidas en onnxruntime CPU; mediana < 20 ms; p95, máximo, hilos y CPU/SO en la evidencia; humo informativo en CI.
- **T05 · Paquete de entrega** (R05): `dist/model-v1.0.0/` con `model.onnx`, `feature_schema.json`, `golden_cases.json`, `baseline_golden_cases.json`, `model_card.md` (con sección «Artefacto ONNX»), `metrics.json` y `SHA256SUMS`; `/dist/` en `.gitignore` y en la prueba de higiene; reproducible; condiciones de «publicable»; lista de verificación para que el responsable publique el release `model-v1.0.0`.
- **T06 · Guía de integración** (R06): `docs/backend-integration.md` (contrato, features, orden de pasos, redondeo en .NET, casos dorados con `Microsoft.ML.OnnxRuntime`, descarga y verificación del release, T05-H1) con una prueba de coherencia contra el ONNX y el esquema.

## F4 — Evidencia

- **T07 · Pruebas y evidencia** (R07): suite completa con cobertura ≥ 80 % del código nuevo; ruff, mypy y pip-audit; `evidence.md` con commit, comandos, resultados, hallazgos y coherencia; verificación del release si el responsable ya lo publicó (o pendiente registrado); actualización de `specs/README.md` y `specs/ARQUITECTURA.md` (lo de ML-06).

## Verificación

Cada fase termina con pruebas en verde, ruff y mypy sin errores y revisión de coherencia, seguridad y licencias antes de pasar a la siguiente. Ninguna task cambia `logreg.json`, `feature_schema.json` ni los params de ML-03/ML-04/ML-05; si el SHA-256 de `logreg.json` no coincide, la task se detiene como BLOCKED.
