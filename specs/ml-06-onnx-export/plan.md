# ML-06 — Artefacto ONNX y contrato con el backend — Plan

| Campo | Valor |
| --- | --- |
| ID | ML-06 |
| Documento | `plan.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |

## Enfoque

Primero la exportación y la paridad; después rendimiento; al final el paquete, la guía y el release con aprobación.

## F2 — Implementación

- **T01 · Exportación verificada** (R01): `export` produce `model.onnx` que onnxruntime carga y ejecuta sobre la partición de prueba.
- **T02 · Paridad** (R02): Puntajes de Python y ONNX dentro de la tolerancia en todos los casos dorados.

## F3 — Calidad, seguridad y datos

- **T03 · Rendimiento** (R03): 200 cuartos en menos de 20 ms en CPU (mediana de 100 corridas).
- **T04 · Paquete y release** (R04): Release `model-vX.Y.Z` con los seis archivos y sus hashes, aprobado por el responsable.
- **T05 · Guía de integración** (R05): `docs/backend-integration.md` con el contrato de entrada y salida, el cálculo de features y cómo correr los casos dorados en .NET.

## F4 — Evidencia

- **T06 · Pruebas y evidencia** (R06): Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Verificación

Cada fase termina con pruebas en verde, ruff y mypy sin errores y revisión de coherencia y de licencias de datos antes de pasar a la siguiente.
