# ML-04 — Modelo base por reglas — Plan

| Campo | Valor |
| --- | --- |
| ID | ML-04 |
| Documento | `plan.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |

## Enfoque

Primero la configuración y la función de puntaje con pruebas de propiedades; después los casos dorados; al final el reporte.

## F1 — Diseño y contrato

- **T02 · Configuración versionada** (R02): Pesos y curvas en un archivo con versión `baseline-x.y.z`; el reporte muestra la versión usada.

## F2 — Implementación

- **T01 · Puntaje y desglose** (R01): Puntaje 0–100 con desglose por factor que suma el total; pruebas de propiedades (monotonía en precio y distancia).
- **T03 · Casos dorados** (R03): Al menos 50 casos (entradas, features, desglose, total) exportados para la prueba de paridad del backend.

## F3 — Calidad, seguridad y datos

- **T04 · Reporte del modelo base** (R04): Distribución de puntajes, ejemplos explicados y métricas de ranking sobre la partición de prueba de ML-03 (marcadas como sintéticas).

## F4 — Evidencia

- **T05 · Pruebas y evidencia** (R05): Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Verificación

Cada fase termina con pruebas en verde, ruff y mypy sin errores y revisión de coherencia y de licencias de datos antes de pasar a la siguiente.
