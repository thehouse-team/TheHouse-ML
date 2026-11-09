# ML-07 — Señales reales de TheHouse — Plan

| Campo | Valor |
| --- | --- |
| ID | ML-07 |
| Documento | `plan.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |

## Enfoque

Primero la spec de exportación en el backend; después ingesta y evaluación real; al final reentrenamiento si mejora.

## F2 — Implementación

- **T01 · Ingesta anonimizada** (R01): Lector de la exportación con validación de esquema y prueba de ausencia de datos personales.
- **T02 · Evaluación real** (R02): Métricas del modelo base y del vigente sobre datos reales con partición temporal.

## F3 — Calidad, seguridad y datos

- **T03 · Reentrenamiento** (R03): Nuevo modelo solo si mejora sobre datos reales; nueva versión por ML-06.

## F4 — Evidencia

- **T04 · Pruebas y evidencia** (R04): Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Verificación

Cada fase termina con pruebas en verde, ruff y mypy sin errores y revisión de coherencia y de licencias de datos antes de pasar a la siguiente.
