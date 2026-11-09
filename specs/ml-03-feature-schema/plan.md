# ML-03 — Esquema de features y estudiantes sintéticos — Plan

| Campo | Valor |
| --- | --- |
| ID | ML-03 |
| Documento | `plan.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |

## Enfoque

Primero el esquema y su prueba contra NV-11; después la función de features y las restricciones; al final el generador y la partición.

## F1 — Diseño y contrato

- **T01 · Esquema de features** (R01): `feature_schema.json` con nombres, tipos, rangos, catálogos y versión; validado contra NV-11 por una prueba.

## F2 — Implementación

- **T02 · Cálculo de features de referencia** (R02): Función pura que, dado estudiante y cuarto, produce las features del par; pruebas con casos límite (presupuesto exacto, distancia 0, sin servicios).
- **T03 · Restricciones duras** (R03): Los pares que violan presupuesto, `livingWith` o mascotas se excluyen; pruebas por cada regla.
- **T04 · Estudiantes sintéticos** (R04): Generador reproducible (misma semilla → mismos perfiles) que respeta rangos y catálogos; reporte de distribuciones.

## F3 — Calidad, seguridad y datos

- **T05 · Conjunto de pares** (R05): Pares candidatos por estudiante con partición por estudiante para entrenamiento, validación y prueba, sin fuga.

## F4 — Evidencia

- **T06 · Pruebas y evidencia** (R06): Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Verificación

Cada fase termina con pruebas en verde, ruff y mypy sin errores y revisión de coherencia y de licencias de datos antes de pasar a la siguiente.
