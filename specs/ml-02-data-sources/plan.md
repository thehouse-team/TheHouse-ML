# ML-02 — Datos: bases nacionales, calibración regional y licencias — Plan

| Campo | Valor |
| --- | --- |
| ID | ML-02 |
| Documento | `plan.md` |
| Versión | 1.0.1 |
| Fecha | 2026-11-12 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | César Yael López Trujillo — 2026-11-12 (M-ML-012): aprueba ML-02 1.0.1 |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |

## Enfoque

Primero registrar fuentes y variables con `dictionary_url` y `variables_status` (F1); después descarga con manifiestos y verificación automática de variables, bases nacionales, calibración regional, entorno de campus y catálogo de cuartos (F2); al final universidades verificadas y data card (F3) y evidencia (F4).

## F1 — Diseño y contrato

- **T01 · Registro de fuentes y licencias** (R01): `data/sources.toml` con URL, edición, fecha, licencia, atribución, `dictionary_url`, `variables_status` y variables registradas; prueba que falla si falta un campo.

## F2 — Implementación

- **T02 · Descarga verificable** (R02): `download` obtiene cada fuente registrada, guarda su SHA-256 en `data/manifests/` y falla si el hash no coincide en descargas posteriores; verifica que cada variable registrada exista en el archivo descargado y la marca `confirmed`.
- **T03 · Bases nacionales ponderadas** (R03): `national_baselines.parquet` con percentiles de renta, proporción de renta y servicios ponderados con el factor de expansión y actualizados con el INPC; pruebas de esquema y de orden de percentiles.
- **T04 · Calibración regional con respaldo** (R04): `regional_calibration.parquet` con los 7 municipios; cada indicador indica si salió del municipio, del estado o del país según la regla de suficiencia; prueba de la regla.
- **T05 · Entorno de campus** (R05): `campus_context.parquet` con conteos del DENUE por categoría SCIAN para las 10 universidades en radios de 1, 2 y 5 km; prueba de consistencia de radios.
- **T06 · Catálogo de cuartos sin datos personales** (R06): `rooms.parquet` sin columnas prohibidas (prueba automática), con coordenadas redondeadas, servicios y tipos mapeados a NV-11 y renta mensual calibrada; los valores no mapeados se cuentan y se reportan.

## F3 — Calidad, seguridad y datos

- **T07 · Universidades verificadas** (R07): `data/universities.csv` con las 10 universidades, id OSM y fecha de verificación; las 3 aproximadas revisadas; diferencias mayores a 200 m reportadas para el backend.
- **T08 · Data card** (R08): `reports/data_card.md` con fuentes, licencias, atribuciones, ponderación, niveles de respaldo, tamaños de muestra, histogramas y limitaciones.

## F4 — Evidencia

- **T09 · Pruebas y evidencia** (R09): Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Verificación

Cada fase termina con pruebas en verde, ruff y mypy sin errores y revisión de coherencia y de licencias de datos antes de pasar a la siguiente.
