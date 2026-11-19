# ML-02 — Datos: bases nacionales, calibración regional y licencias — Plan

| Campo | Valor |
| --- | --- |
| ID | ML-02 |
| Documento | `plan.md` |
| Versión | 1.0.1 |
| Fecha | 2026-11-12 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | César Yael López Trujillo — 2026-11-12 (M-ML-012): aprueba ML-02 1.0.1 |
| Estado de ejecución | **🔄 EN EJECUCIÓN — T01–T05, T07 y T08 completas; T06 DESCARTADA (M-ML-019); T09 sin ejecutar** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |

## Enfoque

Primero registrar fuentes y variables con `dictionary_url` y `variables_status` (F1); después descarga con manifiestos y verificación automática de variables, bases nacionales, calibración regional y entorno de campus (F2); la T06 queda descartada: Inside Airbnb no se descarga ni se verifica y el catálogo de cuartos se traslada a ML-03 como sintético (M-ML-019); al final universidades verificadas y data card (F3) y evidencia (F4).

## F1 — Diseño y contrato

- **T01 · Registro de fuentes y licencias** (R01): `data/sources.toml` con URL, edición, fecha, licencia, atribución, `dictionary_url`, `variables_status` y variables registradas; prueba que falla si falta un campo.

## F2 — Implementación

- **T02 · Descarga verificable** (R02): `download` obtiene cada fuente registrada, guarda su SHA-256 en `data/manifests/` y falla si el hash no coincide en descargas posteriores; verifica que cada variable registrada exista en el archivo descargado y la marca `confirmed`.
- **T03 · Bases nacionales ponderadas** (R03): `national_baselines.parquet` con percentiles de renta, proporción de renta y servicios ponderados con el factor de expansión y actualizados con el INPC; pruebas de esquema y de orden de percentiles.
- **T04 · Calibración regional con respaldo** (R04): `regional_calibration.parquet` con los 7 municipios; cada indicador indica si salió del municipio, del estado o del país según la regla de suficiencia; prueba de la regla.
- **T05 · Entorno de campus** (R05): `campus_context.parquet` con conteos del DENUE por categoría SCIAN para las 10 universidades en radios de 1, 2 y 5 km; prueba de consistencia de radios.
- **T06 · DESCARTADA (M-ML-019)** (R06): Inside Airbnb se marca como `discarded` con `status_reference = "M-ML-019"` y sin URL de `listings.csv.gz`; `download` la omite con `descartada: M-ML-019` y el reporte de variables la muestra como descartada sin borrar las otras secciones. No se produce `rooms.parquet`. Justificación: en México solo publica CDMX (no cubre los 7 municipios), mide rentas turísticas por noche (no cuartos por mes), trae datos personales de anfitriones y sus usos ya están cubiertos por ENVI/ENIGH (M-ML-014), Censo 2020 (M-ML-015) y DENUE (M-ML-018).
- **Nota para ML-03:** generar el catálogo de cuartos de forma sintética con semilla fija y distribuciones documentadas, calibrado con las bases de ML-02, sin datos personales, mapeado a NV-11 y etiquetado como sintético (constitution §8).

## F3 — Calidad, seguridad y datos

- **T07 · Universidades verificadas** (R07): `data/universities.csv` con las 10 universidades, id OSM y fecha de verificación; las 3 aproximadas revisadas; diferencias mayores a 200 m reportadas para el backend.
- **T08 · Data card** (R08): `reports/data_card.md` con fuentes, licencias, atribuciones, ponderación, niveles de respaldo, tamaños de muestra, histogramas y limitaciones.

## F4 — Evidencia

- **T09 · Pruebas y evidencia** (R09): Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Verificación

Cada fase termina con pruebas en verde, ruff y mypy sin errores y revisión de coherencia y de licencias de datos antes de pasar a la siguiente.
