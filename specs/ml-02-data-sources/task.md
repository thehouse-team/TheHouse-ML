# ML-02 — Datos: bases nacionales, calibración regional y licencias — Task list

| Campo | Valor |
| --- | --- |
| ID | ML-02 |
| Documento | `task.md` |
| Versión | 1.0.1 |
| Fecha | 2026-11-12 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | César Yael López Trujillo — 2026-11-12 (M-ML-012): aprueba ML-02 1.0.1 |
| Estado de ejecución | **🔄 EN EJECUCIÓN — T01–T05, T07 y T08 completas; T06 DESCARTADA (M-ML-019); T09 sin ejecutar** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Tasks totales | 9 |
| Tasks completadas | 7 |
| Tasks descartadas | 1 (T06, M-ML-019) |

## Reglas de esta lista

1. Escribir este documento no completa ninguna task; solo lo hace la evidencia reproducible.
2. Una task solo se completa con aceptación, evidencia reproducible y revisión.
3. Ninguna task se ejecuta sin «EJECUTA».
4. Un control no ejecutado no se reporta como PASS.
5. `[-]` marca una task **DESCARTADA** por decisión aprobada: no se ejecuta ni se cuenta como completada.

## F1 — Diseño y contrato

- [x] **T01 · Registro de fuentes y licencias** — Requisito: ML-02-R01 · Aceptación: `data/sources.toml` con URL, edición, fecha, licencia, atribución, `dictionary_url`, `variables_status` y variables registradas; prueba que falla si falta un campo.

## F2 — Implementación

- [x] **T02 · Descarga verificable** — Requisito: ML-02-R02 · Aceptación: `download` obtiene cada fuente registrada, guarda su SHA-256 en `data/manifests/` y falla si el hash no coincide en descargas posteriores; verifica que cada variable registrada exista en el archivo descargado y la marca `confirmed`.
- [x] **T03 · Bases nacionales ponderadas** — Requisito: ML-02-R03 · Aceptación: `national_baselines.parquet` con percentiles de renta, proporción de renta y servicios ponderados con el factor de expansión y actualizados con el INPC; pruebas de esquema y de orden de percentiles.
- [x] **T04 · Calibración regional con respaldo** — Requisito: ML-02-R04 · Aceptación: `regional_calibration.parquet` con los 7 municipios; cada indicador indica si salió del municipio, del estado o del país según la regla de suficiencia; prueba de la regla.
- [x] **T05 · Entorno de campus** — Requisito: ML-02-R05 · Aceptación: `campus_context.parquet` con conteos del DENUE por categoría SCIAN para las 10 universidades en radios de 1, 2 y 5 km; prueba de consistencia de radios. Salida reportada en `reports/denue_campus.md` (M-ML-018).
- [-] **T06 · DESCARTADA (M-ML-019)** — Requisito: ML-02-R06 (ajustado) · Justificación: Inside Airbnb solo publica CDMX en México (no cubre los 7 municipios), mide rentas turísticas por noche y no cuartos por mes, y sus archivos traen datos personales de anfitriones; rentas, servicios y entorno ya están cubiertos por ENVI 2020/ENIGH 2022 (M-ML-014), Censo 2020 (M-ML-015) y DENUE (M-ML-018). Aceptación: la fuente queda como `variables_status = "discarded"` con `status_reference = "M-ML-019"` y sin URL de `listings.csv.gz`; `download` la omite con `descartada: M-ML-019` sin descargarla ni verificarla; el reporte de variables la muestra como descartada conservando las otras secciones. No se produce `rooms.parquet`.

## F3 — Calidad, seguridad y datos

- [x] **T07 · Universidades verificadas** — Requisito: ML-02-R07 · Aceptación: `data/universities.csv` con las 10 universidades, id OSM y fecha de verificación; las 3 aproximadas revisadas; diferencias mayores a 200 m reportadas para el backend. **Ampliada el 2026-11-18 con M-ML-017:** verificación humana en `data/seed/universities_manual.csv` y columnas `final_lat`, `final_lon`, `final_status` y `final_source`.
- [x] **T08 · Data card** — Requisito: ML-02-R08 · Aceptación: `reports/data_card.md` generado automáticamente por `prepare` desde `data/sources.toml`, `data/manifests/`, `data/universities.csv` y los Parquet de `data/processed/`; 11 secciones con fuentes, licencias, atribuciones, ponderación, niveles de respaldo, tamaños de muestra, histogramas y limitaciones.

## F4 — Evidencia

- [ ] **T09 · Pruebas y evidencia** — Requisito: ML-02-R09 · Aceptación: Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Nota para ML-03

Como consecuencia de M-ML-019, ML-03 generará el catálogo de cuartos de forma **sintética** (semilla fija y distribuciones documentadas), calibrado con `national_baselines.parquet` y `regional_calibration.parquet` (renta), el Censo 2020 (servicios) y `campus_context.parquet` (entorno), sin datos personales, mapeado a los catálogos de NV-11 y **etiquetado como sintético** en todo reporte (constitution §8). No se usará ningún dato de Inside Airbnb.

## Matriz task → requisito

| Task | Requisito | Fase | Estado |
| --- | --- | --- | --- |
| T01 | R01 | F1 | COMPLETA |
| T02 | R02 | F2 | COMPLETA |
| T03 | R03 | F2 | COMPLETA |
| T04 | R04 | F2 | COMPLETA |
| T05 | R05 | F2 | COMPLETA |
| T06 | R06 | F2 | DESCARTADA (M-ML-019) |
| T07 | R07 | F3 | COMPLETA |
| T08 | R08 | F3 | COMPLETA |
| T09 | R09 | F4 | ABIERTA |
