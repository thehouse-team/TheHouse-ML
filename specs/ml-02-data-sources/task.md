# ML-02 — Datos: bases nacionales, calibración regional y licencias — Task list

| Campo | Valor |
| --- | --- |
| ID | ML-02 |
| Documento | `task.md` |
| Versión | 1.0.0 |
| Fecha | 2026-11-10 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | Victor Manuel Jose Jose — 2026-11-10 («pues si te apruebo que revises si el censo trae lo de las rentas, pues la region si esta bien, pues si en el municipio creo que hay varias rentas las suficientes y si se usa airbnb usa la licencia y lo demas tambien en pocas palabras si a todo»): ML-02 1.0.0 y sus decisiones de la §10 |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Tasks totales | 9 |
| Tasks completadas | 0 |

## Reglas de esta lista

1. Todas las tasks están abiertas; escribir este documento no completa ninguna.
2. Una task solo se completa con aceptación, evidencia reproducible y revisión.
3. Ninguna task se ejecuta sin «EJECUTA».
4. Un control no ejecutado no se reporta como PASS.

## F1 — Diseño y contrato

- [ ] **T01 · Registro de fuentes y licencias** — Requisito: ML-02-R01 · Aceptación: `data/sources.toml` con URL, edición, fecha, licencia, atribución y variables confirmadas en el diccionario oficial; prueba que falla si falta un campo.

## F2 — Implementación

- [ ] **T02 · Descarga verificable** — Requisito: ML-02-R02 · Aceptación: `download` obtiene cada fuente registrada, guarda su SHA-256 en `data/manifests/` y falla si el hash no coincide en descargas posteriores.
- [ ] **T03 · Bases nacionales ponderadas** — Requisito: ML-02-R03 · Aceptación: `national_baselines.parquet` con percentiles de renta, proporción de renta y servicios ponderados con el factor de expansión y actualizados con el INPC; pruebas de esquema y de orden de percentiles.
- [ ] **T04 · Calibración regional con respaldo** — Requisito: ML-02-R04 · Aceptación: `regional_calibration.parquet` con los 7 municipios; cada indicador indica si salió del municipio, del estado o del país según la regla de suficiencia; prueba de la regla.
- [ ] **T05 · Entorno de campus** — Requisito: ML-02-R05 · Aceptación: `campus_context.parquet` con conteos del DENUE por categoría SCIAN para las 10 universidades en radios de 1, 2 y 5 km; prueba de consistencia de radios.
- [ ] **T06 · Catálogo de cuartos sin datos personales** — Requisito: ML-02-R06 · Aceptación: `rooms.parquet` sin columnas prohibidas (prueba automática), con coordenadas redondeadas, servicios y tipos mapeados a NV-11 y renta mensual calibrada; los valores no mapeados se cuentan y se reportan.

## F3 — Calidad, seguridad y datos

- [ ] **T07 · Universidades verificadas** — Requisito: ML-02-R07 · Aceptación: `data/universities.csv` con las 10 universidades, id OSM y fecha de verificación; las 3 aproximadas revisadas; diferencias mayores a 200 m reportadas para el backend.
- [ ] **T08 · Data card** — Requisito: ML-02-R08 · Aceptación: `reports/data_card.md` con fuentes, licencias, atribuciones, ponderación, niveles de respaldo, tamaños de muestra, histogramas y limitaciones.

## F4 — Evidencia

- [ ] **T09 · Pruebas y evidencia** — Requisito: ML-02-R09 · Aceptación: Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Matriz task → requisito

| Task | Requisito | Fase | Estado |
| --- | --- | --- | --- |
| T01 | R01 | F1 | ABIERTA |
| T02 | R02 | F2 | ABIERTA |
| T03 | R03 | F2 | ABIERTA |
| T04 | R04 | F2 | ABIERTA |
| T05 | R05 | F2 | ABIERTA |
| T06 | R06 | F2 | ABIERTA |
| T07 | R07 | F3 | ABIERTA |
| T08 | R08 | F3 | ABIERTA |
| T09 | R09 | F4 | ABIERTA |
