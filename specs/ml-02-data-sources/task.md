# ML-02 — Datos: fuentes, licencias y preparación — Task list

| Campo | Valor |
| --- | --- |
| ID | ML-02 |
| Documento | `task.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Tasks totales | 7 |
| Tasks completadas | 0 |

## Reglas de esta lista

1. Todas las tasks están abiertas; escribir este documento no completa ninguna.
2. Una task solo se completa con aceptación, evidencia reproducible y revisión.
3. Ninguna task se ejecuta sin «EJECUTA».
4. Un control no ejecutado no se reporta como PASS.

## F2 — Implementación

- [ ] **T01 · Descarga verificable** — Requisito: ML-02-R01 · Aceptación: `download` obtiene la instantánea definida, guarda su SHA-256 en `data/manifests/` y falla si el hash no coincide en descargas posteriores.
- [ ] **T02 · Universidades con coordenadas** — Requisito: ML-02-R02 · Aceptación: Las 10 universidades del seed tienen latitud, longitud e id OSM verificados a mano; se entregan en un CSV versionado para el seed de NV-11.
- [ ] **T03 · Preparación sin datos personales** — Requisito: ML-02-R03 · Aceptación: `prepare` produce el catálogo de cuartos sin columnas personales (prueba automática de columnas prohibidas) y con coordenadas redondeadas.
- [ ] **T04 · Mapeo a catálogos de NV-11** — Requisito: ML-02-R04 · Aceptación: Cada cuarto tiene `placeType`, `propertyTypes` y servicios del catálogo cerrado; los valores no mapeados se cuentan y se reportan.

## F3 — Calidad, seguridad y datos

- [ ] **T05 · Precios y distancias regionales** — Requisito: ML-02-R05 · Aceptación: Rentas mensuales dentro de los rangos regionales documentados y distancias con la distribución medida; histogramas en la data card.
- [ ] **T06 · Data card y licencias** — Requisito: ML-02-R06 · Aceptación: `reports/data_card.md` con fuentes, licencias, atribuciones, transformaciones, conteos y limitaciones.

## F4 — Evidencia

- [ ] **T07 · Pruebas y evidencia** — Requisito: ML-02-R07 · Aceptación: Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Matriz task → requisito

| Task | Requisito | Fase | Estado |
| --- | --- | --- | --- |
| T01 | R01 | F2 | ABIERTA |
| T02 | R02 | F2 | ABIERTA |
| T03 | R03 | F2 | ABIERTA |
| T04 | R04 | F2 | ABIERTA |
| T05 | R05 | F3 | ABIERTA |
| T06 | R06 | F3 | ABIERTA |
| T07 | R07 | F4 | ABIERTA |
