# ML-02 — Datos: fuentes, licencias y preparación — Plan

| Campo | Valor |
| --- | --- |
| ID | ML-02 |
| Documento | `plan.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |

## Enfoque

Primero verificar licencias y registrar fuentes; después descarga con manifiesto; luego preparación, mapeos y reescalado; al final la data card y la entrega de coordenadas al backend.

## F2 — Implementación

- **T01 · Descarga verificable** (R01): `download` obtiene la instantánea definida, guarda su SHA-256 en `data/manifests/` y falla si el hash no coincide en descargas posteriores.
- **T02 · Universidades con coordenadas** (R02): Las 10 universidades del seed tienen latitud, longitud e id OSM verificados a mano; se entregan en un CSV versionado para el seed de NV-11.
- **T03 · Preparación sin datos personales** (R03): `prepare` produce el catálogo de cuartos sin columnas personales (prueba automática de columnas prohibidas) y con coordenadas redondeadas.
- **T04 · Mapeo a catálogos de NV-11** (R04): Cada cuarto tiene `placeType`, `propertyTypes` y servicios del catálogo cerrado; los valores no mapeados se cuentan y se reportan.

## F3 — Calidad, seguridad y datos

- **T05 · Precios y distancias regionales** (R05): Rentas mensuales dentro de los rangos regionales documentados y distancias con la distribución medida; histogramas en la data card.
- **T06 · Data card y licencias** (R06): `reports/data_card.md` con fuentes, licencias, atribuciones, transformaciones, conteos y limitaciones.

## F4 — Evidencia

- **T07 · Pruebas y evidencia** (R07): Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Verificación

Cada fase termina con pruebas en verde, ruff y mypy sin errores y revisión de coherencia y de licencias de datos antes de pasar a la siguiente.
