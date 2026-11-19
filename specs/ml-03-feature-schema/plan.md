# ML-03 — Esquema de features, estudiantes sintéticos y catálogo sintético de cuartos — Plan

| Campo | Valor |
| --- | --- |
| ID | ML-03 |
| Documento | `plan.md` |
| Versión | 1.0.0 |
| Fecha | 2026-11-07 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | César Yael López Trujillo — 2026-11-19 |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |

## Enfoque

Primero el esquema y su prueba contra NV-11; después la función de features y las restricciones; a mitad, el generador de estudiantes y el catálogo sintético de cuartos; al final el conjunto de pares y evidencia.

## F1 — Diseño y contrato

- **T01 · Esquema de features** (R01): `feature_schema.json` con nombres, tipos, rangos, catálogos y versión; validado contra NV-11 por una prueba.

## F2 — Implementación

- **T02 · Cálculo de features de referencia** (R02): Función pura que, dado estudiante y cuarto, produce las features del par; pruebas con casos límite (presupuesto exacto, distancia 0, sin servicios).
- **T03 · Restricciones duras** (R03): Los pares que violan presupuesto, `livingWith` o mascotas se excluyen; pruebas por cada regla.
- **T04 · Estudiantes sintéticos** (R04): Generador reproducible (misma semilla → mismos perfiles) que respeta rangos y catálogos; reporte de distribuciones.
- **T05 · Catálogo sintético de cuartos** (R05): Generador reproducible (misma semilla → mismos cuartos): ~3,000 cuartos entre los 10 campuses del seed, todos a ≤30 km del campus; renta calibrada con `regional_calibration.parquet` y la forma de `rent_histogram.parquet`; servicios por municipio del Censo (agua, drenaje, electricidad, gas, internet→wifi) y priors documentados (laundry, kitchen, furnished, parking, security, cleaning, hot_water); `placeType`, `propertyTypes`, mascotas y `livingWith` de cuarto (el cuarto declara `women`, `men`, `any`) con distribuciones documentadas como supuestos; salida `data/processed/rooms.parquet` y reporte `reports/rooms_catalog.md`.

## F3 — Calidad, seguridad y datos

- **T06 · Conjunto de pares** (R06): Pares candidatos por estudiante con partición por estudiante para entrenamiento, validación y prueba, sin fuga.

## F4 — Evidencia

- **T07 · Pruebas y evidencia** (R07): Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Verificación

Cada fase termina con pruebas en verde, ruff y mypy sin errores y revisión de coherencia y de licencias de datos antes de pasar a la siguiente.
