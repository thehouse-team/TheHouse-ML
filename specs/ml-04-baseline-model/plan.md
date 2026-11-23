# ML-04 — Modelo base por reglas — Plan

| Campo | Valor |
| --- | --- |
| ID | ML-04 |
| Documento | `plan.md` |
| Versión | 1.0.0 |
| Fecha | 2026-11-20 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | César Yael López Trujillo — 2026-11-20 (M-ML-029) |
| Estado de ejecución | **✅ COMPLETA — T01–T07 (7/7), 2026-11-20 a 2026-11-23; cierre M-ML-035, ver `evidence.md`** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Spec | `spec.md` 1.0.0 |

## Enfoque

Primero el contrato: la configuración `baseline-1.0.0` y el esquema 2.0.0 con las features que las curvas necesitan. Después la función de puntaje con pruebas de propiedades, las etiquetas sintéticas (independientes del puntaje, con pesos ocultos distintos) y los casos dorados calculados sin la función de puntaje. Al final, el reporte que fija la referencia de ML-05 y la evidencia. Sin dependencias nuevas: biblioteca estándar, pandas y pyarrow.

## F1 — Diseño y contrato

- **T01 · Configuración versionada** (R01): archivo `baseline-1.0.0` con factores, pesos, curvas y redondeo de la spec §6.2–6.3 (ubicación propuesta: `src/thehouse_ml/models/baseline_params.json`, empaquetado como `feature_schema.json`); cargador que valida que los pesos suman 100 y que cada feature existe en el esquema. Depende de las decisiones 2 y 3.
- **T02 · Esquema 2.0.0 y features nuevas** (R02): versión mayor (constitución §18) con `distance_ratio`, `price_below_min` y `extra_services_ratio` al final de `feature_order`, `changelog` 2.0.0, `metadata.versioning` (agregar, quitar o reordenar features es un cambio mayor), cálculo en `compute_pair_features`, pruebas del esquema (las 7 features anteriores sin cambios, rangos y `training_range`), ampliación de `tests/fixtures/pair_feature_cases.json` con valores calculados a mano y regeneración de `pairs.parquet` con `prepare` (dos corridas con el mismo SHA-256). Depende de la decisión 1. Toca código de ML-03: su «EJECUTA» debe autorizar `src/thehouse_ml/features/`, `tests/` y `data/processed/`. El backend debe actualizar su spec por el cambio mayor (constitución §2.4; aviso del responsable).

## F2 — Implementación

- **T03 · Puntaje y desglose** (R03): función pura en `src/thehouse_ml/models/` que recibe estudiante y cuarto (o las features de un par que ya pasó HC01–HC03) y devuelve `matchScore` (nombre y tipo del `ExplainMatchRequest` de `TheHouse-GenAI`) y `breakdown` (campo nuevo propuesto, spec §6.4); pares excluidos sin puntaje. Pruebas de propiedades: monotonía en precio y en distancia, total entre 0 y 100 y desglose coherente (tolerancias de la §6.3). Depende de T01, T02 y las decisiones 2 y 3.
- **T04 · Simulador de etiquetas sintéticas** (R04): `src/thehouse_ml/synthetic/labels_params.json` 1.0.0 y simulador puro (spec §6.5); la etapa `prepare` escribe `data/processed/labels.parquet` y `reports/labels_synthetic.md`. Pruebas: reproducibilidad byte por byte, pesos ocultos distintos de los del modelo base, relevancia 0–3 con al menos un 3 por estudiante con 2 o más candidatos y advertencia de datos sintéticos en el reporte. Depende de T02 y la decisión 4; los valores exactos se aprueban con su M-ML.
- **T05 · Casos dorados** (R05): al menos 50 casos (entradas, 10 features, desglose y total) en un archivo en git (propuesta: `tests/fixtures/baseline_golden_cases.json`), con valores calculados a mano o con un recuento independiente en las pruebas (aritmética exacta con `fractions.Fraction`, sin llamar a la función de T03). Cubre los límites: renta igual a `budget.min` y por debajo, `budget.min` nulo, `distance_ratio` 0.5, 1 y > 1, sin servicios pedidos, los 11 servicios pedidos, `propertyTypes` nulo y sumas que terminan en .5. Depende de T03.

## F3 — Calidad, seguridad y datos

- **T06 · Reporte del modelo base** (R06): `reports/baseline_report.md` y `reports/baseline_metrics.json` (spec §6.6): distribución de puntajes, ejemplos explicados y NDCG@10 (principal), NDCG@5, Precision@5 y MAP en la partición test, con estudiantes de 2 o más candidatos; intervalos por bootstrap por estudiante (semilla y número de remuestreos documentados); en total, por campus, sin las 3 `seed_sin_verificar` y por rango de presupuesto; todo marcado como sintético. Depende de T03, T04 y la decisión 5; el umbral de relevancia binaria, los parámetros del bootstrap y los rangos de presupuesto se aprueban con su M-ML.

## F4 — Evidencia

- **T07 · Pruebas y evidencia** (R07): pruebas unitarias e integración con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`.

## Verificación

Cada fase termina con pruebas en verde, ruff y mypy sin errores y revisión de coherencia y de licencias de datos antes de pasar a la siguiente. Ningún control no ejecutado se reporta como PASS.
