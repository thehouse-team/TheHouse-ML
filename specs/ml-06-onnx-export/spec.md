# ML-06 — Artefacto ONNX y contrato con el backend

| Campo | Valor |
| --- | --- |
| ID | ML-06 |
| Documento | `spec.md` |
| Versión | 1.0.0 |
| Fecha | 2026-11-24 |
| Estado documental | **✅ APROBADA — 1.0.0** |
| Aprobación humana | César Yael López Trujillo — 2026-11-24 (M-ML-044) |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Plan | `plan.md` 1.0.0 |
| Tasks | `task.md` 1.0.0 |
| Precondición | ML-05 completa ✅ (M-ML-043) con `logreg-1.0.0` promovido (M-ML-041) |
| Relación con el backend | NV-23 carga `model.onnx` con `Microsoft.ML.OnnxRuntime` detrás de `ICompatibilityModel`, calcula las features, aplica las reglas previas y el redondeo, y corre los casos dorados en .NET |

Spec aprobada (1.0.0, M-ML-044). La aprobación no autoriza código, descargas de datos, dependencias ni cambios en GitHub: cada task requiere «EJECUTA» con versión, tasks y rutas. La versión 1.0.0 corrige el borrador 1.0.0-borrador (2026-11-07): estaba pensado para LightGBM, pero el modelo promovido es `logreg-1.0.0` (M-ML-041), guardado como JSON (`thehouse-ml/logreg-json/1`) y no como un objeto de scikit-learn; no definía qué calcula el ONNX y qué calcula el backend; sus casos dorados eran los del baseline; y la distribución del artefacto estaba PENDIENTE.

## 1. Contexto

ML-05 entrega (M-ML-043) el modelo promovido **`logreg-1.0.0`**: una regresión logística aditiva guardada en `models/logreg.json` (fuera de git, sin pickle; SHA-256 `341ea3ec705c2f877b36d1b931eafe2dea4630523560fe8764ce59e0a70b27fa`, verificado contra `models/train_manifest.json`). `LogRegModel.from_dict` la reconstruye y reproduce bit a bit `Pipeline.decision_function` (M-ML-038 decisión 7). El puntaje 0-100 y el desglose están fijados en M-ML-041:

1. **Recorte:** cada una de las 10 features se recorta a su `training_range` (o `range`) de `feature_schema.json` 2.0.0, con los límites guardados en `logreg.json` (`clip.lower`/`clip.upper`), **antes** del escalado.
2. **Escalado** `StandardScaler` (media y escala de `logreg.json`) y **contribución lineal** de cada feature (`coef`).
3. **Suma por factor** en los 6 factores del baseline (presupuesto, distancia, servicios esenciales, tipo de lugar, tipo de inmueble, extras; M-ML-040 decisión 1). `pets_compatible` no pertenece a ningún factor: su término (0) va al término fijo con el intercepto.
4. **Calibración afín por factor:** `puntos_k = 100 · (C_k − min_k) / S`, con `maxPoints` 9.6 / 72.6 / 6.7 / 4.9 / 1.7 / 4.5 (Σ = 100).
5. **Redondeo** de M-ML-029/031: round6 y 1 decimal mitad hacia arriba por factor; total `floor(round6(Σ) + 0.5)`; en .NET `MidpointRounding.AwayFromZero`.
6. **Formato** del baseline: `modelVersion` `logreg-1.0.0`, `featureSchemaVersion` 2.0.0, `matchScore` entero 0-100 y `breakdown` con los 6 factores, `maxPoints` variables y lista `features` en `budget` y `distance` (`value` sin recortar).

El backend no corre Python. El modelo debe llegar como un archivo portable, verificable y versionado (constitución §2.2 y §13), junto con lo necesario para que NV-23 calcule las mismas features, aplique las mismas reglas y compruebe que obtiene los mismos puntajes.

## 2. Objetivo

Exportar `logreg-1.0.0` a ONNX con salidas que permitan al backend reconstruir el `matchScore` y el `breakdown` exactos; demostrar la paridad Python ↔ onnxruntime con casos dorados propios del modelo; medir el presupuesto de inferencia; y entregar un paquete verificable con una guía de integración para NV-23.

## 3. Precondiciones

1. ML-05 completa ✅ (M-ML-043) con `logreg-1.0.0` promovido ✅ (M-ML-041).
2. Aprobación de esta spec y de las decisiones de la §10 ✅ (M-ML-044).
3. «EJECUTA» con versión, tasks y rutas permitidas. T01 agrega dependencias: su «EJECUTA» debe autorizar `pyproject.toml` y `uv.lock`.
4. `models/logreg.json` presente con el SHA-256 de la §1 (si no coincide, la exportación falla; no se reentrena).
5. La publicación del Release de GitHub es una acción humana del responsable (AGENTS.md: «Publicar un artefacto de modelo»); el agente no publica releases.

## 4. Alcance

1. Dependencias `onnx` y `onnxruntime` en `pyproject.toml` y `uv.lock`, con pip-audit limpio (§10, decisión 5).
2. Construcción directa del grafo ONNX de `logreg-1.0.0` desde `logreg.json` (§6.2) y su verificación con onnxruntime en CPU.
3. Paridad Python ↔ onnxruntime en todos los pares de test y en una rejilla fuera del `training_range`.
4. Casos dorados propios de `logreg-1.0.0` con recuento independiente.
5. Prueba de rendimiento: 200 cuartos en < 20 ms (constitución §13.5).
6. Paquete de entrega en `dist/` (fuera de git) generado por la etapa `export`.
7. Guía de integración `docs/backend-integration.md` para NV-23.

## 5. Fuera de alcance

- Código del backend (NV-23) y cambios en NV-11/NV-12.
- **LightGBM** (`models/lightgbm_lambdarank.txt`): referencia del potencial en ML-05, no se exporta en v1 (M-ML-036 decisión 5). Queda como trabajo futuro si se resuelve un desglose aditivo exportable.
- **Exportar `baseline-1.0.0` a ONNX:** el baseline es el respaldo y NV-23 lo implementa en C# a partir de `baseline_params.json` y se valida con sus **81 casos dorados** (`baseline_golden_cases.json`, M-ML-033), que van en el paquete.
- Publicar el Release de GitHub (lo hace el responsable a mano, §6.6) y cualquier cambio de configuración de GitHub (repositorio, ruleset, releases inmutables, secretos).
- Reentrenar, recalibrar o corregir `logreg-1.0.0` (la desviación T05-H1 se conserva; la corrección es `logreg-1.1.0` con datos de ML-07).
- Modificar `feature_schema.json`, `logreg.json`, `train_params.json`, `baseline_params.json` o cualquier params de ML-03/ML-04/ML-05.
- Servir el modelo como API (constitución §2.1).

## 6. Diseño

### 6.1 Reparto de responsabilidades (backend ↔ ONNX)

El ONNX **solo puntúa pares que ya pasaron** todas las reglas previas; no valida, no excluye, no redondea ni ordena.

| Paso | Quién | Regla |
| --- | --- | --- |
| 1. Faltantes | Backend (NV-23) | Resolver los campos opcionales nulos con las reglas del esquema 2.0.0 (M-ML-022: p. ej., `maxDistanceKm` nulo → 30, `transport` nulo → `public`) |
| 2. Perfil incompleto | Backend | Si falta algún campo del quick setup (`budget.max`, `livingWith`, `placeType`), respuesta `incomplete = true` con `missingFields` en ese orden (M-ML-033 decisión 7); se detecta **antes** que las restricciones |
| 3. Restricciones duras | Backend | HC01–HC03 (presupuesto, `livingWith`, mascotas; M-ML-024); un par excluido responde `excluded = true` con `violatedConstraints` y **no** se puntúa (constitución §11.1) |
| 4. Features | Backend | Las 10 features de `feature_schema.json` 2.0.0, en el orden de `feature_order`, con sus fórmulas (haversine, factores de transporte, cocientes); validar el vector (números finitos, dentro del `range`, enteros donde aplique; M-ML-031 decisión 4) |
| 5. Inferencia | **ONNX** | Recorte → escalado → contribuciones → suma por factor → calibración afín (§6.2); devuelve puntos sin redondear de los 6 factores y `raw_total` |
| 6. Redondeo | Backend | round6 y 1 decimal mitad hacia arriba por factor; `matchScore = floor(round6(raw_total) + 0.5)`; `MidpointRounding.AwayFromZero` (nunca `ToEven`); `maxPoints` redondeado con la misma regla |
| 7. JSON | Backend | `modelVersion`, `featureSchemaVersion`, `matchScore`, `breakdown` (6 factores en el orden de ML-04; `feature`/`value` o lista `features` con `value` sin recortar) |
| 8. Ranking | Backend | Por `raw_total` descendente y `room_id` ascendente (M-ML-031 decisión 7) |

### 6.2 Grafo ONNX (decisión 2)

**Opción elegida: (a) construir el grafo directamente con la biblioteca `onnx`** desde los parámetros de `logreg.json`, sin pasar por scikit-learn.

| Opción | A favor | En contra |
| --- | --- | --- |
| **(a) Grafo directo con `onnx`** (recomendada) | Reproduce operación por operación la fórmula de M-ML-041; el desglose por factor sale del grafo; control total de tipos (double), nombres y metadatos; sin dependencia extra | Hay que escribir y probar el grafo (pocas operaciones) |
| (b) Reconstruir el `Pipeline` de scikit-learn desde el JSON y convertirlo con `skl2onnx` | Usa el convertidor nombrado en la constitución §5 | Hay que fijar a mano atributos ajustados (`mean_`, `scale_`, `coef_`) de objetos no entrenados; el convertidor devuelve etiqueta/probabilidad, no el desglose, así que el recorte (sin transformador en scikit-learn, T02-H2) y la calibración por factor igual se agregan a mano; tipos float32 por omisión; dependencia extra |

**Nodos (todo en `tensor(double)`)**, sobre la entrada `features` de forma `[N, 10]` en el orden de `feature_order`:

1. **Recorte por columna:** `Max(features, lower)` y después `Min(·, upper)` con constantes `lower`/`upper` de forma `[10]` (`clip.lower`/`clip.upper` de `logreg.json`). Equivale a `np.clip` por columna; el operador `Clip` de ONNX solo acepta límites escalares, por eso «el nodo `Clip`» de M-ML-043 se implementa con `Max`/`Min`.
2. **Escalado:** `Sub(mean)` y `Div(scale)` con constantes `[10]`.
3. **Contribuciones lineales:** `Mul(coef)` con constante `[10]` (contribución `t_j` de cada feature).
4. **Suma por factor:** `MatMul` con una matriz de asignación `[10, 6]` de ceros y unos (la columna de `pets_compatible` es toda ceros) → `C` de forma `[N, 6]`.
5. **Calibración afín:** `Sub(min_k)` y `Mul(100 / S)` con constantes `[6]` → salida **`factor_points`** `[N, 6]`, en el orden budget, distance, essential_services, place_type, property_type, extras.
6. **Total:** `raw_total = ((((p₁ + p₂) + p₃) + p₄) + p₅) + p₆` con nodos `Add` encadenados de izquierda a derecha (el mismo orden que `raw_total` en Python, M-ML-031 decisión 3) → salida **`raw_total`** `[N]`.

**Salidas del contrato:** `factor_points` y `raw_total`, sin redondear. No se exporta el logit, la probabilidad ni el término fijo (no forman parte del contrato, M-ML-041 decisión 4).

**Metadatos** (`metadata_props`, legibles en .NET con `ModelMetadata.CustomMetadataMap`): `modelVersion` = `logreg-1.0.0`, `featureSchemaVersion` = 2.0.0, `featureOrder`, `factorOrder`, `maxPointsRaw` (6 valores sin redondear), regla de redondeo, SHA-256 de `logreg.json`, commit de git (T02-H8) y `synthetic` = true. **Opset** propuesto 17 (todas las operaciones existen desde el 13), fijado en la decisión M-ML de T01 según la versión de onnxruntime de `uv.lock`.

**Exactitud:** las constantes se copian de `logreg.json` y de `build_breakdown_model` en binary64 sin conversión a float32 (riesgo RK01). La referencia de Python es `score_features_logreg` de `trained_breakdown.py` (`to_dict(include_raw=True)`: `rawPoints`, `rawTotal`). La desviación **T05-H1** (coeficiente negativo de `price_below_min`) se conserva tal cual.

**Determinismo:** dos ejecuciones de `export` con el mismo commit y el mismo `logreg.json` producen un `model.onnx` idéntico byte a byte (sin marcas de tiempo; `producer_name`/`producer_version` fijos).

### 6.3 Paridad Python ↔ onnxruntime

- **Conjuntos:** (1) los casos dorados de §6.4; (2) todos los pares de test de ML-05 (30,755, medición ya hecha en ML-05: solo se comparan puntajes, no se recalculan métricas ni se ajusta nada); (3) una rejilla determinista de vectores fuera del `training_range` (por debajo y por encima de cada límite, una feature a la vez y todas a la vez).
- **Tolerancia:** |Δ| ≤ **1e-4** en cada `factor_points` y en `raw_total` sin redondear; **0** diferencias en los puntos redondeados y en `matchScore` (aplicando la regla de §6.1 paso 6 a las salidas del ONNX). Se reporta además el máximo |Δ| observado (se espera del orden de 1e-12 con double).
- **Python de referencia:** `score_features_logreg` del paquete; la paridad del recorte se comprueba también contra `np.clip` con los límites de `logreg.json`.

### 6.4 Casos dorados de `logreg-1.0.0`

- **Archivo** `tests/fixtures/logreg_golden_cases.json` (en git), escrito por un generador de pruebas con **recuento independiente** (como M-ML-033): no importa `thehouse_ml`; reutiliza el recuento de features de `tests/golden_recount.py`; lee como datos `logreg.json` (recorte, media, escala, coeficientes) y `feature_schema.json`; calcula recorte, escalado, contribuciones, `min_k`/`max_k`, calibración y redondeo con aritmética exacta (`fractions.Fraction` sobre los binary64 del JSON). La prueba exige que el archivo versionado sea **idéntico** a la salida del generador. Se empaqueta como `golden_cases.json`.
- **Casos:** (1) las **mismas entradas que los 81 casos del baseline** (incluidos los 9 excluidos y los 5 incompletos, ahora con `modelVersion` `logreg-1.0.0`); (2) **recorte**: entradas crudas con `distance_km`/`effective_distance_km` > 30 y `distance_ratio` > 60, y vectores con features por debajo de 0 o por encima de su límite (≥ 4); (3) **extremos**: entradas que dan 100 y vectores en las esquinas de la caja (0 no es alcanzable con entradas de la app, T05-H4) (≥ 2); (4) **bordes de redondeo**: factores que terminan en .x5 tras round6 y totales x.49999955, x.4999994 y x.5000004 construidos con la distancia (≥ 6); (5) **zona T05-H1**: `budget.max / budget.min` > 76.2 con renta < `budget.min`, mostrando que el factor sube con la renta (≥ 3); (6) `pets_compatible` en el término fijo (≥ 1). Los casos solo vector (`input.features` sin entradas crudas) se marcan con su categoría y solo prueban la inferencia.
- **Formato por caso:** `id`, `category`, `description`, `input.student`/`input.room` (o `input.features`), `tolerance`, `expected.features` (10, sin recortar), `expected.clippedFeatures` (informativo), `expected.onnx` (`factor_points` [6] y `raw_total`), `expected.result` (JSON de §6.1 paso 7: `breakdown` redondeado y `matchScore`, o `excluded`/`incomplete`). Encabezado con `modelVersion`, SHA-256 de `logreg.json`, `maxPointsRaw`, regla de redondeo, nota de .NET y tolerancias.
- **Tolerancias:** features y `value` 1e-6; `expected.onnx` 1e-4; `points` redondeados y `matchScore` exactos (0).
- **Regla de margen** (M-ML-033 decisión 3): todo valor no exacto queda a ≥ 1e-9 de un punto medio de round6; el generador falla si un caso la incumple (se reemplaza la entrada, no se ajusta el esperado).
- **Sin el modelo oficial** (p. ej. en CI, donde `models/` no existe): la prueba de identidad del archivo y la paridad oficial se omiten con un motivo explícito; en CI la misma exportación y paridad corren con el modelo del perfil `ci` (`models/ci/`). La evidencia registra la corrida con el modelo oficial (un control no ejecutado no se reporta como PASS).

### 6.5 Rendimiento

- **Medición:** onnxruntime en CPU (`CPUExecutionProvider`), sesión creada una vez (la carga no cuenta), lote de **200 cuartos** `[200, 10]` tomados de pares de test, 10 corridas de calentamiento descartadas y **100 corridas** medidas con `time.perf_counter_ns`; criterio **mediana < 20 ms** (constitución §13.5). Se reportan además p95, máximo, la configuración de hilos y el CPU/SO de la máquina.
- **Oficial** en la máquina local documentada en la evidencia; en CI corre como prueba de humo informativa (sin umbral estricto, para no depender del runner).

### 6.6 Paquete y distribución (decisión 4)

- **Etapa `export`** (`python -m thehouse_ml export`, ARQUITECTURA §3) escribe en **`dist/model-v1.0.0/`** (fuera de git; se agrega `/dist/` a `.gitignore` y a `tests/test_repo_hygiene.py`):

| Archivo | Origen |
| --- | --- |
| `model.onnx` | Grafo de §6.2 |
| `feature_schema.json` | Copia byte a byte de `src/thehouse_ml/features/feature_schema.json` 2.0.0 |
| `golden_cases.json` | Copia de `tests/fixtures/logreg_golden_cases.json` (§6.4) |
| `baseline_golden_cases.json` | Copia de `tests/fixtures/baseline_golden_cases.json` (81 casos, respaldo en C#) |
| `model_card.md` | `reports/model_card.md` más una sección «Artefacto ONNX» generada por `export` (opset, entradas, salidas, metadatos, paridad, rendimiento, commit, SHA-256 de `logreg.json`, desviación T05-H1); `reports/model_card.md` no cambia |
| `metrics.json` | Copia byte a byte de `reports/trained_metrics.json` («DATOS SINTÉTICOS») |
| `SHA256SUMS` | SHA-256 de los 6 archivos anteriores, formato `sha256sum` |

- **Publicable** solo con perfil `full`, `logreg.json` con el SHA-256 esperado, paridad y casos dorados en verde y árbol de git limpio; si no, el paquete se marca no publicable (perfil `ci` en `dist/ci/`).
- **Release** `model-v1.0.0`: lo **publica el responsable a mano** (AGENTS.md) con los 7 archivos del paquete, sobre el commit de `main` que lo generó. Es **inmutable**: no se editan ni reemplazan sus archivos; una corrección es una versión nueva (constitución §13.3). Activar los releases inmutables de GitHub sería un cambio de configuración que requiere permiso aparte.
- **Backend:** descarga los archivos del release **por tag** y verifica `SHA256SUMS` en el build; además fija en su repositorio el SHA-256 esperado de `model.onnx` (o de `SHA256SUMS`), para no confiar solo en un archivo descargado del mismo lugar (constitución §14.3).

### 6.7 Guía de integración

`docs/backend-integration.md` (en git, en español) para NV-23: contrato de entrada (`features` `[N, 10]` double en el orden de `feature_order`) y de salida (`factor_points`, `raw_total`, metadatos); cálculo de las 10 features con el esquema 2.0.0 (fórmulas, reglas para faltantes, transporte); orden de los pasos de §6.1 (faltantes → incompleto → restricciones → features → ONNX → redondeo → JSON → ranking); redondeo en .NET (round6 y `Math.Round(x, d, MidpointRounding.AwayFromZero)`, con los bordes de los casos dorados); cómo correr `golden_cases.json` y `baseline_golden_cases.json` en .NET con `Microsoft.ML.OnnxRuntime` (ejemplo de prueba xUnit/NUnit sin dependencias de pago); descarga y verificación del release; y la desviación **T05-H1** (zona, magnitud y que no se corrige en `logreg-1.0.0`). Incluye la nota de la §7 de la model card sobre `maxPoints` para `TheHouse-GenAI`.

## 7. Reglas

1. Sin pickle en ninguna parte del paquete ni del código de exportación (constitución §13.1; `test_no_module_of_the_package_uses_pickle` cubre `src/thehouse_ml/export/`).
2. Un release publicado no se modifica; una corrección es una versión nueva.
3. El ONNX solo recibe pares que pasaron el perfil incompleto y HC01–HC03; nunca puntúa un par excluido.
4. Toda cifra de métricas del paquete es **sintética** y se etiqueta así (constitución §8).
5. No se reentrena ni se recalibra: la exportación lee `logreg.json` y falla si su SHA-256 no es el de la §1.
6. `dist/`, `models/` y `data/` nunca se suben a git.

## 8. Seguridad, datos y privacidad

- SHA-256 de todos los archivos del paquete en `SHA256SUMS`; el backend verifica antes de cargar y fija el hash esperado (§6.6).
- El paquete no contiene datos personales: solo ids sintéticos y agregados (`metrics.json`), entradas inventadas (casos dorados) y parámetros del modelo.
- Dependencias nuevas con licencia registrada en la evidencia (`onnx` Apache-2.0, `onnxruntime` MIT; se confirma en T01), fijadas en `uv.lock` dentro del `exclude-newer` vigente y con pip-audit limpio (constitución §14.2).
- Sin secretos ni red en las pruebas; sin servicios de pago (COST-001).

## 9. Requisitos verificables

| ID | Requisito | Criterio de aceptación | Fase | Tasks |
| --- | --- | --- | --- | --- |
| ML-06-R01 | Dependencias y exportación verificada | `onnx` y `onnxruntime` en `pyproject.toml` y `uv.lock` con `uv lock --check` y pip-audit sin vulnerabilidades (decisión M-ML de T01); `export` verifica el SHA-256 de `logreg.json` contra `train_manifest.json` y la §1, construye el grafo de la §6.2 (double, `Max`/`Min`, `MatMul` por factor, calibración, `Add` encadenado) con sus metadatos; `onnx.checker` sin errores; onnxruntime en CPU lo carga y lo ejecuta sobre los pares de test; dos ejecuciones dan un `model.onnx` idéntico byte a byte. | F1 | T01 |
| ML-06-R02 | Paridad Python ↔ ONNX | En los pares de test, la rejilla fuera del `training_range` y los casos dorados: \|Δ\| ≤ 1e-4 en `factor_points` y `raw_total`; 0 diferencias en puntos redondeados y `matchScore`; máximo \|Δ\| reportado. | F2 | T02 |
| ML-06-R03 | Casos dorados de `logreg-1.0.0` | `tests/fixtures/logreg_golden_cases.json` con recuento independiente (§6.4): las entradas de los 81 casos del baseline más recorte, extremos, bordes de redondeo, zona T05-H1 y término fijo; archivo idéntico a la salida del generador; regla de margen cumplida; paquete y ONNX coinciden con todos dentro de las tolerancias. | F2 | T03 |
| ML-06-R04 | Rendimiento | 200 cuartos en onnxruntime CPU con mediana < 20 ms en 100 corridas (tras 10 de calentamiento, sesión ya cargada), con p95, máximo, hilos y CPU/SO reportados. | F3 | T04 |
| ML-06-R05 | Paquete de entrega | `export` (perfil `full`) escribe en `dist/model-v1.0.0/` los 7 archivos de la §6.6 con `SHA256SUMS` verificable con `sha256sum -c`; copias byte a byte donde aplica; sección «Artefacto ONNX» en `model_card.md`; `/dist/` ignorado por git con prueba de higiene; paquete reproducible (mismos SHA-256 en dos ejecuciones); instrucciones para que el responsable publique el release. | F3 | T05 |
| ML-06-R06 | Guía de integración | `docs/backend-integration.md` con los contenidos de la §6.7, coherente con `feature_schema.json`, M-ML-041 y los casos dorados (una prueba comprueba que los nombres de entradas, salidas, factores y la regla de redondeo citados coinciden con el ONNX y el esquema). | F3 | T06 |
| ML-06-R07 | Pruebas y evidencia | Pruebas unitarias e integración con cobertura ≥ 80 % del código nuevo; ruff, mypy y pip-audit sin errores; `evidence.md` con commit, comandos, resultados, hallazgos, revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`, y verificación del release publicado por el responsable (SHA-256 descargados = `SHA256SUMS`) o, si aún no existe, registrado como pendiente. | F4 | T07 |

## 10. Decisiones

Registradas en `memoria.md` (M-ML-044, APROBADO). Las versiones, licencias y el opset de las dependencias se fijan con la decisión M-ML de la ejecución de T01.

| # | Tema | Propuesta | Estado | Tasks |
| --- | --- | --- | --- | --- |
| 1 | Modelo a exportar | `logreg-1.0.0` (`models/logreg.json`, SHA-256 `341ea3ec…0b27fa` verificado antes de exportar). LightGBM fuera de alcance en v1 (referencia). `baseline-1.0.0` no se exporta: queda como respaldo implementado en C# por NV-23 y validado con sus 81 casos dorados, que van en el paquete | APROBADO (M-ML-044) | T01–T07 |
| 2 | Grafo ONNX | Opción (a): grafo construido con la biblioteca `onnx` (recorte por columna con `Max`/`Min` al `training_range` → escalado → contribuciones lineales → suma por factor → calibración afín), todo en double; salidas `factor_points` [N, 6] y `raw_total` [N] sin redondear; metadatos de la §6.2; opset propuesto 17. El redondeo (round6, 1 decimal por factor, total entero `AwayFromZero`) y el JSON (`modelVersion`, `featureSchemaVersion`, `matchScore`, `breakdown`) los hace el backend con la regla documentada | APROBADO (M-ML-044) | T01–T03, T06 |
| 3 | Responsabilidades | El backend (NV-23) resuelve faltantes, rechaza el perfil incompleto, aplica HC01–HC03, calcula y valida las 10 features con `feature_schema.json` 2.0.0, redondea, arma el JSON y ordena; el ONNX solo puntúa pares que pasaron (§6.1) | APROBADO (M-ML-044) | T03, T06 |
| 4 | Distribución del artefacto | Paquete de 7 archivos en `dist/model-v1.0.0/` (fuera de git) generado por `export`; Release de GitHub `model-v1.0.0` publicado a mano por el responsable e inmutable; el backend descarga por tag, verifica `SHA256SUMS` en el build y fija el SHA-256 esperado de `model.onnx` | APROBADO (M-ML-044) | T05, T06, T07 |
| 5 | Dependencias | `onnx` y `onnxruntime` como dependencias directas en T01, con pip-audit limpio; **sin** `skl2onnx` (opción a). `onnxruntime` está en la constitución §5; el paquete `onnx` no está nombrado de forma explícita (la tabla nombra `skl2onnx`/`onnxmltools`), así que se registra como decisión M-ML junto con las versiones y licencias que resuelva `uv` | APROBADO (M-ML-044) | T01 |
| 6 | Runtime en el backend | `Microsoft.ML.OnnxRuntime` en NV-23 | APROBADO (M-ML-002 y M-ML-006; referencia, no se decide aquí) | T06 |

## 11. Costo — COST-001

| Componente | Costo | Condición |
| --- | --- | --- |
| CPU local y GitHub Actions (repositorio público) | $0 | Sin servicios de pago ni GPU en la nube; `onnx` y `onnxruntime` son de código abierto y corren en CPU |
| Release de GitHub (repositorio público) | $0 | Archivos de pocos KB |

## 12. Riesgos

| ID | Riesgo | Impacto | Mitigación |
| --- | --- | --- | --- |
| ML-06-RK01 | Paridad float32/float64: una constante o una operación en float32 desvía los puntos y cambia un redondeo | Puntajes distintos entre Python, ONNX y .NET | Todo el grafo en `tensor(double)`; prueba de que todas las entradas, constantes y salidas son double; tolerancia 1e-4 y 0 en redondeados; regla de margen en los casos dorados |
| ML-06-RK02 | Cambios en NV-12 (contrato del cuarto) o NV-23 (esquema, reglas) después de exportar | El backend calcula otras features o rompe el contrato | El contrato es `feature_schema.json` 2.0.0 (constitución §2.4): un cambio exige spec en ambos repositorios y, si cambia el esquema, versión mayor y modelo nuevo; la guía y los casos dorados lo detectan |
| ML-06-RK03 | Release mal publicado (archivos equivocados, editados o sobre otro commit) | El backend carga otro artefacto | `SHA256SUMS`, hash esperado fijado en el backend, lista de verificación para el responsable y verificación del release en T07 |
| ML-06-RK04 | Diferencias de cálculo de features entre Python y .NET | Puntajes distintos | Casos dorados con entradas crudas y features esperadas (81 + los nuevos) corridos en ambos lados |
| ML-06-RK05 | Las pruebas con el modelo oficial no corren en CI (`models/` fuera de git) | Una regresión pasa el CI | En CI corren exportación y paridad con el modelo del perfil `ci`; la evidencia registra la corrida oficial; las omisiones dicen su motivo |

## 13. Dependencias

- ML-05 completa (M-ML-043): `models/logreg.json`, `trained_breakdown.py` (`build_breakdown_model`, `score_features_logreg`), `reports/trained_metrics.json` y `reports/model_card.md`.
- ML-04: `baseline_golden_cases.json` (81 casos) y la regla de redondeo (M-ML-029/031/033).
- Constitución §5 y decisión 5: `onnx` y `onnxruntime`.
- NV-23 (backend): integración con `Microsoft.ML.OnnxRuntime`, esquema 2.0.0, recorte, redondeo, `breakdown` y T05-H1 (aviso a cargo del responsable). NV-12: contrato del cuarto (RK03 de ML-03), necesario para producción.
- `TheHouse-GenAI`: la nota de `maxPoints` de la model card (responsable).

## 14. Trazabilidad y cobertura

| Métrica | Valor |
| --- | --- |
| Requisitos | 7 |
| Tasks | 7 |
| Requisitos sin task | 0 |
| Tasks sin requisito | 0 |
| Tasks completadas | 0 |
| Estado documental | **✅ APROBADA — 1.0.0** |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |

## 15. Aprobación

Aprobada por César Yael López Trujillo el 2026-11-24 («Apruebo ML-06 1.0.0», M-ML-044), con las decisiones 1–5 de la §10 en APROBADO y los agregados de la revisión aceptados: recorte con `Max`/`Min` en lugar de `Clip`, `raw_total` con `Add` encadenado en el orden de Python, SHA-256 de `model.onnx` fijado en el repositorio del backend, `maxPoints` en `metadata_props`, `/dist/` en `.gitignore` en T05 y paridad con el modelo oficial en local (con evidencia) y en CI con el modelo del perfil `ci` (RK05 aceptado). Para ejecutar se requiere «EJECUTA» con versión, tasks y rutas permitidas; T01 agrega dependencias, así que su «EJECUTA» debe autorizar `pyproject.toml` y `uv.lock`; T05 modifica `.gitignore` y `tests/test_repo_hygiene.py`; T06 crea `docs/`.
