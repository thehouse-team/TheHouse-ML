# ML-06 — Evidencia de ejecución

| Campo | Valor |
| --- | --- |
| Spec | ML-06 1.0.0 (M-ML-044) |
| Tareas ejecutadas | T01 · Dependencias y exportación verificada (ML-06-R01); T02 · Paridad Python ↔ ONNX (ML-06-R02); T03 · Casos dorados de `logreg-1.0.0` (ML-06-R03) |
| Estado | **T01 completa — PASS WITH WARNINGS** (M-ML-045, APROBADO 2026-11-24); **T02 completa — PASS WITH WARNINGS** (M-ML-046, APROBADO 2026-11-24); **T03 completa — PASS WITH WARNINGS** (M-ML-047, APROBADO 2026-11-25) |
| Autorización (T01) | «EJECUTA ML-06 1.0.0, task T01. Rama: feat/ml-06-export (ya creada desde main actualizado, con la spec ML-06 1.0.0 aprobada en M-ML-044). Se autoriza modificar pyproject.toml y uv.lock SOLO para agregar las dependencias onnx y onnxruntime que pide la spec (respetando exclude-newer 2026-10-27 de uv). Se autoriza crear/modificar los archivos de código y pruebas que la task T01 indique en el plan de ML-06. Ninguna otra ruta fuera de las listadas en el plan» — César Yael López Trujillo, 2026-11-24 |
| Rama | `feat/ml-06-export` (commit base `a046abf`, merge del PR #38) |
| Commit | Sin commit ni push, por instrucción del responsable |
| Fecha de ejecución | 2026-11-24 |
| Aprobación (T01) | «Apruebo ML-06 T01 (César Yael López Trujillo, 2026-11-24)»; en el mismo mensaje: correcto seguir la spec sobre el prompt; autoriza el override de mypy de `onnxruntime` en `pyproject.toml` (en lugar del `type: ignore`); acepta `profile`, «Δ» → «dif» y el sufijo `-dirty`; autoriza `memoria.md` y `task.md` |
| Decisiones | M-ML-044 (APROBADO: decisiones 1, 2 y 5 de la §10 y agregados de la revisión); **M-ML-045 (APROBADO, 2026-11-24)** |

Lecturas previas, en el orden obligatorio de `AGENTS.md`: `constitution.md` (§5, §9, §13, §14, §15) → `AGENTS.md` → `memoria.md` (M-ML-041, M-ML-043, M-ML-044) → `spec.md`, `plan.md` y `task.md` de ML-06 → `specs/ARQUITECTURA.md` → `src/thehouse_ml/models/trained_breakdown.py` (`build_breakdown_model`, `score_matrix`), `models/train.py` (`LogRegModel`, `sha256_file`, `output_dirs`) y `models/trained_eval.py` (verificación de modelos).

## T01 · Dependencias y exportación verificada

### Qué se hizo

1. **Dependencias** con `uv add onnx onnxruntime` (uv 0.13.7, que respeta `exclude-newer = "2026-10-27T00:00:00Z"`): `pyproject.toml` (`onnx>=1.23.2`, `onnxruntime>=1.31.1`) y `uv.lock` (+82 líneas; 60 paquetes resueltos). Sin `skl2onnx` (opción a de la spec §6.2). No se fijó ninguna versión a mano.

   | Paquete | Versión (`uv.lock`) | Tipo | Licencia (metadatos instalados) | Publicado (máx. `upload-time` en `uv.lock`) |
   | --- | --- | --- | --- | --- |
   | `onnx` | 1.23.2 | directa | Apache-2.0 | 2026-10-06 |
   | `onnxruntime` | 1.31.1 | directa | MIT | 2026-10-26 (último artefacto de todo el lock: 2026-10-26T19:26:28Z) |
   | `protobuf` | 7.36.2 | transitiva (onnx, onnxruntime) | BSD-3-Clause | 2026-09-17 |
   | `ml-dtypes` | 0.6.0 | transitiva (onnx) | Apache-2.0 | 2026-08-13 |
   | `flatbuffers` | 25.12.19 | transitiva (onnxruntime) | Apache-2.0 | 2025-12-19 |

   `numpy`, `packaging` y `typing_extensions` ya estaban en el lock. Todas las fechas son anteriores al 2026-10-27.

2. **Verificación de `logreg.json`** antes de exportar (`verify_logreg`): SHA-256 contra `models/train_manifest.json` (perfil `full`) y, en el perfil `full`, contra el valor fijado en la spec §1 (`EXPECTED_LOGREG_SHA256`). Si no coincide, `ExportError` y la CLI sale con código 1 sin escribir nada (no se reentrena). También exige `synthetic: true`.

3. **Grafo ONNX** construido con la biblioteca `onnx` (`src/thehouse_ml/export/onnx_graph.py`), todo en `tensor(double)`, opset 17, IR 8:

   | # | Nodo(s) | Constantes |
   | --- | --- | --- |
   | 1 | `Max(features, clip_lower)` → `Min(·, clip_upper)` | [10] de `clip.lower`/`clip.upper` (no `Clip`) |
   | 2 | `Sub(scaler_mean)` → `Div(scaler_scale)` | [10] |
   | 3 | `Mul(logreg_coef)` | [10] |
   | 4 | `MatMul(factor_assignment)` | [10, 6] de ceros y unos; fila de `pets_compatible` en ceros |
   | 5 | `Sub(factor_min_logit)` → `Mul(points_per_logit)` → **`factor_points`** [N, 6] | [6] `min_k` y [6] `100 / S` |
   | 6 | 6 × `Gather(axis=1)` y 5 × `Add` encadenados → **`raw_total`** [N] | índices escalares int64 |

   `raw_total = ((((p_budget + p_distance) + p_essential_services) + p_place_type) + p_property_type) + p_extras`, el mismo orden que `BreakdownModel.raw_total`. Sin logit, probabilidad ni término fijo. `metadata_props`: `modelVersion`, `featureSchemaVersion`, `featureOrder`, `factorOrder`, `maxPointsRaw`, `roundingRule` (fórmulas y nota .NET de `baseline_params.json`), `logregSha256`, `gitCommit`, `profile`, `synthetic`.

4. **Etapa `export`** (`src/thehouse_ml/export/package.py`, encadenada en `stages.py`): verifica, construye, pasa `onnx.checker.check_model(full_check=True)`, serializa de forma determinista (`SerializeToString(deterministic=True)`, productor `thehouse-ml` 0.1.0, sin marcas de tiempo) y escribe `dist/model-v1.0.0/model.onnx` (perfil `full`) o `dist/ci/model.onnx` (perfil `ci`, con `models/ci/logreg.json` verificado solo contra su manifiesto). Si `data/processed/` existe, ejecuta el modelo en onnxruntime CPU sobre los pares de test. Sin modelo, se omite con mensaje y sin escribir archivos. El resto del paquete (casos dorados, model card, métricas, `SHA256SUMS`) es de T05.

5. **CLI:** `ExportError` se reporta como error (código 1). `ExportError` vive en `thehouse_ml/export/__init__.py` (sin importar `onnx` ni scikit-learn), así que la CLI sigue sin cargar esas bibliotecas hasta ejecutar `export` (verificado: `'onnx' in sys.modules` y `'sklearn' in sys.modules` son `False` tras importar `thehouse_ml.cli`).

### Archivos

| Área | Archivo | Cambio |
| --- | --- | --- |
| Dependencias | `pyproject.toml`, `uv.lock` | `onnx` y `onnxruntime`; override de mypy de `onnxruntime` (autorizado en la aprobación) |
| Código | `src/thehouse_ml/export/__init__.py` | Nuevo: paquete y `ExportError` |
| Código | `src/thehouse_ml/export/onnx_graph.py` | Nuevo: verificación, constantes, grafo, metadatos, serialización y ejecución en onnxruntime |
| Código | `src/thehouse_ml/export/package.py` | Nuevo: etapa `export` (T01: `model.onnx`) |
| Código | `src/thehouse_ml/stages.py` | `export` deja de ser una etapa vacía |
| Código | `src/thehouse_ml/cli.py` | `ExportError` en los errores reportados |
| Pruebas | `tests/unit/test_onnx_graph.py` | Nuevo (22 pruebas) |
| Pruebas | `tests/integration/test_export_stage.py` | Nuevo (7 pruebas; incluye el modelo oficial) |
| Pruebas | `tests/unit/test_cli.py`, `tests/unit/test_stages.py`, `tests/integration/test_installed_package.py` | Esperaban «export: pendiente de ML-06»; ahora el mensaje de omisión. La prueba del punto de entrada usa una raíz temporal (`THEHOUSE_ML_ROOT`) para no escribir en el repositorio |

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `sha256sum models/logreg.json` | `341ea3ec705c2f877b36d1b931eafe2dea4630523560fe8764ce59e0a70b27fa` = spec §1 = `train_manifest.json` (`profile: full`) ✅ |
| `uv add onnx onnxruntime` | onnx 1.23.2, onnxruntime 1.31.1 (+ protobuf 7.36.2, ml-dtypes 0.6.0, flatbuffers 25.12.19) |
| `uv lock --check` | `Resolved 60 packages` — lock coherente ✅ |
| `uv run pip-audit --skip-editable` (el comando del CI) | `No known vulnerabilities found` ✅ |
| `uv run ruff check src tests` | `All checks passed!` ✅ |
| `uv run ruff format --check src tests` | `118 files already formatted` ✅ |
| `uv run mypy` (estricto, `src`; repetido tras el override de `onnxruntime` y sin `type: ignore`) | `Success: no issues found in 54 source files` ✅ |
| `uv run pytest tests/unit/test_onnx_graph.py tests/integration/test_export_stage.py tests/unit/test_cli.py tests/unit/test_stages.py tests/integration/test_installed_package.py tests/test_repo_hygiene.py --cov=thehouse_ml.export` | **98 passed** en 25.71 s, 0 omitidas (la prueba del modelo oficial corrió); cobertura de `thehouse_ml.export` **94 %** (`__init__` 100 %, `onnx_graph` 99 %, `package` 88 %: las líneas sin cubrir son la ejecución sobre pares de test, cubierta por la corrida oficial de abajo) |
| `uv run python -m thehouse_ml export` (perfil `full`, 3 ejecuciones) | `dist/model-v1.0.0/model.onnx`, opset 17, `onnx.checker` sin errores; 30,755 pares de test en onnxruntime CPU con salidas finitas; máx. \|dif\| vs Python `factor_points` 2.842e-14 y `raw_total` 4.263e-14 (informativo; la paridad formal es T02); 8.9 s; salida sin error en la consola cp1252 de Windows |
| `sha256sum dist/model-v1.0.0/model.onnx` (tras cada ejecución) | `99e791c260e57b0ec95ee1e6bdb622362032c702de253de5476a2808817985e8` las 3 veces (4,386 bytes) — **reproducible byte a byte** ✅ |
| Inspección independiente (`onnx.load_from_string` + `check_model(full_check=True)`) | IR 8, opset `("", 17)`; nodos `Max, Min, Sub, Div, Mul, MatMul, Sub, Mul, Gather×6, Add×5`; entrada `features` y salidas `factor_points`/`raw_total` en DOUBLE; inicializadores solo DOUBLE e INT64 (índices); `maxPointsRaw` = [9.569811954389692, 72.60747294686455, 6.733606109723173, 4.8542379171078585, 1.7248277730510626, 4.510043298863674]; `logregSha256` = `341ea3ec…0b27fa`; `gitCommit` = `a046abf25319e9ec97a638282e07490eb9116380-dirty`; `synthetic` = `true` |
| Búsqueda de asignaciones `key/token/secret/password = "<literal>"` en los archivos nuevos (por el falso positivo previo de gitleaks) | Ninguna |
| gitleaks | **No ejecutado localmente** (no instalado); lo corre el CI. No se reporta como PASS |
| Suite completa | **No ejecutada** por instrucción del responsable (~10 min; la corre el CI) |

### Aceptación de T01 (ML-06-R01)

| Criterio | Resultado |
| --- | --- |
| `onnx` y `onnxruntime` en `pyproject.toml` y `uv.lock`, `uv lock --check` y pip-audit limpio | ✅ |
| Decisión M-ML con versiones, licencias y opset | ✅ M-ML-045 (APROBADO) |
| `export` verifica el SHA-256 de `logreg.json` (manifiesto y spec §1) | ✅ (`test_full_profile_rejects_a_model_other_than_logreg_1_0_0`, `test_verify_logreg_*`, corrida oficial) |
| Grafo de la §6.2 en double con `Max`/`Min`, `MatMul` por factor, calibración y `Add` encadenado, con metadatos | ✅ (`test_node_sequence_follows_the_spec`, `test_raw_total_is_a_left_to_right_add_chain`, `test_inputs_outputs_and_float_constants_are_double`, `test_constants_are_copied_from_the_model_in_binary64`, `test_metadata_props`) |
| `onnx.checker` sin errores | ✅ (`full_check=True` en la construcción, en las pruebas y en la inspección independiente) |
| onnxruntime CPU lo carga y lo ejecuta sobre los pares de test | ✅ 30,755 pares, salidas finitas |
| `model.onnx` idéntico en dos ejecuciones | ✅ mismo SHA-256 en 3 ejecuciones; `test_ci_profile_writes_a_reproducible_model`, `test_official_model_exports_and_is_reproducible`, `test_serialization_is_deterministic` |

### Hallazgos

| ID | Clasificación | Hallazgo | Acción |
| --- | --- | --- | --- |
| T01-H1 | Advertencia | `dist/` aún **no está en `.gitignore`** (lo agrega T05 por la spec §6.6); tras exportar, `git status` muestra `?? dist/`. El riesgo de subir `model.onnx` está acotado porque `tests/test_repo_hygiene.py` ya prohíbe versionar `*.onnx` | No hacer `git add dist/`; T05 agrega `/dist/` |
| T01-H2 | Advertencia | El SHA-256 `99e791c2…` corresponde a `gitCommit = a046abf…-dirty` (código de T01 sin commit). Tras el commit el metadato cambia y con él el SHA-256: no es el artefacto publicable. El sufijo `-dirty` solo considera archivos versionados (no `dist/`), para que la exportación sea reproducible | El paquete publicable sale en T05 desde un commit limpio |
| T01-H3 | Resuelto | `onnxruntime` no publica `py.typed`. Primero se usó `# type: ignore[import-untyped]` porque la autorización limitaba `pyproject.toml`; con la aprobación se cambió al override de mypy en `pyproject.toml` (como pide el plan) y se quitó el `type: ignore`. `uv run mypy`: `Success: no issues found in 54 source files`; `ruff check` y `ruff format --check` sin errores | Cerrado (M-ML-045) |
| T01-H4 | Agregado | Metadatos: además de los de la §6.2, `profile` (`full`/`ci`), para que un `model.onnx` del perfil `ci` no pueda confundirse con el oficial (ambos llevan `modelVersion` `logreg-1.0.0`). `maxPointsRaw` va como lista JSON y `roundingRule` como objeto JSON | Aceptado; registrado en M-ML-045 |
| T01-H5 | Decisión técnica | IR 8 fijo (el que corresponde al opset 17; `onnx` 1.23.2 escribiría IR 14 por omisión) para que lo cargue cualquier onnxruntime ≥ 1.13, incluido el de .NET de NV-23 | Registrado en M-ML-045 |
| T01-H6 | Aclaración del «EJECUTA» | El prompt decía «suma por factor con cadena de Add» y «maxPoints en metadata_props»; la spec §6.2 define la suma por factor con `MatMul` y la cadena de `Add` para `raw_total`, y el metadato como `maxPointsRaw`. Se siguió la spec (manda sobre el prompt) | Aceptado por el responsable; registrado en M-ML-045 |
| T01-H7 | Informativo | `onnxruntime` lista `AzureExecutionProvider` como disponible; el código fija `providers=["CPUExecutionProvider"]` y no usa ningún servicio (COST-001 $0) | Ninguna |
| T01-H8 | Control no ejecutado | gitleaks local y suite completa no ejecutados (ver tabla); quedan para el CI del PR | Revisar el CI del PR |

### Decisión registrada (APROBADO, `memoria.md`)

**M-ML-045 — ML-06 T01: dependencias ONNX, opset 17 / IR 8 y etapa `export`** — `onnx` 1.23.2 (Apache-2.0) y `onnxruntime` 1.31.1 (MIT) como dependencias directas, con transitivas `protobuf` 7.36.2 (BSD-3-Clause), `ml-dtypes` 0.6.0 (Apache-2.0) y `flatbuffers` 25.12.19 (Apache-2.0), todas dentro de `exclude-newer` 2026-10-27 y con pip-audit limpio; `onnx` queda registrado como dependencia fuera del texto literal de la constitución §5 (autorizada por M-ML-044 decisión 5); sin `skl2onnx`. Opset 17 e IR 8. `onnxruntime` sin tipos: override de mypy en `pyproject.toml` (T01-H3). Grafo según la spec: `MatMul` para la suma por factor, `Add` encadenado solo para `raw_total`, metadato `maxPointsRaw` (T01-H6). Metadatos de la §6.2 más `profile` (T01-H4). Etapa `export` en `src/thehouse_ml/export/` (`onnx_graph.py`, `package.py`), salida `dist/model-v1.0.0/` (`full`) o `dist/ci/` (`ci`); `gitCommit` con sufijo `-dirty` si hay cambios en archivos versionados.

### Seguridad, datos y licencias

- Sin pickle, joblib ni dill: el grafo se construye desde `logreg.json` (JSON) con `onnx`; `test_no_module_of_the_package_uses_pickle` cubre `src/thehouse_ml/export/` y pasa.
- `models/`, `data/` y `dist/` sin versionar (`git status`: solo `?? dist/`, ver T01-H1). No se modificaron `logreg.json`, `feature_schema.json`, `train_params.json` ni ningún params.
- Sin datos personales ni atributos de género: el ONNX solo recibe las 10 features del esquema 2.0.0 (`pets_compatible` sin peso; el género solo interviene en `livingWith`, que el backend aplica antes, constitución §11.2).
- Sin red en las pruebas, sin secretos, sin servicios de pago.

### Coherencia

| Documento | Comprobación |
| --- | --- |
| `constitution.md` §5, §13.1, §14.2, §15 | ONNX sin pickle; dependencias fijadas en `uv.lock` con pip-audit limpio; CPU local, $0 |
| `constitution.md` §9.2 | Mismo commit y mismo `logreg.json` → mismo `model.onnx` (verificado) |
| `AGENTS.md` | Sin commit ni push; controles no ejecutados declarados como tales; métricas: ninguna nueva (el modelo y sus métricas siguen etiquetados como sintéticos, `synthetic = true`) |
| M-ML-041 / M-ML-044 | Recorte con `Max`/`Min`, calibración afín por factor, `maxPointsRaw` en metadatos, `Add` encadenado en el orden de Python; desviación T05-H1 conservada (no se tocó el coeficiente) |

**Resultado T01: PASS WITH WARNINGS** — T01-H1, T01-H2 y T01-H8 (advertencias). T01 completa con M-ML-045 (APROBADO, César Yael López Trujillo, 2026-11-24).

---

## T02 · Paridad Python ↔ ONNX

| Campo | Valor |
| --- | --- |
| Autorización | «EJECUTA ML-06 1.0.0, task T02» — César Yael López Trujillo, 2026-11-24. Rama `feat/ml-06-t02` (desde `main` con T01 mezclada, PR #39). Solo archivos de código, pruebas y fixtures que indique T02 en el plan, más `evidence.md` y `task.md`; **no** `pyproject.toml`, `uv.lock`, `.gitignore` ni `memoria.md` |
| Commit base | `0cf2e9a` (merge del PR #39) |
| Commit | Sin commit ni push, por instrucción del responsable |
| Fecha de ejecución | 2026-11-24 |
| Estado | **PASS WITH WARNINGS** — T02 completa |
| Aprobación (T02) | «Apruebo ML-06 T02 (César Yael López Trujillo, 2026-11-24)», con decisiones: correcto seguir la spec (casos dorados de logreg en T03); grafo sin cambios y regla `-0` → `0` para NV-23 documentada en T06; errata de la §6.4 al cerrar la spec en T07; `score_matrix` aceptado para la rejilla; `export` solo reporta y T05 debe marcar no publicable si la paridad falla; autoriza `memoria.md` y `task.md` |
| Decisión | **M-ML-046 (APROBADO, 2026-11-24)**, registrada en `memoria.md` |

Lecturas previas: `spec.md` (§6.1, §6.3, §6.4, §9 R02/R03), `plan.md`, `task.md`, esta evidencia (T01) y M-ML-045 en `memoria.md`; código de T01 (`onnx_graph.py`, `package.py`), `trained_breakdown.py` (`score_features_logreg`, `score_matrix`), `baseline_score.py` (`round_half_up`, `factor_points`, `match_score`), `tests/golden_recount.py` y `tests/fixtures/baseline_golden_cases.json`.

### Aclaración del alcance (prompt ↔ spec)

El «EJECUTA» describe T02 como «paridad ONNX vs Python **y casos dorados de logreg con recuento independiente**». En la spec, el generador con recuento independiente y `tests/fixtures/logreg_golden_cases.json` son **T03** (§6.4, ML-06-R03, plan F2); T02 (ML-06-R02) es la paridad en los pares de test, la rejilla fuera del `training_range` y los casos dorados como conjunto de entrada, más la prueba de que todo es double. **Se siguió la spec** (instrucción 1 del «EJECUTA»): no se creó el generador ni `logreg_golden_cases.json`. Como conjunto de casos dorados se usaron los **70 vectores `expected.features` de los casos puntuados de `baseline_golden_cases.json`**, que ya vienen del recuento independiente de ML-04 (`tests/golden_recount.py`, fracciones exactas, sin importar `thehouse_ml`). Cuando exista el archivo de T03, su paridad contra el ONNX es parte de la aceptación de T03 (T02-H2).

### Qué se hizo

1. **`src/thehouse_ml/export/parity.py`** (nuevo):
   - `check_parity(name, model_bytes, model, features)` → `ParityResult`: ejecuta el ONNX en onnxruntime CPU y lo compara con la referencia de Python; reporta máx. \|dif\| en `factor_points` y `raw_total`, filas con algún punto redondeado distinto, filas con `matchScore` distinto, celdas negativas del ONNX y hasta 10 filas con diferencias (vector, puntos y `matchScore` de ambos lados). `passed` exige ≥ 1 fila, \|dif\| ≤ 1e-4 y 0 diferencias redondeadas. **La tolerancia no se ajustó.**
   - **Referencia de Python:** `score_features_logreg` fila por fila (puntos sin redondear, `rawTotal`, puntos redondeados y `matchScore` del paquete). Solo para vectores fuera del `range` del esquema (los rechaza `score_features_logreg`, que valida como el baseline) se usa `score_matrix`, el mismo cálculo interno, con `factor_points`/`match_score` del paquete (T02-H4). El reporte dice cuántas filas fueron por cada camino.
   - **Redondeo de las salidas del ONNX** con la regla del backend (spec §6.1 paso 6): `backend_round` = .NET `Math.Round(x, d, MidpointRounding.AwayFromZero)` (`round_half_up` del paquete sobre la magnitud, con el signo, y `-0.0` normalizado a `0.0`); puntos = round(round6(x), 1); `matchScore` = round(round6(raw_total), 0) recortado a [0, 100] (T02-H1).
   - `out_of_range_grid(lower, upper)`: rejilla determinista de **380 vectores** con los límites del recorte de `logreg.json`: sobre 3 bases (esquina inferior, centro, esquina superior), cada feature sola en cada límite y por debajo/encima con desplazamientos de 1e-9, 1e-6, 0.1, 1 y 10 veces el ancho (360); y todas a la vez por debajo, por encima y alternadas (20).
   - `run_onnx_clipped` / `clip_matches_np_clip`: exponen el tensor intermedio `clipped` del grafo en una copia del modelo y lo comparan **bit a bit** con `np.clip` con los límites leídos de `logreg.json`.
2. **Etapa `export`** (`package.py`): el resumen informativo de T01 (`OnnxRunSummary`) se reemplaza por la paridad formal en los pares de test (si existe `data/processed/`) y en la rejilla, con veredicto PASS/FALLA por conjunto. La etapa no cambia el `model.onnx` ni falla si la paridad falla: la condición de «publicable» es de T05 (T02-H5). Mensaje de consola sin «§» (cp1252).
3. **Pruebas** nuevas: `tests/unit/test_onnx_parity.py` (25, modelo sintético, sin `models/`) y `tests/integration/test_onnx_parity_official.py` (6, modelo oficial; se omiten con motivo explícito sin `models/logreg.json` oficial o sin `data/processed/`).

### Archivos

| Área | Archivo | Cambio |
| --- | --- | --- |
| Código | `src/thehouse_ml/export/parity.py` | Nuevo: comparación, redondeo del backend, rejilla, recorte vs `np.clip` |
| Código | `src/thehouse_ml/export/package.py` | La etapa `export` reporta la paridad formal (pares de test y rejilla); `load_test_pair_features`, `run_on_grid`; se quita `OnnxRunSummary` |
| Pruebas | `tests/unit/test_onnx_parity.py` | Nuevo (25 pruebas) |
| Pruebas | `tests/integration/test_onnx_parity_official.py` | Nuevo (6 pruebas con el modelo oficial) |
| Evidencia | `specs/ml-06-onnx-export/evidence.md` | Esta sección y el encabezado |

No se modificaron `pyproject.toml`, `uv.lock`, `.gitignore`, `memoria.md`, `task.md`, `onnx_graph.py`, `logreg.json`, `feature_schema.json` ni ningún params.

### Resultados de la paridad (modelo oficial `logreg-1.0.0`, SHA-256 `341ea3ec…0b27fa`)

| Conjunto | Vectores | Referencia | máx. \|dif\| `factor_points` | máx. \|dif\| `raw_total` | Dif. puntos redondeados | Dif. `matchScore` | Celdas < 0 del ONNX (mín.) | Recorte = `np.clip` | Resultado |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pares de test de ML-05 | 30,755 | 30,755 `score_features_logreg` | 2.842e-14 | 4.263e-14 | **0** | **0** | 21,102, todas en `place_type` (−2.45e-16) | ✅ bit a bit | **PASS** |
| Rejilla fuera del `training_range` | 380 | 80 `score_features_logreg` + 300 `score_matrix` | 1.421e-14 | 1.421e-14 | **0** | **0** | 243: 107 `distance`, 136 `place_type` (−7.85e-15) | ✅ bit a bit | **PASS** |
| Casos dorados (vectores del recuento independiente del baseline) | 70 | 70 `score_features_logreg` | 1.421e-14 | 1.421e-14 | **0** | **0** | 5 en `place_type` (−2.45e-16) | ✅ bit a bit | **PASS** |

Tolerancias de la spec §6.3: \|dif\| ≤ 1e-4 sin redondear y 0 en redondeados y `matchScore`; cumplidas con margen (máx. observado 4.3e-14, del orden esperado con double). **Ningún caso dio diferencia en `matchScore`.** Tipos: entrada, salidas, las 8 constantes y los 17 tensores intermedios inferidos son `tensor(double)`; solo los 6 índices escalares de `Gather` son int64; ningún float32.

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv run python -m thehouse_ml export` (perfil `full`, 2 ejecuciones) | `logreg.json` SHA-256 `341ea3ec…0b27fa` verificado = spec §1; paridad: pares de test 30,755 → PASS; rejilla 380 → PASS; `model.onnx` SHA-256 `8a407e7a4192d056c95862ce6b7eda2cc22b44c1b2c6b72a91cb56a4d2866eca` las 2 veces (reproducible); `gitCommit` = `0cf2e9a…-dirty` |
| `uv run ruff check src tests` | `All checks passed!` ✅ |
| `uv run ruff format --check src tests` | `121 files already formatted` ✅ |
| `uv run mypy` (estricto) | `Success: no issues found in 55 source files` ✅ |
| `uv run pytest tests/unit/test_onnx_parity.py tests/integration/test_onnx_parity_official.py tests/unit/test_onnx_graph.py tests/integration/test_export_stage.py tests/unit/test_cli.py tests/unit/test_stages.py tests/integration/test_installed_package.py tests/test_repo_hygiene.py --cov=thehouse_ml.export --cov-report=term-missing -rs` | **129 passed** en 13.1 s, **0 omitidas** (corrieron las pruebas con el modelo oficial y con `data/processed/`); cobertura de `thehouse_ml.export` **98 %** (`parity.py` 99 %, `package.py` 96 %, `onnx_graph.py` 99 %) |
| Búsqueda de asignaciones `key/token/secret/password = "<literal>"` en los archivos nuevos o cambiados | Ninguna |
| Búsqueda de `pickle`/`joblib`/`dill` en `src/thehouse_ml/export/` y en las pruebas nuevas | Solo menciones en docstrings («sin pickle») |
| gitleaks | **No ejecutado localmente** (no instalado); lo corre el CI. No se reporta como PASS |
| Suite completa | **No ejecutada** por instrucción del responsable; la corre el CI |
| pip-audit | No aplica: T02 no cambia dependencias |

Plataforma: Windows 11 (10.0.26200), Python 3.12.16, onnxruntime 1.31.1 (`CPUExecutionProvider`).

### Aceptación de T02 (ML-06-R02)

| Criterio | Resultado |
| --- | --- |
| Pares de test: \|dif\| ≤ 1e-4, 0 en redondeados y `matchScore` | ✅ 30,755 pares (`test_parity_on_the_test_pairs`) |
| Rejilla fuera del `training_range` | ✅ 380 vectores (`test_parity_on_the_out_of_range_grid`) |
| Casos dorados | ✅ con los 70 vectores del recuento independiente del baseline (`test_parity_on_the_golden_case_vectors`); **pendiente** la paridad contra `logreg_golden_cases.json` de T03 (T02-H2) |
| Máximo \|dif\| reportado | ✅ tabla de resultados y mensaje de `export` |
| Entradas, constantes y salidas double | ✅ `test_all_inputs_constants_intermediates_and_outputs_are_double` (incluye los tensores intermedios) y las pruebas de T01 |
| Recorte = `np.clip` con los límites de `logreg.json` | ✅ bit a bit en los tres conjuntos |

### Hallazgos

| ID | Clasificación | Hallazgo | Acción |
| --- | --- | --- | --- |
| T02-H1 | Advertencia | El ONNX devuelve **negativos del orden de 1e-16 a 1e-15** en un factor que está en su mínimo, donde Python da 0 exacto (21,102 celdas de `place_type` en test; también `distance` en la rejilla). Está dentro de 1e-4 y no cambia ningún redondeo, pero: (a) `round_half_up` del paquete **rechaza** negativos (por eso las salidas del ONNX se redondean con `backend_round`); (b) en .NET, `Math.Round(-2.45e-16, 6, AwayFromZero)` da **`-0.0`**, que `System.Text.Json` escribe como `-0`. Causa: el ONNX calcula `(C_k − min_k) · (100/S)` con `MatMul`, y Python suma la ganancia término a término | Propuesta en M-ML-046: el backend normaliza `-0` a `0` (o aplica `max(0, x)`) después del redondeo; T06 lo documenta en la guía y T03 incluye un caso dorado con un factor en su mínimo. **No** se cambió el grafo ni la tolerancia |
| T02-H2 | Aclaración del «EJECUTA» | El prompt incluía en T02 los casos dorados con recuento independiente; en la spec son T03 (§6.4, R03). Se siguió la spec | El generador, `logreg_golden_cases.json` y su paridad contra el ONNX quedan para el «EJECUTA» de T03 |
| T02-H3 | Discrepancia documental | La spec §6.4 dice «incluidos los 9 excluidos y los 5 incompletos»; `baseline_golden_cases.json` tiene **70 puntuados, 6 excluidos y 5 incompletos** (81) | Corregir el texto en la próxima versión de la spec o tomar nota en T03; no afecta a T02 |
| T02-H4 | Decisión técnica | 300 de los 380 vectores de la rejilla están fuera del `range` del esquema (p. ej. negativos o > 1 en features acotadas) y `score_features_logreg` los rechaza; para ellos la referencia es `score_matrix` (el mismo cálculo interno) con el redondeo del paquete. El reporte separa ambos conteos | Registrado en M-ML-046 |
| T02-H5 | Informativo | La etapa `export` reporta la paridad (PASS/FALLA) pero no falla ni borra el `model.onnx` si la paridad falla: la condición de «publicable» es de T05 | T05 usa `ParityResult.passed` para marcar el paquete |
| T02-H6 | Advertencia | El `model.onnx` regenerado (`8a407e7a…`, `gitCommit` `0cf2e9a…-dirty`) no es publicable; `dist/` sigue sin estar en `.gitignore` (T01-H1) | No hacer `git add dist/`; paquete publicable en T05 desde un commit limpio |
| T02-H7 | Control no ejecutado | gitleaks local y suite completa | Revisar el CI del PR |

### Decisión registrada (APROBADO, `memoria.md`)

**M-ML-046 — ML-06 T02: paridad Python ↔ ONNX y normalización de `-0` en el backend** — APROBADO (resumen; el texto completo está en `memoria.md`)

- **Decisión:** completar **ML-06 T02** (ML-06-R02) con estas elecciones:
  1. **Módulo** `src/thehouse_ml/export/parity.py` (`check_parity`, `ParityResult`, `out_of_range_grid`, `clip_matches_np_clip`); la etapa `export` reporta la paridad formal en los pares de test y en la rejilla.
  2. **Referencia de Python:** `score_features_logreg`; para vectores fuera del `range` del esquema (solo en la rejilla), `score_matrix` con el redondeo del paquete (T02-H4).
  3. **Redondeo de las salidas del ONNX** como el backend: `Math.Round(x, d, MidpointRounding.AwayFromZero)` sobre round6, con `-0.0` normalizado a `0.0` (T02-H1). **Regla para NV-23** (a documentar en T06): después de redondear cada factor y el total, normalizar `-0` a `0` (o recortar a `max(0, x)`), porque el ONNX puede devolver negativos de ~1e-16 en un factor en su mínimo; el grafo **no** se cambia.
  4. **Rejilla** determinista de 380 vectores (3 bases × 10 features × 12 valores + 20 con todas las features fuera a la vez).
  5. **Casos dorados en T02:** los 70 vectores del recuento independiente de `baseline_golden_cases.json`; la paridad contra `logreg_golden_cases.json` se agrega en T03 (T02-H2).
- **Evidencia:** `specs/ml-06-onnx-export/evidence.md` (T02): 30,755 pares de test, 380 vectores de rejilla y 70 casos dorados con máx. \|dif\| 4.3e-14 y **0** diferencias en puntos redondeados y `matchScore`; recorte idéntico a `np.clip`; todo double; 129 pruebas nuevas o afectadas en verde (0 omitidas), cobertura de `thehouse_ml.export` 98 %; ruff y mypy estricto sin errores.
- **Pendientes:** paridad contra `logreg_golden_cases.json` (T03); corrección del «9 excluidos» de la spec §6.4 (T02-H3); nota de `-0` en la guía (T06); gitleaks y suite completa en el CI del PR.
- **Estado:** APROBADO — César Yael López Trujillo — 2026-11-24. Además: regla para NV-23 «después del redondeo AwayFromZero, normalizar `-0` a `0` en factor_points y matchScore antes de escribir el JSON» (T06); T05 **debe** marcar el paquete no publicable si la paridad falla; errata de la §6.4 al cerrar la spec en T07 sin cambiar la versión 1.0.0.
- **Impacto:** T02 de ML-06 lista para cerrarse (2/7) con la aprobación; sin cambios en el grafo, modelos, `feature_schema.json`, params ni dependencias.

### Seguridad, datos y licencias

- Sin pickle, joblib ni dill; sin dependencias nuevas; sin red ni secretos en las pruebas; sin servicios de pago (COST-001 $0, `CPUExecutionProvider`).
- `models/`, `data/` y `dist/` sin versionar (`git status`: `?? dist/`, ver T02-H6). La paridad solo compara puntajes de los pares de test; no recalcula métricas ni ajusta nada (spec §6.3).
- Sin datos personales: vectores de features sintéticas y entradas inventadas de los casos dorados.

### Coherencia

| Documento | Comprobación |
| --- | --- |
| `constitution.md` §8, §13.1, §13.4, §15 | Métricas: ninguna nueva; sin pickle; paridad Python ↔ runtime demostrada; $0 |
| `AGENTS.md` | Sin commit ni push; controles no ejecutados declarados; spec seguida sobre el prompt y desviación avisada; T02 marcada completa solo tras el «Apruebo» |
| M-ML-029/031/033 | Redondeo round6 → 1 decimal mitad lejos de cero; total `floor(round6(Σ) + 0.5)`; bordes 68.49999999999999 → 69 y 13.349999999999998 → 13.4 probados |
| M-ML-041 / M-ML-044 / M-ML-045 | Grafo de T01 sin cambios (`onnx_graph.py` intacto); `raw_total` con `Add` encadenado; desviación T05-H1 conservada |

**Resultado T02: PASS WITH WARNINGS** — T02-H1 y T02-H6 (advertencias), T02-H2 y T02-H3 (alcance/documental), T02-H7 (controles no ejecutados). T02 completa con M-ML-046 (APROBADO, César Yael López Trujillo, 2026-11-24).

---

## T03 · Casos dorados de `logreg-1.0.0`

| Campo | Valor |
| --- | --- |
| Autorización | «EJECUTA ML-06 1.0.0, task T03» — César Yael López Trujillo, 2026-11-25. Rama `feat/ml-06-t03` (desde `main` con T01 y T02 mezcladas, PR #39 y #40). Solo lo que T03 indica en el plan (generador de casos dorados de logreg y `tests/fixtures/logreg_golden_cases.json`), más `evidence.md` y `task.md`; **no** `pyproject.toml`, `uv.lock`, `.gitignore`, `memoria.md` ni `onnx_graph.py` |
| Commit base | `a0b417f` (merge del PR #40) |
| Commit | Sin commit ni push, por instrucción del responsable |
| Fecha de ejecución | 2026-11-25 |
| Estado | **PASS WITH WARNINGS** — T03 completa |
| Aprobación (T03) | «Apruebo ML-06 T03 (César Yael López Trujillo, 2026-11-25)», con decisiones: T03-H1 aprobado (regla de margen de la spec; los otros 80 casos idénticos); T03-H2, T03-H3 y T03-H5 aprobados; autoriza `tests/fixtures/README.md` y el docstring de `tests/integration/test_onnx_parity_official.py` (T03-H6); los 2 errores de mypy preexistentes en `tests/` quedan para T07; copia como `golden_cases.json` en T05 y errata 9 → 6 en T07; autoriza `memoria.md` y `task.md` |
| Decisión | **M-ML-047 (APROBADO, 2026-11-25)**, registrada en `memoria.md` |

Lecturas previas: `spec.md` (§6.1, §6.3, §6.4, §9 R03), `plan.md`, `task.md`, esta evidencia (T01 y T02) y M-ML-029/031/033, M-ML-040, M-ML-041, M-ML-044, M-ML-045 y M-ML-046 en `memoria.md`; `tests/golden_recount.py`, `tests/build_baseline_golden_cases.py`, `tests/unit/test_baseline_golden_cases.py`, `tests/integration/test_onnx_parity_official.py`; como referencia de formato (sin importarlos en el generador) `trained_breakdown.py` (`to_dict`) y `factor_groups.py`.

**Prompt ↔ spec:** sin contradicciones. El punto 4 del «EJECUTA» («casos excluidos por restricciones duras si la spec lo pide») se resolvió con la spec §6.4: van los 6 excluidos y los 5 incompletos de las entradas del baseline. La spec también pide reutilizar el recuento de features de `tests/golden_recount.py`; se hizo (no forma parte del código bajo prueba).

### Qué se hizo

1. **SHA-256 de `models/logreg.json` verificado antes de empezar:** `341ea3ec705c2f877b36d1b931eafe2dea4630523560fe8764ce59e0a70b27fa` = spec §1. El generador lo vuelve a verificar y falla si no coincide.
2. **Generador `tests/build_logreg_golden_cases.py`** (nuevo), con **recuento independiente**:
   - No importa `thehouse_ml` ni `onnx`, `onnxruntime` o `numpy` (lo comprueba una prueba por AST; además el código fuente no contiene llamadas a `score_features_logreg(`, `score_matrix(`, `build_breakdown_model(` ni `check_parity(`).
   - Features, HC01–HC03 y perfil incompleto con `tests/golden_recount.py` (recuento de ML-04); entradas de los 81 casos con `tests/build_baseline_golden_cases.py`.
   - Parámetros de `models/logreg.json` leídos **como datos** y convertidos a `Fraction` exactas de sus binary64. Antes de usarlos se comprueban `format`, `synthetic`, `transform = standard`, `feature_order` = esquema 2.0.0 y que los 6 factores más `pets_compatible` cubren las 10 features.
   - Cálculo paso a paso de M-ML-041 en aritmética exacta: recorte → `z = (x − mean)/scale` → `t = coef · z` → `C_k` → `min_k`/`max_k` (extremos de cada término en la caja del recorte) → `S = Σ(max_k − min_k)` → `puntos_k = 100 · (C_k − min_k)/S`. Los 6 grupos de M-ML-040 se escribieron aparte de `factor_groups.py`.
   - Redondeo sobre el valor exacto: round6 → 1 decimal mitad hacia arriba por factor; `matchScore = floor(round6(Σ) + 1/2)`; `maxPoints` con la regla de los puntos.
   - **Regla de margen** (M-ML-033 decisión 3): falla si un valor queda a menos de 1e-9 de un punto medio de round6. **Coherencia** (constitución §12): falla si |Σ − `matchScore`| > 1/2 + 5e-7, si |Σ puntos redondeados − Σ| > 0.3 + 3e-6 o si `matchScore` ∉ [0, 100].
   - Los casos de bordes de redondeo se construyen resolviendo la distancia con la pendiente exacta del factor y se verifican a menos de 1e-9 del objetivo.
   - Salida determinista: sin reloj (`generated_at` fijo `2026-11-25`), JSON con orden estable, UTF-8, `\n`.
3. **Fixture `tests/fixtures/logreg_golden_cases.json`** (nuevo, en git; 469,987 bytes; SHA-256 `38a589205b29801761b0ab89ca17c82f53f0152c3035043d468794d4849da48a`): **109 casos**. Encabezado: `modelVersion` `logreg-1.0.0`; `derived_from` (`model_id`, `file`, `format` y `sha256` de `logreg.json`); `featureSchemaVersion` 2.0.0; `featureOrder`, `factorOrder`, `factors` (features, `maxPointsRaw`, `maxPoints`, `minLogit`, `maxLogit`); `maxPointsRaw`; `totalSpanLogit`; regla de redondeo; nota .NET (AwayFromZero, ToEven y `-0`); tolerancias; formato; campus; `synthetic: true`; «DATOS SINTÉTICOS». Cada caso trae `id`, `category`, `description`, `input` (`student`/`room` o `features`), `tolerance` y `expected` (`features` sin recortar, `clippedFeatures`, `onnx.factor_points`/`onnx.raw_total` sin redondear y `result` con `breakdown` redondeado y `matchScore`, o `excluded`/`incomplete`).
4. **Prueba `tests/unit/test_logreg_golden_cases.py`** (nueva, 422 pruebas):
   - **Siempre** (sin el modelo): encabezado, formato, cobertura de la §6.4, mismas entradas que los 81 del baseline, coherencia (los puntos y el `matchScore` son la regla aplicada a `expected.onnx`; el desglose suma el total), casos especiales y que el generador no importa el código bajo prueba.
   - **Con el modelo oficial** (se omiten con motivo explícito si no está o si el SHA-256 no coincide): archivo idéntico a `render()`; paquete de Python (`score_pair_logreg`/`compute_pair_features` en las entradas crudas, `score_features_logreg` en los vectores dentro del `range` y `score_matrix` en los 3 vectores fuera del `range` del esquema); onnxruntime CPU (sesión directa sobre el `model.onnx` de `export_model`) con el redondeo del backend reimplementado en la prueba sobre `Fraction` (AwayFromZero y `-0` → `0`).

### Casos (spec §6.4)

| Categoría | Casos | Mínimo de la spec | Contenido |
| --- | --- | --- | --- |
| `baseline_inputs` | 81 | 81 | Entradas de los 81 casos del baseline: 70 puntuados, 6 excluidos (HC01, HC02, HC03, HC01+HC03 y las tres) y 5 incompletos, con `modelVersion` `logreg-1.0.0`. 80 con la entrada idéntica; 1 reemplazado por la regla de margen (T03-H1) |
| `clipping` | 8 | 4 | Crudos: 35 km a pie con `maxDistanceKm` 0.5 (`distance_km`/`effective_distance_km` 35 > 30 y `distance_ratio` 70 > 60); 31 km en auto (solo `distance_km` y `distance_ratio` recortados); control a 29 km. Vectores: en los límites superiores, por encima (= mismo resultado que en el límite), negativos, acotadas fuera de [0, 1] y a 1e-9 fuera de cada límite |
| `extremes` | 3 | 2 | Entradas crudas que dan 100; esquinas de la caja: mejor (100) y peor (0; no alcanzable desde la app, T05-H4) |
| `factor_minimum` | 2 | — («EJECUTA» punto 4) | `place_type` en su mínimo (el `-0` de T02-H1) y 5 factores en su mínimo a la vez |
| `factor_rounding` | 5 | 6 entre las dos | Presupuesto 4.64999994 → 4.65 → **4.7** y 4.24999983 → 4.25 → **4.3** (ToEven o sin round6: 4.6 y 4.2), con `budget.max`/renta enteros; distancia 40.2499996 → **40.3**, 40.2500004 → **40.3** y 40.2499994 → **40.2** |
| `total_rounding` | 3 | (ver arriba) | Construidos con la distancia: 80.49999955 → **81**, 80.4999994 → **80**, 80.5000004 → **81** (ToEven: 80) |
| `t05_h1_zone` | 4 | 3 | `budget.min` 200, `budget.max` 50,000 (razón 250 > 76.2), renta 0/100/199/200: presupuesto sin redondear 9.4459 < 9.4889 < 9.5316 < 9.5320 (sube con la renta; redondeado 9.4 → 9.5) |
| `fixed_term` | 3 | 1 | Mascota con cuarto que la admite (`pets_compatible` 1); el mismo vector con `pets_compatible` 1 y 0 da `factor_points` y `raw_total` idénticos |

98 casos puntuados, 6 excluidos y 5 incompletos. `matchScore` esperados entre 0 y 100.

### Resultados de la comparación (modelo oficial `logreg-1.0.0`, SHA-256 `341ea3ec…0b27fa`)

| Comparación | Casos | máx. \|dif\| `factor_points` | máx. \|dif\| `raw_total` | Dif. puntos redondeados | Dif. `matchScore` | Resultado |
| --- | --- | --- | --- | --- | --- | --- |
| Paquete de Python vs esperado (recuento exacto) | 106 (incluye 6 excluidos y 5 incompletos: mismos `violatedConstraints` / `missingFields`) + 3 fuera del `range` vía `score_matrix` | 1.421e-14 | 2.842e-14 | **0** | **0** | **PASS** |
| onnxruntime CPU vs esperado | 98 puntuados | 1.421e-14 | 2.842e-14 | **0** | **0** | **PASS** |
| Features del paquete (`compute_pair_features`) vs esperado | 95 con entradas crudas (100 crudos − 5 incompletos; incluye los 6 excluidos) | ≤ 1e-6 (tolerancia) | — | — | — | **PASS** |
| `check_parity` de T02 sobre los 98 vectores (informativo) | 98 (95 `score_features_logreg` + 3 `score_matrix`) | 1.421e-14 | 1.421e-14 | **0** | **0** | PASS |

**Ningún caso falló, así que no hubo que detenerse ni ajustar nada.** El ONNX devolvió **15 celdas negativas en 12 casos** (mín. −7.85e-15; p. ej. `place_type` = −2.45e-16 en `factor_minimum_place_type`, donde el recuento da 0 exacto). Con AwayFromZero y `-0` → `0` dan 0.0 y ningún punto sale como `-0` (T02-H1 / M-ML-046).

### Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `sha256sum models/logreg.json` | `341ea3ec705c2f877b36d1b931eafe2dea4630523560fe8764ce59e0a70b27fa` = spec §1 ✅ |
| `cd tests && uv run python build_logreg_golden_cases.py` (3 ejecuciones, la última tras el formateo de ruff) | `109 casos`; SHA-256 del archivo `38a58920…49da48a` las 3 veces — **determinista** ✅ |
| `uv run pytest tests/unit/test_logreg_golden_cases.py -rs` | **422 passed**, **0 omitidas** (corrieron las pruebas con el modelo oficial) |
| Simulación sin el modelo (copia de la prueba con `OFFICIAL_MODELS` apuntando a un directorio inexistente, en el scratchpad) | **213 passed, 209 skipped** con el motivo «models/logreg.json oficial ausente (fuera de git; en CI se usa el perfil ci, ML-06-RK05)»; sin errores de importación |
| `uv run pytest tests/unit/test_logreg_golden_cases.py tests/unit/test_baseline_golden_cases.py tests/test_repo_hygiene.py` (nuevas + afectadas) | **730 passed** en 3.2 s |
| `uv run ruff check src tests` | `All checks passed!` ✅ |
| `uv run ruff format --check src tests` | `123 files already formatted` ✅ |
| `uv run mypy` (estricto; la configuración cubre `src`) | `Success: no issues found in 55 source files` ✅ |
| `uv run mypy --strict tests/build_logreg_golden_cases.py tests/unit/test_logreg_golden_cases.py` (informativo; `tests/` no está en la configuración de mypy) | 0 errores en los dos archivos nuevos; 2 errores `no-any-return` **preexistentes** en módulos importados (`tests/golden_recount.py:140`, `tests/build_baseline_golden_cases.py:133`), fuera del alcance de T03 |
| Búsqueda de asignaciones `key/token/secret/password = "<literal>"` en los archivos nuevos y en el fixture | Ninguna |
| Búsqueda de `pickle`/`joblib`/`dill` en los archivos nuevos | Ninguna |
| `git status --short --ignored models dist` | `!! models/` (ignorado), `?? dist/` (sin versionar; `/dist/` llega en T05) |
| gitleaks | **No ejecutado localmente** (no instalado); lo corre el CI. No se reporta como PASS |
| Suite completa | **No ejecutada** por instrucción del responsable; la corre el CI |
| pip-audit | No aplica: T03 no cambia dependencias |

Plataforma: Windows 11 (10.0.26200), Python 3.12.16, onnxruntime 1.31.1 (`CPUExecutionProvider`).

### Archivos

| Área | Archivo | Cambio |
| --- | --- | --- |
| Pruebas | `tests/build_logreg_golden_cases.py` | Nuevo: generador con recuento independiente |
| Fixture | `tests/fixtures/logreg_golden_cases.json` | Nuevo: 109 casos |
| Pruebas | `tests/unit/test_logreg_golden_cases.py` | Nuevo: 422 pruebas |
| Evidencia | `specs/ml-06-onnx-export/evidence.md` | Encabezado y esta sección |

No se modificaron `pyproject.toml`, `uv.lock`, `.gitignore`, `memoria.md`, `task.md`, `onnx_graph.py`, `parity.py`, `package.py`, `logreg.json`, `feature_schema.json`, `tests/golden_recount.py`, `tests/build_baseline_golden_cases.py`, `baseline_golden_cases.json` ni ningún params.

### Aceptación de T03 (ML-06-R03)

| Criterio | Resultado |
| --- | --- |
| `tests/fixtures/logreg_golden_cases.json` con recuento independiente (sin importar `thehouse_ml`, `Fraction` sobre los binary64 de `logreg.json`) | ✅ (`test_generator_does_not_import_the_code_under_test`) |
| Entradas de los 81 casos del baseline | ✅ 80 idénticas + 1 reemplazada por la regla de margen de la propia §6.4 (**T03-H1, aprobada en M-ML-047**) |
| Recorte ≥ 4, extremos ≥ 2, bordes de redondeo ≥ 6, zona T05-H1 ≥ 3, término fijo ≥ 1 | ✅ 8, 3, 8 (5 + 3), 4 y 3 (`test_coverage_required_by_spec_6_4`) |
| Archivo idéntico a la salida del generador | ✅ (`test_fixture_is_exactly_what_the_independent_recount_produces`; 3 regeneraciones con el mismo SHA-256) |
| Regla de margen | ✅ (el generador falla si se incumple; se cumple en los 98 casos puntuados) |
| Paquete y ONNX coinciden con todos dentro de las tolerancias | ✅ 0 diferencias en puntos redondeados y `matchScore`; máx. \|dif\| 2.8e-14 |
| Omisión con motivo sin el modelo oficial | ✅ (simulación: 209 omitidas con motivo, 213 estructurales en verde) |
| Etiquetado como derivado de `logreg-1.0.0` (model id y SHA-256) | ✅ `derived_from` y `modelVersion` |
| Desglose redondeado coherente con el total (constitución §12) | ✅ en el generador y en `test_rounded_breakdown_is_consistent_with_the_raw_outputs` |
| Paridad de T02 contra `logreg_golden_cases.json` (pendiente T02-H2) | ✅ cubierta por `test_onnxruntime_matches` (98 casos) |

### Hallazgos

| ID | Clasificación | Hallazgo | Acción |
| --- | --- | --- | --- |
| T03-H1 | **Desviación (aprobada, M-ML-047)** | La entrada del caso del baseline `null_all_optional_partial_room` (12 km) deja el factor de distancia de logreg en **53.13840450060224**, a **6.0e-10** del punto medio de round6 53.1384045: incumple la regla de margen (≥ 1e-9). La spec §6.4 pide «las mismas entradas que los 81 casos» y también «se reemplaza la entrada, no se ajusta el esperado» | Se aplicó la regla de margen de la §6.4: caso `null_all_optional_partial_room_margin` con la misma entrada salvo la distancia (12 → 12.01 km), descrito como reemplazo; los otros 80 son idénticos. El esperado no se ajustó. Propuesto en M-ML-047 |
| T03-H2 | Informativo | Los ids `total_x4999994_down` y `total_x5000004_up` ya existen en el baseline (con otros totales); los casos de logreg llevan el prefijo `logreg_` (`logreg_total_x49999955_up`, etc.) | Ninguna |
| T03-H3 | Decisión técnica | Cota de coherencia: con round6 antes del entero, \|Σ − `matchScore`\| puede llegar a 1/2 + 5e-7 (80.49999955 → 81); la primera versión del generador usaba 1/2 y lo rechazaba. Se corrigió la cota del generador (no un esperado). La cota de M-ML-041 (máx. 0.49997 en test) no cambia | Registrado en M-ML-047 |
| T03-H4 | Advertencia (conocida) | onnxruntime: 15 celdas negativas en 12 casos (mín. −7.85e-15), donde el recuento da 0 exacto. Con AwayFromZero y `-0` → `0` no hay diferencias | Regla de M-ML-046 para NV-23 (guía en T06); `factor_minimum_place_type` queda como caso dorado del `-0` |
| T03-H5 | Decisión técnica | 3 casos solo vector (`clip_vector_below_zero`, `clip_vector_bounded_features_out`, `clip_vector_epsilon_outside`) están fuera del `range` del esquema: el backend los rechaza antes del ONNX y `score_features_logreg` también (se prueba). Su referencia de Python es `score_matrix` con el redondeo del paquete, como en T02-H4 | Registrado en M-ML-047; las descripciones lo dicen para NV-23 |
| T03-H6 | Resuelto tras la aprobación | `tests/fixtures/README.md` no menciona `logreg_golden_cases.json`, y el docstring de `tests/integration/test_onnx_parity_official.py` dice «hasta que T03 entregue…» (su conjunto dorado siguen siendo los 70 vectores del baseline). No se tocaron: no están en la autorización | Autorizado en la aprobación: `README.md` documenta `logreg_golden_cases.json` y su generador; el docstring remite a `tests/unit/test_logreg_golden_cases.py`. Tras el cambio: `ruff check`/`ruff format --check` sin errores y 736 pruebas afectadas en verde (`test_logreg_golden_cases`, `test_onnx_parity_official`, `test_baseline_golden_cases`, `test_repo_hygiene`) |
| T03-H7 | Informativo | El fixture depende del modelo oficial y de la distancia en binary64 (haversine con `atan2` de la plataforma). La prueba de identidad solo corre donde está `models/logreg.json` (local); en CI se omite (RK05) y corren las 213 pruebas estructurales | Ninguna; si cambian `logreg.json` o una coordenada de `data/universities.csv`, se regenera |
| T03-H8 | Informativo | Errata de la §6.4 confirmada: las entradas del baseline dan 70 puntuados, **6** excluidos y 5 incompletos (no 9), igual que en T02-H3 | Errata en T07 (M-ML-046) |
| T03-H9 | Control no ejecutado | gitleaks local y suite completa | Revisar el CI del PR |
| T03-H10 | Pendiente para T05 | La etapa `export` aún no copia el fixture como `golden_cases.json` en `dist/model-v1.0.0/` | T05 |

### Decisión registrada (APROBADO, `memoria.md`)

**M-ML-047 — ML-06 T03: casos dorados de `logreg-1.0.0` con recuento independiente**

- **Decisión:** completar **ML-06 T03** (ML-06-R03) con estas elecciones:
  1. **Generador** `tests/build_logreg_golden_cases.py`: no importa `thehouse_ml`, `onnx`, `onnxruntime` ni `numpy`. Reutiliza `tests/golden_recount.py` (features, HC01–HC03 y perfil incompleto) y las entradas de `tests/build_baseline_golden_cases.py`. Lee `models/logreg.json` como datos tras verificar el SHA-256 `341ea3ec…0b27fa` y falla sin él. Calcula paso a paso M-ML-041 con `Fraction` exactas de los binary64 (recorte → escalado → contribución → suma por factor → `min_k`/`max_k`, `S` → calibración) y redondea sobre el valor exacto (round6 → 1 decimal mitad hacia arriba; total `floor(round6(Σ) + 1/2)`). Los 6 grupos de M-ML-040 están escritos aparte. Salida determinista (`generated_at` fijo).
  2. **Fixture** `tests/fixtures/logreg_golden_cases.json` (109 casos): 81 con las entradas del baseline, 8 de recorte, 3 de extremos, 2 de factor en su mínimo (`-0`), 5 de redondeo por factor, 3 de redondeo del total, 4 de la zona T05-H1 y 3 del término fijo. Etiquetado con `modelVersion` y `derived_from` (`model_id`, archivo, formato y SHA-256 de `logreg.json`) y como sintético. Formato de la §6.4 (`expected.features`, `clippedFeatures`, `onnx`, `result`).
  3. **Regla de margen sobre las entradas del baseline (T03-H1):** cuando una entrada del baseline incumple la regla de margen con logreg, se reemplaza la entrada (no el esperado) con un id propio (`<id>_margin`) y se describe el cambio. Hoy solo `null_all_optional_partial_room` (12 → 12.01 km); los otros 80 son idénticos. Esto precisa «las mismas entradas que los 81 casos» de la §6.4.
  4. **Ids** de los totales de logreg con prefijo `logreg_` para no repetir ids del baseline (T03-H2).
  5. **Coherencia del desglose** (constitución §12) en el generador y en la prueba: \|Σ − `matchScore`\| ≤ 1/2 + 5e-7 (por round6, T03-H3), \|Σ puntos − Σ\| ≤ 0.3 + 3e-6 y `matchScore` ∈ [0, 100].
  6. **Prueba** `tests/unit/test_logreg_golden_cases.py`. Sin el modelo corren las pruebas de formato, cobertura, coherencia y no importación. Con el modelo oficial corren la identidad del archivo, el paquete de Python (`score_pair_logreg`, `score_features_logreg`; `score_matrix` solo para los 3 vectores fuera del `range` del esquema, T03-H5) y onnxruntime con el redondeo del backend (AwayFromZero, `-0` → `0`) reimplementado sobre `Fraction`. Sin el modelo se omiten con motivo explícito (RK05).
- **Evidencia:** `specs/ml-06-onnx-export/evidence.md` (T03). 109 casos; paquete y onnxruntime con 0 diferencias en puntos redondeados y `matchScore` y máx. \|dif\| 2.8e-14; 3 regeneraciones idénticas (SHA-256 `38a58920…`); 422 pruebas en verde con el modelo (0 omitidas) y 213 en verde + 209 omitidas con motivo sin él; ruff y mypy estricto sin errores.
- **Pendientes:** copia como `golden_cases.json` en el paquete (T05); nota de `-0` y casos dorados en la guía (T06); errata de la §6.4 y 2 errores de mypy preexistentes en `tests/` (T07); gitleaks y suite completa en el CI del PR.
- **Estado:** APROBADO — César Yael López Trujillo — 2026-11-25 (texto completo en `memoria.md`).
- **Impacto:** T03 de ML-06 lista para cerrarse (3/7) con la aprobación. Sin cambios en el grafo, modelos, `feature_schema.json`, params, dependencias ni `.gitignore`.

### Seguridad, datos y licencias

- Sin pickle, joblib ni dill; sin dependencias nuevas; sin red ni secretos; sin servicios de pago (COST-001 $0, `CPUExecutionProvider`).
- `models/`, `data/` y `dist/` sin versionar (`models/` ignorado; `?? dist/`, `/dist/` en T05). El fixture solo trae parámetros derivados del modelo (`maxPointsRaw`, `minLogit`/`maxLogit`, `totalSpanLogit`), no el modelo ni datos de entrenamiento.
- Sin datos personales: perfiles y cuartos inventados (los mismos campos que `baseline_golden_cases.json`) y coordenadas públicas de campus de `data/universities.csv`.

### Coherencia

| Documento | Comprobación |
| --- | --- |
| `constitution.md` §8, §12, §13.1, §13.4, §15 | Etiquetado como sintético; el desglose suma el total (comprobado en cada caso); sin pickle; paridad Python ↔ runtime con casos dorados; $0 |
| `AGENTS.md` | Sin commit ni push; controles no ejecutados declarados; desviación T03-H1 avisada y aprobada; T03 marcada en `task.md` solo tras el «Apruebo» |
| Spec §6.4 / R03 | Recuento independiente, categorías y mínimos, formato, tolerancias, regla de margen, identidad del archivo y omisión con motivo |
| M-ML-029/031/033 | round6 → mitad hacia arriba; total `floor(round6(Σ) + 0.5)`; regla de margen ≥ 1e-9 |
| M-ML-040 / M-ML-041 | 6 factores con sus features; `pets_compatible` en el término fijo; calibración afín sobre el recorte; desviación T05-H1 conservada y mostrada en 4 casos |
| M-ML-044 / M-ML-045 / M-ML-046 | `onnx_graph.py` y `parity.py` intactos; `-0` → `0` en el redondeo del backend; T02-H2 resuelto |

**Resultado T03: PASS WITH WARNINGS** — T03-H1 (desviación aprobada), T03-H4 (advertencia conocida), T03-H6 y T03-H10 (pendientes fuera de las rutas autorizadas o de otras tasks), T03-H9 (controles no ejecutados). T03 completa con M-ML-047 (APROBADO, César Yael López Trujillo, 2026-11-25).
