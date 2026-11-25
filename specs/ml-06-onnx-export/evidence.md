# ML-06 — Evidencia de ejecución

| Campo | Valor |
| --- | --- |
| Spec | ML-06 1.0.0 (M-ML-044) |
| Tareas ejecutadas | T01 · Dependencias y exportación verificada (ML-06-R01) |
| Estado | **T01 completa — PASS WITH WARNINGS** (M-ML-045, APROBADO 2026-11-24) |
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
