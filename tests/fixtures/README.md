# Fixtures de ML-02 T02

Archivos **inventados** y muy pequeños para probar la descarga sin red.
No contienen datos reales del INEGI.

- `envi_2020_base_de_datos_csv.zip`: contiene `Bases de datos/TVIVIENDA.csv` en latin-1 con
  todas las variables confirmadas de ENVI.
- `envi_sin_p5_5.zip`: igual, pero sin `P5_5`; sirve para probar una variable faltante.
- `enigh2022_ns_viviendas_csv.zip`: contiene `viviendas.csv` en UTF-8 con BOM con las
  variables confirmadas de ENIGH.
- `envi_2020_fd.xlsx`: bytes simulados; solo se descarga y se hashea, nunca se lee (T02 no
  necesita openpyxl).

# Casos de ML-03 T02

- `pair_feature_cases.json`: casos por tabla de las features del par (entrada con los nombres de
  `feature_schema.json` y salida esperada con tolerancia 1e-6). Entradas **sintéticas**; las salidas se
  calcularon a mano (cuartos sobre el meridiano del campus). Se reutilizarán como casos dorados del
  backend en ML-06.

# Casos de ML-03 T03

- `hard_constraint_cases.json`: casos por tabla de las restricciones duras (`HC01_budget`, `HC02_living_with`,
  `HC03_pets`): entrada con los nombres de `feature_schema.json` y salida esperada (`allowed` y la lista de
  restricciones violadas en el orden del esquema) o `expected_error` para el quick setup incompleto. Entradas
  **sintéticas**; las salidas se escribieron a mano. Se reutilizarán como casos dorados del backend en ML-06.

# Casos de ML-04 T02

- `pair_feature_cases.json` pasa a **2.0.0** (esquema 2.0.0, M-ML-029): los 35 casos de ML-03 conservan entradas y
  valores de las 7 features y agregan `distance_ratio`, `price_below_min` y `extra_services_ratio`; 13 casos nuevos
  cubren `maxDistanceKm` nulo, distancia igual a `maxDistanceKm` y a la mitad, `budget.min` nulo, renta igual, menor,
  mucho menor y 0 frente a `budget.min`, sin servicios pedidos, los 11 pedidos y cuartos sin o con todos los extras.
  Los valores nuevos salen de un recuento independiente (fracciones exactas y la regla del meridiano), sin el código
  bajo prueba.
- `hard_constraint_cases.json` solo cambia `schema_version` a 2.0.0 (las restricciones y los casos no cambian).

# Casos dorados de ML-04 T05

- `baseline_golden_cases.json`: 81 casos dorados del modelo base `baseline-1.0.0` (esquema 2.0.0) para la paridad
  Python ↔ backend (.NET, NV-23) y, en ML-06, Python ↔ onnxruntime: entrada (estudiante de NV-11 con nulos explícitos y
  cuarto, con los nombres del esquema), las 10 features, y `matchScore` con `breakdown`, `excluded` con
  `violatedConstraints` o `incomplete` con `missingFields` (perfil sin `budget.max`, `livingWith` o `placeType`). Entradas **sintéticas**. Los esperados salen de un recuento independiente
  (`tests/golden_recount.py`, fracciones exactas, sin importar `thehouse_ml`) y se regeneran con
  `uv run python tests/build_baseline_golden_cases.py`; la prueba exige que el archivo sea idéntico a esa salida.
  En ML-06 van en el paquete como `baseline_golden_cases.json` (respaldo en C#).

# Casos dorados de ML-06 T03

- `logreg_golden_cases.json`: 109 casos dorados del modelo promovido `logreg-1.0.0` (esquema 2.0.0) para la paridad
  Python ↔ onnxruntime ↔ backend (.NET, NV-23): las entradas de los 81 casos del baseline (una reemplazada por la
  regla de margen: `null_all_optional_partial_room_margin`, M-ML-047) más recorte, extremos, factores en su mínimo
  (`-0`), bordes de redondeo, zona T05-H1 y término fijo; algunos casos son solo vector (`input.features`). Cada caso
  trae las features sin recortar y recortadas, las salidas esperadas del ONNX sin redondear (`factor_points`,
  `raw_total`) y el JSON del backend (`breakdown` redondeado y `matchScore`, o `excluded`/`incomplete`). El encabezado
  lo identifica como derivado de `logreg-1.0.0` (`derived_from`: id, archivo y SHA-256 de `models/logreg.json`).
  Entradas **sintéticas**; modelo entrenado con datos sintéticos. Los esperados salen de un recuento independiente
  (`tests/build_logreg_golden_cases.py`: reutiliza `tests/golden_recount.py`, lee `models/logreg.json` como datos y
  usa fracciones exactas, sin importar `thehouse_ml`) y se regeneran con
  `cd tests && uv run python build_logreg_golden_cases.py` (requiere el modelo oficial, fuera de git); la prueba
  `tests/unit/test_logreg_golden_cases.py` exige que el archivo sea idéntico a esa salida y lo omite con motivo sin el
  modelo. En ML-06 se empaqueta como `golden_cases.json` junto al ONNX (T05).
