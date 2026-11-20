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
