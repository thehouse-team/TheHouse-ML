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
