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
