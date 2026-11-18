# Entorno de campus — ML-02 T05

Generado por `uv run python -m thehouse_ml prepare`. **Atribución: "Fuente: INEGI, DENUE 05/2026".**
Salida de datos: `data/processed/campus_context.parquet` (fuera de git). INEGI no respalda este proyecto.

Conteos **agregados** de establecimientos a 1 000, 2 000 y 5 000 m de cada campus,
con distancia haversine (el límite exacto del radio sí cuenta). Se usan los dos estados,
Hidalgo (13) y Estado de México (15), porque Tepeji y Jilotepec están cerca de la frontera.
Solo se publican conteos: ni nombres, ni teléfonos, ni correos, ni direcciones.

Categorías por prefijo SCIAN (`codigo_act`): `comida` = 722; `abarrotes_super` = 4611, 4621; `farmacia` = 464111, 464112; `salud` = 621, 622; `papeleria_copias` = 465311, 561431; `lavanderia` = 812210; `gimnasio` = 713943; `transporte` = 485; `bancos` = 522; `total` = todos.

## Conteos a 2 km por universidad y categoría

| Universidad | Código | comida | abarrotes_super | farmacia | salud | papeleria_copias | lavanderia | gimnasio | transporte | bancos | total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Universidad Tecnológica de Tula-Tepeji | `UTTT` | 27 | 54 | 3 | 3 | 10 | 2 | 1 | 1 | 1 | 197 |
| Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Tepeji del Río | `UAEH-ESTE` | 47 | 192 | 6 | 8 | 35 | 5 | 3 | 1 | 0 | 578 |
| Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Tlahuelilpan | `UAEH-ESTL` | 166 | 229 | 21 | 40 | 39 | 6 | 6 | 4 | 5 | 1,311 |
| Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Atotonilco de Tula | `UAEH-ESAT` | 140 | 172 | 23 | 43 | 23 | 5 | 6 | 0 | 2 | 1,016 |
| Instituto Tecnológico Superior del Occidente del Estado de Hidalgo | `ITSOEH` | 343 | 512 | 41 | 112 | 76 | 13 | 18 | 9 | 11 | 2,922 |
| Tecnológico de Estudios Superiores de Jilotepec | `TESJI` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Universidad Mexiquense del Bicentenario, Unidad de Estudios Superiores Jilotepec | `UMB-JILO` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Universidad Mexiquense del Bicentenario, Unidad de Estudios Superiores Villa del Carbón | `UMB-VDC` | 144 | 208 | 16 | 55 | 37 | 4 | 5 | 1 | 12 | 1,366 |
| Universidad Canadiense | `UCAN` | 497 | 567 | 54 | 194 | 76 | 40 | 20 | 14 | 57 | 3,808 |
| Universidad del Nuevo México | `UNE-TULA` | 196 | 218 | 20 | 48 | 39 | 20 | 7 | 5 | 1 | 1,168 |

## Universidades con coordenada verificada

### `UTTT` — Universidad Tecnológica de Tula-Tepeji

- `final_status`: `ok_osm`; coordenada final: `20.0084106, -99.3449608`.

| Radio (m) | comida | abarrotes_super | farmacia | salud | papeleria_copias | lavanderia | gimnasio | transporte | bancos | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 2,000 | 27 | 54 | 3 | 3 | 10 | 2 | 1 | 1 | 1 | 197 |
| 5,000 | 390 | 630 | 38 | 115 | 106 | 37 | 16 | 14 | 11 | 2,830 |

### `UAEH-ESTE` — Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Tepeji del Río

- `final_status`: `ok_manual`; coordenada final: `19.91703, -99.36116`.

| Radio (m) | comida | abarrotes_super | farmacia | salud | papeleria_copias | lavanderia | gimnasio | transporte | bancos | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 2 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 7 |
| 2,000 | 47 | 192 | 6 | 8 | 35 | 5 | 3 | 1 | 0 | 578 |
| 5,000 | 418 | 774 | 49 | 114 | 103 | 24 | 18 | 5 | 18 | 3,426 |

### `UAEH-ESTL` — Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Tlahuelilpan

- `final_status`: `corregida_manual`; coordenada final: `20.132062, -99.235535`.

| Radio (m) | comida | abarrotes_super | farmacia | salud | papeleria_copias | lavanderia | gimnasio | transporte | bancos | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 132 | 149 | 17 | 31 | 30 | 6 | 6 | 3 | 5 | 967 |
| 2,000 | 166 | 229 | 21 | 40 | 39 | 6 | 6 | 4 | 5 | 1,311 |
| 5,000 | 459 | 803 | 77 | 147 | 129 | 15 | 16 | 7 | 18 | 4,237 |

### `UAEH-ESAT` — Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Atotonilco de Tula

- `final_status`: `ok_manual`; coordenada final: `20.01584, -99.23601`.

| Radio (m) | comida | abarrotes_super | farmacia | salud | papeleria_copias | lavanderia | gimnasio | transporte | bancos | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 31 | 35 | 6 | 11 | 7 | 2 | 2 | 0 | 0 | 193 |
| 2,000 | 140 | 172 | 23 | 43 | 23 | 5 | 6 | 0 | 2 | 1,016 |
| 5,000 | 439 | 641 | 55 | 94 | 83 | 28 | 17 | 3 | 5 | 3,287 |

### `ITSOEH` — Instituto Tecnológico Superior del Occidente del Estado de Hidalgo

- `final_status`: `corregida_manual`; coordenada final: `20.2333739, -99.209633`.

| Radio (m) | comida | abarrotes_super | farmacia | salud | papeleria_copias | lavanderia | gimnasio | transporte | bancos | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 236 | 307 | 34 | 89 | 49 | 11 | 14 | 5 | 9 | 2,007 |
| 2,000 | 343 | 512 | 41 | 112 | 76 | 13 | 18 | 9 | 11 | 2,922 |
| 5,000 | 670 | 967 | 84 | 201 | 149 | 25 | 27 | 10 | 21 | 5,292 |

### `UMB-JILO` — Universidad Mexiquense del Bicentenario, Unidad de Estudios Superiores Jilotepec

- `final_status`: `ok_manual`; coordenada final: `19.91248, -99.57849`.

| Radio (m) | comida | abarrotes_super | farmacia | salud | papeleria_copias | lavanderia | gimnasio | transporte | bancos | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 2,000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 5,000 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 10 |

### `UNE-TULA` — Universidad del Nuevo México

- `final_status`: `ok_osm`; coordenada final: `20.0310934, -99.3363626`.

| Radio (m) | comida | abarrotes_super | farmacia | salud | papeleria_copias | lavanderia | gimnasio | transporte | bancos | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 100 | 114 | 12 | 27 | 19 | 11 | 6 | 2 | 1 | 589 |
| 2,000 | 196 | 218 | 20 | 48 | 39 | 20 | 7 | 5 | 1 | 1,168 |
| 5,000 | 813 | 1,029 | 79 | 268 | 152 | 72 | 38 | 19 | 63 | 6,017 |

## Universidades con coordenada sin verificar (`seed_sin_verificar`)

### `TESJI` — Tecnológico de Estudios Superiores de Jilotepec

- `final_status`: `seed_sin_verificar`; coordenada final: `19.91953, -99.57719`.

| Radio (m) | comida | abarrotes_super | farmacia | salud | papeleria_copias | lavanderia | gimnasio | transporte | bancos | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 2,000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 5,000 | 5 | 10 | 0 | 4 | 0 | 1 | 0 | 0 | 1 | 68 |

### `UMB-VDC` — Universidad Mexiquense del Bicentenario, Unidad de Estudios Superiores Villa del Carbón

- `final_status`: `seed_sin_verificar`; coordenada final: `19.72791, -99.46312`.

| Radio (m) | comida | abarrotes_super | farmacia | salud | papeleria_copias | lavanderia | gimnasio | transporte | bancos | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 93 | 134 | 11 | 45 | 24 | 3 | 4 | 1 | 12 | 958 |
| 2,000 | 144 | 208 | 16 | 55 | 37 | 4 | 5 | 1 | 12 | 1,366 |
| 5,000 | 160 | 279 | 19 | 60 | 50 | 4 | 6 | 1 | 12 | 1,592 |

### `UCAN` — Universidad Canadiense

- `final_status`: `seed_sin_verificar`; coordenada final: `20.0517, -99.34507`.

| Radio (m) | comida | abarrotes_super | farmacia | salud | papeleria_copias | lavanderia | gimnasio | transporte | bancos | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 342 | 389 | 41 | 155 | 49 | 23 | 15 | 10 | 39 | 2,656 |
| 2,000 | 497 | 567 | 54 | 194 | 76 | 40 | 20 | 14 | 57 | 3,808 |
| 5,000 | 763 | 953 | 78 | 254 | 135 | 69 | 36 | 18 | 60 | 5,742 |

> **Advertencia:** la coordenada de estas universidades no está verificada. Se mantienen las
> del seed y no deben usarse como coordenadas verificadas hasta que el backend confirme el
> nombre y la ubicación (en particular `UCAN`). Sus conteos son solo orientativos.
