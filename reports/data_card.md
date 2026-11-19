# Data card — ML-02

Generada automáticamente por `uv run python -m thehouse_ml prepare`
(`thehouse_ml.data_card.render_data_card`). Todas las cifras proceden de
`data/sources.toml`, `data/manifests/`, `data/universities.csv` y los Parquet de
`data/processed/`; ninguna está escrita a mano.

## 1. Propósito y alcance

Datos públicos para el **Smart Match** de TheHouse en la zona de Tula/Tepeji:
7 municipios (5 en Hidalgo, 2 en Estado de México) y 10 universidades.
Las fuentes describen viviendas y su entorno, no cuartos: el catálogo de cuartos será
**sintético en ML-03**, calibrado con estas bases y etiquetado como tal (M-ML-019).

## 2. Fuentes

Tamaños y SHA-256 se leen de los manifiestos de `data/manifests/`.

| id | producto | edición | URL | licencia | archivos (tamaño y SHA-256) | estado | decisión |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `inegi_censo_2020_viviendas_ca` | INEGI — Censo de Población y Vivienda 2020, muestra del cuestionario ampliado (Viviendas_CA) | Censo 2020, muestra del cuestionario ampliado, tabla Viviendas_CA | https://www.inegi.org.mx/programas/ccpv/2020/#microdatos | Términos de libre uso de la información del INEGI (citar a INEGI como autor y el producto; no sugerir respaldo de INEGI) | `Censo2020_CA_hgo_csv.zip`: 17,225,011 B, `5d0ff2b338acb93f8558ef2fed0500c1b4af004cdfc5ee34fb1fec7faa21e865`<br>`Censo2020_CA_mex_csv.zip`: 41,189,169 B, `178a2a5c3a0b7b08a1ac878a5e94ca4e6c9b9d5851d98b8b07fe2f9d35672c2b`<br>`diccionario_cuestionario_ampliado_cpv2020.xlsx`: 95,642 B, `e728cdac1872be856ba15c2902a77fb6c03e60c2d98619b7ae8988cc82afb6dc`<br>`Censo2020_clasificaciones_CPV.xlsx`: 210,951 B, `64c7d6974392132258d77b8697e92cb9f39ba4cd303fc5a9c0f32861f174f381` | confirmed | M-ML-015 |
| `inegi_denue_vigente` | INEGI — Directorio Estadístico Nacional de Unidades Económicas (DENUE) | DENUE 05/2026 | https://www.inegi.org.mx/app/descarga/?ti=6 | Términos de Libre Uso de la Información del INEGI | `denue_13_csv.zip`: 15,059,273 B, `d0747958bc364177ea8d281047334cffd4d3505247f43967e08da2ca9c1d9ca5`<br>`denue_15_1_csv.zip`: 50,572,588 B, `23021fc24a2046e1aa3ad8c21137ca57988ded26b6f293ec0c929f59dd2cb1ae`<br>`denue_15_2_csv.zip`: 30,236,017 B, `ad53e5fb65f15b491e140a80e79b207b95eb0320e4a69822d5a3cefb7652bc20` | confirmed | M-ML-018 |
| `inegi_enigh_2022` | INEGI — Encuesta Nacional de Ingresos y Gastos de los Hogares (ENIGH) 2022 | ENIGH 2022 | https://www.inegi.org.mx/programas/enigh/nc/2022/ | Términos de libre uso de la información del INEGI (citar a INEGI como autor y el producto; no sugerir respaldo de INEGI) | `enigh2022_ns_viviendas_csv.zip`: 1,712,569 B, `6ebd189feb30657dda1ebc95ab8815d13717dac4022d95672ec17ccf56095bd4` | confirmed | M-ML-013, M-ML-014 |
| `inegi_envi_2020` | INEGI — Encuesta Nacional de Vivienda (ENVI) 2020 | ENVI 2020 (INEGI, Infonavit, SHF) | https://www.inegi.org.mx/programas/envi/2020/ | Términos de libre uso de la información del INEGI (citar a INEGI como autor y el producto; no sugerir respaldo de INEGI) | `envi_2020_base_de_datos_csv.zip`: 7,322,530 B, `36b8e3d04f37722974e0d48edb2b4708b4a5e55c9c3faa551b9b75fda95cc40b`<br>`envi_2020_fd.xlsx`: 242,100 B, `208fc3d2c70a4cc2b346479813fd98c0481fa5c490d5c8bc2ca8970694befbcf` | confirmed | M-ML-013, M-ML-014 |
| `inegi_inpc` | INEGI — Índice Nacional de Precios al Consumidor (INPC) | INPC base segunda quincena de julio de 2018 = 100 (serie vigente); indicador mensual | https://www.inegi.org.mx/programas/inpc/2018/ | Términos de libre uso de la información del INEGI (citar a INEGI como autor y el producto; no sugerir respaldo de INEGI) | `inpc_indicador_mensual_csv.zip`: 33,157 B, `dc1aeb86aed8bfcbbc4661a3da385b67fb074f8d5c62bd53df22752d54ae24af` | confirmed | M-ML-014 |
| `inside_airbnb_cdmx_2026_09_23` | Inside Airbnb — Ciudad de México (DESCARTADA) | Instantánea Ciudad de México 2026-09-23 — DESCARTADA (M-ML-019) | https://insideairbnb.com/get-the-data/ | Creative Commons Attribution 4.0 International (CC BY 4.0); una descarga, sin scraping ni republicar microdatos | — (descartada, sin descarga ni hash) | DESCARTADA (M-ML-019) | M-ML-019 |
| `openstreetmap` | OpenStreetMap | Datos colaborativos vigentes al momento de verificación de campus | https://www.openstreetmap.org/ | Open Database License (ODbL) 1.0; atribución y compartir-igual para bases derivadas | `UTTT.json`: 561 B, `44be6d20cd3dc4bbf0c3410ede622dad88548827dd7f28133e3d8d649d07e4c8`<br>`UAEH-ESTE.json`: 2 B, `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`<br>`UAEH-ESTL.json`: 2 B, `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`<br>`UAEH-ESAT.json`: 2 B, `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`<br>`ITSOEH.json`: 627 B, `a9b8b59321c2b2f6b1e5e9e87015985228b5c07ffe0a3ad234623f4b03297261`<br>`TESJI.json`: 2 B, `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`<br>`UMB-JILO.json`: 2 B, `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`<br>`UMB-VDC.json`: 2 B, `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`<br>`UCAN.json`: 2 B, `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`<br>`UNE-TULA.json`: 549 B, `7d2f84cba43a20069e93fc7e706af71b2a535ede76aef80ae11dcc79c9aac411` | confirmed | M-ML-016, M-ML-017 |
| `thehouse_seed_nv07_nv08_nv11` | Seed de TheHouse (NV-07, NV-08, NV-11) | Seed de universidades y rangos de referencia definido en NV-07, NV-08 y NV-11 del backend | https://github.com/thehouse-team/TheHouse-Backend | Datos propios del proyecto TheHouse | — (sin manifiesto) | pending | M-ML-016 |

## 3. Atribuciones

Se cita a cada productor como autor de su producto, sin sugerir que respalda el proyecto.
- Fuente: INEGI, Censo de Población y Vivienda 2020, muestra del cuestionario ampliado.
- Fuente: INEGI, DENUE 05/2026
- Fuente: INEGI, Encuesta Nacional de Ingresos y Gastos de los Hogares (ENIGH) 2022.
- Fuente: INEGI, Encuesta Nacional de Vivienda (ENVI) 2020.
- Fuente: INEGI, Índice Nacional de Precios al Consumidor (INPC), indicador mensual.
- © OpenStreetMap contributors (ODbL).
- INEGI no respalda este proyecto: solo se le cita como autor de los productos anteriores.

## 4. Ponderación y deflactación

Toda estimación de encuesta usa su factor de expansión; los conteos sin ponderar solo
deciden si la muestra alcanza (spec §6, M-ML-003).

| Encuesta | factor de expansión | diseño (estrato / UPM) |
| --- | --- | --- |
| `inegi_censo_2020_viviendas_ca` | `FACTOR` | `ESTRATO` / `UPM` |
| `inegi_envi_2020` | `FACTOR` | `EST_DIS` / `UPM_DIS` |
| `inegi_enigh_2022` | `factor` | `est_dis` / `upm` |

Las rentas se expresan en **pesos de jul-2024** (último mes del INPC de datos
abiertos; M-ML-014). El deflactor es el INPC general y sale de `national_baselines.parquet`:

| Fuente | periodo del levantamiento | deflactor |
| --- | --- | ---: |
| `inegi_envi_2020` | 2020-10, 2020-11, 2020-12 | 1.248112 |
| `inegi_enigh_2022` | 2022-08, 2022-09, 2022-10, 2022-11 | 1.088793 |

## 5. Niveles de respaldo

Regla de suficiencia: al menos **30 viviendas rentadas en muestra** y
**CV ≤ 30 %**. Si no se cumple, se usa el estado y, si tampoco, el país
(M-ML-003, M-ML-015). Cada fila registra su nivel y su fuente.

| Municipio | indicador | nivel | fuente | estimación | viviendas en muestra | CV |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Atotonilco de Tula | agua_dentro | municipio | Censo | 81.6 % | 204 | 3.5 % |
| Atotonilco de Tula | agua_entubada | municipio | Censo | 98.2 % | 204 | 0.7 % |
| Atotonilco de Tula | drenaje | municipio | Censo | 99.2 % | 204 | 0.3 % |
| Atotonilco de Tula | electricidad | municipio | Censo | 99.9 % | 204 | 0.1 % |
| Atotonilco de Tula | gas | municipio | Censo | 94.8 % | 204 | 0.6 % |
| Atotonilco de Tula | internet | municipio | Censo | 44.7 % | 204 | 5.2 % |
| Atotonilco de Tula | renta | estado | ENVI | $1,872 | 164 | — |
| Atotonilco de Tula | rentadas | municipio | Censo | 11.6 % | 204 | 10.1 % |
| Mixquiahuala de Juárez | agua_dentro | municipio | Censo | 76.5 % | 211 | 3.5 % |
| Mixquiahuala de Juárez | agua_entubada | municipio | Censo | 96.4 % | 211 | 2.3 % |
| Mixquiahuala de Juárez | drenaje | municipio | Censo | 95.8 % | 211 | 2.2 % |
| Mixquiahuala de Juárez | electricidad | municipio | Censo | 97.4 % | 211 | 2.0 % |
| Mixquiahuala de Juárez | gas | municipio | Censo | 91.2 % | 211 | 2.0 % |
| Mixquiahuala de Juárez | internet | municipio | Censo | 43.0 % | 211 | 4.0 % |
| Mixquiahuala de Juárez | renta | estado | ENVI | $1,872 | 164 | — |
| Mixquiahuala de Juárez | rentadas | municipio | Censo | 8.6 % | 211 | 11.2 % |
| Tepeji del Río de Ocampo | agua_dentro | municipio | Censo | 60.6 % | 385 | 6.2 % |
| Tepeji del Río de Ocampo | agua_entubada | municipio | Censo | 98.8 % | 385 | 0.3 % |
| Tepeji del Río de Ocampo | drenaje | municipio | Censo | 97.8 % | 385 | 0.6 % |
| Tepeji del Río de Ocampo | electricidad | municipio | Censo | 99.5 % | 385 | 0.3 % |
| Tepeji del Río de Ocampo | gas | municipio | Censo | 92.6 % | 385 | 0.8 % |
| Tepeji del Río de Ocampo | internet | municipio | Censo | 37.6 % | 385 | 6.0 % |
| Tepeji del Río de Ocampo | renta | estado | ENVI | $1,872 | 164 | — |
| Tepeji del Río de Ocampo | rentadas | municipio | Censo | 12.5 % | 385 | 14.7 % |
| Tlahuelilpan | agua_dentro | municipio | Censo | 80.1 % | 134 | 2.3 % |
| Tlahuelilpan | agua_entubada | municipio | Censo | 99.5 % | 134 | 0.3 % |
| Tlahuelilpan | drenaje | municipio | Censo | 99.4 % | 134 | 0.2 % |
| Tlahuelilpan | electricidad | municipio | Censo | 99.5 % | 134 | 0.2 % |
| Tlahuelilpan | gas | municipio | Censo | 96.6 % | 134 | 0.7 % |
| Tlahuelilpan | internet | municipio | Censo | 42.6 % | 134 | 6.8 % |
| Tlahuelilpan | renta | estado | ENVI | $1,872 | 164 | — |
| Tlahuelilpan | rentadas | municipio | Censo | 9.8 % | 134 | 13.6 % |
| Tula de Allende | agua_dentro | municipio | Censo | 80.7 % | 364 | 3.9 % |
| Tula de Allende | agua_entubada | municipio | Censo | 98.6 % | 364 | 0.5 % |
| Tula de Allende | drenaje | municipio | Censo | 98.5 % | 364 | 0.5 % |
| Tula de Allende | electricidad | municipio | Censo | 99.3 % | 364 | 0.5 % |
| Tula de Allende | gas | municipio | Censo | 95.5 % | 364 | 0.9 % |
| Tula de Allende | internet | municipio | Censo | 46.9 % | 364 | 5.1 % |
| Tula de Allende | renta | estado | ENVI | $1,872 | 164 | — |
| Tula de Allende | rentadas | municipio | Censo | 11.1 % | 364 | 10.8 % |
| Jilotepec | agua_dentro | municipio | Censo | 65.5 % | 138 | 8.3 % |
| Jilotepec | agua_entubada | municipio | Censo | 93.0 % | 138 | 1.8 % |
| Jilotepec | drenaje | municipio | Censo | 90.9 % | 138 | 2.0 % |
| Jilotepec | electricidad | municipio | Censo | 97.8 % | 138 | 0.8 % |
| Jilotepec | gas | municipio | Censo | 90.2 % | 138 | 1.7 % |
| Jilotepec | internet | municipio | Censo | 35.1 % | 138 | 14.0 % |
| Jilotepec | renta | estado | ENVI | $2,496 | 271 | — |
| Jilotepec | rentadas | municipio | Censo | 7.8 % | 138 | 21.6 % |
| Villa del Carbón | agua_dentro | municipio | Censo | 41.1 % | 94 | 9.8 % |
| Villa del Carbón | agua_entubada | municipio | Censo | 97.7 % | 94 | 0.5 % |
| Villa del Carbón | drenaje | municipio | Censo | 87.6 % | 94 | 2.4 % |
| Villa del Carbón | electricidad | municipio | Censo | 97.5 % | 94 | 0.7 % |
| Villa del Carbón | gas | municipio | Censo | 72.7 % | 94 | 4.9 % |
| Villa del Carbón | internet | municipio | Censo | 18.8 % | 94 | 11.3 % |
| Villa del Carbón | renta | estado | ENVI | $2,496 | 271 | — |
| Villa del Carbón | rentadas | municipio | Censo | 5.0 % | 94 | 21.8 % |

El CV se muestra solo para proporciones (linealización de Taylor con ESTRATO/UPM). La renta es una mediana ponderada; su CV no se calcula y se toma del nivel estatal porque ningún municipio alcanza la suficiencia.

## 6. Tamaños de muestra

Viviendas en muestra (conteo sin ponderar) y viviendas expandidas (suma del factor).
Las celdas con menos de 30 viviendas en muestra se marcan `suprimido`
y no publican ninguna de las dos cifras (spec §8).

| Fuente | ámbito | viviendas en muestra | viviendas expandidas |
| --- | --- | ---: | ---: |
| `inegi_enigh_2022` | 01 | 444 | 73,967 |
| `inegi_enigh_2022` | 02 | 749 | 290,896 |
| `inegi_enigh_2022` | 03 | 470 | 57,627 |
| `inegi_enigh_2022` | 04 | 236 | 28,404 |
| `inegi_enigh_2022` | 05 | 564 | 152,182 |
| `inegi_enigh_2022` | 06 | 597 | 57,304 |
| `inegi_enigh_2022` | 07 | 228 | 149,927 |
| `inegi_enigh_2022` | 08 | 507 | 165,249 |
| `inegi_enigh_2022` | 09 | 508 | 681,204 |
| `inegi_enigh_2022` | 10 | 244 | 57,661 |
| `inegi_enigh_2022` | 11 | 308 | 228,046 |
| `inegi_enigh_2022` | 12 | 182 | 79,006 |
| `inegi_enigh_2022` | Hidalgo | 213 | 88,536 |
| `inegi_enigh_2022` | 14 | 574 | 587,926 |
| `inegi_enigh_2022` | Estado de México | 387 | 706,980 |
| `inegi_enigh_2022` | 16 | 294 | 195,889 |
| `inegi_enigh_2022` | 17 | 312 | 79,704 |
| `inegi_enigh_2022` | 18 | 407 | 73,367 |
| `inegi_enigh_2022` | 19 | 428 | 294,526 |
| `inegi_enigh_2022` | 20 | 122 | 60,933 |
| `inegi_enigh_2022` | 21 | 253 | 228,015 |
| `inegi_enigh_2022` | 22 | 445 | 110,280 |
| `inegi_enigh_2022` | 23 | 579 | 151,200 |
| `inegi_enigh_2022` | 24 | 250 | 91,125 |
| `inegi_enigh_2022` | 25 | 363 | 87,620 |
| `inegi_enigh_2022` | 26 | 288 | 124,169 |
| `inegi_enigh_2022` | 27 | 197 | 60,115 |
| `inegi_enigh_2022` | 28 | 356 | 190,540 |
| `inegi_enigh_2022` | 29 | 236 | 37,638 |
| `inegi_enigh_2022` | 30 | 254 | 276,778 |
| `inegi_enigh_2022` | 31 | 156 | 48,477 |
| `inegi_enigh_2022` | 32 | 286 | 50,882 |
| `inegi_enigh_2022` | Nacional | 11,437 | 5,566,173 |
| `inegi_enigh_2022` | tamaño de localidad 1 | 6,384 | 3,633,685 |
| `inegi_enigh_2022` | tamaño de localidad 2 | 2,156 | 1,027,793 |
| `inegi_enigh_2022` | tamaño de localidad 3 | 1,280 | 574,919 |
| `inegi_enigh_2022` | tamaño de localidad 4 | 1,617 | 329,776 |
| `inegi_enigh_2022` | tipo de vivienda no especificado (&) | 52 | 29,643 |
| `inegi_enigh_2022` | tipo de vivienda 1 | 9,294 | 4,161,291 |
| `inegi_enigh_2022` | tipo de vivienda 2 | 993 | 762,980 |
| `inegi_enigh_2022` | tipo de vivienda 3 | 928 | 510,812 |
| `inegi_enigh_2022` | tipo de vivienda 4 | 40 | 22,401 |
| `inegi_enigh_2022` | tipo de vivienda 5 | 130 | 79,046 |
| `inegi_envi_2020` | 01 | 334 | 74,352 |
| `inegi_envi_2020` | 02 | 394 | 254,058 |
| `inegi_envi_2020` | 03 | 387 | 57,527 |
| `inegi_envi_2020` | 04 | 157 | 24,226 |
| `inegi_envi_2020` | 05 | 227 | 131,229 |
| `inegi_envi_2020` | 06 | 413 | 57,666 |
| `inegi_envi_2020` | 07 | 234 | 192,955 |
| `inegi_envi_2020` | 08 | 275 | 184,666 |
| `inegi_envi_2020` | 09 | 383 | 668,517 |
| `inegi_envi_2020` | 10 | 180 | 53,541 |
| `inegi_envi_2020` | 11 | 266 | 233,552 |
| `inegi_envi_2020` | 12 | 157 | 86,788 |
| `inegi_envi_2020` | Hidalgo | 164 | 81,275 |
| `inegi_envi_2020` | 14 | 442 | 599,269 |
| `inegi_envi_2020` | Estado de México | 271 | 739,380 |
| `inegi_envi_2020` | 16 | 275 | 213,281 |
| `inegi_envi_2020` | 17 | 246 | 84,507 |
| `inegi_envi_2020` | 18 | 315 | 68,941 |
| `inegi_envi_2020` | 19 | 284 | 289,486 |
| `inegi_envi_2020` | 20 | 112 | 72,356 |
| `inegi_envi_2020` | 21 | 228 | 235,598 |
| `inegi_envi_2020` | 22 | 265 | 101,846 |
| `inegi_envi_2020` | 23 | 427 | 143,900 |
| `inegi_envi_2020` | 24 | 225 | 107,629 |
| `inegi_envi_2020` | 25 | 166 | 87,207 |
| `inegi_envi_2020` | 26 | 229 | 126,896 |
| `inegi_envi_2020` | 27 | 149 | 59,773 |
| `inegi_envi_2020` | 28 | 270 | 182,328 |
| `inegi_envi_2020` | 29 | 171 | 33,228 |
| `inegi_envi_2020` | 30 | 171 | 247,240 |
| `inegi_envi_2020` | 31 | 160 | 72,739 |
| `inegi_envi_2020` | 32 | 193 | 53,458 |
| `inegi_envi_2020` | Nacional | 8,170 | 5,619,414 |
| `inegi_envi_2020` | tamaño de localidad 1 | 5,048 | 3,700,647 |
| `inegi_envi_2020` | tamaño de localidad 2 | 1,668 | 1,092,482 |
| `inegi_envi_2020` | tamaño de localidad 3 | 987 | 531,144 |
| `inegi_envi_2020` | tamaño de localidad 4 | 467 | 295,141 |
| `inegi_envi_2020` | tipo de vivienda 1 | 5,291 | 3,146,223 |
| `inegi_envi_2020` | tipo de vivienda 2 | 1,111 | 947,319 |
| `inegi_envi_2020` | tipo de vivienda 3 | 267 | 220,405 |
| `inegi_envi_2020` | tipo de vivienda 4 | 765 | 744,577 |
| `inegi_envi_2020` | tipo de vivienda 5 | 719 | 551,450 |
| `inegi_envi_2020` | tipo de vivienda 6 | suprimido | suprimido |
| `inegi_envi_2020` | tipo de vivienda 7 | suprimido | suprimido |

Los tamaños por municipio están en la sección 5; el detalle completo vive en
`data/processed/national_baselines.parquet` y `data/processed/regional_calibration.parquet` (fuera de git).

## 7. Histogramas

Barras en texto: la longitud es proporcional a la frecuencia del intervalo más frecuente.
Los intervalos con menos de 30 viviendas en muestra se agrupan con los
adyacentes antes de dibujarse; la suma no cambia.

### Renta mensual ponderada ENVI (pesos de jul-2024)

```text
# Nacional — ENVI 2020, intervalos de 500 pesos
0-499        | ██                                             48,347 (n=90)
500-999      | ███████████████████                           467,614 (n=753)
1,000-1,499  | ████████████████████████████████████████      980,850 (n=1,458)
1,500-1,999  | ███████████████████████████████████████       953,667 (n=1,479)
2,000-2,499  | ██████████████████████████████████            844,209 (n=1,305)
2,500-2,999  | ███████                                       175,521 (n=236)
3,000-3,499  | █████████████████████                         505,335 (n=730)
3,500-3,999  | ███████████████                               365,071 (n=527)
4,000-4,499  | █████████                                     225,071 (n=304)
4,500-4,999  | █████████                                     221,237 (n=285)
5,000-5,999  | ████                                          109,664 (n=145)
6,000-6,499  | ██████                                        146,345 (n=202)
6,500-6,999  | ██                                             41,745 (n=51)
7,000-7,499  | █████                                         114,348 (n=136)
7,500-8,499  | █                                              32,549 (n=45)
8,500-8,999  | ██                                             57,299 (n=75)
9,000-9,999  | ███                                            74,855 (n=94)
10,000+      | ██████████                                    255,687 (n=255)
Total expandido |                                             5,619,414
```

Las rentas se reportan en cifras redondas (2,000, 2,500…); al deflactarlas a pesos de jul-2024 (factor 1.248112) se desplazan de intervalo (2,500 → 3,120), por lo que algunos intervalos quedan casi vacíos. No es un error de procesamiento

```text
# Hidalgo — ENVI 2020, intervalos de 500 pesos
500-1,499    | ████████████████████████████████████████       32,597 (n=65)
1,500-1,999  | ████████████████████                           16,127 (n=31)
2,000-7,499  | ████████████████████████████████████████       32,551 (n=68)
Total expandido |                                                81,275
```

```text
# Estado de México — ENVI 2020, intervalos de 500 pesos
500-1,499    | ████████████████████████████████████          165,766 (n=68)
1,500-1,999  | ██████████████████████████████                137,990 (n=51)
2,000-2,499  | ████████████████████████████                  129,732 (n=49)
2,500-3,499  | ███████████████████████████                   121,903 (n=45)
3,500+       | ████████████████████████████████████████      183,989 (n=58)
Total expandido |                                               739,380
```

### Negocios totales a 2,000 m por universidad (DENUE)

```text
#  Universidad |                                          negocios totales
ITSOEH       | ███████████████████████████████               2,922
TESJI        | █                                                 1
UAEH-ESAT    | ███████████                                   1,016
UAEH-ESTE    | ██████                                          578
UAEH-ESTL    | ██████████████                                1,311
UCAN         | ████████████████████████████████████████      3,808
UMB-JILO     | █                                                 1
UMB-VDC      | ██████████████                                1,366
UNE-TULA     | ████████████                                  1,168
UTTT         | ██                                              197
```

## 8. Universidades

Coordenada final y procedencia según la verificación de la T07 (M-ML-016, M-ML-017).

| code | universidad | final_status | fuente de la coordenada | latitud | longitud |
| --- | --- | --- | --- | ---: | ---: |
| `ITSOEH` | Instituto Tecnológico Superior del Occidente del Estado de Hidalgo | `corregida_manual` | nominatim T07 | 20.2333739 | -99.2096330 |
| `TESJI` | Tecnológico de Estudios Superiores de Jilotepec | `seed_sin_verificar` | seed | 19.91953 | -99.57719 |
| `UAEH-ESAT` | Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Atotonilco de Tula | `ok_manual` | seed | 20.01584 | -99.23601 |
| `UAEH-ESTE` | Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Tepeji del Río | `ok_manual` | seed | 19.91703 | -99.36116 |
| `UAEH-ESTL` | Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Tlahuelilpan | `corregida_manual` | way/550950316 | 20.132062 | -99.235535 |
| `UCAN` | Universidad Canadiense | `seed_sin_verificar` | seed | 20.0517 | -99.34507 |
| `UMB-JILO` | Universidad Mexiquense del Bicentenario, Unidad de Estudios Superiores Jilotepec | `ok_manual` | seed | 19.91248 | -99.57849 |
| `UMB-VDC` | Universidad Mexiquense del Bicentenario, Unidad de Estudios Superiores Villa del Carbón | `seed_sin_verificar` | seed | 19.72791 | -99.46312 |
| `UNE-TULA` | Universidad del Nuevo México | `ok_osm` | osm:way/458178077 | 20.0310934 | -99.3363626 |
| `UTTT` | Universidad Tecnológica de Tula-Tepeji | `ok_osm` | osm:way/386380501 | 20.0084106 | -99.3449608 |

### Advertencia

**3 universidades sin verificar:** `TESJI`, `UMB-VDC`, `UCAN`.
Conservan la coordenada del seed y no deben tratarse como verificadas; el backend
debe confirmar su nombre y ubicación (en particular `UCAN`). Sus conteos son orientativos.

## 9. Transformaciones

| etapa | entrada | salida |
| --- | --- | --- |
| `download` | `data/sources.toml` | `data/manifests/inegi_censo_2020_viviendas_ca.json`, `data/manifests/inegi_denue_vigente.json`, `data/manifests/inegi_enigh_2022.json`, `data/manifests/inegi_envi_2020.json`, `data/manifests/inegi_inpc.json`, `data/manifests/openstreetmap.json`, `reports/variables_check.md` |
| `prepare` nacional | ENVI 2020, ENIGH 2022 e INPC (en `data/raw/`) | `data/processed/national_baselines.parquet`, `data/processed/rent_histogram.parquet`, `reports/national_baselines.md` |
| `prepare` regional | Censo 2020 y `data/processed/national_baselines.parquet` | `data/processed/regional_calibration.parquet`, `reports/regional_calibration.md` |
| `prepare` DENUE | DENUE 05/2026, `data/universities.csv` | `data/processed/campus_context.parquet`, `reports/denue_campus.md` |
| `prepare` universidades | seed NV-07/NV-11, respuestas OSM y revisión humana | `data/universities.csv`, `reports/universities_check.md` |
| data card | registro, manifiestos, CSV y Parquet anteriores | `reports/data_card.md` |

## 10. Limitaciones

- Las encuestas son de 2020, 2022, anteriores a jul-2024; sus montos se deflactaron al INPC de ese mes y la serie no continúa después de esa fecha.
- La renta no se estima a nivel municipal: no cumple la regla de suficiencia y se usa la estimación  estatal de ENVI 2020.
- 3 universidades conservan una coordenada sin verificar (M-ML-017).
- El DENUE cuenta establecimientos, no calidad, precios ni disponibilidad de cuartos.
- No hay datos reales de cuartos: el catálogo se genera sintético en ML-03 (M-ML-019).
- Inside Airbnb — Ciudad de México (DESCARTADA) queda descartada por M-ML-019.

## 11. Privacidad

- Del DENUE **nunca se leen** nombres, razones sociales, teléfonos, correos, sitios web ni direcciones: `nom_estab`, `raz_social`, `telefono`, `correoelec`, `www`, `tipo_vial`, `nom_vial`, `numero_ext`, `numero_int`, `colonia`, `codigo_postal`, `nombre_act`, `clee`.
- Solo se leen las variables públicas registradas: `id`, `codigo_act`, `per_ocu`, `cve_ent`, `cve_mun`, `latitud`, `longitud`, `fecha_alta`; se publican únicamente conteos agregados.
- Ninguna celda con menos de 30 viviendas en muestra se publica (spec §8).
- A git solo entran agregados y metadatos: `sources.toml`, manifiestos, `data/universities.csv`
  y los reportes. Los datos crudos y los Parquet quedan fuera del repositorio (constitution §6).

