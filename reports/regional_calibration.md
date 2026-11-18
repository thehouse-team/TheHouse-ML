# Calibración regional — ML-02 T04

Generado por `uv run python -m thehouse_ml prepare`. Fuente: INEGI, Censo de Población y Vivienda 2020,
muestra del cuestionario ampliado (tabla Viviendas, solo Hidalgo 13 y Estado de México 15), citado como
autor del producto. INEGI no respalda este proyecto. La renta procede de la ENVI 2020 (M-ML-015).

## Los 7 municipios

Cada celda muestra el valor y el nivel del que salió: `municipio`, `estado` (entidad completa) o `nacional`.
Los porcentajes están ponderados con `FACTOR`; se excluyen los códigos 9 y blanco del denominador.

| Clave | Municipio | Rentadas | Internet | Electricidad | Agua entubada | Agua dentro | Drenaje | Gas | Renta p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13013 | Atotonilco de Tula | 11.6 % (mun) | 44.7 % (mun) | 99.9 % (mun) | 98.2 % (mun) | 81.6 % (mun) | 99.2 % (mun) | 94.8 % (mun) | $1,872 (edo) |
| 13041 | Mixquiahuala de Juárez | 8.6 % (mun) | 43.0 % (mun) | 97.4 % (mun) | 96.4 % (mun) | 76.5 % (mun) | 95.8 % (mun) | 91.2 % (mun) | $1,872 (edo) |
| 13063 | Tepeji del Río de Ocampo | 12.5 % (mun) | 37.6 % (mun) | 99.5 % (mun) | 98.8 % (mun) | 60.6 % (mun) | 97.8 % (mun) | 92.6 % (mun) | $1,872 (edo) |
| 13070 | Tlahuelilpan | 9.8 % (mun) | 42.6 % (mun) | 99.5 % (mun) | 99.5 % (mun) | 80.1 % (mun) | 99.4 % (mun) | 96.6 % (mun) | $1,872 (edo) |
| 13076 | Tula de Allende | 11.1 % (mun) | 46.9 % (mun) | 99.3 % (mun) | 98.6 % (mun) | 80.7 % (mun) | 98.5 % (mun) | 95.5 % (mun) | $1,872 (edo) |
| 15045 | Jilotepec | 7.8 % (mun) | 35.1 % (mun) | 97.8 % (mun) | 93.0 % (mun) | 65.5 % (mun) | 90.9 % (mun) | 90.2 % (mun) | $2,496 (edo) |
| 15112 | Villa del Carbón | 5.0 % (mun) | 18.8 % (mun) | 97.5 % (mun) | 97.7 % (mun) | 41.1 % (mun) | 87.6 % (mun) | 72.7 % (mun) | $2,496 (edo) |

## Detalle de la estimación

`n` es el conteo sin ponderar de viviendas válidas para el indicador; `Σ w` es la suma de sus factores.
`ee` es el error estándar y `cv = ee / estimación`. La renta no trae `ee`/`cv` porque la ENVI 2020
de T03 no publica varianza para sus percentiles; se marcan con `—`.

| Municipio | Indicador | Nivel | Fuente | n | Σ w | Rentadas | Estimación | ee | cv |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Atotonilco de Tula | agua_dentro | municipio | Censo | 1,690 | 17,545 | 204 | 81.6 % | 0.0285 | 3.5 % |
| Atotonilco de Tula | agua_entubada | municipio | Censo | 1,690 | 17,545 | 204 | 98.2 % | 0.0067 | 0.7 % |
| Atotonilco de Tula | drenaje | municipio | Censo | 1,690 | 17,545 | 204 | 99.2 % | 0.0031 | 0.3 % |
| Atotonilco de Tula | electricidad | municipio | Censo | 1,690 | 17,545 | 204 | 99.9 % | 0.0015 | 0.1 % |
| Atotonilco de Tula | gas | municipio | Censo | 1,690 | 17,545 | 204 | 94.8 % | 0.0056 | 0.6 % |
| Atotonilco de Tula | internet | municipio | Censo | 1,690 | 17,545 | 204 | 44.7 % | 0.0231 | 5.2 % |
| Atotonilco de Tula | renta | estado | ENVI | 164 | 81,275 | 164 | 1,872 | — | — |
| Atotonilco de Tula | rentadas | municipio | Censo | 1,690 | 17,545 | 204 | 11.6 % | 0.0117 | 10.1 % |
| Mixquiahuala de Juárez | agua_dentro | municipio | Censo | 2,390 | 12,659 | 211 | 76.5 % | 0.0267 | 3.5 % |
| Mixquiahuala de Juárez | agua_entubada | municipio | Censo | 2,390 | 12,659 | 211 | 96.4 % | 0.0223 | 2.3 % |
| Mixquiahuala de Juárez | drenaje | municipio | Censo | 2,390 | 12,659 | 211 | 95.8 % | 0.0212 | 2.2 % |
| Mixquiahuala de Juárez | electricidad | municipio | Censo | 2,390 | 12,659 | 211 | 97.4 % | 0.0198 | 2.0 % |
| Mixquiahuala de Juárez | gas | municipio | Censo | 2,390 | 12,659 | 211 | 91.2 % | 0.0185 | 2.0 % |
| Mixquiahuala de Juárez | internet | municipio | Censo | 2,390 | 12,659 | 211 | 43.0 % | 0.0173 | 4.0 % |
| Mixquiahuala de Juárez | renta | estado | ENVI | 164 | 81,275 | 164 | 1,872 | — | — |
| Mixquiahuala de Juárez | rentadas | municipio | Censo | 2,390 | 12,659 | 211 | 8.6 % | 0.0097 | 11.2 % |
| Tepeji del Río de Ocampo | agua_dentro | municipio | Censo | 2,319 | 24,572 | 385 | 60.6 % | 0.0374 | 6.2 % |
| Tepeji del Río de Ocampo | agua_entubada | municipio | Censo | 2,319 | 24,572 | 385 | 98.8 % | 0.0031 | 0.3 % |
| Tepeji del Río de Ocampo | drenaje | municipio | Censo | 2,319 | 24,572 | 385 | 97.8 % | 0.0060 | 0.6 % |
| Tepeji del Río de Ocampo | electricidad | municipio | Censo | 2,319 | 24,572 | 385 | 99.5 % | 0.0029 | 0.3 % |
| Tepeji del Río de Ocampo | gas | municipio | Censo | 2,319 | 24,572 | 385 | 92.6 % | 0.0071 | 0.8 % |
| Tepeji del Río de Ocampo | internet | municipio | Censo | 2,319 | 24,572 | 385 | 37.6 % | 0.0228 | 6.0 % |
| Tepeji del Río de Ocampo | renta | estado | ENVI | 164 | 81,275 | 164 | 1,872 | — | — |
| Tepeji del Río de Ocampo | rentadas | municipio | Censo | 2,319 | 24,572 | 385 | 12.5 % | 0.0183 | 14.7 % |
| Tlahuelilpan | agua_dentro | municipio | Censo | 1,436 | 4,869 | 134 | 80.1 % | 0.0183 | 2.3 % |
| Tlahuelilpan | agua_entubada | municipio | Censo | 1,436 | 4,869 | 134 | 99.5 % | 0.0026 | 0.3 % |
| Tlahuelilpan | drenaje | municipio | Censo | 1,436 | 4,869 | 134 | 99.4 % | 0.0019 | 0.2 % |
| Tlahuelilpan | electricidad | municipio | Censo | 1,436 | 4,869 | 134 | 99.5 % | 0.0020 | 0.2 % |
| Tlahuelilpan | gas | municipio | Censo | 1,436 | 4,869 | 134 | 96.6 % | 0.0064 | 0.7 % |
| Tlahuelilpan | internet | municipio | Censo | 1,435 | 4,866 | 134 | 42.6 % | 0.0290 | 6.8 % |
| Tlahuelilpan | renta | estado | ENVI | 164 | 81,275 | 164 | 1,872 | — | — |
| Tlahuelilpan | rentadas | municipio | Censo | 1,436 | 4,869 | 134 | 9.8 % | 0.0133 | 13.6 % |
| Tula de Allende | agua_dentro | municipio | Censo | 2,396 | 32,244 | 364 | 80.7 % | 0.0319 | 3.9 % |
| Tula de Allende | agua_entubada | municipio | Censo | 2,396 | 32,244 | 364 | 98.6 % | 0.0049 | 0.5 % |
| Tula de Allende | drenaje | municipio | Censo | 2,396 | 32,244 | 364 | 98.5 % | 0.0050 | 0.5 % |
| Tula de Allende | electricidad | municipio | Censo | 2,396 | 32,244 | 364 | 99.3 % | 0.0049 | 0.5 % |
| Tula de Allende | gas | municipio | Censo | 2,396 | 32,244 | 364 | 95.5 % | 0.0089 | 0.9 % |
| Tula de Allende | internet | municipio | Censo | 2,396 | 32,244 | 364 | 46.9 % | 0.0240 | 5.1 % |
| Tula de Allende | renta | estado | ENVI | 164 | 81,275 | 164 | 1,872 | — | — |
| Tula de Allende | rentadas | municipio | Censo | 2,396 | 32,244 | 364 | 11.1 % | 0.0120 | 10.8 % |
| Jilotepec | agua_dentro | municipio | Censo | 2,175 | 24,036 | 138 | 65.5 % | 0.0541 | 8.3 % |
| Jilotepec | agua_entubada | municipio | Censo | 2,175 | 24,036 | 138 | 93.0 % | 0.0167 | 1.8 % |
| Jilotepec | drenaje | municipio | Censo | 2,175 | 24,036 | 138 | 90.9 % | 0.0183 | 2.0 % |
| Jilotepec | electricidad | municipio | Censo | 2,175 | 24,036 | 138 | 97.8 % | 0.0083 | 0.8 % |
| Jilotepec | gas | municipio | Censo | 2,174 | 24,024 | 138 | 90.2 % | 0.0157 | 1.7 % |
| Jilotepec | internet | municipio | Censo | 2,174 | 24,024 | 138 | 35.1 % | 0.0491 | 14.0 % |
| Jilotepec | renta | estado | ENVI | 271 | 739,380 | 271 | 2,496 | — | — |
| Jilotepec | rentadas | municipio | Censo | 2,175 | 24,036 | 138 | 7.8 % | 0.0169 | 21.6 % |
| Villa del Carbón | agua_dentro | municipio | Censo | 2,065 | 13,004 | 94 | 41.1 % | 0.0403 | 9.8 % |
| Villa del Carbón | agua_entubada | municipio | Censo | 2,065 | 13,004 | 94 | 97.7 % | 0.0048 | 0.5 % |
| Villa del Carbón | drenaje | municipio | Censo | 2,065 | 13,004 | 94 | 87.6 % | 0.0212 | 2.4 % |
| Villa del Carbón | electricidad | municipio | Censo | 2,065 | 13,004 | 94 | 97.5 % | 0.0067 | 0.7 % |
| Villa del Carbón | gas | municipio | Censo | 2,065 | 13,004 | 94 | 72.7 % | 0.0354 | 4.9 % |
| Villa del Carbón | internet | municipio | Censo | 2,065 | 13,004 | 94 | 18.8 % | 0.0213 | 11.3 % |
| Villa del Carbón | renta | estado | ENVI | 271 | 739,380 | 271 | 2,496 | — | — |
| Villa del Carbón | rentadas | municipio | Censo | 2,065 | 13,004 | 94 | 5.0 % | 0.0109 | 21.8 % |

## Método

- **Proporción:** `p = Σ(w_i y_i) / Σ(w_i x_i)`, con `x_i = 1` si el código no es 9/blanco y
  `y_i = 1` si además está en los códigos afirmativos confirmados (M-ML-015).
- **Varianza (linealización de Taylor para razón, UPM con reemplazo):**
  `l_i = w_i (y_i - p x_i) / Σ(w_j x_j)`; agregando por estrato `h` y UPM `j`, `L_hj = Σ l_i`,
  `Var(p) = Σ_h [ n_h / (n_h - 1) · Σ_j (L_hj - media(L_h*))² ]`, `ee = √Var(p)` y `cv = ee / p`.
- Los estratos con una sola UPM no aportan varianza y se cuentan en la columna `n_strata_single_psu`
  del Parquet; es una limitación documentada del diseño con reemplazo.
- **Suficiencia:** el nivel municipal solo se usa con `>= 30` viviendas rentadas en la muestra y
  `cv <= 30 %`; si no, se usa la entidad completa (`13`/`15`). El respaldo nacional de ENVI/ENIGH de
  T03 existe para la proporción de rentadas; T03 no publica prevalencias de servicios (M-ML-014(c)),
  así que para esos indicadores la entidad es el último nivel disponible. Cada fila registra su nivel.
- **Renta:** el Censo no pregunta el monto. Para cada municipio se copian los percentiles p10-p90 en
  pesos de julio de 2024 de la fila estatal de ENVI 2020 en `national_baselines.parquet` (T03),
  con `level = estado` y `source = ENVI` (M-ML-014, M-ML-015).
