# DATOS SINTÉTICOS — NO SON PREFERENCIAS REALES

> **DATOS SINTÉTICOS — ETIQUETAS SINTÉTICAS.** Estas etiquetas **no son preferencias reales** de ningún
> estudiante: salen de una **utilidad oculta documentada más ruido** (constitución §8.2), calculada sobre
> estudiantes y cuartos sintéticos. Un modelo entrenado con ellas aprende esa función y ese ruido, no el
> comportamiento de personas reales; ninguna métrica calculada con ellas es desempeño real. Hay riesgo de
> **circularidad** (ML-04-RK02): la utilidad usa las mismas features que el modelo base. Se **reemplazan con
> señales reales en ML-07**.

Generado por `uv run python -m thehouse_ml prepare` (`thehouse_ml.synthetic.labels_stage`). Parámetros:
`src/thehouse_ml/synthetic/labels_params.json` versión 1.0.0 (decisión M-ML-032 (APROBADO)). Entrada: `data/processed/pairs.parquet` (esquema de features 2.0.0).

## 1. Resumen

- Pares etiquetados: **196,605** de 2,000 estudiantes (semilla 20261121); una etiqueta por par de `pairs.parquet`, ninguna fuera de los candidatos.
- `data/processed/labels.parquet` (fuera de git): `student_id`, `room_id`, `split`, `relevance`, `synthetic`, `labels_params_version`.
- Utilidad oculta y pesos ocultos por estudiante, aparte y fuera de git: `data/processed/labels_hidden_utility.parquet` y `data/processed/labels_hidden_weights.parquet`. No son entradas de ningún modelo.
- Spearman entre el puntaje del modelo base (baseline-1.0.0, suma sin redondear de `score_features`) y la utilidad oculta: **0.513** (rango documentado [0.3, 0.9]: **cumple**).

## 2. Utilidad oculta

Utilidad = Σ peso oculto * curva(feature) + ruido gaussiano. Los pesos base y dos curvas difieren del
modelo base para que las etiquetas no sean una copia de las reglas.

| Factor | Peso baseline-1.0.0 | Peso base oculto | Curva del modelo base | Curva oculta |
| --- | ---: | ---: | --- | --- |
| `budget` | 30 | 20 | `price_below_min` por tramos (0, 1) · (1, 0) | `1 - price_to_budget` |
| `distance` | 25 | 35 | `distance_ratio` por tramos (0, 1) · (0.5, 1) · (1, 0) | `exp(-effective_distance_km / 3)` |
| `essential_services` | 20 | 20 | `services_coverage` por tramos (0, 0) · (1, 1) | `services_coverage` |
| `place_type` | 10 | 10 | `place_type_match` por tramos (0, 0) · (1, 1) | `place_type_match` |
| `property_type` | 10 | 5 | `property_type_match` por tramos (0, 0) · (1, 1) | `property_type_match` |
| `extras` | 5 | 10 | `extra_services_ratio` por tramos (0, 0) · (1, 1) | `extra_services_ratio` |

- **Variación por estudiante:** cada peso base * `exp(0.3 · z)`, `z` normal estándar, renormalizado para sumar 100.
- **Ruido:** normal de media 0 y desviación 5 puntos por par.
- **Aleatoriedad:** `random.Random("labels:<seed>:<student_id>")`: 6 normales para los multiplicadores y después una por candidato en orden de `room_id`.
- El simulador **no** usa el modelo base ni ningún atributo sensible o de género (constitución §11).

## 3. Pesos ocultos por estudiante

Distribución entre estudiantes (percentiles con interpolación lineal); en cada estudiante suman 100.

| Factor | Peso base | Mín. | p25 | Mediana | p75 | Máx. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `budget` | 20 | 7.7 | 15.9 | 19.5 | 23.4 | 48.0 |
| `distance` | 35 | 15.8 | 29.5 | 34.4 | 39.5 | 63.0 |
| `essential_services` | 20 | 6.6 | 16.3 | 19.3 | 23.1 | 41.8 |
| `place_type` | 10 | 3.2 | 7.9 | 9.8 | 12.0 | 24.9 |
| `property_type` | 5 | 1.6 | 3.9 | 4.8 | 6.0 | 14.5 |
| `extras` | 10 | 3.5 | 7.9 | 9.8 | 11.9 | 26.6 |

## 4. Relevancia graduada 0-3

Dentro de los `n` candidatos de cada estudiante, ordenados por utilidad descendente (empates por `room_id`
ascendente), la posición `r` recibe 3 si `r ≤ ⌈0.1·n⌉`, 2 si `r ≤ ⌈0.2·n⌉`, 1 si `r ≤ ⌈0.3·n⌉` y 0 en otro caso.
Todo estudiante con al menos un candidato tiene al menos un 3. Ejemplos:

| Candidatos `n` | Relevancia 3 | Relevancia 2 | Relevancia 1 | Relevancia 0 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 0 | 0 |
| 2 | 1 | 0 | 0 | 1 |
| 3 | 1 | 0 | 0 | 2 |
| 4 | 1 | 0 | 1 | 2 |
| 5 | 1 | 0 | 1 | 3 |
| 10 | 1 | 1 | 1 | 7 |
| 200 | 20 | 20 | 20 | 140 |

Estudiantes por número de candidatos: 1 → **1**, 2 → **0**, 3-9 → **23**, ≥ 10 → **1,976**.

## 5. Distribución de relevancia

### 5.1 Total y por partición

| Partición | Estudiantes | Pares | Relevancia 3 | Relevancia 2 | Relevancia 1 | Relevancia 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | 1,400 | 135,993 | 14,097 (10.4 %) | 13,543 (10.0 %) | 13,653 (10.0 %) | 94,700 (69.6 %) |
| `validation` | 300 | 29,857 | 3,092 (10.4 %) | 2,972 (10.0 %) | 3,002 (10.1 %) | 20,791 (69.6 %) |
| `test` | 300 | 30,755 | 3,173 (10.3 %) | 3,073 (10.0 %) | 3,088 (10.0 %) | 21,421 (69.7 %) |
| **Total** | 2,000 | 196,605 | 20,362 (10.4 %) | 19,588 (10.0 %) | 19,743 (10.0 %) | 136,912 (69.6 %) |

### 5.2 Por campus

| Campus | Estudiantes | Pares | Relevancia 3 | Relevancia 2 | Relevancia 1 | Relevancia 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `UTTT` | 200 | 18,152 | 1,885 (10.4 %) | 1,809 (10.0 %) | 1,827 (10.1 %) | 12,631 (69.6 %) |
| `UAEH-ESTE` | 200 | 19,704 | 2,036 (10.3 %) | 1,963 (10.0 %) | 1,982 (10.1 %) | 13,723 (69.6 %) |
| `UAEH-ESTL` | 200 | 20,927 | 2,164 (10.3 %) | 2,090 (10.0 %) | 2,101 (10.0 %) | 14,572 (69.6 %) |
| `UAEH-ESAT` | 200 | 19,390 | 2,010 (10.4 %) | 1,936 (10.0 %) | 1,944 (10.0 %) | 13,500 (69.6 %) |
| `ITSOEH` | 200 | 19,924 | 2,074 (10.4 %) | 1,984 (10.0 %) | 2,006 (10.1 %) | 13,860 (69.6 %) |
| `TESJI` (sin verificar) | 200 | 19,911 | 2,060 (10.3 %) | 1,980 (9.9 %) | 1,988 (10.0 %) | 13,883 (69.7 %) |
| `UMB-JILO` | 200 | 18,695 | 1,945 (10.4 %) | 1,858 (9.9 %) | 1,874 (10.0 %) | 13,018 (69.6 %) |
| `UMB-VDC` (sin verificar) | 200 | 20,343 | 2,109 (10.4 %) | 2,026 (10.0 %) | 2,039 (10.0 %) | 14,169 (69.7 %) |
| `UCAN` (sin verificar) | 200 | 20,933 | 2,152 (10.3 %) | 2,089 (10.0 %) | 2,109 (10.1 %) | 14,583 (69.7 %) |
| `UNE-TULA` | 200 | 18,626 | 1,927 (10.3 %) | 1,853 (9.9 %) | 1,873 (10.1 %) | 12,973 (69.6 %) |
| **Total** | 2,000 | 196,605 | 20,362 (10.4 %) | 19,588 (10.0 %) | 19,743 (10.0 %) | 136,912 (69.6 %) |
| **Total sin `seed_sin_verificar`** | 1,400 | 135,418 | 14,041 (10.4 %) | 13,493 (10.0 %) | 13,607 (10.0 %) | 94,277 (69.6 %) |

## 6. No circularidad: correlación con el modelo base

Spearman (Pearson de rangos promedio) entre la suma sin redondear del modelo base y la utilidad oculta con
ruido. Positiva: las etiquetas premian lo mismo que las reglas en lo general. Claramente menor que 1: el
orden de las etiquetas no es el del modelo base (ML-04-RK02).

| Medida | Valor |
| --- | ---: |
| Sobre todos los pares (la que se comprueba) | **0.513** |
| Rango documentado (`check.spearman_range`) | [0.3, 0.9] |
| Mediana por estudiante (1,999 estudiantes con series no constantes) | 0.547 |
| p25 - p75 por estudiante | 0.377 - 0.685 |

## 7. Privacidad y limitaciones

- Solo ids sintéticos (`S…`, `R…`), la partición y la relevancia; sin datos personales (constitución §7) ni
  género, edad u origen (constitución §11).
- Pesos, curvas, ruido y cortes son SUPUESTOS documentados (M-ML-032), no calibrados con conducta real.
- Las métricas de ranking que se calculen con estas etiquetas (T06, ML-05) son sobre **datos sintéticos** y
  se reportan como tales; se reemplazan con señales reales en ML-07.
