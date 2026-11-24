# ML-05 T02 — Validación cruzada y entrenamiento

> **DATOS SINTÉTICOS — ETIQUETAS SINTÉTICAS.** Métricas de validación cruzada sobre **train + validación** con las etiquetas sintéticas de ML-04 (M-ML-032): el modelo aprende la utilidad oculta del simulador más su ruido; **no** es desempeño real (constitución §8.2 y §20.3). La partición **test no se evalúa aquí**: se mide una sola vez en T03.

Generado por `python -m thehouse_ml train` (spec ML-05 1.0.0, T02). Parámetros: `src/thehouse_ml/models/train_params.json` 1.0.0 (M-ML-038 (APROBADO)). Métricas en `reports/train_cv_metrics.json`; modelos y asignación de pliegues en `models/` (fuera de git).

Perfil: **`full`** (corrida oficial: rejilla completa).

## Datos

- Esquema de features 2.0.0; `labels_params` 1.0.0; entradas (solo `training_data.TRAINING_INPUTS`; nunca la utilidad ni los pesos ocultos):

| Archivo | SHA-256 |
| --- | --- |
| `data/processed/pairs.parquet` | `1f5c4d167c2e45c571681857e6b6b34d2889f9cf5d3578729b6557f0b0431cae` |
| `data/processed/labels.parquet` | `b720f05602fb19cd1d68b8ee9a38d3a593b4cca6eb4985ae3a6a99984d9cdd54` |
| `data/processed/students.parquet` | `d4ad048ac31392bf98d33437d4bf91ba07839ffeb83d1ebec9842fb49ef373c1` |

| Partición | Pares | Estudiantes | Excluidos (< 2 candidatos) |
| --- | ---: | ---: | ---: |
| `train` | 135,992 | 1,399 | 1 |
| `validation` | 29,857 | 300 | 0 |
| `test` | 30,755 | 300 | 0 |
| **total** | 196,604 | 1,999 | 1 |

Pliegues y búsqueda: **1,699 estudiantes** y **165,849 pares** de train + validation. Objetivo de la regresión logística: `relevance ≥ 2` (`baseline_eval_params.json`, M-ML-034).

## Pliegues

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261124)` con una fila por estudiante (grupo = `student_id`, estrato = campus). Ningún estudiante está en dos pliegues; test no participa.

| Pliegue | `ITSOEH` | `TESJI` | `UAEH-ESAT` | `UAEH-ESTE` | `UAEH-ESTL` | `UCAN` | `UMB-JILO` | `UMB-VDC` | `UNE-TULA` | `UTTT` | Estudiantes | Pares |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 340 | 32,234 |
| P1 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 340 | 33,999 |
| P2 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 340 | 36,237 |
| P3 | 34 | 33 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 339 | 33,014 |
| P4 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 340 | 30,365 |

## Rejilla y NDCG@10 por pliegue

NDCG@10 medio de los estudiantes retenidos de cada pliegue (`thehouse_ml.evaluation.ranking`, puntaje descendente y desempate por `room_id`, igual que el baseline). Se elige el mayor NDCG@10 medio; desempate: menor desviación estándar muestral y después la primera de la rejilla (ordenada de la más simple a la más compleja).

### Regresión logística (8 configuraciones)

Pointwise, recorte de cada feature a su `training_range` (o `range`), `StandardScaler` (`standard`) o `KBinsDiscretizer` por cuantiles con 5 intervalos y one-hot (`quantile_bins`), `LogisticRegression(solver=lbfgs, max_iter=2000, tol=0.0001)`.

| Configuración | P0 | P1 | P2 | P3 | P4 | Media | Desv. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logreg:standard:C=0.01` **(elegida)** | 0.8664 | 0.8570 | 0.8616 | 0.8626 | 0.8640 | 0.8623 | 0.0035 |
| `logreg:standard:C=0.1` | 0.8671 | 0.8576 | 0.8622 | 0.8617 | 0.8629 | 0.8623 | 0.0034 |
| `logreg:standard:C=1` | 0.8667 | 0.8579 | 0.8619 | 0.8612 | 0.8629 | 0.8621 | 0.0032 |
| `logreg:standard:C=10` | 0.8667 | 0.8576 | 0.8618 | 0.8612 | 0.8631 | 0.8621 | 0.0033 |
| `logreg:quantile_bins:C=0.01` | 0.7145 | 0.7084 | 0.7069 | 0.7098 | 0.7158 | 0.7111 | 0.0039 |
| `logreg:quantile_bins:C=0.1` | 0.7124 | 0.7061 | 0.7047 | 0.7084 | 0.7146 | 0.7093 | 0.0042 |
| `logreg:quantile_bins:C=1` | 0.7125 | 0.7058 | 0.7048 | 0.7082 | 0.7153 | 0.7093 | 0.0045 |
| `logreg:quantile_bins:C=10` | 0.7125 | 0.7059 | 0.7047 | 0.7086 | 0.7154 | 0.7094 | 0.0045 |

Avisos de no convergencia en la búsqueda: 0; en el modelo final: 0.

### LightGBM LambdaRank (8 configuraciones)

`label_gain` [0, 1, 3, 7], `min_data_in_leaf` 50, restricciones monotónicas -1 en `price_to_budget`, `distance_km`, `effective_distance_km`, `distance_ratio`, `num_threads` 1, `deterministic` = true, `force_row_wise` = true, `seed` 20261124; sin parada temprana.

| Configuración | P0 | P1 | P2 | P3 | P4 | Media | Desv. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lightgbm:leaves=15:lr=0.05:trees=100` | 0.8658 | 0.8522 | 0.8556 | 0.8571 | 0.8593 | 0.8580 | 0.0051 |
| `lightgbm:leaves=15:lr=0.05:trees=300` **(elegida)** | 0.8733 | 0.8597 | 0.8651 | 0.8661 | 0.8684 | 0.8665 | 0.0049 |
| `lightgbm:leaves=15:lr=0.1:trees=100` | 0.8741 | 0.8576 | 0.8626 | 0.8635 | 0.8681 | 0.8652 | 0.0062 |
| `lightgbm:leaves=15:lr=0.1:trees=300` | 0.8725 | 0.8582 | 0.8637 | 0.8635 | 0.8679 | 0.8652 | 0.0053 |
| `lightgbm:leaves=31:lr=0.05:trees=100` | 0.8682 | 0.8559 | 0.8593 | 0.8602 | 0.8606 | 0.8609 | 0.0045 |
| `lightgbm:leaves=31:lr=0.05:trees=300` | 0.8726 | 0.8586 | 0.8638 | 0.8640 | 0.8671 | 0.8652 | 0.0051 |
| `lightgbm:leaves=31:lr=0.1:trees=100` | 0.8725 | 0.8605 | 0.8638 | 0.8643 | 0.8675 | 0.8657 | 0.0045 |
| `lightgbm:leaves=31:lr=0.1:trees=300` | 0.8702 | 0.8576 | 0.8604 | 0.8625 | 0.8662 | 0.8634 | 0.0049 |

## Hiperparámetros elegidos

| Familia | Configuración | NDCG@10 medio ± desv. |
| --- | --- | ---: |
| Regresión logística | `logreg:standard:C=0.01` | 0.8623 ± 0.0035 |
| LightGBM LambdaRank | `lightgbm:leaves=15:lr=0.05:trees=300` | 0.8665 ± 0.0049 |

## Comparación por pliegue con el baseline

El baseline `baseline-1.0.0` (`score_features` y `rank` de ML-04) no se entrena: se mide en los mismos estudiantes retenidos de cada pliegue. Es la entrada de la condición 1 de la spec §6.5, que se evalúa en T03 junto con la medición en test.

| Pliegue | Estudiantes | Baseline | Regresión logística | Δ | LightGBM | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 340 | 0.5396 | 0.8664 | +0.3267 | 0.8733 | +0.3336 |
| P1 | 340 | 0.5470 | 0.8570 | +0.3100 | 0.8597 | +0.3128 |
| P2 | 340 | 0.5453 | 0.8616 | +0.3163 | 0.8651 | +0.3198 |
| P3 | 339 | 0.5241 | 0.8626 | +0.3386 | 0.8661 | +0.3420 |
| P4 | 340 | 0.5345 | 0.8640 | +0.3295 | 0.8684 | +0.3339 |
| **Media ± desv.** | | 0.5381 ± 0.0093 | 0.8623 ± 0.0035 | +0.3242 | 0.8665 ± 0.0049 | +0.3284 |

Pliegues por encima del baseline: regresión logística 5/5, LightGBM 5/5 (informativo; la decisión de promoción es de T03).

## Modelos finales

Reentrenados con los hiperparámetros elegidos sobre todo train + validation (1,699 estudiantes, 165,849 pares). Sin pickle: la regresión logística en JSON (recorte, transformación, coeficientes e intercepto) y LightGBM en su texto nativo (`Booster.save_model`). Manifiesto con la asignación de pliegues: `models/train_manifest.json`.

| Archivo (fuera de git) | Bytes | SHA-256 |
| --- | ---: | --- |
| `models/logreg.json` | 1,853 | `341ea3ec705c2f877b36d1b931eafe2dea4630523560fe8764ce59e0a70b27fa` |
| `models/lightgbm_lambdarank.txt` | 543,515 | `5eb8d22dd771e04444383c092413c02501726d6ea1362111673b87ecd9e6086d` |

Verificación monotónica de LightGBM: en 200 pares equiespaciados, cada feature restringida se recorre en 21 valores de su `training_range` con lo demás fijo; filas en las que el puntaje sube: `price_to_budget` 0, `distance_km` 0, `effective_distance_km` 0, `distance_ratio` 0.

### Coeficientes de la regresión logística final

Intercepto: -4.133224.

| Feature | Recorte | Media | Escala | Coeficiente |
| --- | --- | ---: | ---: | ---: |
| `distance_km` | [0, 30] | 5.206677 | 6.303973 | -3.349741 |
| `effective_distance_km` | [0, 30] | 2.567704 | 3.697290 | -1.869938 |
| `price_to_budget` | [0, 1] | 0.556731 | 0.232874 | -0.996049 |
| `services_coverage` | [0, 1] | 0.732956 | 0.185477 | 0.565531 |
| `place_type_match` | [0, 1] | 0.298916 | 0.457783 | 1.006235 |
| `property_type_match` | [0, 1] | 0.695162 | 0.460339 | 0.359535 |
| `pets_compatible` | [0, 1] | 1.000000 | 1.000000 | 0.000000 |
| `distance_ratio` | [0, 60] | 1.331539 | 3.421818 | -0.100585 |
| `price_below_min` | [0, 1] | 0.137947 | 0.219381 | -0.012314 |
| `extra_services_ratio` | [0, 1] | 0.482977 | 0.215048 | 0.439171 |
