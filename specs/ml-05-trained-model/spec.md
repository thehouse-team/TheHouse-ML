# ML-05 — Modelo entrenado y evaluación

| Campo | Valor |
| --- | --- |
| ID | ML-05 |
| Documento | `spec.md` |
| Versión | 1.0.0 |
| Fecha | 2026-11-23 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | César Yael López Trujillo — 2026-11-23 (M-ML-036) |
| Estado de ejecución | **🟡 EN EJECUCIÓN — T01 a T05 completas (5/7); ver `evidence.md`** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Plan | `plan.md` 1.0.0 |
| Tasks | `task.md` 1.0.0 |
| Precondición | ML-04 completa ✅ (M-ML-035) |
| Relación con el backend | Si se promueve un modelo, ML-06 lo exporta a ONNX para NV-23; si no, el backend sigue con `baseline-1.0.0` |

Spec aprobada (1.0.0, M-ML-036). La aprobación no autoriza código, descargas de datos, dependencias ni cambios en GitHub: cada task requiere «EJECUTA» con versión, tasks y rutas. La versión 1.0.0 corrige el borrador 1.0.0-borrador (2026-11-07): R01/T01 pedía generar etiquetas, que ya existen (ML-04 T04, `labels.parquet`, M-ML-032); no estaba definida la validación cruzada frente a la partición 70/15/15 de ML-03; «supera de forma consistente» era ambiguo; faltaban las dependencias (scikit-learn y LightGBM están en la constitución §5 pero no en `pyproject.toml`), y el desglose del modelo entrenado estaba PENDIENTE aunque la constitución §12 exige que sume el total.

## 1. Contexto

ML-04 entrega (M-ML-035) el esquema `feature_schema.json` **2.0.0** (10 features), el modelo base por reglas `baseline-1.0.0` con `matchScore` y `breakdown`, las etiquetas sintéticas de relevancia 0–3 (`data/processed/labels.parquet`, `labels_params.json` 1.0.0, M-ML-032), las métricas reutilizables de `src/thehouse_ml/evaluation/ranking.py` y la referencia que ML-05 debe superar: `reports/baseline_metrics.json`, partición test de ML-03 (300 estudiantes), **NDCG@10 0.527 [0.503, 0.550]**, con las métricas por estudiante en `per_student_baseline` para comparaciones pareadas. Todo es **sintético**.

Las etiquetas vienen de una utilidad oculta que difiere del baseline en pesos y en la forma de 2 factores (presupuesto `1 − price_to_budget` y distancia `exp(−effective_distance_km / 3)`, M-ML-032): hay algo que un modelo entrenado puede aprender y las reglas no capturan. El segmento `budget.max` > $3,500 es el más débil del baseline (NDCG@10 0.468); que se deba a que el baseline puntúa igual cualquier renta dentro del presupuesto es una **hipótesis** que ML-05 confirma o refuta (M-ML-035, herencia 4).

## 2. Objetivo

Entrenar con rigor dos familias de modelo, compararlas contra el modelo base con un criterio de promoción medible y, si un modelo explicable lo supera, dejarlo listo para ML-06 con su puntaje 0–100 y su desglose por factor que suma el total. Si ninguno cumple, se queda el baseline y se documenta: es un resultado válido (constitución §10.1).

## 3. Precondiciones

1. ML-04 completa ✅ (M-ML-035).
2. Aprobación de esta spec y de las decisiones de la §10 ✅ (M-ML-036).
3. «EJECUTA» con versión, tasks y rutas permitidas. T01 agrega dependencias: su «EJECUTA» debe autorizar `pyproject.toml` y `uv.lock`.

## 4. Alcance

1. Dependencias `scikit-learn` y `lightgbm` (constitución §5) en `pyproject.toml` y `uv.lock`, con pip-audit limpio.
2. Carga de pares y etiquetas existentes (`pairs.parquet` + `labels.parquet`); **no** se generan etiquetas nuevas.
3. Validación cruzada de 5 pliegues agrupada por estudiante y estratificada por campus dentro de train + validación; la partición test solo para la medición final.
4. Dos modelos: regresión logística (pointwise, aditiva y explicable) y LightGBM LambdaRank con restricciones monotónicas (referencia del potencial).
5. Evaluación contra el modelo base y decisión de promoción con el criterio de la §6.5.
6. Ablación por grupo de features y análisis por segmentos, incluida la hipótesis del segmento > $3,500.
7. Calibración del puntaje a 0–100 y desglose aditivo por factor del modelo explicable.
8. Model card `reports/model_card.md`.

## 5. Fuera de alcance

- Generar o modificar etiquetas, pesos ocultos, `labels_params.json` o cualquier parámetro de ML-03/ML-04 (estudiantes, cuartos, pares, `feature_schema.json`, `baseline_params.json`, `baseline_eval_params.json`).
- Exportación a ONNX, paridad Python ↔ onnxruntime y presupuesto de inferencia (ML-06).
- Promover LightGBM en v1 (§6.6).
- Afirmaciones de desempeño real (ML-07).

## 6. Diseño

### 6.1 Datos y etiquetas

- **Única fuente de etiquetas:** `data/processed/labels.parquet` (M-ML-032), unido a `pairs.parquet` por `student_id` y `room_id`; se valida `synthetic = true` y `labels_params_version = "1.0.0"`.
- **Prohibido** leer `data/processed/labels_hidden_utility.parquet` y `data/processed/labels_hidden_weights.parquet` en entrenamiento, validación, selección de features o hiperparámetros, calibración o evaluación (fuga de la etiqueta, M-ML-035 herencia 3). **Prueba obligatoria** (como `test_evaluate_never_reads_the_hidden_label_files` de ML-04 T06): espía `pd.read_parquet`, borra esos dos archivos y comprueba que `train` y `evaluate` dan un resultado idéntico, y que sus nombres no aparecen en los módulos de ML-05.
- **Entradas del modelo:** exactamente las 10 columnas de `feature_order` del esquema 2.0.0, en ese orden (lo que recibirá el ONNX y calculará NV-23); sin `utility`, `relevance`, `split`, campus, presupuesto ni identificadores. `pets_compatible` es constante (1) en los candidatos: entra en el vector pero no aporta señal.
- **Estudiantes con menos de 2 candidatos** se excluyen de entrenamiento, validación cruzada y evaluación, y se cuentan (M-ML-035 herencia 5; hoy 1: `S1078`, train).

### 6.2 Validación cruzada y partición de prueba

- **Pliegues:** 5 pliegues **agrupados por estudiante** y **estratificados por campus** (`StratifiedGroupKFold`, `shuffle=True`, semilla fija) sobre los estudiantes de **train + validación** de ML-03 (1,400 + 300 = 1,700). Ningún estudiante aparece en dos pliegues; la asignación de pliegues se guarda en el reporte (recuento por pliegue y campus).
- **Uso de los pliegues:** elegir hiperparámetros, comparar modelos contra el baseline por pliegue y la ablación. En cada pliegue el baseline se evalúa sobre los mismos estudiantes retenidos con las mismas funciones (`thehouse_ml.evaluation`): no se entrena, así que no hay fuga.
- **Modelo final de cada familia:** se reentrena con los hiperparámetros elegidos sobre todo train + validación.
- **Partición test** (300 estudiantes de ML-03): solo para la **medición final**, **una vez** por modelo final, sin ajustar nada con ella (§7 regla 2): ni hiperparámetros, ni features, ni calibración, ni umbrales. Si después de medir en test se cambia algo del modelo, la medición se invalida y se documenta como hallazgo.

### 6.3 Modelos

| Modelo | Papel | Objetivo | Detalles |
| --- | --- | --- | --- |
| (a) Regresión logística (scikit-learn) | **Candidato a promoción**: modelo aditivo explicable | Pointwise: `y = 1` si `relevance ≥ 2` (mismo umbral que Precision@5 y MAP de M-ML-034) | Transformaciones dentro de un `Pipeline` de scikit-learn con **solo transformadores que tengan convertidor en `skl2onnx`** (ML-06 lo exporta); recorte de cada feature a su `training_range` del esquema; interacciones solo entre features del **mismo** factor (§6.6), nunca entre factores |
| (b) LightGBM LambdaRank | **Referencia del potencial** (no se promueve en v1) | Ranking por estudiante con la relevancia graduada 0–3 (`label_gain` 2^rel − 1, igual que NDCG) | Restricciones monotónicas: el puntaje **no sube** al subir `price_to_budget`, `distance_km`, `effective_distance_km` ni `distance_ratio` (−1); el resto sin restricción; `deterministic = true` y un hilo para reproducibilidad |

- **Búsqueda de hiperparámetros** pequeña, en rejilla, documentada en `src/thehouse_ml/models/train_params.json` 1.0.0 (cada valor con `value` y `basis`, como M-ML-032/034) y **solo con los pliegues**; se elige la configuración con mayor NDCG@10 medio entre pliegues (desempate: menor desviación, después la más simple). Propuesta (SUPUESTO, se fija con la decisión M-ML de T02):
  - Regresión logística: `C` ∈ {0.01, 0.1, 1, 10} × transformación ∈ {estandarizada (`StandardScaler`), discretizada por cuantiles (`KBinsDiscretizer`, 5 intervalos, one-hot)}: 8 configuraciones.
  - LightGBM: `num_leaves` ∈ {15, 31} × `learning_rate` ∈ {0.05, 0.1} × `n_estimators` ∈ {100, 300}, `min_child_samples` = 50: 8 configuraciones.
  - Semilla de pliegues y modelos **20261124** (distinta de 20261119–20261123 de ML-03/ML-04).
- **Persistencia sin pickle** (constitución §13.1): la regresión logística se guarda como JSON (parámetros de las transformaciones y coeficientes) y LightGBM en su formato de texto nativo, en `models/` (fuera de git). ML-06 produce el ONNX.

### 6.4 Métricas

- NDCG@10 (principal), NDCG@5, Precision@5 y MAP por estudiante con las funciones y definiciones de M-ML-034 (`thehouse_ml.evaluation.ranking`), umbral `relevance ≥ 2`, ranking por puntaje descendente con desempate `room_id` ascendente.
- En validación cruzada: media ± desviación entre pliegues y el valor de cada pliegue, del baseline y de los dos modelos (constitución §10.3).
- En test: valor con intervalo por bootstrap por estudiante y **diferencia pareada** contra `per_student_baseline` de `reports/baseline_metrics.json`, con los parámetros de `baseline_eval_params.json` 1.0.0 (1,000 remuestreos, percentil 95 %, semilla 20261122, `bootstrap:<seed>:<segment_id>`). Una prueba comprueba que los estudiantes evaluados de test son exactamente los 300 de `per_student_baseline` y que recalcular el baseline reproduce sus métricas por estudiante.

### 6.5 Criterio de promoción

Un candidato se promueve solo si cumple **las tres** condiciones:

1. **Validación cruzada:** NDCG@10 del candidato **mayor** que el del baseline en **cada uno de los 5 pliegues** (mismos estudiantes retenidos).
2. **Prueba:** el intervalo del 95 % (bootstrap pareado por estudiante contra `per_student_baseline`) de la diferencia NDCG@10 candidato − baseline tiene su **límite inferior > 0**.
3. **Explicabilidad:** ofrece un **desglose aditivo por factor que suma el total** (constitución §12), verificado en T05 sobre todos los pares de test.

Si ninguno cumple, **se queda el baseline** (`baseline-1.0.0`) y la model card lo documenta con las métricas (constitución §10.1); ML-06 exporta entonces el baseline. Con la §6.6, solo la regresión logística puede cumplir la condición 3.

### 6.6 Puntaje 0–100 y desglose del modelo explicable

- **Solo se promueve un modelo aditivo explicable** (la regresión logística). Su logit es `b₀ + Σⱼ tⱼ(xⱼ)`, donde cada término `tⱼ` depende de una sola feature (o de features del mismo factor).
- **Recorte al `training_range`:** antes de puntuar, cada feature se recorta a su `training_range` de `feature_schema.json` 2.0.0 (`clip(x, min, max)`), con **la misma regla en Python y en el backend** (NV-23). Así cada término está acotado, cada factor queda en [0, `maxPoints`], el total en [0, 100] y la suma de los factores cuadra con el total. El recorte forma parte del pipeline exportable (ML-06).
- **Factores:** los 6 del baseline (constitución §12; M-ML-029): presupuesto (`price_to_budget`, `price_below_min`), distancia (`distance_km`, `effective_distance_km`, `distance_ratio`), servicios esenciales (`services_coverage`), tipo de lugar (`place_type_match`), tipo de inmueble (`property_type_match`) y extras (`extra_services_ratio`); `pets_compatible` es constante en los candidatos y su término se suma a la constante.
- **Calibración a 0–100** (propuesta, SUPUESTO que se fija con la decisión M-ML de T05): transformación afín creciente del logit con los rangos alcanzables de cada factor sobre el `training_range` del esquema: `puntosₖ = 100 × (Cₖ − minₖ) / Σₘ (maxₘ − minₘ)`, con `Cₖ` la suma de los términos del factor `k`. Así cada factor está en [0, `maxPointsₖ`], con `maxPointsₖ = 100 × (maxₖ − minₖ) / Σₘ (maxₘ − minₘ)`, los factores **suman el total** antes de redondear y el orden del ranking coincide con el del logit **para entradas dentro del `training_range`** (fuera del rango, el puntaje es el de la entrada recortada: dos cuartos que solo difieren más allá del límite empatan y se desempatan por `room_id`); las métricas no cambian con la calibración y una prueba lo verifica. La probabilidad (sigmoide) se reporta como dato, pero no es el puntaje porque no es aditiva.
- **Redondeo:** las reglas de M-ML-029 (decisión 3) y M-ML-031 (decisión 10): puntos por factor con 1 decimal mitad hacia arriba tras round6; total `floor(round6(Σ) + 0.5)`; tolerancias `|Σ sin redondear − matchScore| ≤ 0.5` y `|Σ reportada − Σ sin redondear| ≤ 0.3`.
- **Formato:** el de ML-04 §6.4 con `modelVersion` propio (propuesta `logreg-1.0.0`), `featureSchemaVersion` 2.0.0, `matchScore` y `breakdown` con los 6 factores en el mismo orden; `maxPoints` por factor sale de la calibración (no 30/25/20/10/10/5). Un factor con varias features lleva `features` (lista de nombre y valor) en lugar de `feature`/`value`; el cambio se coordina con `TheHouse-GenAI` y el backend (responsable), como el `breakdown` de ML-04.
- **LightGBM** se reporta como referencia del potencial; si solo LightGBM supera al baseline, **no se promueve en v1** y se registra como trabajo futuro (p. ej., un desglose aditivo por contribuciones exportable).

### 6.7 Ablación y segmentos

- **Ablación** (solo con los pliegues, nunca con test): para cada uno de los 6 grupos de la §6.6, se reentrena cada modelo con sus hiperparámetros elegidos (sin nueva búsqueda) sin las features del grupo y se reporta ΔNDCG@10 (media ± desviación entre pliegues) contra el modelo completo.
- **Segmentos** (de la medición única en test; constitución §11.3): total, por campus, sin las 3 `seed_sin_verificar` y por rango de `budget.max` (≤ $2,000, $2,001–$3,500, > $3,500; M-ML-034), para baseline, regresión logística y LightGBM, con la diferencia pareada contra el baseline y su intervalo, y el número de estudiantes por segmento.
- **Hipótesis del segmento > $3,500** (M-ML-035 herencia 4): se **confirma** si, para el candidato, la mejora ΔNDCG@10 en > $3,500 es la mayor de los tres rangos y su intervalo del 95 % excluye el 0, **y** la ablación del grupo presupuesto reduce el NDCG@10; en otro caso se **refuta** o queda **no concluyente**, y se reporta tal cual.

## 7. Reglas

1. Toda métrica sobre etiquetas sintéticas se etiqueta «DATOS SINTÉTICOS» en reportes, JSON y model card (constitución §8 y §20.3).
2. Sin búsqueda de hiperparámetros, selección de features, calibración ni ningún ajuste con la partición de prueba: test se usa una vez por modelo final (§6.2).
3. Nunca se leen la utilidad ni los pesos ocultos (§6.1).
4. Las entradas del modelo son las 10 features de `feature_order` del esquema 2.0.0; ningún atributo sensible (constitución §11.2).
5. Los pares excluidos por HC01–HC03 no reciben puntaje (constitución §11.1): el modelo solo puntúa candidatos.
6. Sin pickle (constitución §13.1).

## 8. Seguridad, datos y privacidad

- Solo datos sintéticos de `data/processed/` (fuera de git); `models/` fuera de git; reportes sin datos personales.
- Dependencias nuevas solo de la constitución §5 (scikit-learn BSD-3-Clause, LightGBM MIT), fijadas en `uv.lock` dentro del `exclude-newer` vigente de `pyproject.toml` y con pip-audit limpio (constitución §14.2). Su licencia se registra en la evidencia.
- El análisis por segmentos va en la model card para detectar sesgos (constitución §11.3).

## 9. Requisitos verificables

| ID | Requisito | Criterio de aceptación | Fase | Tasks |
| --- | --- | --- | --- | --- |
| ML-05-R01 | Dependencias y carga de datos | `scikit-learn` y `lightgbm` en `pyproject.toml` y `uv.lock` (constitución §5) con `uv lock --check` y pip-audit sin vulnerabilidades; cargador que une `pairs.parquet` y `labels.parquet` (valida `synthetic = true`, `labels_params_version` 1.0.0 y las 10 columnas de `feature_order`), excluye y cuenta estudiantes con < 2 candidatos; **prueba obligatoria** de que nunca se leen `labels_hidden_utility.parquet` ni `labels_hidden_weights.parquet`; sin generación de etiquetas. | F1 | T01 |
| ML-05-R02 | Validación cruzada y entrenamiento reproducible | 5 pliegues agrupados por estudiante y estratificados por campus sobre train + validación (sin estudiantes compartidos, recuento por pliegue y campus); rejilla de `train_params.json` evaluada solo con los pliegues; modelo final de cada familia reentrenado en train + validación; LightGBM con las 4 restricciones monotónicas; `train` produce los mismos pliegues, hiperparámetros, coeficientes, modelo y métricas con la misma semilla y datos (SHA-256 iguales en dos corridas); sin pickle. | F2 | T02 |
| ML-05-R03 | Evaluación contra el baseline y decisión de promoción | Tabla de NDCG@10, NDCG@5, Precision@5 y MAP por pliegue y media ± desviación (baseline, regresión logística, LightGBM); medición única en test con intervalos y diferencia pareada contra `per_student_baseline`; evaluación de las 3 condiciones de la §6.5 por candidato y decisión (promovido o se queda el baseline) en `reports/trained_metrics.json` y `reports/trained_report.md`, marcados como sintéticos. | F2 | T03 |
| ML-05-R04 | Ablación y segmentos | ΔNDCG@10 por grupo de features (6 grupos, solo pliegues) para los dos modelos; métricas y diferencias pareadas por campus, sin `seed_sin_verificar` y por rango de presupuesto en test; hipótesis del segmento > $3,500 confirmada, refutada o no concluyente según la §6.7. | F3 | T04 |
| ML-05-R05 | Calibración 0–100 y desglose | Cada feature se recorta a su `training_range` antes de puntuar (misma regla en Python y en el backend); puntaje 0–100 de la regresión logística con la calibración de la §6.6 y desglose por los 6 factores que suma el total (tolerancias de M-ML-029/031) en todos los pares de test; ranking idéntico al del logit para entradas dentro del `training_range`; pruebas de propiedades (incluidas entradas fuera del rango) (total en [0, 100], cada factor en [0, `maxPoints`], coherencia del redondeo); pipeline con solo transformadores con convertidor en `skl2onnx` (lista verificada por prueba). | F3 | T05 |
| ML-05-R06 | Model card | `reports/model_card.md` con «DATOS SINTÉTICOS»: uso previsto, datos y etiquetas (M-ML-032), modelos e hiperparámetros, métricas contra el baseline (pliegues y test), ablación, segmentos, desglose de ejemplo, limitaciones (circularidad, etiquetas sintéticas), decisión de promoción y trabajo futuro. | F3 | T06 |
| ML-05-R07 | Pruebas y evidencia | Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; pip-audit limpio; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`. | F4 | T07 |

## 10. Decisiones

Registradas en `memoria.md` (M-ML-036, APROBADO). Los valores marcados SUPUESTO en la §6.3 (rejilla y semilla) y la §6.6 (calibración) se fijan con la decisión M-ML de la ejecución de su task (T02 y T05); hasta entonces esas tasks no pueden cerrarse.

| # | Tema | Decisión | Estado | Tasks |
| --- | --- | --- | --- | --- |
| 1 | Etiquetas | Se reutiliza `labels.parquet` de ML-04 (M-ML-032) como única fuente; prohibido leer `labels_hidden_utility.parquet` y `labels_hidden_weights.parquet`, con prueba obligatoria; la generación de etiquetas sale del alcance | APROBADO | T01–T07 |
| 2 | Validación | 5 pliegues agrupados por estudiante y estratificados por campus dentro de train + validación para elegir hiperparámetros y comparar modelos; test solo para la medición final, una vez, sin ajustar nada con ella | APROBADO | T02–T04 |
| 3 | Modelos | (a) Regresión logística pointwise (`relevance ≥ 2`), aditiva y explicable, con transformaciones dentro de un pipeline exportable a ONNX en ML-06; (b) LightGBM LambdaRank con restricciones monotónicas (no sube con `price_to_budget`, `distance_km`, `effective_distance_km`, `distance_ratio`) como referencia del potencial; búsqueda de hiperparámetros pequeña, documentada y solo con los pliegues (rejilla y semilla: SUPUESTO, se fijan en T02) | APROBADO | T02–T05 |
| 4 | Criterio de promoción | Supera al baseline en NDCG@10 en los 5 pliegues **y** el intervalo del 95 % (bootstrap pareado por estudiante contra `per_student_baseline`) de la diferencia en test es > 0 **y** ofrece un desglose aditivo que suma el total (constitución §12); si ninguno cumple, se queda el baseline y se documenta (constitución §10) | APROBADO | T03, T05, T06 |
| 5 | Desglose | Solo se promueve un modelo aditivo explicable (regresión logística): cada feature se recorta a su `training_range` antes de puntuar (misma regla en Python y en NV-23); puntaje calibrado a 0–100 y desglose por las contribuciones aditivas agrupadas en los 6 factores del baseline, reescaladas para sumar el total con el redondeo de M-ML-029/031 (método de calibración: SUPUESTO, se fija en T05); LightGBM como referencia y, si solo él supera al baseline, no se promueve en v1 y queda como trabajo futuro | APROBADO | T03, T05, T06 |
| 6 | Dependencias | `scikit-learn` y `lightgbm` (constitución §5) en `pyproject.toml` y `uv.lock`, con pip-audit limpio, en la primera task (T01) | APROBADO | T01 |

## 11. Costo — COST-001

| Componente | Costo | Condición |
| --- | --- | --- |
| CPU local y GitHub Actions (repositorio público) | $0 | Sin servicios de pago ni GPU en la nube; scikit-learn y LightGBM son de código abierto y corren en CPU |

## 12. Riesgos

| ID | Riesgo | Impacto | Mitigación |
| --- | --- | --- | --- |
| ML-05-RK01 | Sobreajuste a etiquetas sintéticas (circularidad, ML-04-RK02): el modelo aprende la función del simulador | Métricas sin valor real | Validación cruzada por estudiante, restricciones monotónicas, test usado una vez, «DATOS SINTÉTICOS» en todo reporte; se reemplaza con señales reales en ML-07 |
| ML-05-RK02 | Ningún modelo se promueve | No hay modelo entrenado para ML-06 | Es un resultado válido: se queda `baseline-1.0.0` y se documenta en la model card (constitución §10.1); ML-06 exporta el baseline |
| ML-05-RK03 | Paridad del modelo en ONNX: una transformación sin convertidor o con resultado distinto en onnxruntime | Bloquea ML-06 | Pipeline restringido a transformadores con convertidor en `skl2onnx` (prueba en T05); la paridad se verifica en ML-06 |

## 13. Dependencias

- ML-04 completa (M-ML-035): esquema 2.0.0, `labels.parquet`, `baseline-1.0.0`, `evaluation/ranking.py`, `baseline_eval_params.json` y `reports/baseline_metrics.json`.
- Constitución §5: scikit-learn y LightGBM (se agregan en T01).
- `TheHouse-GenAI` y backend: si se promueve la regresión logística, el formato del `breakdown` con `maxPoints` variables y la lista `features` por factor se coordina con `TheHouse-GenAI` y con **NV-23** del backend, que además aplica el mismo recorte al `training_range` antes de puntuar (responsable).

## 14. Trazabilidad y cobertura

| Métrica | Valor |
| --- | --- |
| Requisitos | 7 |
| Tasks | 7 |
| Requisitos sin task | 0 |
| Tasks sin requisito | 0 |
| Tasks completadas | 5 (T01, T02, T03, T04, T05; rejilla y semilla fijadas en M-ML-038; T03 con M-ML-039: regresión logística candidata a promoción, sujeta a T05; T04 con M-ML-040: 4 métricas por pliegue, ablación de los 6 grupos, segmentos e hipótesis > $3,500 confirmada con salvedades T04-H2/H3). T05 con M-ML-041: calibración afín 0-100 y desglose de `logreg-1.0.0` que suma el total en los 30,755 pares de test; **regresión logística PROMOVIDA** (baseline de respaldo), con la no monotonía del presupuesto (`budget.max/budget.min` > 76.2) como desviación aceptada T05-H1 y T05-H2 para la model card) |
| Estado documental | **✅ APROBADA** |
| Estado de ejecución | **🟡 EN EJECUCIÓN — T01 a T05 completas (5/7); ver `evidence.md`** |

## 15. Aprobación

Aprobada por César Yael López Trujillo el 2026-11-23 («Apruebo ML-05 1.0.0», M-ML-036), con las decisiones 1–6 de la §10 en APROBADO y las aclaraciones de la revisión (recorte al `training_range`, propuestas de calibración, formato de `breakdown`, regla de la hipótesis > $3,500, bootstrap, semilla, persistencia y nombres de salida). Para ejecutar se requiere «EJECUTA» con versión, tasks y rutas permitidas; T01 agrega dependencias, así que su «EJECUTA» debe autorizar `pyproject.toml` y `uv.lock`.
