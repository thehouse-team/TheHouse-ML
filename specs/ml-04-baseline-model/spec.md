# ML-04 — Modelo base por reglas

| Campo | Valor |
| --- | --- |
| ID | ML-04 |
| Documento | `spec.md` |
| Versión | 1.0.0 |
| Fecha | 2026-11-20 |
| Estado documental | **✅ APROBADA** |
| Aprobación humana | César Yael López Trujillo — 2026-11-20 (M-ML-029) |
| Estado de ejecución | **🟡 EN EJECUCIÓN — T01 y T02 completas (2/7); ver `evidence.md`** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Plan | `plan.md` 1.0.0 |
| Tasks | `task.md` 1.0.0 |
| Precondición | ML-03 completa ✅ (M-ML-028) |
| Relación con el backend | Es el respaldo determinista de NV-23 y la base del desglose que explica `TheHouse-GenAI`; el esquema 2.0.0 cambia las entradas del ONNX y lo que calcula NV-23 (constitución §2.4 y §18) |

Spec aprobada (1.0.0, M-ML-029). La aprobación no autoriza código, descargas de datos, dependencias ni cambios en GitHub: cada task requiere «EJECUTA» con versión, tasks y rutas. La versión 1.0.0 corrige el borrador 1.0.0-borrador (2026-11-07): las curvas usaban `maxDistanceKm` y `budget.min`, que no están en las 7 features del esquema 1.1.0; «modalidad» y «extras» no tenían feature en NV-11; `pets_compatible` es constante (1) dentro de los candidatos (M-ML-028); R04 pedía métricas de ranking sin etiquetas en `pairs.parquet` (M-ML-027), y la suma del desglose y el redondeo no estaban definidos.

## 1. Contexto

Smart Match debe ser explicable y funcionar desde el primer día. Un modelo por reglas con pesos documentados da un puntaje 0–100 con desglose y es la referencia contra la que se mide cualquier modelo entrenado (constitución §10.1).

ML-03 entrega (M-ML-028) el esquema `feature_schema.json` **1.1.0** con 7 features del par, las restricciones duras HC01–HC03 y `data/processed/pairs.parquet`: 2,000 estudiantes sintéticos, 196,605 pares candidatos (cuartos del campus que pasan HC01–HC03, hasta 200 por estudiante) y partición por estudiante train 1,400 · validation 300 · test 300, **sin etiquetas**. Dentro de los candidatos `pets_compatible` vale siempre 1 y `price_to_budget` ≤ 1.

## 2. Objetivo

Definir y construir un puntaje determinista 0–100 con desglose por factor (`baseline-1.0.0`), las etiquetas sintéticas de relevancia que necesitan ML-04 y ML-05 para medir rankings, los casos dorados para la paridad del backend y el reporte que fija la referencia que ML-05 debe superar.

## 3. Precondiciones

1. ML-03 completa ✅ (M-ML-028).
2. Aprobación de esta spec («Apruebo 1.0.0») y de las decisiones de la §10.
3. «EJECUTA» con versión, tasks y rutas permitidas.

## 4. Alcance

1. Configuración versionada del modelo base (`baseline-1.0.0`): factores, pesos, curvas y redondeo.
2. Esquema de features **2.0.0** con 3 features nuevas al final del orden y su cálculo en `compute_pair_features`.
3. Función de puntaje con desglose por factor: `matchScore` compatible con `TheHouse-GenAI` y `breakdown` como campo nuevo propuesto.
4. Simulador de etiquetas sintéticas de relevancia graduada 0–3.
5. Al menos 50 casos dorados para la paridad del backend.
6. Reporte del modelo base con métricas de ranking en la partición de prueba, marcado como sintético.

## 5. Fuera de alcance

- Modelos entrenados (ML-05) y exportación ONNX (ML-06).
- Ajustar los pesos del modelo base con las etiquetas sintéticas (§7 regla 3).
- Quitar `pets_compatible` del esquema (sería un cambio mayor; lo decide ML-05 si hace falta).
- Señales reales (ML-07).

## 6. Diseño

### 6.1 Esquema 2.0.0: features nuevas

**Versión mayor** (constitución §18): agregar features cambia las entradas del ONNX y lo que calcula NV-23, así que el backend debe actualizar su spec (constitución §2.4). La 1.1.0 de M-ML-023 fue menor porque solo amplió rangos. El cambio solo **agrega** tres features al **final** de `feature_order`; las 7 existentes conservan nombre, fórmula, rango y posición. T02 también actualiza `metadata.versioning` del esquema para decir que agregar, quitar o reordenar features es un cambio mayor. Se implementa en T02, no al aprobar esta spec.

| Orden | Feature | Fórmula | Rango |
| --- | --- | --- | --- |
| 8 | `distance_ratio` | `distance_km / maxDistanceKm` resuelto (nulo → 30, M-ML-022). Usa la distancia **real**, no la efectiva, porque `maxDistanceKm` ya refleja el transporte que eligió el estudiante. `maxDistanceKm` ≥ 0.5 en NV-11, así que no hay división entre cero. | mínimo 0, **sin límite superior** (`"max": null`); `training_range` documentado con el máximo observado en `pairs.parquet` (la distancia no es restricción dura: 26.3 % de los pares supera `maxDistanceKm`) |
| 9 | `price_below_min` | `(budget.min − price_mxn) / budget.min` si `budget.min` resuelto > 0 y `price_mxn < budget.min`; en otro caso 0 (`budget.min` nulo → 0, M-ML-022) | 0–1 |
| 10 | `extra_services_ratio` | `|room.services − student.essentialServices| / |catálogo − student.essentialServices|`, es decir, servicios del cuarto que el estudiante **no** pidió entre los servicios del catálogo (11) que no pidió; 0 si pidió los 11 (`essentialServices` nulo → conjunto vacío, denominador 11) | 0–1 |

T02 actualiza `compute_pair_features`, `metadata.changelog` (2.0.0) y las pruebas del esquema, regenera `data/processed/pairs.parquet` con `prepare` (mismas semillas; las 7 columnas anteriores no cambian) y amplía `tests/fixtures/pair_feature_cases.json` con valores de las 3 features calculados a mano.

### 6.2 Factores, pesos y curvas (`baseline-1.0.0`)

Cada factor vale `peso × f(feature)` con `f` en [0, 1]; el total máximo es 100.

| Factor (`id`) | Peso | Feature | Curva `f` |
| --- | --- | --- | --- |
| Presupuesto (`budget`) | 30 | `price_below_min` | `1 − price_below_min` (1.0 si la renta está en [`budget.min`, `budget.max`]; decae linealmente por debajo de `budget.min`) |
| Distancia (`distance`) | 25 | `distance_ratio` | 1.0 si `distance_ratio` ≤ 0.5; `2 × (1 − distance_ratio)` si 0.5 < `distance_ratio` ≤ 1; 0 si `distance_ratio` > 1 |
| Servicios esenciales (`essential_services`) | 20 | `services_coverage` | identidad (1 si no pidió servicios, esquema 1.1.0) |
| Tipo de lugar (`place_type`) | 10 | `place_type_match` | identidad (0 o 1) |
| Tipo de inmueble (`property_type`) | 10 | `property_type_match` | identidad (0 o 1; 1 si `propertyTypes` es nulo o vacío) |
| Extras (`extras`) | 5 | `extra_services_ratio` | identidad |

- **Correspondencia con la constitución §12:** «modalidad» es el factor `property_type` (tipo de inmueble: casa, departamento o residencia estudiantil, el único dato de NV-11 que queda para ese factor) y «extras» son los servicios del cuarto que el estudiante no pidió.
- **Features que no puntúan:** `pets_compatible` (constante 1 en los candidatos: HC03 ya es restricción dura, M-ML-028); `price_to_budget` (HC01 ya excluye la renta sobre `budget.max` y el factor de presupuesto usa `price_below_min`); `effective_distance_km` (el factor de distancia usa `distance_ratio`, que ya refleja el transporte). Siguen en el esquema para ML-05.
- **Pares excluidos:** el puntaje solo se calcula para pares que pasan HC01–HC03 (`check_hard_constraints`, M-ML-024). Un par excluido no recibe `matchScore`: el resultado indica las restricciones violadas (constitución §11.1).
- **Sin atributos sensibles:** ningún factor usa género, edad u origen; el género no existe en el perfil ni en el esquema (constitución §11.2).
- **Monotonía:** con lo demás fijo, el factor de presupuesto no disminuye al subir la renta por debajo de `budget.min` y es constante entre `budget.min` y `budget.max` (una renta muy por debajo del mínimo del estudiante no es un mejor match); el factor de distancia no aumenta al subir `distance_km`.

### 6.3 Redondeo

1. Los puntos de cada factor se calculan en punto flotante de 64 bits sin redondear.
2. **Total** (`matchScore`): entero 0–100 = redondeo **mitad hacia arriba** de la suma de los puntos sin redondear, después de redondear esa suma a 6 decimales para absorber el error de punto flotante (`floor(round6(Σ) + 0.5)`). En .NET equivale a `Math.Round(x, MidpointRounding.AwayFromZero)` en los dos pasos; el `Math.Round` por omisión (mitad al par) **no** es compatible.
3. **Desglose:** los puntos de cada factor se reportan con **1 decimal**, mitad hacia arriba.
4. **Coherencia:** `|Σ puntos sin redondear − matchScore| ≤ 0.5` y `|Σ puntos reportados − Σ puntos sin redondear| ≤ 0.3` (6 factores × 0.05). La suma de los puntos reportados puede diferir del total hasta en 0.8; el reporte y la ficha lo explican así (constitución §12: el desglose suma el total antes de redondear).
5. **Orden del ranking:** por la suma sin redondear, de mayor a menor; los empates se rompen por `room_id` ascendente.

### 6.4 Formato de salida

Verificado por el responsable en `TheHouse-GenAI` PR #4: `ExplainMatchRequest` recibe `matchScore` (entero 0–100), `PreferencesIn` y `PropertyIn`, y **no** tiene desglose. Por eso `matchScore` conserva ese nombre y tipo, y `breakdown` es un **campo nuevo propuesto** que el responsable coordinará con `TheHouse-GenAI` (§13); hasta entonces, `TheHouse-GenAI` puede ignorarlo sin romper su contrato.

```json
{
  "modelVersion": "baseline-1.0.0",
  "featureSchemaVersion": "2.0.0",
  "matchScore": 78,
  "breakdown": [
    {"factor": "budget", "points": 25.5, "maxPoints": 30, "feature": "price_below_min", "value": 0.15},
    {"factor": "distance", "points": 17.0, "maxPoints": 25, "feature": "distance_ratio", "value": 0.66},
    {"factor": "essential_services", "points": 13.3, "maxPoints": 20, "feature": "services_coverage", "value": 0.666667},
    {"factor": "place_type", "points": 10.0, "maxPoints": 10, "feature": "place_type_match", "value": 1},
    {"factor": "property_type", "points": 10.0, "maxPoints": 10, "feature": "property_type_match", "value": 1},
    {"factor": "extras", "points": 2.1, "maxPoints": 5, "feature": "extra_services_ratio", "value": 0.428571}
  ]
}
```

Suma sin redondear 77.976190 → `matchScore` 78; suma de los puntos reportados 77.9 (dentro de la tolerancia de 6.3). Los factores van siempre en el orden de la tabla 6.2. Si la coordinación con `TheHouse-GenAI` cambia los nombres de los campos de `breakdown`, se adoptan sus nombres sin cambiar los valores.

### 6.5 Etiquetas sintéticas

- **Simulador** puro con semilla fija y parámetros versionados en `src/thehouse_ml/synthetic/labels_params.json` 1.0.0 (cada parámetro con `value` y `basis`, como M-ML-025 a M-ML-027), aleatoriedad con `random.Random("labels:<seed>:<student_id>")` y semilla distinta de las de estudiantes (20261119), cuartos (20261905) y pares (20261120).
- **Utilidad oculta:** cada estudiante sintético tiene pesos ocultos **distintos** de los del modelo base: pesos de referencia documentados (propuesta: presupuesto 20 · distancia 35 · servicios 20 · tipo de lugar 10 · tipo de inmueble 5 · extras 10) con variación aleatoria por estudiante (propuesta: multiplicador lognormal con sigma 0.3 por factor, renormalizado a 100). Utilidad = Σ peso oculto × `f` de la §6.2 + ruido gaussiano (propuesta: sigma 5 puntos).
- **Relevancia graduada 0–3** dentro de los candidatos de cada estudiante, por posición de la utilidad (propuesta: 10 % superior → 3, siguiente 20 % → 2, siguiente 30 % → 1, resto → 0, con redondeo hacia arriba, así que todo estudiante con 2 o más candidatos tiene al menos un 3). Empates por `room_id`.
- **Salida:** `data/processed/labels.parquet` (fuera de git) con `student_id`, `room_id`, `split`, `utility`, `relevance`, `synthetic = true` y la versión de `labels_params.json`, para las tres particiones (ML-05 entrena con train). Reporte `reports/labels_synthetic.md` con «DATOS SINTÉTICOS».
- **Advertencia obligatoria** en todo reporte que las use: las etiquetas **no son preferencias reales**; son una función de utilidad documentada más ruido (constitución §8.2), con riesgo de circularidad (ML-04-RK02), y se reemplazan con señales reales en ML-07.
- Los valores marcados «propuesta» son SUPUESTOS documentados y se fijan con la decisión M-ML de la ejecución de T04.

### 6.6 Reporte del modelo base

`reports/baseline_report.md` y `reports/baseline_metrics.json`, encabezados con «DATOS SINTÉTICOS — ETIQUETAS SINTÉTICAS» y con la versión `baseline-1.0.0`, la del esquema y la de `labels_params.json`:

1. Distribución de `matchScore` y de los puntos por factor en los pares candidatos.
2. Ejemplos explicados con su desglose (estudiantes y cuartos sintéticos, sin datos personales).
3. Métricas de ranking por estudiante en la partición **test** de ML-03 (300 estudiantes), las de la constitución §10.2:
   - **NDCG@10 (principal)** y NDCG@5, con ganancia `2^rel − 1` y descuento `log2(posición + 1)`;
   - **Precision@5** y **MAP**, con relevancia binaria: un cuarto es relevante si `relevance ≥ 2` (SUPUESTO, se fija en T06);
   - calculadas solo con estudiantes de **2 o más candidatos** (se reporta cuántos quedan fuera).
4. **Intervalos** (constitución §10.3): bootstrap por estudiante (se remuestrean estudiantes con reemplazo, no pares) con semilla fija y número de remuestreos documentados en el reporte y en `reports/baseline_metrics.json` (propuesta, SUPUESTO que se fija en T06: 1,000 remuestreos, intervalo percentil del 95 %, `random.Random("bootstrap:<seed>")`); cada métrica se reporta con su intervalo.
5. **Segmentos** (constitución §11.3): en total, por campus, sin las 3 `seed_sin_verificar` y por **rango de presupuesto** (`budget.max`; propuesta, SUPUESTO que se fija en T06: ≤ $2,000, $2,001–$3,500 y > $3,500), con el número de estudiantes de cada segmento.
6. Estas métricas son la **referencia que ML-05 debe superar** (constitución §10.1: sin mejora demostrada no se promueve).

## 7. Reglas

1. Mismo resultado para la misma entrada en Python y en el backend (casos dorados, §6.3).
2. Pesos, curvas y redondeo solo cambian con una versión nueva del modelo base (`baseline-x.y.z`).
3. Los pesos de `baseline-1.0.0` se fijan antes de generar las etiquetas y no se ajustan con ellas: las etiquetas sirven para medir, no para calibrar el modelo base.
4. Las métricas sobre etiquetas sintéticas nunca se presentan como desempeño real (constitución §8 y §20.3).
5. Las 3 features nuevas solo se agregan al final de `feature_order`; cualquier otro cambio del esquema requiere otra versión y otra decisión.

## 8. Seguridad, datos y privacidad

- Sin atributos sensibles como factor ni en el simulador de etiquetas (constitución §11).
- Solo datos sintéticos: `pairs.parquet` y `labels.parquet` quedan en `data/processed/` (fuera de git); los reportes y casos dorados no contienen datos personales.
- Sin dependencias nuevas: biblioteca estándar, pandas y pyarrow (constitución §5).

## 9. Requisitos verificables

| ID | Requisito | Criterio de aceptación | Fase | Tasks |
| --- | --- | --- | --- | --- |
| ML-04-R01 | Configuración versionada | Factores, pesos (total 100), curvas y reglas de redondeo de la §6.2–6.3 en un archivo con versión `baseline-1.0.0` validado al cargarlo (pesos suman 100, cada feature existe en el esquema); el reporte y la salida muestran la versión usada. | F1 | T01 |
| ML-04-R02 | Esquema 2.0.0 y features nuevas | `feature_schema.json` **2.0.0** (versión mayor) con `distance_ratio`, `price_below_min` y `extra_services_ratio` al final de `feature_order`, `changelog` y `metadata.versioning` actualizados; `compute_pair_features` las calcula; las 7 features anteriores no cambian; `pairs.parquet` regenerado de forma reproducible; casos de `pair_feature_cases.json` ampliados con valores calculados a mano. | F1 | T02 |
| ML-04-R03 | Puntaje y desglose | Puntaje 0–100 (`matchScore`) con desglose por factor (`breakdown`, campo nuevo propuesto) en el formato de la §6.4; pares excluidos sin puntaje; redondeo de la §6.3; pruebas de propiedades: monotonía en precio y en distancia, total entre 0 y 100 y desglose coherente (tolerancias de la §6.3). | F2 | T03 |
| ML-04-R04 | Etiquetas sintéticas | Simulador con semilla fija y `labels_params.json` versionado; pesos ocultos distintos de los del modelo base más ruido; relevancia 0–3 dentro de los candidatos de cada estudiante; `labels.parquet` fuera de git, reproducible byte por byte y etiquetado como sintético, con la advertencia de la §6.5. | F2 | T04 |
| ML-04-R05 | Casos dorados | Al menos 50 casos (entradas, features, desglose y total) calculados a mano o con un recuento independiente (no con la función de T03), con casos límite de la §6.2–6.3, exportados para la prueba de paridad del backend. | F2 | T05 |
| ML-04-R06 | Reporte del modelo base | Distribución de puntajes, ejemplos explicados y NDCG@10 (principal), NDCG@5, Precision@5 y MAP en la partición test, con estudiantes de 2 o más candidatos (cuántos quedan fuera); intervalos por bootstrap por estudiante con semilla y número de remuestreos documentados; en total, por campus, sin las 3 `seed_sin_verificar` y por rango de presupuesto; todo marcado como sintético; registrado como referencia de ML-05. | F3 | T06 |
| ML-04-R07 | Pruebas y evidencia | Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`. | F4 | T07 |

## 10. Decisiones

Se registran en `memoria.md` (M-ML-029, APROBADO). Los valores marcados SUPUESTO en la §6.5 y la §6.6 se fijan con la decisión M-ML de la ejecución de su task (T04 y T06); hasta entonces esas tasks no pueden cerrarse.

| # | Tema | Decisión | Estado | Tasks |
| --- | --- | --- | --- | --- |
| 1 | Esquema 2.0.0 | **Versión mayor** (constitución §18) que agrega `distance_ratio`, `price_below_min` y `extra_services_ratio` al final de `feature_order` (§6.1): cambia las entradas del ONNX y lo que calcula NV-23, así que el backend actualiza su spec (constitución §2.4). Se implementa en T02 con regeneración de `pairs.parquet`, casos ampliados y `metadata.versioning` actualizado | APROBADO | T02–T07 |
| 2 | Factores, pesos y curvas | `baseline-1.0.0`: presupuesto 30, distancia 25, servicios esenciales 20, tipo de lugar 10, tipo de inmueble 10, extras 5 con las curvas de la §6.2; «modalidad» de la constitución §12 = tipo de inmueble (`property_type_match`); `pets_compatible`, `price_to_budget` y `effective_distance_km` no puntúan; pares excluidos por HC01–HC03 sin puntaje; sin atributos sensibles ni género | APROBADO | T01, T03–T07 |
| 3 | Redondeo y formato | Puntos por factor con 1 decimal (mitad hacia arriba); total entero 0–100 = `floor(round6(Σ) + 0.5)` sobre la suma sin redondear, en .NET con `MidpointRounding.AwayFromZero`; tolerancias del desglose ≤ 0.5 y ≤ 0.3 (§6.3); orden del ranking por la suma sin redondear y `room_id`; salida con `matchScore` (nombre y tipo del `ExplainMatchRequest` de `TheHouse-GenAI`) y `breakdown` como campo nuevo propuesto (§6.4) | APROBADO | T01, T03, T05 |
| 4 | Etiquetas sintéticas | Simulador de la §6.5 con pesos ocultos distintos de los del modelo base, variación por estudiante, ruido y relevancia 0–3 por posición; `labels.parquet` fuera de git; declaradas como no reales y reemplazadas en ML-07. Pesos ocultos y parámetros: SUPUESTO, se aprueban en T04 (`labels_params.json` 1.0.0) | APROBADO | T04, T06 |
| 5 | Reporte y métricas de referencia | En test, con estudiantes de 2 o más candidatos: NDCG@10 (principal), NDCG@5, Precision@5 y MAP (constitución §10.2); intervalos por bootstrap por estudiante con semilla y número de remuestreos documentados (§10.3); en total, por campus, sin las `seed_sin_verificar` y por rango de presupuesto (§11.3); referencia que ML-05 debe superar (§6.6). Umbral de relevancia binaria, parámetros del bootstrap y rangos de presupuesto: SUPUESTO, se fijan en T06 | APROBADO | T06 |

## 11. Costo — COST-001

| Componente | Costo | Condición |
| --- | --- | --- |
| CPU local y GitHub Actions (repositorio público) | $0 | Sin servicios de pago ni GPU en la nube |

## 12. Riesgos

| ID | Riesgo | Impacto | Mitigación |
| --- | --- | --- | --- |
| ML-04-RK01 | Pesos arbitrarios | Recomendaciones poco útiles | Validación con el equipo y comparación posterior con señales reales (ML-07) |
| ML-04-RK02 | Circularidad de las etiquetas sintéticas: la utilidad oculta usa las mismas features que el modelo base | Métricas optimistas o sin sentido; ML-05 aprendería la función del simulador | Pesos ocultos distintos de los del modelo base, variación por estudiante y ruido (§6.5); los pesos del modelo base no se ajustan con las etiquetas (§7 regla 3); advertencia en todo reporte y reemplazo con señales reales en ML-07 |
| ML-04-RK03 | Paridad Python/backend: el backend calcula distinto (redondeo, punto flotante, faltantes) | Puntajes distintos en la app y en el reporte | ≥ 50 casos dorados con casos límite y redondeo mitad hacia arriba documentado (§6.3), entregados al backend para su prueba en .NET |

## 13. Dependencias

- ML-03 completa (M-ML-028).
- `TheHouse-Backend`: el esquema **2.0.0** (versión mayor) agrega 3 features que NV-23 debe calcular y cambia las entradas del ONNX; el backend debe actualizar su spec (constitución §2.4). Aviso a cargo del responsable.
- `TheHouse-GenAI`: `ExplainMatchRequest` (PR #4) recibe `matchScore` (entero 0–100), `PreferencesIn` y `PropertyIn`, sin desglose; agregar `breakdown` (§6.4) es un campo nuevo propuesto que el responsable coordinará con `TheHouse-GenAI`.

## 14. Trazabilidad y cobertura

| Métrica | Valor |
| --- | --- |
| Requisitos | 7 |
| Tasks | 7 |
| Requisitos sin task | 0 |
| Tasks sin requisito | 0 |
| Tasks completadas | 2 (T01, T02) |
| Estado documental | **✅ APROBADA** |
| Estado de ejecución | **🟡 EN EJECUCIÓN — T01 y T02 completas (2/7); ver `evidence.md`** |

## 15. Aprobación

Aprobada por César Yael López Trujillo el 2026-11-20 («Apruebo ML-04 1.0.0», M-ML-029), con las decisiones 1–5 de la §10 en APROBADO. Para ejecutar se requiere «EJECUTA» con versión, tasks y rutas permitidas; T02 modifica código y datos de ML-03 (`src/thehouse_ml/features/`, `tests/fixtures/pair_feature_cases.json` y `data/processed/pairs.parquet`), por lo que su «EJECUTA» debe autorizar esas rutas.
