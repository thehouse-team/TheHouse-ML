# TheHouse ML — Constitution

| Campo | Valor |
| --- | --- |
| Versión | 1.0.0 |
| Fecha | 2026-11-09 |
| Estado | **✅ APROBADA** |
| Aprobación humana | Victor Manuel Jose Jose — 2026-11-09 («si crealo», en respuesta a «¿Creo el repo TheHouse-ML con la constitución 1.0.0?»); responsable de `TheHouse-ML` |
| Decisión | M-ML-001 |

Autoridad: máxima norma de `TheHouse-ML` dentro de las instrucciones humanas y políticas del entorno. Orden de autoridad: `constitution.md` → `AGENTS.md` → `memoria.md` → specs.

## 1. Propósito

`TheHouse-ML` es el proyecto de Machine Learning de TheHouse. Produce el **modelo de compatibilidad de Smart Match**: dado el perfil de búsqueda de un estudiante y un cuarto, calcula un puntaje de 0 a 100 con desglose por factor. Es un proyecto separado del backend, con su propio código, datos, pruebas, evaluación y evidencia, para presentarse y evaluarse por sí mismo.

## 2. Alcance y relación con TheHouse-Backend

1. `TheHouse-ML` **no se despliega**: no expone API, no tiene base de datos en producción y no corre en Azure.
2. Su entregable es un **artefacto de modelo versionado e inmutable** (ONNX) con su esquema de entrada, su ficha (model card) y su reporte de evaluación.
3. El backend carga ese artefacto dentro del monolito, detrás del puerto `ICompatibilityModel` de NV-23. Así se respeta la regla del backend de "monolito modular, no microservicios" (M-BE-004/M-BE-022).
4. El contrato entre los dos repositorios es el **esquema de features**: las preferencias de NV-11 y los datos públicos del cuarto. Un cambio de esquema requiere cambio de spec en ambos repositorios.
5. La IA generativa no es parte de este proyecto: vive en el servicio separado `TheHouse-GenAI`, que recibe el puntaje ya calculado y su desglose y los explica, sin calcularlos ni modificarlos (constitución de `TheHouse-GenAI`, Artículo 2).

## 3. SDD obligatorio — SDD-001

Todo cambio de código, datos, modelo o CI se hace por spec: `spec.md` → `plan.md` → `task.md` → aprobación humana → «EJECUTA» → implementación → `evidence.md`. Sin spec aprobada no se escribe código ni se descargan datos.

## 4. Specs independientes — SDD-002

Cada spec (`specs/ml-XX-*`) tiene sus requisitos, tasks, criterios de aceptación y evidencia. Aprobar una spec no aprueba otra. El índice `specs/README.md` muestra el estado de todas.

## 5. Stack

| Área | Elección |
| --- | --- |
| Lenguaje | Python 3.12 |
| Entorno y dependencias | `uv` con `pyproject.toml` y `uv.lock` (versiones fijadas) |
| Datos | pandas y pyarrow (Parquet) |
| Modelos | scikit-learn (base y regresión logística) y LightGBM (ranking) |
| Exportación | ONNX (`skl2onnx` / `onnxmltools`) verificado con `onnxruntime` |
| Calidad | ruff (lint y formato), mypy (tipos), pytest con cobertura |
| Seguridad | pip-audit, gitleaks |
| Exploración | Jupyter, solo para análisis; ningún resultado oficial depende de un notebook |

Agregar una dependencia fuera de esta tabla requiere decisión M-ML.

## 6. Estructura del repositorio

```text
src/thehouse_ml/     código del paquete (datos, features, modelos, evaluación, exportación)
tests/               pruebas unitarias y de integración
data/raw/            datos descargados (fuera de git)
data/processed/      datos derivados (fuera de git)
data/manifests/      manifiestos con fuente, licencia, fecha y SHA-256 (en git)
models/              artefactos generados (fuera de git; se publican como release)
reports/             reportes de evaluación y model cards (en git)
notebooks/           exploración (sin salidas guardadas)
specs/               specs SDD
```

## 7. Datos: fuentes, licencias y privacidad

1. Solo se usan fuentes con licencia que permita el uso académico y la creación de obras derivadas, registradas en `memoria.md` con licencia, atribución y restricciones.
2. Los datos crudos de terceros **no se suben al repositorio** ni se republican; se descargan con un script reproducible y se verifican con SHA-256 del manifiesto.
3. Prohibido hacer scraping de portales cuyos términos lo impidan o no lo autoricen expresamente.
4. Sin datos personales: no se procesan nombres, correos, teléfonos, fotos ni identificadores de anfitriones o usuarios; si una fuente los trae, se eliminan en la primera transformación.
5. Los datos reales de TheHouse (favoritos y visitas) solo llegan como exportación anonimizada y agregada definida por una spec del backend.

## 8. Datos sintéticos honestos

1. Los estudiantes sintéticos se generan desde el esquema de preferencias de NV-11 con distribuciones documentadas y semilla fija.
2. Si las etiquetas se generan con una función de utilidad, el reporte lo declara y explica que el modelo aprende esa función más el ruido: las métricas sobre datos sintéticos no se presentan como desempeño real.
3. El desempeño real solo se afirma con señales reales (ML-07) o con una evaluación humana documentada.

## 9. Reproducibilidad

1. Semillas fijas, dependencias fijadas en `uv.lock` y manifiestos de datos con SHA-256.
2. Un comando reproduce cada etapa (`descargar → preparar → entrenar → evaluar → exportar`) y el mismo commit con los mismos datos produce el mismo artefacto y las mismas métricas.
3. Cada artefacto registra commit, versión de datos, hiperparámetros y métricas.

## 10. Evaluación

1. Todo modelo entrenado se compara contra el **modelo base por reglas** (ML-04); sin mejora demostrada no se promueve.
2. Métricas de ranking por estudiante: NDCG@10, Precision@5 y MAP; partición por estudiante (sin fuga entre entrenamiento y prueba) con validación cruzada.
3. Se reportan intervalos o variación entre particiones, no solo un número.

## 11. Equidad y restricciones

1. `livingWith` y el presupuesto máximo son **restricciones duras**: un cuarto que no las cumple no se recomienda, sin importar el puntaje.
2. El modelo no usa atributos sensibles (género, edad, origen) como señal de puntaje; el género solo interviene en la restricción `livingWith` que elige el propio estudiante.
3. El reporte de evaluación incluye un análisis por segmentos (universidad, rango de presupuesto) para detectar sesgos.

## 12. Explicabilidad

Cada puntaje incluye un desglose por factor (presupuesto, distancia, servicios, tipo de lugar, modalidad y extras) que suma el total. La explicación debe ser comprensible para el estudiante y es la entrada que `TheHouse-GenAI` redacta en lenguaje natural.

## 13. Contrato del artefacto

1. Formato ONNX, sin pickle: el backend nunca deserializa objetos Python.
2. Se publica con `feature_schema.json` (nombres, tipos, rangos y catálogos), `model_card.md` y `metrics.json`.
3. Versión semántica del modelo (`modelVersion`); un artefacto publicado no se modifica, se reemplaza por una versión nueva.
4. Pruebas de paridad: el mismo conjunto de casos dorados produce el mismo puntaje en Python y en `onnxruntime` (tolerancia documentada); los casos se entregan al backend para su prueba en .NET.
5. Presupuesto de inferencia: puntuar 200 cuartos en menos de 20 ms en CPU.

## 14. Seguridad

1. Sin secretos en el repositorio; si alguna etapa los necesitara, se usan secretos de GitHub Actions.
2. Dependencias fijadas y revisadas con pip-audit en cada PR.
3. Ningún artefacto se carga desde fuentes no verificadas; los artefactos publicados llevan SHA-256.

## 15. COST-001 — costo obligatorio $0

Entrenamiento y evaluación en CPU local o en GitHub Actions (repositorio público). Sin servicios de pago, GPU en la nube ni plataformas de ML administradas. Un plan gratuito no prueba por sí solo costo $0: se exige evidencia vigente.

## 16. Testing

Pruebas unitarias de transformaciones, features y reglas; pruebas de integración del pipeline con datos de muestra; pruebas de paridad del artefacto; cobertura mínima del 80 % en `src/thehouse_ml`.

## 17. CI

GitHub Actions en cada PR: ruff, mypy, pytest con cobertura, pip-audit, gitleaks y una corrida del pipeline con datos de muestra. Los checks son obligatorios en el ruleset de `main`.

## 18. Versionamiento

Versión semántica para el paquete y para cada modelo. Los cambios de esquema de features son cambios mayores.

## 19. Flujo de Git

Ramas cortas hacia `main` con pull request, checks obligatorios y una aprobación de otra persona. Prohibido hacer commit directo a `main` después del commit inicial.

## 20. Reglas para agentes

1. Leer `constitution.md`, `AGENTS.md` y `memoria.md` antes de cada tarea.
2. No ejecutar sin «EJECUTA»; no descargar datos sin spec aprobada.
3. No presentar métricas sintéticas como desempeño real.
4. Revisar coherencia documental y seguridad en cada iteración.

## 21. Prohibiciones y cambios de gobierno

Prohibido: subir datos crudos de terceros, procesar datos personales, usar pickle como formato de intercambio, scraping no autorizado y costos no aprobados. Cambiar esta constitución requiere aprobación humana explícita de la nueva versión («Apruebo X.Y.Z»).
