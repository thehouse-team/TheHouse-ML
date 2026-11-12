# Guía para trabajar en TheHouse-ML

Esta guía es para quien implemente el proyecto. Explica **cómo** se trabaja; **qué** se construye está en las specs y **por qué** en la constitution.

Si algo de esta guía contradice a `constitution.md`, `AGENTS.md`, `memoria.md` o una spec, mandan ellos (en ese orden).

## 1. Antes de empezar

Lee en este orden (unos 30 minutos):

1. [`README.md`](README.md): qué es el proyecto.
2. [`specs/ARQUITECTURA.md`](specs/ARQUITECTURA.md): cómo encaja con el backend y GenAI, el pipeline y qué lleva cada spec.
3. [`constitution.md`](constitution.md): reglas que no se negocian (datos, licencias, evaluación, costo $0).
4. [`AGENTS.md`](AGENTS.md): qué requiere autorización.
5. [`memoria.md`](memoria.md): decisiones tomadas y pendientes.
6. La spec que vas a trabajar: `spec.md`, `plan.md` y `task.md`.

Para instalar: [`INSTALACION.txt`](INSTALACION.txt).

## 2. Roles

| Rol | Quién | Qué hace |
| --- | --- | --- |
| Responsable | César Yael López Trujillo | Aprueba specs y decisiones («Apruebo X.Y.Z»), autoriza ejecución («EJECUTA»), aprueba releases de modelos |
| Implementador | Quien tome la spec | Implementa las tasks autorizadas en una rama, con pruebas y evidencia |
| Revisor | Victor Manuel Jose Jose o Isai Axel Paniagua Francisco | Revisa y aprueba el pull request; nadie aprueba su propio PR |

## 3. Flujo SDD, paso a paso

```text
1. Spec en borrador        → specs/ml-XX-*/ con spec.md, plan.md y task.md
2. Revisión                → el responsable lee y resuelve las decisiones de la §10
3. Aprobación              → «Apruebo ML-XX X.Y.Z» (se registra en memoria.md)
4. Autorización            → «EJECUTA» con spec, versión, tasks y rutas permitidas
5. Rama                    → feat/ml-XX-nombre-corto desde main actualizado
6. Por cada task           → implementar → probar → revisar coherencia → revisar seguridad y licencias
7. Marcar la task          → [x] en task.md y COMPLETA en la matriz, solo con evidencia
8. evidence.md             → commit, comandos, resultados, cobertura, hallazgos, resultado final
9. Pull request            → plantilla completa, CI en verde, aprobación de otra persona
10. Cierre                 → estado de la spec en ✅ COMPLETA y índice specs/README.md actualizado
```

Reglas:

- **Nada se programa sin «EJECUTA».** Ni "un borrador rápido". Si la spec no alcanza, se cambia la spec primero.
- **Una task fuera del plan no se hace.** Si aparece trabajo nuevo, se anota como hallazgo y se propone una nueva versión de la spec.
- **Un control que no corriste no es PASS.** Si no pudiste ejecutar algo, dilo en la evidencia.
- **Cambiar la constitution o AGENTS** requiere aprobar la nueva versión.

## 4. Ramas, commits y pull requests

| Tipo | Rama | Ejemplo de commit |
| --- | --- | --- |
| Implementación de una spec | `feat/ml-02-datos` | `feat(ML-02): bases nacionales ponderadas del Censo 2020` |
| Corrección | `fix/ml-02-hash-manifiesto` | `fix(ML-02): verificar SHA-256 antes de descomprimir` |
| Documentos y specs | `docs/ml-03-1.0.0` | `docs(ML-03): spec 1.0.0 para revisión` |
| Pruebas o CI | `test/…`, `ci/…` | `ci: agregar pip-audit` |

- Nunca commit directo a `main` (constitution §19).
- Un PR por spec o por bloque coherente de tasks; si pasa de ~600 líneas de código, divídelo.
- El PR usa la plantilla de `.github/pull_request_template.md`.

## 5. Pruebas

Objetivo: cobertura **≥ 80 %** de `src/thehouse_ml` (constitution §16), y que cada requisito tenga al menos una prueba que lo demuestre.

| Tipo | Dónde | Qué prueba | Ejemplo |
| --- | --- | --- | --- |
| Unitaria | `tests/unit/` | Una función pura | Percentiles ordenados, regla de suficiencia, mapeo de servicios |
| Integración | `tests/integration/` | Una etapa completa con datos de muestra | `prepare` sobre archivos de `tests/fixtures/` produce el Parquet esperado |
| Higiene | `tests/test_repo_hygiene.py` | Reglas del repositorio | Ningún archivo de `data/raw`, `data/processed` o `models` en git |
| Privacidad | `tests/unit/test_privacy*.py` | Columnas prohibidas | El catálogo de cuartos no tiene nombres, URLs ni datos del anfitrión |
| Paridad | `tests/parity/` (ML-06) | Python y ONNX dan lo mismo | Casos dorados dentro de la tolerancia |

Reglas:

1. **Sin red en las pruebas.** Ninguna prueba descarga nada; se usan archivos pequeños en `tests/fixtures/`.
2. **Fixtures sintéticos.** Los archivos de muestra imitan el formato de INEGI o Inside Airbnb pero con valores inventados; nunca se copian datos reales al repositorio.
3. **Determinismo.** Toda aleatoriedad recibe una semilla; la misma semilla da el mismo resultado.
4. **Nombres que describen el comportamiento:** `test_municipio_sin_muestra_usa_el_estado`, no `test_1`.
5. **Una prueba falla por una sola razón.** Si necesitas muchos `assert` distintos, probablemente son varias pruebas.
6. **Las pruebas se escriben con la task, no después.** Una task sin prueba no está completa.

Comandos (disponibles cuando ML-01 esté implementada):

```bash
uv run pytest                       # todas
uv run pytest tests/unit            # solo unitarias
uv run pytest --cov=thehouse_ml --cov-report=term-missing
```

## 6. Calidad y seguridad

Todo esto corre en el CI de cada PR y debe pasar también en tu máquina antes de abrirlo:

```bash
uv run ruff format --check .        # formato
uv run ruff check .                 # lint
uv run mypy src                     # tipos (modo estricto)
uv run pytest --cov=thehouse_ml     # pruebas y cobertura
uv run pip-audit                    # vulnerabilidades de dependencias
```

`gitleaks` corre en el CI para detectar secretos.

## 7. Datos

- `data/raw/`, `data/processed/` y `models/` **nunca** van a git.
- Antes de descargar algo: la fuente debe estar en `memoria.md` (M-ML-003) como APROBADO y en `data/sources.toml` con su licencia.
- Toda variable de INEGI se confirma en el diccionario oficial de su producto; su nombre exacto se anota en `data/sources.toml`.
- Toda estimación de encuesta se pondera con el factor de expansión.
- Se cita a INEGI como autor y el nombre del producto; nunca se sugiere que INEGI respalda el proyecto.
- Las métricas con datos sintéticos se etiquetan como sintéticas en todo reporte.

## 8. Evidencia (`evidence.md`)

Se crea en la carpeta de la spec al ejecutarla, con la plantilla de `specs/_plantilla/evidence.md`. Debe permitir que otra persona repita lo que hiciste:

- Commit y rama.
- Comandos exactos y su resultado (pruebas: total y aprobadas; cobertura).
- Trazabilidad requisito → prueba.
- Hallazgos y desviaciones del plan (si hubo, ¿quién las aprobó?).
- Revisión de coherencia (constitution, AGENTS, memoria) y de seguridad y licencias.
- Resultado: **PASS**, **PASS WITH WARNINGS** (con las advertencias) o **BLOCKED** (con la causa).

## 9. Definición de terminado

Una spec está terminada cuando:

- [ ] Todas sus tasks están en `[x]` con evidencia.
- [ ] `ruff`, `mypy`, `pytest` (cobertura ≥ 80 %) y `pip-audit` pasan en local y en el CI.
- [ ] Ningún dato ni modelo entró a git.
- [ ] Las salidas de la tabla de "Resultados esperados" existen y cumplen su verificación.
- [ ] `evidence.md` completo con resultado final.
- [ ] Encabezados de `spec.md`, `plan.md` y `task.md` en ✅ COMPLETA y `specs/README.md` actualizado.
- [ ] PR aprobado por otra persona y mezclado.

## 10. Resultados esperados por spec

| Spec | Al terminar debe existir | Cómo comprobarlo |
| --- | --- | --- |
| ML-01 | Paquete instalable, CLI con 5 etapas, CI, prueba de higiene | `uv sync --frozen` sin errores; `uv run python -m thehouse_ml --help` lista `download`, `prepare`, `train`, `evaluate`, `export`; CI en verde en el PR |
| ML-02 | `data/sources.toml`, manifiestos, `national_baselines`, `regional_calibration`, `campus_context`, `rooms`, `data/universities.csv`, `reports/data_card.md` | `uv run python -m thehouse_ml download` y `prepare` terminan sin error; pruebas de esquema, privacidad y suficiencia en verde; tabla §6.2 de la spec |
| ML-03 | `feature_schema.json`, generador de estudiantes, pares con partición por estudiante | Prueba contra los catálogos de NV-11; misma semilla → mismos estudiantes; ningún estudiante en dos particiones |
| ML-04 | Puntaje 0–100 con desglose, pesos versionados, ≥ 50 casos dorados, reporte | El desglose suma el total; el puntaje no sube al subir la renta ni la distancia; reporte en `reports/` |
| ML-05 | Modelos entrenados, `metrics.json`, `reports/model_card.md` | Tabla NDCG@10, Precision@5 y MAP (media ± desviación) contra el modelo base; decisión de promoción escrita |
| ML-06 | Release `model-vX.Y.Z` con 6 archivos y `SHA256SUMS`, `docs/backend-integration.md` | Paridad Python ↔ ONNX en todos los casos dorados; 200 cuartos en < 20 ms |
| ML-07 | Ingesta anonimizada, métricas reales | Prueba de ausencia de datos personales; métricas con partición temporal |

## 11. Pendientes fuera de este repositorio

Estos cambios los hace el equipo del backend y bloquean specs de aquí:

| Pendiente | Repositorio | Bloquea |
| --- | --- | --- |
| Constitution 1.2.0 del backend (M-BE-062: repositorios satélite sin despliegue) | `TheHouse-Backend` | Integración formal de ML-06 |
| NV-23: cargar el ONNX con `Microsoft.ML.OnnxRuntime` en lugar de entrenar con ML.NET | `TheHouse-Backend` | Uso del modelo en la app |
| NV-24: consumir `TheHouse-GenAI` para las explicaciones | `TheHouse-Backend` | Explicaciones en la app |
| Spec de exportación anonimizada de favoritos y visitas | `TheHouse-Backend` | ML-07 |
| Corregir en el seed de NV-11 las coordenadas que ML-02 reporte | `TheHouse-Backend` | Distancias exactas |

## 12. Qué no hacer

- Programar sin «EJECUTA» o fuera de las tasks autorizadas.
- Subir datos, modelos, notebooks con salidas o secretos.
- Hacer scraping de portales o usar una fuente sin licencia registrada.
- Usar pickle para entregar modelos (solo ONNX).
- Presentar métricas de datos sintéticos como desempeño real.
- Agregar dependencias fuera de la tabla de la constitution §5 sin decisión M-ML.
- Aprobar tu propio PR o mezclar sin la aprobación de otra persona.
