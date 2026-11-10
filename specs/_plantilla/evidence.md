# ML-XX — Evidencia

| Campo | Valor |
| --- | --- |
| ID | ML-XX |
| Documento | `evidence.md` |
| Versión de la spec | X.Y.Z |
| Fecha | AAAA-MM-DD |
| Autorización | «EJECUTA» — quién, cuándo, tasks y rutas |
| Rama y commit | `feat/ml-XX-…` (`abc1234`) → PR #N |
| Resultado | **PASS** / **PASS WITH WARNINGS**: advertencias / **BLOCKED**: causa |

## 1. Cambios

| Área | Archivo | Cambio |
| --- | --- | --- |
| … | `src/thehouse_ml/…` | … |

## 2. Comandos y resultados

| Comando | Resultado |
| --- | --- |
| `uv sync --frozen` | |
| `uv run ruff format --check .` | |
| `uv run ruff check .` | |
| `uv run mypy src` | |
| `uv run pytest --cov=thehouse_ml` (dos corridas) | pruebas: … / … · cobertura: … % |
| `uv run pip-audit` | |
| Etapas del pipeline ejecutadas | |

## 3. Resultados esperados

| Salida (spec §6.1) | ¿Existe? | Verificación |
| --- | --- | --- |
| … | Sí / No | Prueba o comando y su resultado |

## 4. Trazabilidad requisito → prueba

| Requisito | Pruebas |
| --- | --- |
| R01 | `test_…` |

## 5. Datos y licencias

- Fuentes usadas, edición, fecha y SHA-256 (ver `data/manifests/`).
- Atribuciones incluidas.
- Confirmación de que ningún dato ni modelo entró a git.

## 6. Hallazgos y desviaciones

| Hallazgo | Impacto | Decisión (quién aprobó) |
| --- | --- | --- |
| … | … | … |

## 7. Revisión de coherencia y seguridad

- Coherente con `constitution.md`, `AGENTS.md` y `memoria.md`: …
- Secretos, datos personales y licencias: …
- Métricas sintéticas etiquetadas como tales: …

## 8. CI

| Paso | Resultado |
| --- | --- |
| CI del PR (run …) | |
| Revisión y merge | Aprobado por …, mezclado en `main` (`…`) |
