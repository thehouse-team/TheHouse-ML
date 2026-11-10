## Spec y autorización

- Spec: `ML-XX` versión `X.Y.Z`
- Tasks de este PR: `T0X`, `T0Y`
- Autorización «EJECUTA»: <!-- quién y cuándo; para PR de solo documentos, escribe "solo documentos" -->

## Qué cambia

<!-- Resumen corto por capa o etapa del pipeline. -->

## Evidencia

| Control | Resultado |
| --- | --- |
| `uv run ruff format --check .` y `uv run ruff check .` | |
| `uv run mypy src` | |
| `uv run pytest --cov=thehouse_ml` | pruebas: … / … · cobertura: … % |
| `uv run pip-audit` | |
| CI del PR | |

## Lista de revisión

- [ ] Solo se tocaron las tasks y rutas autorizadas.
- [ ] Cada requisito tocado tiene al menos una prueba.
- [ ] Ningún archivo de `data/raw`, `data/processed` o `models` en el PR.
- [ ] Sin secretos, tokens ni datos personales.
- [ ] Fuentes nuevas registradas en `memoria.md` y `data/sources.toml` con su licencia.
- [ ] Métricas con datos sintéticos etiquetadas como sintéticas.
- [ ] `task.md` y `evidence.md` actualizados; desviaciones explicadas.
- [ ] Coherencia revisada con `constitution.md`, `AGENTS.md` y `memoria.md`.
