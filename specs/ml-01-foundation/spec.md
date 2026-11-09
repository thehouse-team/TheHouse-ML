# ML-01 — Base del proyecto

| Campo | Valor |
| --- | --- |
| ID | ML-01 |
| Documento | `spec.md` |
| Versión | 1.0.0-borrador |
| Fecha | 2026-11-07 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Aprobación humana | Pendiente |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |
| Autoridad | `constitution.md` → `AGENTS.md` → `memoria.md` |
| Plan | `plan.md` 1.0.0-borrador |
| Tasks | `task.md` 1.0.0-borrador |
| Precondición | Constitution y AGENTS 1.0.0 aprobadas (M-ML-001) |
| Relación con el backend | Ninguna directa; fija las reglas que usan ML-02 a ML-07 |

Spec en borrador para revisión humana. No autoriza código, descargas de datos, dependencias ni cambios en GitHub: la ejecución requiere aprobación y «EJECUTA».

## 1. Contexto

`TheHouse-ML` empieza vacío. Antes de tocar datos o modelos necesita estructura, entorno reproducible, calidad automática y un ruleset igual de estricto que el del backend.

## 2. Objetivo

Dejar un repositorio Python reproducible, con calidad y seguridad automáticas en cada PR y un comando por etapa del pipeline, sin datos ni modelos todavía.

## 3. Precondiciones

1. Constitution y AGENTS 1.0.0 aprobadas.
2. Repositorio `thehouse-team/TheHouse-ML` creado con permiso explícito.
3. Autorización «EJECUTA».

## 4. Alcance

1. Paquete `src/thehouse_ml` con `pyproject.toml` y `uv.lock`.
2. Configuración de ruff, mypy y pytest con cobertura.
3. CLI `python -m thehouse_ml <etapa>` con etapas vacías (`download`, `prepare`, `train`, `evaluate`, `export`).
4. `.gitignore` que excluye `data/raw`, `data/processed` y `models`.
5. Workflow de CI y ruleset de `main`.
6. README con instrucciones de instalación y uso.

## 5. Fuera de alcance

- Descarga de datos (ML-02).
- Cualquier modelo (ML-04 en adelante).

## 6. Diseño

- `uv` fija Python 3.12 y todas las dependencias.
- CLI con `argparse` (sin dependencias extra); cada etapa lee y escribe rutas configurables.
- CI: `uv sync --frozen`, ruff, mypy, pytest con cobertura, pip-audit y gitleaks; corrida del pipeline con datos de muestra cuando existan.
- Ruleset de `main` con checks obligatorios, una aprobación de otra persona y sin bypass, igual que el backend.

## 7. Reglas

1. Ningún archivo de datos o modelo entra al repositorio.
2. Las dependencias solo se agregan por decisión M-ML o la tabla de la constitution.
3. Licencia del código: decisión pendiente (§10).

## 8. Seguridad, datos y privacidad

- gitleaks y pip-audit obligatorios en cada PR.
- Sin secretos en el repositorio ni en los logs de CI.

## 9. Requisitos verificables

| ID | Requisito | Criterio de aceptación | Fase | Tasks |
| --- | --- | --- | --- | --- |
| ML-01-R01 | Entorno reproducible | `uv sync --frozen` instala el entorno desde cero en Windows y Linux; `python -m thehouse_ml --help` lista las cinco etapas. | F1 | T01 |
| ML-01-R02 | Calidad automática | ruff, mypy (modo estricto en `src`) y pytest con cobertura corren localmente y en CI sin errores. | F2 | T02 |
| ML-01-R03 | Exclusión de datos y modelos | `.gitignore` excluye `data/raw`, `data/processed` y `models`; una prueba de CI falla si alguno aparece en el árbol de git. | F3 | T03 |
| ML-01-R04 | CI y ruleset | Workflow con los checks de la §6 en verde en un PR; ruleset de `main` activo con checks obligatorios y una aprobación. | F3 | T04 |
| ML-01-R05 | Pruebas y evidencia | Pruebas unitarias e integración de lo implementado con cobertura ≥ 80 % del código nuevo; ruff y mypy sin errores; `evidence.md` con commit, comandos, resultados, hallazgos y revisión de coherencia con `constitution.md`, `AGENTS.md` y `memoria.md`. | F4 | T05 |

## 10. Decisiones pendientes

Se registran en `memoria.md` al aprobarse; mientras estén PENDIENTE, las tasks que dependen de ellas quedan BLOCKED.

| Tema | Propuesta | Estado |
| --- | --- | --- |
| Gestor de entorno | `uv` | PROPUESTO |
| Licencia del código | MIT | PENDIENTE |
| Nombre del repositorio | `thehouse-team/TheHouse-ML` | PROPUESTO |

## 11. Costo — COST-001

| Componente | Costo | Condición |
| --- | --- | --- |
| CPU local y GitHub Actions (repositorio público) | $0 | Sin servicios de pago ni GPU en la nube |

## 12. Riesgos

| ID | Riesgo | Impacto | Mitigación |
| --- | --- | --- | --- |
| ML-01-RK01 | Diferencias Windows/Linux | Pipeline no reproducible | CI en Linux y pruebas locales en Windows |
| ML-01-RK02 | Datos subidos por error | Violación de licencias | `.gitignore` y prueba de árbol de git |

## 13. Dependencias

- Ninguna spec previa.

## 14. Trazabilidad y cobertura

| Métrica | Valor |
| --- | --- |
| Requisitos | 5 |
| Tasks | 5 |
| Requisitos sin task | 0 |
| Tasks sin requisito | 0 |
| Tasks completadas | 0 |
| Estado documental | **🟡 BORRADOR — PENDIENTE DE APROBACIÓN** |
| Estado de ejecución | **⛔ NO EJECUTAR HASTA «EJECUTA»** |

## 15. Aprobación pendiente

Para aprobar se requiere la revisión humana de esta versión y la resolución (o aceptación explícita como PENDIENTE) de las decisiones de la §10. Para ejecutar se requiere además «EJECUTA» con versión, tasks y rutas permitidas.
