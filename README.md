# TheHouse ML

Modelo de compatibilidad de **Smart Match** para TheHouse: dado el perfil de búsqueda de un estudiante y un cuarto, calcula un puntaje de 0 a 100 con desglose por factor (presupuesto, distancia, servicios, tipo de lugar, modalidad y extras).

Es un proyecto separado del backend. No se despliega: entrega un artefacto ONNX versionado que `TheHouse-Backend` carga en NV-23.

## Estado

Constitution y AGENTS 1.0.0 aprobados (2026-11-09). Las specs ML-01 a ML-07 están en borrador: antes de escribir código se requiere aprobar cada spec y dar «EJECUTA».

Las explicaciones en lenguaje natural del puntaje son de [`TheHouse-GenAI`](https://github.com/thehouse-team/TheHouse-GenAI); este repositorio solo calcula el puntaje.

## Para empezar

- **Instalar y descargar:** [`INSTALACION.txt`](INSTALACION.txt).
- **Cómo se trabaja (SDD, ramas, pruebas, evidencia, definición de terminado):** [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **Qué hace el proyecto y qué lleva cada spec:** [`specs/ARQUITECTURA.md`](specs/ARQUITECTURA.md).

Uso rápido (Python 3.12 y [uv](https://docs.astral.sh/uv/)):

```bash
uv sync --frozen
uv run python -m thehouse_ml --help
uv run pytest
```

El código está bajo licencia MIT ([`LICENSE`](LICENSE)); los datos de terceros conservan su propia licencia.

## Documentos

| Documento | Contenido |
| --- | --- |
| [`constitution.md`](constitution.md) | Reglas de fondo: alcance, stack, datos, evaluación, equidad, contrato del artefacto y costo $0 |
| [`AGENTS.md`](AGENTS.md) | Cómo trabajan los agentes: autorizaciones, flujo, datos y cierre |
| [`memoria.md`](memoria.md) | Decisiones `M-ML-XXX` (fuentes de datos, stack, método) |
| [`specs/ARQUITECTURA.md`](specs/ARQUITECTURA.md) | Qué hace el proyecto, relación con el backend y GenAI, pipeline, estructura del código y qué lleva cada spec |
| [`specs/README.md`](specs/README.md) | Índice y orden de las specs ML-01 a ML-07 |
| [`specs/_plantilla/`](specs/_plantilla/) | Plantillas de `spec.md`, `plan.md`, `task.md` y `evidence.md` para specs nuevas |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Guía de trabajo para quien implemente |
| [`INSTALACION.txt`](INSTALACION.txt) | Instalación, verificación y problemas comunes |

## Datos y atribución

Los datos de terceros no se incluyen en este repositorio; se descargan con un script reproducible.

Fuentes previstas, pendientes de verificar su licencia antes de descargar (ver `memoria.md`, M-ML-003):
- [INEGI](https://www.inegi.org.mx/): Censo de Población y Vivienda 2020 (muestra), ENVI 2020, ENIGH 2022, DENUE e INPC, bajo sus [Términos de libre uso](https://www.inegi.org.mx/inegi/terminos.html). Fuente: INEGI. TheHouse no está respaldado por INEGI.
- [Inside Airbnb](https://insideairbnb.com/) (Ciudad de México).
- [OpenStreetMap](https://www.openstreetmap.org/copyright) (© colaboradores de OpenStreetMap, ODbL).
- Perfiles de estudiantes sintéticos generados desde el esquema de preferencias de TheHouse.
