# TheHouse ML

Modelo de compatibilidad de **Smart Match** para TheHouse: dado el perfil de búsqueda de un estudiante y un cuarto, calcula un puntaje de 0 a 100 con desglose por factor (presupuesto, distancia, servicios, tipo de lugar, modalidad y extras).

Es un proyecto separado del backend. No se despliega: entrega un artefacto ONNX versionado que `TheHouse-Backend` carga en NV-23.

## Estado

Constitution y AGENTS 1.0.0 aprobados (2026-11-09). Las specs ML-01 a ML-07 están en borrador: antes de escribir código se requiere aprobar cada spec y dar «EJECUTA».

Las explicaciones en lenguaje natural del puntaje son de [`TheHouse-GenAI`](https://github.com/thehouse-team/TheHouse-GenAI); este repositorio solo calcula el puntaje.

## Documentos

| Documento | Contenido |
| --- | --- |
| [`constitution.md`](constitution.md) | Reglas de fondo: alcance, stack, datos, evaluación, equidad, contrato del artefacto y costo $0 |
| [`AGENTS.md`](AGENTS.md) | Cómo trabajan los agentes: autorizaciones, flujo, datos y cierre |
| [`memoria.md`](memoria.md) | Decisiones `M-ML-XXX` (fuentes de datos, stack, método) |
| [`specs/README.md`](specs/README.md) | Índice y orden de las specs ML-01 a ML-07 |

## Datos y atribución

Los datos de terceros no se incluyen en este repositorio; se descargan con un script reproducible.

Fuentes previstas, pendientes de verificar su licencia antes de descargar (ver `memoria.md`):
- [Inside Airbnb](https://insideairbnb.com/) (Ciudad de México).
- [OpenStreetMap](https://www.openstreetmap.org/copyright) (© colaboradores de OpenStreetMap, ODbL).
- Perfiles de estudiantes sintéticos generados desde el esquema de preferencias de TheHouse.
