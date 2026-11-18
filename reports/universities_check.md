# Verificación de coordenadas de universidades — ML-02 T07

Generado por `uv run python -m thehouse_ml prepare` a partir del seed de TheHouse-Backend
(`TheHouse.Application/Common/UniversitySeedData.cs`, commit `b7d75955513320a0a97aa5daf6b3bbe92e9514d2`)
y de las respuestas de Nominatim (OpenStreetMap), una consulta por universidad.

**Atribución: © OpenStreetMap contributors (ODbL).**

Distancia haversine entre la coordenada del seed y el elemento educativo de OSM más cercano.
`ok` si la distancia es ≤ 200 m; `revisar` si es mayor; `sin_coincidencia`
si no hay un elemento OSM con `category/class=amenity` y `type=university|college|school`.
Las coincidencias dudosas se confirman en openstreetmap.org antes de usarlas en la T05.

| Código | Universidad | Seed (lat, lon) | OSM (lat, lon) | Distancia (m) | Estado |
| --- | --- | --- | --- | ---: | --- |
| `UTTT` | Universidad Tecnológica de Tula-Tepeji | 20.00903, -99.34471 | 20.0084106, -99.3449608 | 73.69 | ok |
| `UAEH-ESTE` | Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Tepeji del Río | 19.91703, -99.36116 | — | — | sin_coincidencia |
| `UAEH-ESTL` | Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Tlahuelilpan | 20.12991, -99.23041 | — | — | sin_coincidencia |
| `UAEH-ESAT` | Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Atotonilco de Tula | 20.01584, -99.23601 | — | — | sin_coincidencia |
| `ITSOEH` | Instituto Tecnológico Superior del Occidente del Estado de Hidalgo | 20.20551, -99.22178 | 20.2333739, -99.2096330 | 3347.54 | revisar |
| `TESJI` | Tecnológico de Estudios Superiores de Jilotepec | 19.91953, -99.57719 | — | — | sin_coincidencia |
| `UMB-JILO` | Universidad Mexiquense del Bicentenario, Unidad de Estudios Superiores Jilotepec | 19.91248, -99.57849 | — | — | sin_coincidencia |
| `UMB-VDC` | Universidad Mexiquense del Bicentenario, Unidad de Estudios Superiores Villa del Carbón | 19.72791, -99.46312 | — | — | sin_coincidencia |
| `UCAN` | Universidad Canadiense | 20.0517, -99.34507 | — | — | sin_coincidencia |
| `UNE-TULA` | Universidad del Nuevo México | 20.03108, -99.33629 | 20.0310934, -99.3363626 | 7.73 | ok |

## Universidades a corregir en NV-11

- `UAEH-ESTE` — Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Tepeji del Río: **sin_coincidencia** (sin candidato educativo).
- `UAEH-ESTL` — Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Tlahuelilpan: **sin_coincidencia** (sin candidato educativo).
- `UAEH-ESAT` — Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Atotonilco de Tula: **sin_coincidencia** (sin candidato educativo).
- `ITSOEH` — Instituto Tecnológico Superior del Occidente del Estado de Hidalgo: **revisar** (3347.54 m).
- `TESJI` — Tecnológico de Estudios Superiores de Jilotepec: **sin_coincidencia** (sin candidato educativo).
- `UMB-JILO` — Universidad Mexiquense del Bicentenario, Unidad de Estudios Superiores Jilotepec: **sin_coincidencia** (sin candidato educativo).
- `UMB-VDC` — Universidad Mexiquense del Bicentenario, Unidad de Estudios Superiores Villa del Carbón: **sin_coincidencia** (sin candidato educativo).
- `UCAN` — Universidad Canadiense: **sin_coincidencia** (sin candidato educativo).
