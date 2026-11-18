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

## Verificación humana (openstreetmap.org)

Revisión del 2026-11-18 registrada en `data/seed/universities_manual.csv` (M-ML-017).
Solo se guardan coordenadas tomadas de OpenStreetMap; nunca de Google Maps, por su licencia.
`final_source` indica el origen de la coordenada final: `osm:<tipo>/<id>` si salió de Nominatim,
`seed` si se conserva la del backend o la referencia anotada por el revisor.

| Código | Decisión | Coordenada final | Referencia OSM | Nota |
| --- | --- | --- | --- | --- |
| `UAEH-ESTE` | ok_seed | 19.91703, -99.36116 | — | El marcador del seed cae sobre "Escuela Superior de Tepeji del Río UAEH" en OSM |
| `UAEH-ESTL` | corregida | 20.132062, -99.235535 | way/550950316 | Edificio "Universidad Autonoma Del Estado de Hidalgo" junto a AREA VERDE UAEH; el seed está a ~590 m |
| `UAEH-ESAT` | ok_seed | 20.01584, -99.23601 | — | El marcador del seed cae sobre "UAEH Plantel Atotonilco de Tula" en OSM |
| `ITSOEH` | corregida | 20.2333739, -99.2096330 | nominatim T07 | Se confirma el candidato de Nominatim; el seed del backend está a ~3.3 km |
| `TESJI` | no_localizada | 19.91953, -99.57719 | — | Sin resultados en Nominatim ni elemento visible en el mapa |
| `UMB-JILO` | ok_seed | 19.91248, -99.57849 | — | El marcador del seed cae sobre "Unidad de Estudios Superiores Jilotepec" en OSM |
| `UMB-VDC` | no_localizada | 19.72791, -99.46312 | — | Sin resultados; el seed cae en Plaza Hidalgo (centro del municipio) |
| `UCAN` | no_localizada | 20.0517, -99.34507 | — | Sin elemento en OSM; posiblemente sea "Instituto Canadiense A.C." (carretera Tepetitlán–Tula, El Tesoro); el nombre debe confirmarlo el equipo del backend |

## Correcciones para NV-11 (backend)

Sustituye la detección automática de la sección anterior por las decisiones humanas definitivas:

- `UAEH-ESTL` — Universidad Autónoma del Estado de Hidalgo, Escuela Superior de Tlahuelilpan: **corregida**. usar `20.132062, -99.235535` (way/550950316). Edificio "Universidad Autonoma Del Estado de Hidalgo" junto a AREA VERDE UAEH; el seed está a ~590 m
- `ITSOEH` — Instituto Tecnológico Superior del Occidente del Estado de Hidalgo: **corregida**. usar `20.2333739, -99.2096330` (nominatim T07). Se confirma el candidato de Nominatim; el seed del backend está a ~3.3 km
- `TESJI` — Tecnológico de Estudios Superiores de Jilotepec: **no_localizada**. mantener el seed `19.91953, -99.57719` marcado `seed_sin_verificar`. Sin resultados en Nominatim ni elemento visible en el mapa
- `UMB-VDC` — Universidad Mexiquense del Bicentenario, Unidad de Estudios Superiores Villa del Carbón: **no_localizada**. mantener el seed `19.72791, -99.46312` marcado `seed_sin_verificar`. Sin resultados; el seed cae en Plaza Hidalgo (centro del municipio)
- `UCAN` — Universidad Canadiense: **no_localizada**. mantener el seed `20.0517, -99.34507` marcado `seed_sin_verificar`. Sin elemento en OSM; posiblemente sea "Instituto Canadiense A.C." (carretera Tepetitlán–Tula, El Tesoro); el nombre debe confirmarlo el equipo del backend

T05 debe reportar por separado las universidades `seed_sin_verificar` y no usarlas como coordenadas
verificadas hasta que el backend confirme el nombre y la ubicación (en particular `UCAN`).
