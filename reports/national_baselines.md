# Bases nacionales de renta — ML-02 T03

Generado por `uv run python -m thehouse_ml prepare`. Fuente: INEGI, Encuesta Nacional de Vivienda (ENVI) 2020
y Encuesta Nacional de Ingresos y Gastos de los Hogares (ENIGH) 2022, citadas como autor del producto.
INEGI no respalda este proyecto.

## Pesos de julio de 2024

- Todas las rentas en pesos ajustados están expresadas en **pesos de julio de 2024**;
  es el último mes disponible en el INPC de datos abiertos (límite documentado, M-ML-014).
- Índice general INPC (2024-07) = 136.003 (base 2Q jul 2018 = 100).
- ENVI 2020: promedio oct-dic 2020 = 108.967;
  factor = 1.248112.
- ENIGH 2022: promedio ago-nov 2022 = 124.912;
  factor = 1.088793.

> Limitación: julio de 2024 es el último mes del INPC publicado en el archivo de datos abiertos.
> Si se ejecuta más adelante con una serie más reciente, hay que actualizar el mes de referencia.

## Resumen: nacional, Hidalgo (13) y Estado de México (15)

Los percentiles de la tabla están en **pesos de julio de 2024**. `p50 (año)` es la mediana
en pesos del año del levantamiento (2020 u 2022). `% rentadas` es la proporción ponderada.

| Fuente | Ámbito | p50 (año) | p10 | p25 | p50 | p75 | p90 | % rentadas | n | Σ factores |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ENVI 2020 | Nacional | 2,000 | 1,123 | 1,498 | 2,496 | 3,744 | 6,865 | 16.4 % | 8,170 | 5,619,414 |
| ENVI 2020 | Hidalgo | 1,500 | 811 | 1,248 | 1,872 | 2,995 | 3,744 | 9.9 % | 164 | 81,275 |
| ENVI 2020 | Estado de México | 2,000 | 1,248 | 1,623 | 2,496 | 3,495 | 4,992 | 16.3 % | 271 | 739,380 |
| ENIGH 2022 | Nacional | 2,000 | 1,089 | 1,633 | 2,178 | 3,811 | 6,533 | 15.1 % | 11,437 | 5,566,173 |
| ENIGH 2022 | Hidalgo | 2,000 | 980 | 1,470 | 2,178 | 2,722 | 3,484 | 9.9 % | 213 | 88,536 |
| ENIGH 2022 | Estado de México | 2,000 | 1,089 | 1,633 | 2,178 | 3,266 | 4,355 | 14.7 % | 387 | 706,980 |

## Método

- **Percentil ponderado:** el menor valor cuya suma acumulada de factores (valores ordenados de
  menor a mayor, empatados agrupados) alcanza `q * suma de factores`; se usa `q` = 0.10, 0.25, 0.50,
  0.75 y 0.90. Los montos de 2024 se obtienen multiplicando cada renta por el factor del deflactor.
- **Viviendas rentadas:** ENVI `P5_1 == "1"`; ENIGH `tenencia == "1"`. La proporción usa como
  denominador el total de viviendas del grupo, también ponderado.
- **Renta válida:** ENVI `P5_2` entre 100 y 50000; se excluyen 0 (`Nada`), 999999888 (`No responde`),
  999999999 (`No sabe`) y blanks. ENIGH `renta > 0` entre quienes pagan renta.
- **Niveles:** nacional, entidad (`ENT` / primeros 2 dígitos de `ubica_geo`), tamaño de localidad
  (`TLOC` / `tam_loc`) y clase o tipo de vivienda (`P4_3` / `tipo_viv`).
- **`n_muestra`** cuenta las viviendas rentadas con renta válida (sin ponderar); `Σ factores` es la
  suma de sus factores. `n_total` y `Σ factores total` son el denominador de la proporción.
- Los grupos sin renta válida o sin peso total no generan fila.
