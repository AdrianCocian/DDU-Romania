# National boundary of Romania

`ro_frontiera_poligon.shp` (with `.shx`, `.dbf`, `.prj`, `.cpg`) is the polygon of the state border of Romania.
The pipeline uses it to select the grid cells of the study area (step 02) and to draw the outline on the maps.

| | |
|---|---|
| Layer | "Frontieră România poligon" |
| Author | Agenția Națională de Cadastru și Publicitate Imobiliară (ANCPI), the National Agency for Cadastre and Land Registration of Romania |
| Version | 23 December 2021 (attribute `version` = 2021-12-23) |
| Obtained from | geo-spatial.org, "România: seturi de date vectoriale generale", section "Frontiera de stat a României": https://geo-spatial.org/vechi/download/romania-seturi-vectoriale |
| Coordinate system | Stereo 70, EPSG:3844 (step 02 reprojects it to EPSG:4326, the grid of ERA5-Land) |
| Content | one polygon; attributes `id`, `country` = RO, `name` = România, `version` |
| Changes | none; the files are distributed as downloaded |

## Licence

The MIT licence of this repository covers the code only. This boundary file is not covered by it.

According to the page it was downloaded from, the data were created by ANCPI, published on the INSPIRE Geoportal
of Romania and on the open data portal of the Government of Romania, and can be reused under the terms of the
Romanian Open Government Licence (Licența pentru Guvernare Deschisă, OGL-ROU-1.0; text:
http://data.gov.ro/base/images/logoinst/OGL-ROU-1.0.pdf).

Attribution: Conține informații publice în baza Licenței pentru Guvernare Deschisă v1.0 (contains public information
under the Open Government Licence v1.0). Source: ANCPI, state border of Romania, version 23.12.2021, processed and
distributed by the geo-spatial.org community.

## Use in the article

Step 02 keeps every cell of the 0.1° ERA5-Land grid that the polygon touches (`CLIP_ALL_TOUCHED = True` in
`config.py`). The polygon touches 2,937 cells. In the published run 25 of them, all on the Black Sea coast and the
coastal lagoons (longitude 28.6° E or more), carry no value, because ERA5-Land provides values over land only; this
leaves the 2,912 cells of the article. `tests/test_stats_published.py` checks the cell list of the published run
against this file.

To repeat the analysis for another region, point the environment variable `DDU_SHP_RO` to another polygon file and
change `BBOX` in `config.py` (see `docs/reproducibility.md`).
