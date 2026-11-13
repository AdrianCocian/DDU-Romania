# Data sources

## ERA5-Land hourly 2 m air temperature

- Provider: Copernicus Climate Change Service (C3S), Climate Data Store (CDS), https://cds.climate.copernicus.eu
- Dataset: `reanalysis-era5-land`, variable `2m_temperature`, hourly, 2005–2024, 0.1° grid (about 9 km);
  request box 48.5 N, 19.9 E, 43.5 N, 30.1 E.
- Data citation: Muñoz Sabater J. ERA5-Land hourly data from 1950 to present. Copernicus Climate Change Service
  (C3S) Climate Data Store (CDS); 2019. https://doi.org/10.24381/cds.e2161bac
- Article describing the data set (cited in the manuscript): Muñoz-Sabater J, et al. ERA5-Land: a state-of-the-art
  global reanalysis dataset for land applications. Earth Syst Sci Data. 2021;13:4349–83.
- Licence: CC BY 4.0. On 2 July 2025 the "Licence to use Copernicus Products" of the Copernicus data stores was
  replaced by the Creative Commons Attribution licence (ECMWF announcement:
  https://forum.ecmwf.int/t/cc-by-licence-to-replace-licence-to-use-copernicus-products-on-02-july-2025/13464).
- Access: free CDS account; the licence of the data set is accepted once in the account. The hourly files are not
  redistributed in this repository; step 1 downloads them.
- Attribution for the derived tables in `results/`: contains modified Copernicus Climate Change Service information
  2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus
  information or data it contains.
- Completeness: step 3b counts the hours without a value inside the study area for every year and writes
  `out/07_reports/qc_gaps_summary.csv`. Stage 3 of `tools/verify_release.py` archives this report for the published
  run in `results/` (see `results/README.md`).

## National boundary of Romania

- `data/boundaries/ro_frontiera_poligon.shp`: layer "Frontieră România poligon", ANCPI, version 23.12.2021,
  Stereo 70 (EPSG:3844), downloaded from geo-spatial.org.
- Licence: Romanian Open Government Licence (OGL-ROU-1.0); not covered by the MIT licence of the code.
  Attribution and details: `data/boundaries/README.md`.
- Use: selects the cells of the study area (cells touched by the polygon) and draws the outline on the maps.

## Outputs archived in this repository

- `results/run_20260908_131817_6a83be93/`: tables and per-cell values of the run reported in the manuscript; see
  `results/README.md`.
