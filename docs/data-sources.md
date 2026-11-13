# Data sources

## ERA5-Land hourly 2 m air temperature

- Provider: Copernicus Climate Change Service (C3S), Climate Data Store (CDS), https://cds.climate.copernicus.eu
- Dataset: `reanalysis-era5-land`, variable `2m_temperature`, hourly, 2005–2024, ~0.1° (~9 km) grid.
- Reference: Muñoz-Sabater J, et al. ERA5-Land: a state-of-the-art global reanalysis dataset for land
  applications. Earth Syst Sci Data. 2021;13:4349–83.
- Access: free CDS account; the ERA5-Land licence must be accepted once in the account. The raw data are not
  redistributed in this repository; step 1 downloads them.

## Romania boundary

- `data/boundaries/ro_frontiera_poligon.shp` (EPSG:4326), used to clip the grid and to draw map outlines.

## Outputs archived in this repository

- `results/run_20260908_131817_6a83be93/`: tables and per-cell season lengths of the run reported in the
  manuscript; see `results/README.md`.
