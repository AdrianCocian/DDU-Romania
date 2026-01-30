# DDU Romania Pipeline (2005–2024)

This repository contains a reproducible, portable pipeline to quantify temperature-driven climatic suitability for Dirofilaria spp. transmission in Romania using Dirofilaria Development Units (DDU). The workflow ingests ERA5-Land hourly near-surface air temperature, clips to Romania, computes DDU-based season metrics (start/end/length/EIP days) using multiple calculation methods, and produces tables, plots, and maps for national-scale analysis.

Scope and assumptions
- The framework models temperature-bounded development only, within existing mosquito populations.
- It does not model vector abundance, biting rate, species composition, host density, prophylaxis, or intervention effects.
- Outputs should be interpreted as climatic suitability indicators, not observed infection prevalence.

Key outputs
- Per-year per-pixel season metrics (start DOY, end DOY, season length, total EIP days)
- Method comparison (consensus median across methods + pairwise deltas)
- WIN30 risk classes maps (fixed bins by season length)
- WIN30 summary workbook with distribution diagnostics and trend tests

Repository structure
- boundaries/                 Romania boundary shapefile (EPSG:4326)
- scripts/                    Step scripts (01–08)
- docs/                       Documentation (overview, reproducibility, sources)
- out/                        Generated outputs (gitignored)
- config.py                   Portable configuration and directory layout
- environment.yml             Conda environment for the pipeline
- run_pipeline.py             Orchestrator (steps 0–8)
- CITATION.cff                Citation metadata
- LICENSE                     Code license

Requirements
- Conda (recommended) and a working C/C++ toolchain if required by geospatial deps on your OS.
- Copernicus Climate Data Store (CDS) account and API key stored locally in .cdsapirc.

CDS API key setup (.cdsapirc)
Create the file in your home directory (do not commit it to Git):
- Windows: C:\Users\<USER>\.cdsapirc
- Linux/macOS: ~/.cdsapirc

CDS help pages
```text
https://cds.climate.copernicus.eu/
https://cds.climate.copernicus.eu/api-how-to
