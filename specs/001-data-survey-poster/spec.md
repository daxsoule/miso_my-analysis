# Analysis Specification: Recent Deployment Data Survey & Poster Figures

**Directory**: `specs/001-data-survey-poster`
**Created**: 2026-02-02
**Status**: Draft
**Input**: Survey all recent MISO deployment data (2022-2024 and 2024-2025) to identify records with consistent high hydrothermal temperatures and produce poster-quality figures.

## Research Question(s)

1. Which recent MISO temperature records (2022-2025) show consistent high-temperature hydrothermal venting suitable for further analysis?
2. How do vent temperatures compare across instruments and deployments during the 2022-2025 period?

**Hypothesis**: The 2024-2025 deployment expanded coverage from International District (3 vents) to primarily ASHES field (5 vents) plus El Guapo (Top). Inferno (ASHES) and El Guapo (Top) show consistent high-temperature venting across deployments, while newly instrumented ASHES vents (Virgin, Trevi, Hell) may show different thermal behavior than their International District counterparts.

## Data Description

### Primary Data: 2022-2024 Deployment

- **Source**: MISO high-temperature sensors deployed at ASHES and International District vents (Axial Seamount)
- **Coverage**: Aug 2022 – Jun 2024 (~22 months)
- **Format**: CSV with datetime + temperature columns
- **Access**: `/home/jovyan/my_data/axial/axial_miso/2022_2024_MISO/`
- **Files**:
  | File | Vent | Temp Range | Notes |
  |------|------|------------|-------|
  | `MISOTEMP_2017-002_Axial2022_inferno.csv` | Inferno | 278–294°C | Stable |
  | `MISOTEMP_2017-019_Axial2022_hel.csv` | Hell | 296–316°C | Cooling trend |
  | `MISO_2017-006_Axial2022_ElGuapo.csv` | El Guapo | 103–315°C | Most dynamic |
- **Known issues**: Missing-data flag `-888.88`; deployment settling transients in first ~24 hrs above 200°C
- **Subdirectory files** (`hell_2022_2024/`, `inferno_2022_2024/`): Internal reference thermistors (21-22°C). These are **not** vent temperatures — they are redundant with the logger data and should be excluded from analysis.

### Primary Data: 2024-2025 Deployment

- **Source**: MISO HiT (High-Temperature) sensors, 2024 deployment at Axial Seamount
- **Coverage**: Jun/Jul 2024 – Aug 2025 (~13 months)
- **Format**: CSV with multiple header formats (see below)
- **Access**: `/home/jovyan/my_data/axial/axial_miso/2024_2025_MISO/`
- **HiT deployment files** (contain vent thermocouple J-Type temperatures):
  | File | Instrument | Vent | Field | Header Format | Temp Column | J-Type Range | % >100°C | Assessment |
  |------|-----------|------|-------|--------------|-------------|--------------|----------|------------|
  | `HiT_2023-002_Axial_2024_Deployment.csv` | MISO 2023-002 | Hell | ASHES | Format A | `J-Type` | 0–80°C | 0% | Low-temp / ambient |
  | `HiT_2023-005_Axial_2024_Deployment_0.csv` | MISO 2023-005 | Inferno | ASHES | Format B | `Temp` | 0–315°C | 99% | **High-temp vent** |
  | `HiT_2023-007_2024_Axial_Deployment.csv` | MISO 2023-007 | Virgin | ASHES | Format B | `Temp` | 0–294°C | 32% | Intermittent / partial |
  | `HiT_2023-009_Axial_OOI_2024_deployment.csv` | MISO 2023-009 | El Guapo (Top) | Int. District | Format C | J-Type (col 2) | -14–343°C | 89% | **High-temp vent** |
  | `HiT_2023-010_Axial_2024_Deployment.csv` | MISO 2023-010 | Trevi / Mkr156 | ASHES | Format C | J-Type (col 2) | 0–1229°C | 45% | Suspect max; needs QC |
  | `HiT_2023-012_Axial_OOI_2024_deployment.csv` | MISO 2023-012 | Vixen / Marker 218 | ASHES | Format C | J-Type (col 2) | 22–24°C | 0% | Only 34 rows; trivial |
- **Logger files** (`2212*_logger-*.csv`): Internal reference thermistors (2-33°C). These are **not** vent temperatures and should be excluded from the primary survey.
- **Known issues**:
  - Missing-data flag `-888.88` for temperature, `-103` to `-115` for reference
  - Three different CSV header formats require format-specific parsers
  - Vent names now mapped (see table above): 5 ASHES vents + 1 International District vent
  - `HiT_2023-010` has a max of 1229°C which exceeds physical limits (~400°C for black smokers); likely sensor malfunction
  - `HiT_2023-012` has only 34 rows — likely failed or aborted deployment

**Header Format Reference:**

| Format | Files | Header Row | Date Column | Temp Column |
|--------|-------|-----------|-------------|-------------|
| A | 2023-002 | Row 1: `idx#,DateTime,J-Type,IntTemp,...` | `DateTime` | `J-Type` |
| B | 2023-005, 2023-007 | Row 1: `idx,DateTime,Temp,RefTempC,...` | `DateTime` | `Temp` |
| C | 2023-009, 2023-010, 2023-012 | Row 1: Plot title; Row 2: detailed headers | Col 1 (`Date Time, GMT+00:00`) | Col 2 (J-Type) |

## Methods Overview

1. **Data loading**: Parse all HiT deployment files from both deployment periods using format-specific readers. Exclude logger/reference thermistor files. Handle missing-data flags (`-888.88`) by replacing with NaN.

2. **Quality control**: Apply existing MAD-based spike removal (from `analysis.py`). Flag and cap physically impossible values (>400°C). Note deployment settling transients.

3. **Survey classification**: For each instrument record, compute summary statistics (median, mean, range, % time above 100°C) and classify as:
   - **High-temperature vent** (median >100°C, >80% of data above 100°C)
   - **Intermittent / partial** (some data above 100°C but inconsistent)
   - **Low-temperature / ambient** (consistently <100°C)

4. **Visualization**: Create overview and poster-quality figures showing all records.

**Justification**: A comprehensive data survey is needed before further analysis to ensure no usable high-temperature records are overlooked and to identify quality issues early. Poster figures require careful formatting for readability at conference scale.

## Expected Outputs

### Figures

- **Figure 1 — Data Survey Overview**: All instrument time series plotted together (subplots or stacked) showing the full time range of both deployments. Color-coded by classification (high-temp, intermittent, low-temp). Includes instrument ID labels. Purpose: verify all data has been considered.

- **Figure 2 — High-Temperature Vent Comparison**: Only records classified as consistent high-temperature vents. Daily-averaged temperatures overlaid on a single axis. Both deployment periods shown. Purpose: identify which records to carry forward for detailed analysis.

- **Figure 3 — Poster Figure: Temperature Time Series with Context**: Publication/poster-quality figure showing selected high-temperature vents with BPR differential uplift overlay. Follows constitution figure standards (PNG, 300 DPI, Magma palette, 7" width). Purpose: conference poster.

### Tables/Statistics

- **Table 1 — Instrument Summary**: For each file: instrument ID, date range, sample count, temperature statistics (min, median, mean, max), % above 100°C, classification, and notes on quality issues.

### Key Metrics

- Number of records with consistent high-temperature venting
- Temperature ranges and variability for each high-temp record
- Temporal overlap between deployments
- Any instruments that can be linked across deployments (same vent, different sensors)

## Validation Approach

- High-temp vents should show temperatures in the 100–400°C range (constitution: "High-temp vents typically 100-400°C; ambient ~2°C")
- Values exceeding 400°C should be flagged as sensor artifacts
- 2022-2024 results should be consistent with existing `analysis.py` outputs (Inferno 278-294°C, Hell 296-316°C, El Guapo 103-315°C)
- Check for temporal continuity — large gaps or sudden jumps indicate deployment/recovery events
- Cross-reference: instruments at the same vent across deployments should show broadly similar temperature ranges

## Completion Criteria

- [ ] All HiT deployment files from both 2022-2024 and 2024-2025 loaded and parsed
- [ ] Logger/reference thermistor files explicitly excluded with documented rationale
- [ ] Summary table produced for all instruments
- [ ] Classification applied (high-temp / intermittent / low-temp)
- [ ] Figure 1 (survey overview) generated
- [ ] Figure 2 (high-temp comparison) generated
- [ ] Figure 3 (poster figure with BPR overlay) generated
- [ ] All figures meet constitution standards (PNG, 300 DPI, labeled axes)
- [ ] Results reproducible from raw data via script
- [ ] Quality issues documented (2023-010 suspect max, 2023-012 insufficient data)

## Assumptions & Limitations

**Assumptions**:
- Vent-to-instrument mapping for 2024-2025 provided by PI: Hell (2023-002), Inferno (2023-005), Virgin (2023-007), El Guapo Top (2023-009), Trevi/Mkr156 (2023-010), Vixen/Marker218 (2023-012)
- The J-Type thermocouple column represents the vent fluid temperature in all HiT files
- UTC timezone applies to all records (per constitution)
- The existing QC approach (MAD spike removal, settling window) from `analysis.py` is appropriate for the new data

**Limitations**:
- This survey covers recent deployments only (2022-2025); historical data (pre-2022) is out of scope
- Poster figure design choices (layout, annotation) may require iteration with user feedback

## Notes

- The 2022-2024 data is already processed by `analysis.py` with outputs in `outputs/data/` and `outputs/figures/`. This spec adds the 2024-2025 data and creates a unified survey.
- **Inferno (ASHES)** (MISO 2023-005, median 309°C, 99% >100°C) and **El Guapo Top** (MISO 2023-009, median 341°C, 89% >100°C) are strong candidates for high-temperature vent analysis.
- **Virgin (ASHES)** (MISO 2023-007, 32% >100°C) may represent a vent with variable flow or a sensor that was intermittently displaced.
- **Trevi / Mkr156** (MISO 2023-010) needs careful QC — the 1229°C max is unphysical and the 45% above 100°C suggests mixed behavior.
- **Hell (ASHES)** (MISO 2023-002, max 80°C) — notably lower temperature than Hell in the 2022-2024 International District deployment (296-316°C). This is a *different* Hell vent in the ASHES field.
- **Vixen / Marker 218** (MISO 2023-012) — only 34 rows, effectively a failed deployment.
- The 2024-2025 deployment shifts geographic focus from International District to ASHES, with El Guapo (Top) as the only continuing ID vent.
- **Tiny Tower / MISO 2017-002** — instrument was flooded during deployment and did not produce data. No data file exists.
- **TODO: 2010-2011 deformation data** — The BPR data only starts from 2015 (OOI cabled array). To overlay deformation on the April 2011 eruption figure, need to obtain campaign pressure or GPS data from that era.
