# MISO Hydrothermal Vent Temperature Analysis

## Research Context

This project analyzes high-resolution hydrothermal vent temperature time series
from Axial Seamount to understand magma-hydrothermal interactions. Temperature
sensors deployed inside vent orifices capture the thermal signature of
subsurface processes, including tidal modulation that constrains crustal
permeability.

**Research questions:**
- How does hydrothermal discharge vary, in space and time, in relation to
  magma movements and associated changes in upflow-zone permeability?
- How does venting temperature respond to magma movements across different
  vent fields with varying proximity to magma sources?
- Is there a temporal relationship between variations in focused vent
  temperature (MISO) and diffuse flow temperature (TMPSF) at the ASHES
  vent field?

**Vent fields studied:**
- ASHES (Inferno, Hell, Virgin, Trevi/Mkr156, Vixen/Mkr218)
- International District (Hell, El Guapo, El Guapo Top)
- Coquille (Casper, Vixen)
- CASM

Note: "Hell" exists as distinct vents in both ASHES and International District.

**Vent field coordinates:**

| Vent Field | Latitude | Longitude | Depth (m) |
|------------|----------|-----------|-----------|
| ASHES | 45° 56.0186'N | 130° 00.8203'W | 1540 |
| CASM | 45° 59.332'N | 130° 1.632'W | 1580 |
| Coquille | 45° 55.0448'N | 129° 59.5793'W | 1538 |
| International District | 45° 55.5786'N | 129° 58.7394'W | 1522 |
| Trevi | 45° 56.777'N | 129° 59.023'W | 1520 |

Decimal degrees (for code):

| Vent Field | Lat (°N) | Lon (°W) | Depth (m) |
|------------|----------|----------|-----------|
| ASHES | 45.9336 | -130.0137 | 1540 |
| CASM | 45.9889 | -130.0272 | 1580 |
| Coquille | 45.9174 | -129.9930 | 1538 |
| International District | 45.9263 | -129.9790 | 1522 |
| Trevi | 45.9463 | -129.9838 | 1520 |

**Related projects:**
- `../my-analysis_botpt/` - Differential uplift (volcanic deformation)
- `../earthquakes_my-analysis/` - Seismicity patterns
- `../my-analysis_tmpsf/` - ASHES diffuse flow temperatures (TMPSF)

## Core Principles

### I. Reproducibility

Analysis should be fully reproducible from raw data to final outputs.
Scripts run without manual intervention. Random seeds are fixed and
documented. Environment dependencies are explicit (requirements.txt,
environment.yml, or equivalent).

### II. Data Integrity

Raw data is immutable - all transformations produce new files, never
overwrite sources. Data lineage is traceable through the analysis chain.
Missing or suspect values are flagged, not silently dropped or filled.

### III. Provenance

Every output links back to: the code that produced it, the input data,
and key parameter choices. Figures and tables can be regenerated from
tracked artifacts. If you can't trace how a number was made, it doesn't
belong in the paper.

## Data Sources

### 1. MISO Temperature Sensors

- **Description**: High-resolution temperature loggers deployed in vent orifices
- **Local path**: `/home/jovyan/my_data/axial/axial_miso/`
- **Temporal coverage**: 2001-2025 (varies by vent)
- **Sampling interval**: 10-40 minutes (varies by instrument)

**Directory structure:**

| Directory | Description |
|-----------|-------------|
| `2022_2024_MISO/` | Recent deployment (Inferno, Hell, El Guapo) |
| `2024_2025_MISO/` | Current deployment |
| `castle/` | Castle vent (2001-2020) |
| `casper/` | Casper vent |
| `diva/` | Diva vent |
| `hell-inferno/` | Hell/Inferno vents |
| `trevi/` | Trevi vent |
| `virgin/` | Virgin vent |
| `vixen/` | Vixen vent |
| `CASM/` | CASM vent field |

**File formats:**

| Format | Pattern | Columns |
|--------|---------|---------|
| CSV | `MISOTEMP_*.csv`, `HiT_*.csv` | datetime, temperature, reference temp |
| TXT | `MISO*-Axial-*.txt`, `HOBO*-Axial-*.txt` | Date Time, Temperature (°C) |
| MAT | `*-ALL-*.mat` | MATLAB format (legacy) |

### 2. OOI Bottom Pressure Recorder Data (from sibling project)

- **Description**: Differential uplift time series
- **Path**: `../my-analysis_botpt/outputs/data/differential_uplift_*.parquet`
- **Temporal coverage**: 2015-01-01 to 2026-01-16

### 3. Earthquake Catalog (from sibling project)

- **Description**: Wilcock earthquake catalog
- **Path**: `../earthquakes_my-analysis/outputs/data/earthquake_daily.parquet`
- **Temporal coverage**: 2015-01-22 to 2026-01-25

### 4. TMPSF Diffuse Flow Temperatures (from sibling project)

- **Description**: OOI TMPSF (Temperature Mooring Sea Floor) instrument measuring diffuse hydrothermal flow at the ASHES vent field. 24 thermistor channels, QC-filtered and averaged.
- **Location**: ASHES vent field (45.933653°N, 130.013688°W) — co-located with 2024-2025 MISO deployments (Inferno, Hell, Virgin, Trevi, Vixen)
- **Reference designator**: RS03ASHS-MJ03B-07-TMPSFA301
- **Path**: `../my-analysis_tmpsf/outputs/data/`
  - `tmpsf_2015-2026_daily.parquet` — daily averaged temperatures (24 channels)
  - `tmpsf_2015-2026_hourly.parquet` — hourly averaged temperatures
  - `channel_statistics.csv` — per-channel summary (mean, std, regime)
- **Temporal coverage**: 2015-01-01 to 2026-01-22
- **Temperature range**: 2–5°C (diffuse flow; 12 "hot" channels, 12 "cool" channels)
- **Known issues**: Channel 06 failed in 2017; Channel 02 offset/drift 2015-2017

### Usage

```python
import pandas as pd

# Load MISO temperature data
temp = pd.read_csv('path/to/MISOTEMP_file.csv', parse_dates=['datetime'])

# Load BPR differential uplift
bpr = pd.read_parquet('../my-analysis_botpt/outputs/data/differential_uplift_daily.parquet')

# Load earthquake daily counts
eq = pd.read_parquet('../earthquakes_my-analysis/outputs/data/earthquake_daily.parquet')

# Load TMPSF diffuse flow temperatures
tmpsf = pd.read_parquet('../my-analysis_tmpsf/outputs/data/tmpsf_2015-2026_daily.parquet')

# Merge for multi-instrument analysis
merged = bpr.join(eq, how='inner').join(tmpsf, how='inner')
```

## Technical Environment

- **Language**: Python 3.12
- **Key packages**: pandas, matplotlib, scipy (for tidal analysis)
- **Working directory**: `/home/jovyan/repos/specKitScience/miso_my-analysis/`
- **Compute environment**: JupyterHub

## Coordinate Systems & Units

- **Temperature**: Degrees Celsius (°C)
- **Time zone**: UTC (assumed; verify with deployment logs)
- **Datetime formats**:
  - CSV: `MM/DD/YY HH:MM:SS AM/PM`
  - TXT: `MM/DD/YY HH:MM:SS.0`
- **Missing data**: NaN (numpy/pandas convention)

## Figure Standards

- **Format**: PNG, 300 DPI
- **Color palette**: Magma (colorblind-safe), consistent with sibling projects
- **Dimensions**: 7" width (double column)
- **Required elements**: Axis labels with units, timestamps, vent field labels

## Quality Checks

- **Temperature range**: High-temp vents typically 100-400°C; ambient ~2°C
- **Temporal consistency**: Check for gaps, duplicate timestamps
- **Sensor drift**: Compare reference temperature where available
- **Known events**: April 2015 eruption should show thermal signature
- **Tidal signal**: Expect semi-diurnal (~12.42 hr) modulation in temperature

## Project Structure

```
miso_my-analysis/
├── .gitignore
├── .specify/
│   ├── features/                   # Future analysis features
│   │   └── NNN-feature-name/
│   │       ├── plan.md
│   │       ├── spec.md
│   │       └── tasks.md
│   ├── memory/
│   │   └── constitution.md         # This document
│   ├── scripts/bash/               # Helper scripts
│   └── templates/                  # Spec/plan templates
├── README.md                       # Project documentation
├── analysis.py                     # Main analysis script
├── requirements.txt                # Python dependencies
└── outputs/
    ├── data/
    │   └── vent_temperature_*.parquet
    ├── figures/
    │   └── temperature_timeseries_*.png
    └── notebooks/
        ├── README.md
        ├── miso_analysis.ipynb
        └── environment.yml
```

## Project Notes

- BPR and earthquake data are pre-processed in sibling projects
- Key event: April 2015 eruption - expect thermal response across vent fields
- Tidal analysis requires sufficient temporal resolution (≤1 hour sampling)
- Multiple file formats require unified parsing approach
- Consider instrument-specific calibration factors
