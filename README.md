# MISO Hydrothermal Vent Temperature Analysis

Analyzing high-resolution hydrothermal vent temperature time series at Axial Seamount to understand magma-hydrothermal interactions.

## Overview

This project studies temperature variability at multiple hydrothermal vents in the International District vent field at Axial Seamount. By comparing vent temperatures with volcanic deformation (BPR) and seismicity data, we investigate how hydrothermal discharge responds to magmatic processes.

**Key Finding:** Hell vent shows a strong anti-correlation with volcanic inflation—as the magma chamber pressurizes, vent temperature decreases. This suggests pressure-driven changes in upflow-zone permeability.

![Hell Temperature vs Deformation](outputs/figures/hell_bpr_normalized.png)

## Research Questions

1. How does hydrothermal discharge vary in relation to magma movements and changes in upflow-zone permeability?
2. Do differences in crustal strain lead to differences in permeability among vent fields?
3. How does venting temperature respond differently across vents with varying proximity to magma sources?

## Vents Analyzed

| Vent | Temperature Range | Characteristics |
|------|------------------|-----------------|
| Inferno | 278-294°C | Relatively stable, gradual variations |
| Hell | 296-316°C | Hottest vent, steady cooling trend |
| El Guapo | 103-315°C | Most dynamic, large temperature swings |

## Data Sources

| Source | Description | Path |
|--------|-------------|------|
| MISO Sensors | High-resolution temperature loggers | `/home/jovyan/my_data/axial/axial_miso/` |
| BPR | Differential uplift (sibling project) | `../my-analysis_botpt/outputs/data/` |
| Earthquakes | Daily seismicity (sibling project) | `../earthquakes_my-analysis/outputs/data/` |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/daxsoule/miso_my-analysis.git
cd miso_my-analysis

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python analysis.py
```

## Outputs

### Data Products

| File | Description |
|------|-------------|
| `outputs/data/{vent}_temperature.parquet` | Full temperature time series |
| `outputs/data/{vent}_daily.parquet` | Daily aggregates (mean, min, max, std) |

### Figures

For each vent (inferno, hell, el_guapo):

| File | Description |
|------|-------------|
| `{vent}_temperature.png` | Temperature time series |
| `{vent}_bpr_overlay.png` | Temperature + BPR (dual axis) |
| `{vent}_bpr_normalized.png` | Temperature + BPR (z-score normalized) |
| `{vent}_earthquake_overlay.png` | Temperature + seismicity (dual axis) |
| `{vent}_earthquake_normalized.png` | Temperature + seismicity (z-score normalized) |

## Quality Control

### Settling Window
- Exclude first 24 hours after sensor first exceeds 200°C
- Removes deployment transients and equilibration artifacts

### MAD Spike Removal
- Rolling 24-hour median absolute deviation (MAD)
- Flag points > 5× scaled MAD from rolling median
- More robust to outliers than standard deviation

| Vent | Spikes Removed |
|------|----------------|
| Inferno | 152 (0.16%) |
| Hell | 31 (0.03%) |
| El Guapo | 333 (0.35%) |

## Normalization

Z-score normalization enables direct comparison of signals with different units:

```python
z = (x - mean) / std
```

This transforms both temperature and deformation to have mean=0 and std=1, revealing when anomalies in both signals coincide.

## Project Structure

```
miso_my-analysis/
├── .gitignore
├── .specify/
│   └── memory/constitution.md      # Project standards
├── README.md                       # This file
├── analysis.py                     # Main analysis script
├── requirements.txt                # Python dependencies
└── outputs/
    ├── data/                       # Parquet data products
    └── figures/                    # PNG visualizations (15 total)
```

## Related Projects

- `../my-analysis_botpt/` - BPR differential uplift analysis
- `../earthquakes_my-analysis/` - Earthquake catalog analysis

Together these three projects form a multi-instrument view of Axial Seamount's magma-hydrothermal system.

## References

- German, C. R., et al. (2016). Hydrothermal impacts on trace element and isotope ocean biogeochemistry. *Phil. Trans. R. Soc. A*, 374(2081).
- Nooner, S. L., & Chadwick, W. W. (2016). Inflation-predictable behavior and co-eruption deformation at Axial Seamount. *Science*, 354(6318), 1399-1403.
- OOI Cabled Array: https://oceanobservatories.org/array/cabled-array/

## License

This analysis uses data from the MISO project and the Ocean Observatories Initiative.

## Author

Dax Soule
January 2026
