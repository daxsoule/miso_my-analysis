# Implementation Plan: Data Survey & Poster Figures

**Spec**: `specs/001-data-survey-poster/spec.md`
**Created**: 2026-02-02

## Pipeline Overview

Single script (`survey.py`) that:
1. Loads all 2022-2024 and 2024-2025 HiT deployment data
2. Applies QC (missing flags, physical caps, MAD spike removal)
3. Computes summary statistics and classifies each record
4. Generates three figures: survey overview, high-temp comparison, poster figure with BPR

## Script: `survey.py`

**Location**: Project root (alongside `analysis.py`)

**Phases**:
1. **Load & parse** — Format-specific CSV readers for all 9 instrument files (3 from 2022-2024, 6 from 2024-2025)
2. **QC** — Replace missing flags, cap >400°C, MAD spike removal, settling window
3. **Summary table** — Print/save instrument statistics and classification
4. **Figure 1** — Survey overview: all instruments as subplots
5. **Figure 2** — High-temp comparison: daily means overlaid
6. **Figure 3** — Poster figure: high-temp vents + BPR overlay

**Reuse from `analysis.py`**: `remove_spikes_mad()` function (copy into survey.py for self-containment).

## Data Flow

```
Raw CSVs (2022-2024 + 2024-2025)
  → Load with format-specific parsers
  → QC (flag removal, cap, MAD)
  → Summary stats + classification
  → Figures 1-3
  → outputs/figures/survey_*.png
  → outputs/data/survey_summary.csv
```

## Environment

Uses existing `requirements.txt` (pandas, matplotlib, numpy, scipy, pyarrow). No new dependencies.
