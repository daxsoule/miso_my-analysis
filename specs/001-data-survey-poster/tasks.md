# Tasks: Data Survey & Poster Figures

## Phase 1: Data Loading & QC

- [x] T001: Write `survey.py` — load all 9 instrument files from both deployments with format-specific parsers, apply QC, compute summary table, classify records
  - Script: `survey.py`
  - Inputs: Raw CSVs from `2022_2024_MISO/` and `2024_2025_MISO/`
  - Outputs: Summary table printed + saved to `outputs/data/survey_summary.csv`

- [x] T002: QC review — verify summary stats match known values (Inferno 278-294°C, Hell 296-316°C, El Guapo 103-315°C for 2022-2024) and flag issues in 2024-2025 data
  - Depends: T001
  - Result: 2022-2024 values match. Virgin reclassified from High-temp to Intermittent (late dropoff). Trevi has 1 capped value >400°C and 905 spikes. Vixen failed (34 rows).

## Phase 2: Figures

- [x] T003: Figure 1 — survey overview (all instruments as subplots)
  - Script: `survey.py`
  - Output: `outputs/figures/survey_overview.png`
  - Depends: T001
  - Note: Vixen excluded (no deployed data)

- [x] T004: Figure 2 — high-temp vent comparison (daily means overlaid)
  - Script: `survey.py`
  - Output: `outputs/figures/survey_hightemp_comparison.png`
  - Depends: T002

- [x] T005: Figure 3 — poster figure (high-temp vents + BPR overlay)
  - Script: `survey.py`
  - Output: `outputs/figures/poster_temp_bpr.png`
  - Depends: T004
  - Note: Same-color grouping for El Guapo/El Guapo Top and Inferno across deployments. Solid=2022-2024, dashed=2024-2025. Legend below plot.
