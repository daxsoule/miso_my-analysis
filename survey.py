"""
MISO Temperature Data Survey — Recent Deployments (2022-2025)

Loads all HiT deployment data from 2022-2024 and 2024-2025,
applies quality control, classifies records by temperature,
and generates survey and poster figures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# --- Font configuration: Helvetica for all figures ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

# --- Paths ---
DATA_2022 = Path("/home/jovyan/my_data/axial/axial_miso/2022_2024_MISO")
DATA_2024 = Path("/home/jovyan/my_data/axial/axial_miso/2024_2025_MISO")
DATA_HISTORICAL = Path("/home/jovyan/my_data/axial/axial_miso")
BPR_PATH = Path("/home/jovyan/repos/specKitScience/my-analysis_botpt/outputs/data/differential_uplift_daily.parquet")
TMPSF_PATH = Path("/home/jovyan/repos/specKitScience/my-analysis_tmpsf/outputs/data/tmpsf_2015-2026_daily.parquet")

OUTPUT_DIR = Path(__file__).parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures"

# --- QC parameters ---
MISSING_FLAG = -888.88
PHYSICAL_MAX = 400.0  # °C — cap unphysical values
DEPLOYMENT_THRESHOLD = 200  # °C — sensor is in vent
SETTLING_HOURS = 24
MAD_THRESHOLD = 5.0
MAD_WINDOW_HOURS = 24

# --- Instrument registry ---
# Each entry: (file_path, vent_name, field, deployment, format, temp_col, date_col)
INSTRUMENTS = [
    # 2022-2024 deployment (International District)
    {
        "file": DATA_2022 / "MISOTEMP_2017-002_Axial2022_inferno.csv",
        "instrument": "MISO 2017-002",
        "vent": "Inferno",
        "field": "ASHES",
        "deployment": "2022-2024",
        "format": "miso_2022",
        "temp_col": "infernoTemp",
        "date_col": "datetime",
        "has_index": False,
    },
    {
        "file": DATA_2022 / "MISOTEMP_2017-019_Axial2022_hel.csv",
        "instrument": "MISO 2017-019",
        "vent": "Hell",
        "field": "International District",
        "deployment": "2022-2024",
        "format": "miso_2022",
        "temp_col": "hell_temp1",
        "date_col": "datetime",
        "has_index": False,
    },
    {
        "file": DATA_2022 / "MISO_2017-006_Axial2022_ElGuapo.csv",
        "instrument": "MISO 2017-006",
        "vent": "El Guapo",
        "field": "International District",
        "deployment": "2022-2024",
        "format": "miso_2022",
        "temp_col": "Tem",
        "date_col": "datetime",
        "has_index": True,
    },
    # 2024-2025 deployment (mostly ASHES)
    {
        "file": DATA_2024 / "HiT_2023-002_Axial_2024_Deployment.csv",
        "instrument": "MISO 2023-002",
        "vent": "Hell",
        "field": "ASHES",
        "deployment": "2024-2025",
        "format": "hit_A",  # idx#,DateTime,J-Type,IntTemp,...
        "temp_col": "J-Type",
        "date_col": "DateTime",
    },
    {
        "file": DATA_2024 / "HiT_2023-005_Axial_2024_Deployment_0.csv",
        "instrument": "MISO 2023-005",
        "vent": "Inferno",
        "field": "ASHES",
        "deployment": "2024-2025",
        "format": "hit_B",  # idx,DateTime,Temp,RefTempC,...
        "temp_col": "Temp",
        "date_col": "DateTime",
    },
    {
        "file": DATA_2024 / "HiT_2023-007_2024_Axial_Deployment.csv",
        "instrument": "MISO 2023-007",
        "vent": "Virgin",
        "field": "ASHES",
        "deployment": "2024-2025",
        "format": "hit_B",
        "temp_col": "Temp",
        "date_col": "DateTime",
    },
    {
        "file": DATA_2024 / "HiT_2023-009_Axial_OOI_2024_deployment.csv",
        "instrument": "MISO 2023-009",
        "vent": "El Guapo (Top)",
        "field": "International District",
        "deployment": "2024-2025",
        "format": "hit_C",  # Row 1: plot title; Row 2: headers; data col 2 = J-Type
    },
    {
        "file": DATA_2024 / "HiT_2023-010_Axial_2024_Deployment.csv",
        "instrument": "MISO 2023-010",
        "vent": "Trevi / Mkr156",
        "field": "ASHES",
        "deployment": "2024-2025",
        "format": "hit_C",
    },
    {
        "file": DATA_2024 / "HiT_2023-012_Axial_OOI_2024_deployment.csv",
        "instrument": "MISO 2023-012",
        "vent": "Vixen / Mkr218",
        "field": "ASHES",
        "deployment": "2024-2025",
        "format": "hit_C",
    },
]

# --- Colors: one per vent label (consistent across deployments) ---
# Colorblind-safe palette (Okabe-Ito)
VENT_COLORS = {
    "Inferno": "#D55E00",      # Vermillion
    "Hell": "#E69F00",         # Orange
    "El Guapo": "#0072B2",     # Blue
    "El Guapo (Top)": "#56B4E9",  # Sky blue
    "Virgin": "#CC79A7",       # Reddish purple
    "Trevi / Mkr156": "#009E73",  # Bluish green
    "Vixen / Mkr218": "#F0E442",  # Yellow
    "Casper": "#0072B2",       # Blue
    "Diva": "#D55E00",         # Vermillion
}

# --- Historical instruments (2010-2011 eruption period) ---
HISTORICAL_INSTRUMENTS = [
    {
        "file": DATA_HISTORICAL / "casper" / "MISO104-Chip1-Axial-2010-Casper.txt",
        "instrument": "MISO 104",
        "vent": "Casper",
        "field": "ASHES",
        "deployment": "2010-2011",
        "format": "miso_historical_tab",
    },
    {
        "file": DATA_HISTORICAL / "diva" / "MISO129-Chip1-Axial-2010-Diva.txt",
        "instrument": "MISO 129",
        "vent": "Diva",
        "field": "ASHES",
        "deployment": "2010-2011",
        "format": "miso_historical_tab",
    },
]


def remove_spikes_mad(series, window_hours=24, threshold=5.0):
    """Remove spikes using rolling median and MAD."""
    cleaned = series.copy()
    samples_per_hour = max(1, int(len(series) / max(1, (series.index[-1] - series.index[0]).total_seconds() / 3600)))
    window = max(3, window_hours * samples_per_hour)
    rolling_median = cleaned.rolling(window=window, center=True, min_periods=1).median()
    deviation = (cleaned - rolling_median).abs()
    rolling_mad = deviation.rolling(window=window, center=True, min_periods=1).median()
    scaled_mad = 1.4826 * rolling_mad
    is_spike = deviation > (threshold * scaled_mad)
    n_spikes = int(is_spike.sum())
    if n_spikes > 0:
        cleaned[is_spike] = np.nan
    return cleaned, n_spikes


def load_instrument(config):
    """Load a single instrument's data, returning standardized DataFrame."""
    fmt = config["format"]
    path = config["file"]

    if fmt == "miso_2022":
        if config.get("has_index"):
            df = pd.read_csv(path, encoding="utf-8-sig", index_col=0)
            df["datetime"] = pd.to_datetime(df["datetime"], format="%m/%d/%y %I:%M:%S %p")
        else:
            df = pd.read_csv(path, encoding="utf-8-sig")
            df["datetime"] = pd.to_datetime(df["datetime"], format="%m/%d/%y %I:%M:%S %p")
        temp = pd.to_numeric(df[config["temp_col"]], errors="coerce")
        dt = df["datetime"]

    elif fmt == "hit_A":
        df = pd.read_csv(path, encoding="utf-8-sig")
        temp = pd.to_numeric(df[config["temp_col"]], errors="coerce")
        dt = pd.to_datetime(df[config["date_col"]], format="%m/%d/%y %I:%M:%S %p")

    elif fmt == "hit_B":
        df = pd.read_csv(path, encoding="utf-8-sig")
        temp = pd.to_numeric(df[config["temp_col"]], errors="coerce")
        dt = pd.to_datetime(df[config["date_col"]], format="%m/%d/%y %I:%M:%S %p")

    elif fmt == "hit_C":
        df = pd.read_csv(path, skiprows=1, encoding="utf-8-sig")
        # J-Type temperature is column index 2, datetime is column index 1
        temp = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        dt = pd.to_datetime(df.iloc[:, 1], format="%m/%d/%y %I:%M:%S %p")

    else:
        raise ValueError(f"Unknown format: {fmt}")

    # Build standardized DataFrame
    out = pd.DataFrame({"temperature": temp.values, "datetime": dt.values})
    out = out.set_index("datetime").sort_index()

    # Replace missing flag
    out["temperature"] = out["temperature"].replace(MISSING_FLAG, np.nan)

    # Cap unphysical values
    n_capped = int((out["temperature"] > PHYSICAL_MAX).sum())
    out.loc[out["temperature"] > PHYSICAL_MAX, "temperature"] = np.nan

    # Settling window QC
    hot_mask = out["temperature"] > DEPLOYMENT_THRESHOLD
    if hot_mask.any():
        first_hot = out[hot_mask].index.min()
        stable_start = first_hot + pd.Timedelta(hours=SETTLING_HOURS)
        out["deployed"] = (out.index >= stable_start) & (out["temperature"] > 50)
    else:
        out["deployed"] = out["temperature"] > 50

    # MAD spike removal on deployed data
    n_spikes = 0
    deployed_mask = out["deployed"]
    if deployed_mask.sum() > 10:
        temp_cleaned, n_spikes = remove_spikes_mad(
            out.loc[deployed_mask, "temperature"],
            window_hours=MAD_WINDOW_HOURS,
            threshold=MAD_THRESHOLD,
        )
        out.loc[deployed_mask, "temperature"] = temp_cleaned

    # Metadata
    out.attrs["instrument"] = config["instrument"]
    out.attrs["vent"] = config["vent"]
    out.attrs["field"] = config["field"]
    out.attrs["deployment"] = config["deployment"]
    out.attrs["n_capped"] = n_capped
    out.attrs["n_spikes"] = n_spikes

    return out


def load_historical_instrument(config):
    """Load historical MISO data (2010-2011 format)."""
    path = config["file"]
    fmt = config["format"]

    if fmt == "miso_historical_tab":
        # Tab-delimited: "Date Time\tTemperature   (*C)"
        df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
        df.columns = ["datetime", "temperature"]
        df["datetime"] = pd.to_datetime(df["datetime"], format="%m/%d/%y %H:%M:%S.%f")
    else:
        raise ValueError(f"Unknown format: {fmt}")

    out = pd.DataFrame({"temperature": df["temperature"].values, "datetime": df["datetime"].values})
    out = out.set_index("datetime").sort_index()

    # Cap unphysical values
    n_capped = int((out["temperature"] > PHYSICAL_MAX).sum())
    out.loc[out["temperature"] > PHYSICAL_MAX, "temperature"] = np.nan

    # Settling window QC
    hot_mask = out["temperature"] > DEPLOYMENT_THRESHOLD
    if hot_mask.any():
        first_hot = out[hot_mask].index.min()
        stable_start = first_hot + pd.Timedelta(hours=SETTLING_HOURS)
        out["deployed"] = (out.index >= stable_start) & (out["temperature"] > 50)
    else:
        out["deployed"] = out["temperature"] > 50

    # MAD spike removal on deployed data
    n_spikes = 0
    deployed_mask = out["deployed"]
    if deployed_mask.sum() > 10:
        temp_cleaned, n_spikes = remove_spikes_mad(
            out.loc[deployed_mask, "temperature"],
            window_hours=MAD_WINDOW_HOURS,
            threshold=MAD_THRESHOLD,
        )
        out.loc[deployed_mask, "temperature"] = temp_cleaned

    # Metadata
    out.attrs["instrument"] = config["instrument"]
    out.attrs["vent"] = config["vent"]
    out.attrs["field"] = config["field"]
    out.attrs["deployment"] = config["deployment"]
    out.attrs["n_capped"] = n_capped
    out.attrs["n_spikes"] = n_spikes

    return out


def add_figure_caption(fig, caption_text, fontsize=20):
    """Add a caption below the figure with proper spacing."""
    # Create a separate axes for the caption at the bottom
    caption_ax = fig.add_axes([0.05, 0.02, 0.9, 0.18])  # [left, bottom, width, height]
    caption_ax.axis('off')
    caption_ax.text(0.5, 1.0, caption_text, ha="center", va="top", fontsize=fontsize,
                    wrap=True, transform=caption_ax.transAxes,
                    multialignment="center")


def fig_historical_eruption(records, fig_path, eruption_date=None):
    """Figure: Historical vent temperatures around the 2011 eruption."""
    # Create figure with space for 24pt caption below
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_axes([0.1, 0.30, 0.85, 0.60])  # [left, bottom, width, height] - plot area

    for rec in records:
        deployed = rec[rec["deployed"]]
        vent = rec.attrs["vent"]
        color = VENT_COLORS.get(vent, "#333333")
        label = f"{vent}"
        daily = deployed["temperature"].resample("D").mean()
        ax.plot(daily.index, daily.values, color=color, linewidth=1.2, alpha=0.85, label=label)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_title("Vent Temperatures Around the April 2011 Eruption\nAxial Seamount", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    # Legend in lower left, inside plot
    ax.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.9)

    # Eruption annotation
    if eruption_date:
        ax.axvline(eruption_date, color="#CC0000", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.annotate("April 6, 2011\neruption", xy=(eruption_date, ax.get_ylim()[1]),
                    xytext=(5, -5), textcoords="offset points",
                    fontsize=9, color="#CC0000", va="top", fontweight="bold")

    # Figure caption (24pt font for poster)
    caption = (
        "Daily mean vent fluid temperatures at Casper and Diva vents (ASHES field) spanning the "
        "April 6, 2011 Axial Seamount eruption. Y-axis: temperature (°C). Both vents maintained "
        "stable temperatures (~310–320°C) for 7 months pre-eruption. Diva dropped ~70°C immediately "
        "post-eruption with partial recovery; Casper remained stable. Drops to ~150°C at end = "
        "instrument recovery."
    )
    add_figure_caption(fig, caption, fontsize=20)

    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def compute_summary(records):
    """Compute summary statistics for all loaded instruments."""
    rows = []
    for rec in records:
        deployed = rec[rec["deployed"]]["temperature"].dropna()
        all_temp = rec["temperature"].dropna()

        if len(deployed) > 0:
            pct_above_100 = 100 * (deployed > 100).sum() / len(deployed)
            median_t = deployed.median()
            mean_t = deployed.mean()
            min_t = deployed.min()
            max_t = deployed.max()
            date_start = deployed.index.min()
            date_end = deployed.index.max()
        else:
            pct_above_100 = 0
            median_t = mean_t = min_t = max_t = np.nan
            date_start = all_temp.index.min() if len(all_temp) > 0 else pd.NaT
            date_end = all_temp.index.max() if len(all_temp) > 0 else pd.NaT

        # Classification
        # Check for late-record dropoff: if the last 20% of deployed data
        # has a median below 100°C, classify as Intermittent even if overall
        # stats look high-temp (catches cases like Virgin that cool late)
        late_dropoff = False
        if len(deployed) > 100:
            tail_frac = deployed.iloc[-len(deployed)//5:]
            if tail_frac.median() < 100:
                late_dropoff = True

        if median_t > 100 and pct_above_100 > 80 and not late_dropoff:
            classification = "High-temp"
        elif pct_above_100 > 5:
            classification = "Intermittent"
        else:
            classification = "Low-temp"

        notes = []
        if rec.attrs["n_capped"] > 0:
            notes.append(f"{rec.attrs['n_capped']} values capped >400°C")
        if rec.attrs["n_spikes"] > 0:
            notes.append(f"{rec.attrs['n_spikes']} spikes removed")
        if len(all_temp) < 100:
            notes.append(f"Only {len(all_temp)} total samples")

        rows.append({
            "Instrument": rec.attrs["instrument"],
            "Vent": rec.attrs["vent"],
            "Field": rec.attrs["field"],
            "Deployment": rec.attrs["deployment"],
            "Start": date_start,
            "End": date_end,
            "N_samples": len(deployed),
            "Min_C": round(min_t, 1) if not np.isnan(min_t) else np.nan,
            "Median_C": round(median_t, 1) if not np.isnan(median_t) else np.nan,
            "Mean_C": round(mean_t, 1) if not np.isnan(mean_t) else np.nan,
            "Max_C": round(max_t, 1) if not np.isnan(max_t) else np.nan,
            "Pct_above_100C": round(pct_above_100, 1),
            "Classification": classification,
            "Notes": "; ".join(notes) if notes else "",
        })

    return pd.DataFrame(rows)


def fig_survey_overview(records, fig_path):
    """Figure 1: All instruments as subplots (skip records with no deployed data)."""
    plotable = [r for r in records if r[r["deployed"]].shape[0] > 0]
    n = len(plotable)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2 * n), dpi=300, sharex=True)
    if n == 1:
        axes = [axes]

    # Find global date range for shared x-axis
    all_starts = []
    all_ends = []
    for rec in plotable:
        deployed = rec[rec["deployed"]]
        if len(deployed) > 0:
            all_starts.append(deployed.index.min())
            all_ends.append(deployed.index.max())
    xmin = min(all_starts) - pd.Timedelta(days=14)
    xmax = max(all_ends) + pd.Timedelta(days=14)

    for ax, rec in zip(axes, plotable):
        deployed = rec[rec["deployed"]]
        vent = rec.attrs["vent"]
        dep = rec.attrs["deployment"]
        color = VENT_COLORS.get(vent, "#333333")
        label = f"{vent} ({rec.attrs['field']}, {dep})"

        daily = deployed["temperature"].resample("D").mean()
        ax.plot(daily.index, daily.values, color=color, linewidth=0.8)
        ax.set_ylabel("°C", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.set_xlim(xmin, xmax)

    axes[-1].set_xlabel("Date")

    # Deployment change annotation on all panels
    deploy_change = pd.Timestamp("2024-06-26")
    for ax in axes:
        ax.axvline(deploy_change, color="#666666", linestyle=":", linewidth=0.8, alpha=0.7)

    fig.suptitle("MISO Temperature Survey — All Recent Instruments (2022–2025)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def fig_hightemp_comparison(records, summary, fig_path):
    """Figure 2: High-temp vents overlaid (daily means)."""
    high_recs = [r for r, s in zip(records, summary.itertuples())
                 if s.Classification == "High-temp"]

    # Create figure with space for caption and legend outside
    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax = fig.add_axes([0.08, 0.35, 0.65, 0.55])  # Leave room on right for legend

    for rec in high_recs:
        deployed = rec[rec["deployed"]]
        vent = rec.attrs["vent"]
        dep = rec.attrs["deployment"]
        color = VENT_COLORS.get(vent, "#333333")
        label = f"{vent} ({rec.attrs['field']}, {dep})"
        daily = deployed["temperature"].resample("D").mean()
        ax.plot(daily.index, daily.values, color=color, linewidth=1, alpha=0.85, label=label)

    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("High-Temperature Vents — Daily Mean Comparison (2022–2025)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)

    # Deployment change annotation
    deploy_change = pd.Timestamp("2024-06-26")
    ax.axvline(deploy_change, color="#666666", linestyle=":", linewidth=1, alpha=0.7)
    ax.annotate("Deployment\nchange", xy=(deploy_change, ax.get_ylim()[1]),
                xytext=(5, -5), textcoords="offset points",
                fontsize=8, color="#666666", va="top")

    # Figure caption (24pt font for poster)
    caption = (
        "Daily mean temperatures from high-temperature vents at Axial Seamount (2022–2025). "
        "Y-axis: temperature (°C). Inferno (ASHES) shows stable ~285–310°C across both deployments. "
        "Hell and El Guapo (International District) show greater variability, with El Guapo "
        "exhibiting dramatic swings (100–315°C). El Guapo Top is the hottest and most stable (~341°C). "
        "Vertical dashed line marks deployment change (June 2024)."
    )
    add_figure_caption(fig, caption, fontsize=20)

    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def fig_poster_bpr(records, summary, bpr, fig_path, tmpsf=None):
    """Figure 3: Poster figure — high-temp vents + BPR overlay, with TMPSF panel."""
    high_recs = [r for r, s in zip(records, summary.itertuples())
                 if s.Classification == "High-temp"]

    # Poster-specific styling: group by vent identity across deployments
    # Same base color for same vent, solid=2022-2024, dashed=2024-2025
    # Uses colorblind-safe Okabe-Ito palette
    POSTER_STYLE = {
        ("Inferno", "2022-2024"):      {"color": "#D55E00", "ls": "-",  "lw": 1.4, "label": "Inferno (ASHES, 2022–24)"},
        ("Inferno", "2024-2025"):      {"color": "#D55E00", "ls": "--", "lw": 1.4, "label": "Inferno (ASHES, 2024–25)"},
        ("Hell", "2022-2024"):         {"color": "#E69F00", "ls": "-",  "lw": 1.4, "label": "Hell (ID, 2022–24)"},
        ("El Guapo", "2022-2024"):     {"color": "#0072B2", "ls": "-",  "lw": 1.4, "label": "El Guapo (ID, 2022–24)"},
        ("El Guapo (Top)", "2024-2025"): {"color": "#56B4E9", "ls": "--", "lw": 1.4, "label": "El Guapo Top (ID, 2024–25)"},
    }

    # Two panels if TMPSF available, otherwise single (with space for caption)
    if tmpsf is not None:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 14), dpi=300,
                                        height_ratios=[3, 1], sharex=True)
        fig.subplots_adjust(bottom=0.16, top=0.94, hspace=0.12)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 10), dpi=300)
        fig.subplots_adjust(bottom=0.25)

    for rec in high_recs:
        deployed = rec[rec["deployed"]]
        vent = rec.attrs["vent"]
        dep = rec.attrs["deployment"]
        style = POSTER_STYLE.get((vent, dep), {
            "color": "#333333", "ls": "-", "lw": 1.2, "label": f"{vent} ({dep})"
        })
        daily = deployed["temperature"].resample("D").mean()
        ax1.plot(daily.index, daily.values,
                 color=style["color"], linestyle=style["ls"],
                 linewidth=style["lw"], alpha=0.85, label=style["label"])

    ax1.set_ylabel("Vent Temperature (°C)", fontsize=11)
    ax1.tick_params(labelsize=9)

    # Constrain x-axis to temperature data range (with small padding)
    all_starts = []
    all_ends = []
    for rec in high_recs:
        deployed = rec[rec["deployed"]]
        if len(deployed) > 0:
            all_starts.append(deployed.index.min())
            all_ends.append(deployed.index.max())
    if all_starts:
        xmin = min(all_starts) - pd.Timedelta(days=30)
        xmax = max(all_ends) + pd.Timedelta(days=30)
        ax1.set_xlim(xmin, xmax)

    # BPR on right axis
    if bpr is not None:
        ax2 = ax1.twinx()
        bpr_cm = bpr["differential_m"] * 100
        ax2.plot(bpr.index, bpr_cm,
                 color="#0868AC", alpha=0.5, linewidth=1.5, linestyle="--",
                 label="Differential uplift")
        ax2.set_ylabel("Differential uplift (cm)", color="#0868AC", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="#0868AC", labelsize=9)

        # Constrain y-axis to visible BPR range (within the x-axis window)
        bpr_visible = bpr_cm.loc[xmin:xmax].dropna()
        if len(bpr_visible) > 0:
            ypad = (bpr_visible.max() - bpr_visible.min()) * 0.05
            ax2.set_ylim(bpr_visible.min() - ypad, bpr_visible.max() + ypad)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
    else:
        all_lines, all_labels = ax1.get_legend_handles_labels()

    # TMPSF panel
    if tmpsf is not None:
        # Mean of hot channels (excluding ch06 which has sensor issues)
        hot_channel_nums = [1, 2, 3, 5, 7, 8, 10, 12, 13, 14, 16]  # ch06 excluded
        hot_cols = [f"temperature{n:02d}" for n in hot_channel_nums if f"temperature{n:02d}" in tmpsf.columns]
        tmpsf_hot_mean = tmpsf[hot_cols].mean(axis=1)

        ax3.plot(tmpsf_hot_mean.index, tmpsf_hot_mean.values,
                 color="#7A0177", linewidth=0.8, alpha=0.85)
        ax3.set_ylabel("TMPSF (°C)", fontsize=11)
        ax3.set_xlabel("Date", fontsize=11)
        ax3.tick_params(labelsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Diffuse Flow — ASHES (TMPSF hot channel mean, excl. ch06)",
                      fontsize=10, loc="left", style="italic")

        # Constrain TMPSF y-axis to visible range
        tmpsf_visible = tmpsf_hot_mean.loc[xmin:xmax].dropna()
        if len(tmpsf_visible) > 0:
            ypad = (tmpsf_visible.max() - tmpsf_visible.min()) * 0.1
            ax3.set_ylim(tmpsf_visible.min() - ypad, tmpsf_visible.max() + ypad)

        # Deployment change annotation on TMPSF panel
        deploy_change = pd.Timestamp("2024-06-26")  # ~when 2022-2024 ended, 2024-2025 began
        if xmin <= deploy_change <= xmax:
            ax3.axvline(deploy_change, color="#666666", linestyle=":", linewidth=1, alpha=0.7)
            ax3.annotate("Deployment\nchange", xy=(deploy_change, tmpsf_visible.max()),
                         xytext=(5, -5), textcoords="offset points",
                         fontsize=7, color="#666666", va="top")
    else:
        ax1.set_xlabel("Date", fontsize=11)

    # Deployment change annotation on main panel
    deploy_change = pd.Timestamp("2024-06-26")
    if xmin <= deploy_change <= xmax:
        ax1.axvline(deploy_change, color="#666666", linestyle=":", linewidth=1, alpha=0.7)

    # Legend in main temperature panel (top), not TMPSF panel
    ax1.legend(all_lines, all_labels,
               loc="lower right", ncol=2, fontsize=7, frameon=True, framealpha=0.9)

    ax1.set_title("Hydrothermal Vent Temperatures and Volcanic Deformation\nAxial Seamount (2022–2025)",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Panel labels (a), (b) for reference in text
    ax1.text(-0.08, 1.02, "(a)", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="bottom")
    if tmpsf is not None:
        ax3.text(-0.08, 1.05, "(b)", transform=ax3.transAxes, fontsize=14, fontweight="bold", va="bottom")

    # Figure caption (24pt font for poster) - use dedicated axes to avoid overlap
    caption = (
        "(a) Daily mean focused vent temperatures (°C) from high-temperature vents at "
        "Axial Seamount with differential seafloor uplift (cm, right axis). "
        "(b) TMPSF diffuse flow temperature (°C) from ASHES field hot channels (excl. ch06). "
        "Vertical dashed line marks deployment change (June 2024). Inferno and El Guapo "
        "show continuity across deployments. BPR shows steady inflation through 2025."
    )
    caption_ax = fig.add_axes([0.05, 0.01, 0.9, 0.13])
    caption_ax.axis("off")
    caption_ax.text(0.5, 0.95, caption, ha="center", va="top", fontsize=20,
                    wrap=True, transform=caption_ax.transAxes, multialignment="center")

    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load all instruments
    records = []
    for config in INSTRUMENTS:
        vent = config["vent"]
        inst = config["instrument"]
        print(f"Loading {vent} ({inst})...")
        try:
            rec = load_instrument(config)
            records.append(rec)
            deployed = rec[rec["deployed"]]
            n_dep = len(deployed)
            if n_dep > 0:
                temp = deployed["temperature"].dropna()
                print(f"  Deployed samples: {n_dep:,}  |  "
                      f"Temp: {temp.min():.1f}–{temp.max():.1f}°C  |  "
                      f"Capped: {rec.attrs['n_capped']}  Spikes: {rec.attrs['n_spikes']}")
            else:
                print(f"  No deployed samples (total rows: {len(rec)})")
        except Exception as e:
            print(f"  ERROR loading {vent} ({inst}): {e}")

    # Summary table
    print("\n" + "=" * 90)
    print("INSTRUMENT SUMMARY")
    print("=" * 90)
    summary = compute_summary(records)
    print(summary.to_string(index=False))
    summary.to_csv(DATA_DIR / "survey_summary.csv", index=False)
    print(f"\nSaved: {DATA_DIR / 'survey_summary.csv'}")

    # Load BPR
    bpr = None
    if BPR_PATH.exists():
        bpr = pd.read_parquet(BPR_PATH)
        print(f"\nBPR data loaded: {bpr.index.min()} to {bpr.index.max()}")

    # Load TMPSF
    tmpsf = None
    if TMPSF_PATH.exists():
        tmpsf = pd.read_parquet(TMPSF_PATH)
        print(f"TMPSF data loaded: {tmpsf.index.min()} to {tmpsf.index.max()}")

    # Figures
    print("\nGenerating figures...")
    fig_survey_overview(records, FIG_DIR / "survey_overview.png")
    fig_hightemp_comparison(records, summary, FIG_DIR / "survey_hightemp_comparison.png")
    fig_poster_bpr(records, summary, bpr, FIG_DIR / "poster_temp_bpr_tmpsf.png", tmpsf=tmpsf)

    # Load and plot historical data (2010-2011 eruption period)
    print("\nLoading historical instruments (2010-2011)...")
    historical_records = []
    for config in HISTORICAL_INSTRUMENTS:
        vent = config["vent"]
        inst = config["instrument"]
        print(f"Loading {vent} ({inst})...")
        try:
            rec = load_historical_instrument(config)
            historical_records.append(rec)
            deployed = rec[rec["deployed"]]
            n_dep = len(deployed)
            if n_dep > 0:
                temp = deployed["temperature"].dropna()
                print(f"  Deployed samples: {n_dep:,}  |  "
                      f"Temp: {temp.min():.1f}–{temp.max():.1f}°C  |  "
                      f"Capped: {rec.attrs['n_capped']}  Spikes: {rec.attrs['n_spikes']}")
            else:
                print(f"  No deployed samples (total rows: {len(rec)})")
        except Exception as e:
            print(f"  ERROR loading {vent} ({inst}): {e}")

    if historical_records:
        eruption_date = pd.Timestamp("2011-04-06")  # April 2011 eruption
        fig_historical_eruption(historical_records, FIG_DIR / "eruption_2011_casper_diva.png",
                                eruption_date=eruption_date)

    print("\nDone!")


if __name__ == "__main__":
    main()
