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
FIG_DIR = OUTPUT_DIR / "figures" / "poster"

# --- QC parameters ---
MISSING_FLAG = -888.88
PHYSICAL_MAX = 400.0  # °C — cap unphysical values
DEPLOYMENT_THRESHOLD = 200  # °C — sensor is in vent
SETTLING_HOURS = 24
MAD_THRESHOLD = 5.0
MAD_WINDOW_HOURS = 24

# --- Poster figure styling (optimized for viewing at 3-6 feet) ---
POSTER_DPI = 600            # high DPI for sharp text on posters
POSTER_TITLE_SIZE = 24
POSTER_LABEL_SIZE = 18      # axis labels
POSTER_TICK_SIZE = 16       # tick labels
POSTER_LEGEND_SIZE = 14
POSTER_ANNOT_SIZE = 14      # annotations
POSTER_PANEL_LABEL_SIZE = 22  # (a), (b) labels
POSTER_CAPTION_SIZE = 18    # figure captions (smaller than title)
POSTER_LINE_WIDTH = 2.0     # main data lines
POSTER_ANNOT_LINE_WIDTH = 2.0  # vertical annotation lines
POSTER_SPINE_WIDTH = 2.0    # plot bounding box

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
    "Castle": "#009E73",       # Bluish green
    "Trevi": "#E69F00",        # Orange
    "Vixen": "#CC79A7",        # Reddish purple
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

# --- 2015-2019 instruments (pre-inflation period) ---
INSTRUMENTS_2015_2019 = [
    {
        "file": DATA_HISTORICAL / "castle" / "castle_2001-2022.csv",
        "instrument": "Castle combined",
        "vent": "Castle",
        "field": "ASHES",
        "deployment": "2015-2019",
        "format": "castle_csv",
    },
    {
        "file": DATA_HISTORICAL / "vixen" / "MISO103-Chip1-Axial-2015-Vixen.txt",
        "instrument": "MISO 103",
        "vent": "Vixen",
        "field": "ASHES",
        "deployment": "2015-2019",
        "format": "miso_historical_tab",
    },
    {
        "file": DATA_HISTORICAL / "trevi" / "MISO101-Chip1-Axial-2015-Trevi.txt",
        "instrument": "MISO 101",
        "vent": "Trevi",
        "field": "ASHES",
        "deployment": "2015-2019",
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
    """Load historical MISO data (various formats)."""
    path = config["file"]
    fmt = config["format"]

    if fmt == "miso_historical_tab":
        # Tab-delimited: "Date Time\tTemperature   (*C)"
        df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
        df.columns = ["datetime", "temperature"]
        df["datetime"] = pd.to_datetime(df["datetime"], format="%m/%d/%y %H:%M:%S.%f")
    elif fmt == "castle_csv":
        # Combined castle CSV: datetime index, Temperature column
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.reset_index()
        df.columns = ["datetime", "temperature"]
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


def set_spine_width(ax, width=POSTER_SPINE_WIDTH):
    """Set the linewidth for all spines (plot bounding box) of an axes."""
    for spine in ax.spines.values():
        spine.set_linewidth(width)


def add_figure_caption(fig, caption_text, fontsize=POSTER_CAPTION_SIZE):
    """Add a caption below the figure with proper spacing."""
    # Create a separate axes for the caption at the bottom
    caption_ax = fig.add_axes([0.05, 0.02, 0.9, 0.18])  # [left, bottom, width, height]
    caption_ax.axis('off')
    caption_ax.text(0.5, 1.0, caption_text, ha="center", va="top", fontsize=fontsize,
                    wrap=True, transform=caption_ax.transAxes,
                    multialignment="center")


def fig_historical_eruption(records, fig_path, eruption_date=None):
    """Figure: Historical vent temperatures around the 2011 eruption."""
    # Create figure with space for caption below
    fig = plt.figure(figsize=(10, 8), dpi=POSTER_DPI)
    ax = fig.add_axes([0.1, 0.28, 0.85, 0.62])  # [left, bottom, width, height] - plot area

    # Cut off data at local maximum (mid-July 2011) to exclude instrument recovery
    data_end = pd.Timestamp("2011-07-18")

    for rec in records:
        deployed = rec[rec["deployed"]]
        vent = rec.attrs["vent"]
        color = VENT_COLORS.get(vent, "#333333")
        label = f"{vent}"
        daily = deployed["temperature"].resample("D").mean()
        daily = daily[daily.index <= data_end]  # Trim to local max
        ax.plot(daily.index, daily.values, color=color, linewidth=POSTER_LINE_WIDTH, alpha=0.85, label=label)

    ax.set_xlabel("Date", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Temperature (°C)", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
    ax.set_title("Vent Temperatures Around the April 2011 Eruption\nAxial Seamount", fontsize=POSTER_TITLE_SIZE, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=POSTER_TICK_SIZE)
    set_spine_width(ax)

    # Clean date formatting - 2 month intervals
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    # Tighten y-axis to focus on eruption response (230-330°C)
    ax.set_ylim(230, 330)

    # Set x-axis limits with balanced padding on both sides
    ax.set_xlim(left=pd.Timestamp("2010-08-15"), right=pd.Timestamp("2011-08-01"))

    # Legend in lower left (away from eruption signal)
    ax.legend(loc="lower left", fontsize=POSTER_LEGEND_SIZE, frameon=True, framealpha=0.9)

    # Eruption annotation
    if eruption_date:
        ax.axvline(eruption_date, color="#CC0000", linestyle="--", linewidth=POSTER_ANNOT_LINE_WIDTH, alpha=0.8)
        ax.annotate("April 6, 2011\neruption", xy=(eruption_date, 328),
                    xytext=(5, -5), textcoords="offset points",
                    fontsize=POSTER_ANNOT_SIZE, color="#CC0000", va="top", fontweight="bold")

    # Figure caption (20pt font for poster)
    caption = (
        "Vent temperatures spanning the April 6, 2011 Axial Seamount eruption. "
        "Y-axis: temperature (°C); x-axis: date. Colors distinguish Casper (blue) and Diva (orange) vents in ASHES field. "
        "Both vents maintained stable temperatures (~310–320°C) for 7 months pre-eruption. "
        "Diva dropped ~70°C immediately post-eruption with gradual recovery; Casper remained stable throughout."
    )
    add_figure_caption(fig, caption, fontsize=POSTER_CAPTION_SIZE)

    fig.savefig(fig_path, dpi=POSTER_DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {fig_path}")


def fig_2015_2019(records, fig_path, eruption_date=None):
    """Figure: Castle, Vixen, Trevi temperatures spanning 2015-2019 (includes April 2015 eruption)."""
    # Create figure with space for caption below
    fig = plt.figure(figsize=(10, 8), dpi=POSTER_DPI)
    ax = fig.add_axes([0.1, 0.28, 0.85, 0.62])

    # Time window
    time_start = pd.Timestamp("2015-01-01")
    time_end = pd.Timestamp("2019-01-01")

    for rec in records:
        deployed = rec[rec["deployed"]]
        # Filter to time window
        deployed = deployed[(deployed.index >= time_start) & (deployed.index <= time_end)]
        if len(deployed) == 0:
            continue

        vent = rec.attrs["vent"]
        color = VENT_COLORS.get(vent, "#333333")
        label = f"{vent}"
        daily = deployed["temperature"].resample("D").mean()
        ax.plot(daily.index, daily.values, color=color, linewidth=POSTER_LINE_WIDTH, alpha=0.85, label=label)

    ax.set_xlabel("Date", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Temperature (°C)", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
    ax.set_title("ASHES Vent Temperatures (2015–2019)\nAxial Seamount", fontsize=POSTER_TITLE_SIZE, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=POSTER_TICK_SIZE)
    set_spine_width(ax)

    # Clean date formatting - 6 month intervals
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    # Set x-axis limits with balanced padding
    ax.set_xlim(left=time_start - pd.Timedelta(days=45), right=time_end + pd.Timedelta(days=45))

    # Legend in upper right
    ax.legend(loc="upper right", fontsize=POSTER_LEGEND_SIZE, frameon=True, framealpha=0.9)

    # April 2015 eruption annotation
    if eruption_date:
        ax.axvline(eruption_date, color="#CC0000", linestyle="--", linewidth=POSTER_ANNOT_LINE_WIDTH, alpha=0.8)
        ax.annotate("April 24, 2015\neruption", xy=(eruption_date, ax.get_ylim()[1]),
                    xytext=(5, -5), textcoords="offset points",
                    fontsize=POSTER_ANNOT_SIZE, color="#CC0000", va="top", fontweight="bold")

    # Figure caption
    caption = (
        "Daily mean vent temperatures from Castle, Vixen, and Trevi vents (ASHES field) spanning 2015–2019. "
        "Y-axis: temperature (°C); x-axis: date. Colors distinguish Castle (green), Vixen (purple), and Trevi (orange). "
        "The April 24, 2015 eruption is marked. Castle shows stable high temperatures (~260°C); "
        "Vixen and Trevi show lower, more variable temperatures (~150–200°C)."
    )
    add_figure_caption(fig, caption, fontsize=POSTER_CAPTION_SIZE)

    fig.savefig(fig_path, dpi=POSTER_DPI, bbox_inches="tight", pad_inches=0.1)
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
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), dpi=POSTER_DPI, sharex=True)
    fig.subplots_adjust(bottom=0.08, top=0.95)
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
    xmin = min(all_starts) - pd.Timedelta(days=45)
    xmax = max(all_ends) + pd.Timedelta(days=45)

    for ax, rec in zip(axes, plotable):
        deployed = rec[rec["deployed"]]
        vent = rec.attrs["vent"]
        dep = rec.attrs["deployment"]
        color = VENT_COLORS.get(vent, "#333333")
        # Abbreviated labels: ID = International District, 22-24 = 2022-2024
        field_abbr = "ID" if rec.attrs["field"] == "International District" else rec.attrs["field"]
        dep_abbr = dep.replace("2022-2024", "22–24").replace("2024-2025", "24–25")
        label = f"{vent} ({field_abbr}, {dep_abbr})"

        daily = deployed["temperature"].resample("D").mean()
        ax.plot(daily.index, daily.values, color=color, linewidth=POSTER_LINE_WIDTH)
        ax.set_ylabel("°C", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
        ax.set_title(label, fontsize=POSTER_LABEL_SIZE, fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=POSTER_TICK_SIZE)
        ax.set_xlim(xmin, xmax)
        set_spine_width(ax)

    axes[-1].set_xlabel("Date", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
    # Clean date formatting (6-month intervals)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Deployment change annotation on all panels
    deploy_change = pd.Timestamp("2024-06-26")
    for i, ax in enumerate(axes):
        ax.axvline(deploy_change, color="#666666", linestyle=":", linewidth=POSTER_ANNOT_LINE_WIDTH, alpha=0.7)
        # Add text annotation on first panel only
        if i == 0:
            ax.annotate("Deployment\nchange", xy=(deploy_change, ax.get_ylim()[1]),
                        xytext=(-5, -5), textcoords="offset points",
                        fontsize=POSTER_ANNOT_SIZE, color="#666666", va="top", ha="right")

    fig.suptitle("MISO Temperature Survey\nAll Recent Instruments (2022–2025)",
                 fontsize=POSTER_TITLE_SIZE, fontweight="bold", y=0.99)

    # Figure caption
    caption = (
        "Daily mean vent temperatures from all MISO deployments at Axial Seamount (2022–2025). "
        "Each panel shows one instrument; y-axis is temperature (°C). "
        "Abbreviations: ID = International District, ASHES = vent field name. "
        "High-temp vents (Inferno, Hell, El Guapo) maintain 250–340°C. "
        "Intermittent vents (Virgin, Trevi) show cooling trends. "
        "Vertical dotted line marks deployment change (June 2024)."
    )
    # Add caption at the very bottom of the figure
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=POSTER_CAPTION_SIZE - 2,
             wrap=True, multialignment="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])  # Leave space at bottom for caption
    fig.savefig(fig_path, dpi=POSTER_DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {fig_path}")


def fig_hightemp_comparison(records, summary, fig_path):
    """Figure 2: High-temp vents overlaid (daily means)."""
    high_recs = [r for r, s in zip(records, summary.itertuples())
                 if s.Classification == "High-temp"]

    # Create figure with space for caption
    fig = plt.figure(figsize=(10, 6), dpi=POSTER_DPI)
    ax = fig.add_axes([0.10, 0.35, 0.85, 0.55])

    # Find date range for x-axis padding
    all_starts = []
    all_ends = []
    for rec in high_recs:
        deployed = rec[rec["deployed"]]
        if len(deployed) > 0:
            all_starts.append(deployed.index.min())
            all_ends.append(deployed.index.max())
    if all_starts:
        xmin = min(all_starts) - pd.Timedelta(days=45)
        xmax = max(all_ends) + pd.Timedelta(days=45)

    for rec in high_recs:
        deployed = rec[rec["deployed"]]
        vent = rec.attrs["vent"]
        dep = rec.attrs["deployment"]
        color = VENT_COLORS.get(vent, "#333333")
        # Abbreviated labels: ID = International District, 22-24 = 2022-2024
        field_abbr = "ID" if rec.attrs["field"] == "International District" else rec.attrs["field"]
        dep_abbr = dep.replace("2022-2024", "22–24").replace("2024-2025", "24–25")
        label = f"{vent} ({field_abbr}, {dep_abbr})"
        daily = deployed["temperature"].resample("D").mean()
        ax.plot(daily.index, daily.values, color=color, linewidth=POSTER_LINE_WIDTH, alpha=0.85, label=label)

    ax.set_xlabel("Date", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Temperature (°C)", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
    ax.set_title("High-Temperature Vents\nDaily Mean Comparison (2022–2025)", fontsize=POSTER_TITLE_SIZE, fontweight="bold")
    ax.legend(loc="lower left", bbox_to_anchor=(0.62, 0.02), ncol=1, fontsize=POSTER_LEGEND_SIZE - 2, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=POSTER_TICK_SIZE)
    set_spine_width(ax)
    # Clean date formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    # Balanced x-axis padding
    if all_starts:
        ax.set_xlim(xmin, xmax)

    # Deployment change annotation
    deploy_change = pd.Timestamp("2024-06-26")
    ax.axvline(deploy_change, color="#666666", linestyle=":", linewidth=POSTER_ANNOT_LINE_WIDTH, alpha=0.7)
    ax.annotate("Deployment\nchange", xy=(deploy_change, ax.get_ylim()[1]),
                xytext=(-5, -5), textcoords="offset points",
                fontsize=POSTER_ANNOT_SIZE, color="#666666", va="top", ha="right")

    # Figure caption (24pt font for poster)
    caption = (
        "Daily mean temperatures from high-temperature vents at Axial Seamount (2022–2025). "
        "Y-axis: temperature (°C). Inferno (ASHES) shows stable ~285–310°C across both deployments. "
        "Hell and El Guapo (International District) show greater variability, with El Guapo "
        "exhibiting dramatic swings (100–315°C). El Guapo Top is the hottest and most stable (~341°C). "
        "Vertical dashed line marks deployment change (June 2024)."
    )
    add_figure_caption(fig, caption, fontsize=POSTER_CAPTION_SIZE)

    fig.savefig(fig_path, dpi=POSTER_DPI, bbox_inches="tight", pad_inches=0.1)
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
        ("Inferno", "2022-2024"):      {"color": "#D55E00", "ls": "-",  "lw": POSTER_LINE_WIDTH, "label": "Inferno (ASHES, 22–24)"},
        ("Inferno", "2024-2025"):      {"color": "#D55E00", "ls": "--", "lw": POSTER_LINE_WIDTH, "label": "Inferno (ASHES, 24–25)"},
        ("Hell", "2022-2024"):         {"color": "#E69F00", "ls": "-",  "lw": POSTER_LINE_WIDTH, "label": "Hell (ID, 22–24)"},
        ("El Guapo", "2022-2024"):     {"color": "#0072B2", "ls": "-",  "lw": POSTER_LINE_WIDTH, "label": "El Guapo (ID, 22–24)"},
        ("El Guapo (Top)", "2024-2025"): {"color": "#56B4E9", "ls": "--", "lw": POSTER_LINE_WIDTH, "label": "El Guapo Top (ID, 24–25)"},
    }

    # Two panels if TMPSF available, otherwise single (with space for caption)
    if tmpsf is not None:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 15), dpi=POSTER_DPI,
                                        height_ratios=[3, 1], sharex=True)
        fig.subplots_adjust(bottom=0.18, top=0.94, right=0.88, hspace=0.12)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 10), dpi=POSTER_DPI)
        fig.subplots_adjust(bottom=0.25, right=0.82)

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

    ax1.set_ylabel("Vent Temperature (°C)", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
    ax1.tick_params(labelsize=POSTER_TICK_SIZE)

    # Constrain x-axis to temperature data range (with small padding)
    all_starts = []
    all_ends = []
    for rec in high_recs:
        deployed = rec[rec["deployed"]]
        if len(deployed) > 0:
            all_starts.append(deployed.index.min())
            all_ends.append(deployed.index.max())
    if all_starts:
        xmin = min(all_starts) - pd.Timedelta(days=45)
        xmax = max(all_ends) + pd.Timedelta(days=45)
        ax1.set_xlim(xmin, xmax)

    # BPR on right axis
    if bpr is not None:
        ax2 = ax1.twinx()
        bpr_cm = bpr["differential_m"] * 100
        ax2.plot(bpr.index, bpr_cm,
                 color="#0868AC", alpha=0.5, linewidth=POSTER_LINE_WIDTH + 0.5, linestyle="--",
                 label="Differential uplift")
        ax2.set_ylabel("Differential uplift (cm)", color="#0868AC", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="#0868AC", labelsize=POSTER_TICK_SIZE)
        set_spine_width(ax2)

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
        # 2nd highest of all channels at each timestep (excluding ch06 which has sensor issues)
        # This avoids single-sensor artifacts while still showing high values
        all_channel_nums = [n for n in range(1, 25) if n != 6]  # ch06 excluded
        all_cols = [f"temperature{n:02d}" for n in all_channel_nums if f"temperature{n:02d}" in tmpsf.columns]
        # Get 2nd highest by sorting and taking index -2
        tmpsf_2nd = tmpsf[all_cols].apply(lambda row: row.nlargest(2).iloc[-1], axis=1)

        ax3.plot(tmpsf_2nd.index, tmpsf_2nd.values,
                 color="#7A0177", linewidth=POSTER_LINE_WIDTH, alpha=0.85)
        ax3.set_ylabel("TMPSF (°C)", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
        ax3.set_xlabel("Date", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
        ax3.tick_params(labelsize=POSTER_TICK_SIZE)
        # Clean date formatting for x-axis
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Diffuse Flow — ASHES (TMPSF 2nd highest channel, excl. ch06)",
                      fontsize=POSTER_LABEL_SIZE, loc="left", style="italic")
        set_spine_width(ax3)

        # Constrain TMPSF y-axis to visible range
        tmpsf_visible = tmpsf_2nd.loc[xmin:xmax].dropna()
        if len(tmpsf_visible) > 0:
            ypad = (tmpsf_visible.max() - tmpsf_visible.min()) * 0.1
            ax3.set_ylim(tmpsf_visible.min() - ypad, tmpsf_visible.max() + ypad)

        # Deployment change annotation on TMPSF panel
        deploy_change = pd.Timestamp("2024-06-26")  # ~when 2022-2024 ended, 2024-2025 began
        if xmin <= deploy_change <= xmax:
            ax3.axvline(deploy_change, color="#666666", linestyle=":", linewidth=POSTER_ANNOT_LINE_WIDTH, alpha=0.7)
            ax3.annotate("Deployment\nchange", xy=(deploy_change, tmpsf_visible.max()),
                         xytext=(5, -5), textcoords="offset points",
                         fontsize=POSTER_ANNOT_SIZE, color="#666666", va="top")
    else:
        ax1.set_xlabel("Date", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
        # Clean date formatting for x-axis
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Deployment change annotation on main panel
    deploy_change = pd.Timestamp("2024-06-26")
    if xmin <= deploy_change <= xmax:
        ax1.axvline(deploy_change, color="#666666", linestyle=":", linewidth=POSTER_ANNOT_LINE_WIDTH, alpha=0.7)

    # Legend in lower right, to the right of deployment change line (vertical)
    ax1.legend(all_lines, all_labels,
               loc="lower left", bbox_to_anchor=(0.62, 0.02), ncol=1,
               fontsize=POSTER_LEGEND_SIZE - 2, frameon=True, framealpha=0.9)

    ax1.set_title("Hydrothermal Vent Temperatures and Volcanic Deformation\nAxial Seamount (2022–2025)",
                  fontsize=POSTER_TITLE_SIZE, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    set_spine_width(ax1)

    # Panel labels (a), (b) for reference in text
    ax1.text(-0.08, 1.02, "(a)", transform=ax1.transAxes, fontsize=POSTER_PANEL_LABEL_SIZE, fontweight="bold", va="bottom")
    if tmpsf is not None:
        ax3.text(-0.08, 1.05, "(b)", transform=ax3.transAxes, fontsize=POSTER_PANEL_LABEL_SIZE, fontweight="bold", va="bottom")

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
    caption_ax.text(0.5, 0.95, caption, ha="center", va="top", fontsize=POSTER_CAPTION_SIZE,
                    wrap=True, transform=caption_ax.transAxes, multialignment="center")

    fig.savefig(fig_path, dpi=POSTER_DPI, bbox_inches="tight", pad_inches=0.1)
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

    # Load and plot 2015-2019 data (Castle, Vixen, Trevi)
    print("\nLoading 2015-2019 instruments (Castle, Vixen, Trevi)...")
    records_2015 = []
    for config in INSTRUMENTS_2015_2019:
        vent = config["vent"]
        inst = config["instrument"]
        print(f"Loading {vent} ({inst})...")
        try:
            rec = load_historical_instrument(config)
            records_2015.append(rec)
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

    if records_2015:
        eruption_2015 = pd.Timestamp("2015-04-24")  # April 2015 eruption
        fig_2015_2019(records_2015, FIG_DIR / "ashes_2015_2019_castle_vixen_trevi.png",
                      eruption_date=eruption_2015)

    print("\nDone!")


if __name__ == "__main__":
    main()
