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
import textwrap

# --- Font configuration: Helvetica for all figures ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

# --- Paths ---
DATA_2022 = Path("/home/jovyan/my_data/axial/axial_miso/2022_2024_MISO")
DATA_2024 = Path("/home/jovyan/my_data/axial/axial_miso/2024_2025_MISO")
DATA_HISTORICAL = Path("/home/jovyan/my_data/axial/axial_miso")
BPR_PATH = Path("/home/jovyan/repos/specKitScience/my-analysis_botpt/outputs/data/differential_uplift_daily.parquet")
BPR_HIST_PATH = Path("/home/jovyan/my_data/axial/axial_botpt/Axial-BPR-LTplot-combo-2010-2017.csv")
BPR_NANO_PATH = Path("/home/jovyan/my_data/axial/axial_botpt/MJ03F-NANO-1dayMeans-06Jul2017-01Dec2025.csv")
TMPSF_PATH = Path("/home/jovyan/repos/specKitScience/my-analysis_tmpsf/outputs/data/tmpsf_2015-2026_daily.parquet")

OUTPUT_DIR = Path(__file__).parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "poster"
EXPLORATORY_DIR = OUTPUT_DIR / "figures" / "exploratory"

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
POSTER_CAPTION_SIZE = 22    # figure captions
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
        "field": "ASHES",
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
    "Hell": "#E69F00",         # Orange - ASHES field
    "El Guapo": "#0072B2",     # Blue
    "El Guapo (Top)": "#56B4E9",  # Sky blue
    "Virgin": "#CC79A7",       # Reddish purple
    "Trevi / Mkr156": "#009E73",  # Bluish green
    "Vixen / Mkr218": "#F0E442",  # Yellow
    "Casper": "#009E73",       # Bluish green
    "Diva": "#D55E00",         # Vermillion
    "Castle": "#009E73",       # Bluish green
    "Trevi": "#E69F00",        # Orange
    "Vixen": "#D55E00",        # Vermillion
    "Escargot": "#CC79A7",     # Reddish purple
}

# --- Historical instruments (2010-2011 eruption period) ---
HISTORICAL_INSTRUMENTS = [
    {
        "file": DATA_HISTORICAL / "casper" / "MISO104-Chip1-Axial-2010-Casper.txt",
        "instrument": "MISO 104",
        "vent": "Casper",
        "field": "Coquille",
        "deployment": "2010-2011",
        "format": "miso_historical_tab",
    },
    {
        "file": DATA_HISTORICAL / "diva" / "MISO129-Chip1-Axial-2010-Diva.txt",
        "instrument": "MISO 129",
        "vent": "Diva",
        "field": "International District",
        "deployment": "2010-2011",
        "format": "miso_historical_tab",
    },
]

# UNUSED: 2015-2019 instruments (pre-inflation period) - Removed per user request
# # --- 2015-2019 instruments (pre-inflation period) ---
# INSTRUMENTS_2015_2019 = [
#     {
#         "file": DATA_HISTORICAL / "vixen" / "MISO103-Chip1-Axial-2015-Vixen.txt",
#         "instrument": "MISO 103",
#         "vent": "Vixen",
#         "field": "Coquille",
#         "deployment": "2015-2019",
#         "format": "miso_historical_tab",
#     },
#     {
#         "file": DATA_HISTORICAL / "trevi" / "MISO101-Chip1-Axial-2015-Trevi.txt",
#         "instrument": "MISO 101",
#         "vent": "Trevi",
#         "field": "ASHES",
#         "deployment": "2015-2019",
#         "format": "miso_historical_tab",
#     },
# ]


# --- 2015 eruption instruments (Vixen, Casper, Escargot) ---
INSTRUMENTS_2015_ERUPTION = [
    {
        "file": DATA_HISTORICAL / "vixen" / "Vixen-ALL-2001-2017.mat",
        "instrument": "MISO (compiled)",
        "vent": "Vixen",
        "field": "Coquille",
        "deployment": "2014-2017",
        "format": "mat_all",
        "time_start": "2014-09-01",
        "time_end": "2017-08-31",
        "skip_spike_filter": True,
    },
    {
        "file": DATA_HISTORICAL / "casper" / "Casper-ALL-2006-2015.mat",
        "instrument": "MISO (compiled)",
        "vent": "Casper",
        "field": "Coquille",
        "deployment": "2014-2015",
        "format": "mat_all",
        "time_start": "2014-09-01",
        "time_end": "2015-08-31",
        "skip_spike_filter": True,
    },
    {
        "files": [
            Path("/home/jovyan/ooi/kdata/RS03INT1-MJ03C-10-TRHPHA301-streamed-trhph_sample/"
                 "deployment0001_RS03INT1-MJ03C-10-TRHPHA301-streamed-trhph_sample_"
                 "20140927T064607.055936-20150709T235946.064065.nc"),
            Path("/home/jovyan/ooi/kdata/RS03INT1-MJ03C-10-TRHPHA301-streamed-trhph_sample/"
                 "deployment0002_RS03INT1-MJ03C-10-TRHPHA301-streamed-trhph_sample_"
                 "20150711T213941.594888-20160725T235941.850078.nc"),
            Path("/home/jovyan/ooi/kdata/RS03INT1-MJ03C-10-TRHPHA301-streamed-trhph_sample/"
                 "deployment0003_RS03INT1-MJ03C-10-TRHPHA301-streamed-trhph_sample_"
                 "20160727T050905.011982-20170814T225158.561216.nc"),
            Path("/home/jovyan/ooi/kdata/RS03INT1-MJ03C-10-TRHPHA301-streamed-trhph_sample/"
                 "deployment0004_RS03INT1-MJ03C-10-TRHPHA301-streamed-trhph_sample_"
                 "20170822T165705.775934-20190703T013257.827590.nc"),
        ],
        "instrument": "OOI TRHPHA301",
        "vent": "Escargot",
        "field": "International District",
        "deployment": "2014-2019",
        "format": "ooi_trhph",
        "time_start": "2014-09-01",
        "time_end": "2019-07-03",
        "deploy_threshold": 0,
        "mad_threshold": 2.0,
        "max_drop_per_hour": 5.0,
    },
]


def remove_dropouts(series, max_drop_per_hour=5.0):
    """Remove sudden temperature dropouts by rate-of-change filtering.

    When temperature drops faster than max_drop_per_hour, marks data as NaN
    until temperature recovers to within 10% of the pre-dropout rolling median.

    Parameters
    ----------
    series : pd.Series
        Temperature time series with datetime index
    max_drop_per_hour : float
        Maximum allowable temperature drop in °C per hour (default: 5.0)

    Returns
    -------
    pd.Series
        Series with dropouts replaced by NaN
    int
        Number of points removed
    """
    cleaned = series.copy()

    # Estimate sampling interval in hours
    dt_hours = series.index.to_series().diff().median().total_seconds() / 3600
    max_drop = max_drop_per_hour * dt_hours

    # Rolling median as reference for "normal" temperature
    samples_per_day = max(1, int(24 / dt_hours))
    rolling_med = cleaned.rolling(window=samples_per_day, center=True, min_periods=1).median()

    diff = cleaned.diff()
    in_dropout = False
    pre_dropout_level = np.nan
    n_removed = 0

    for i in range(len(cleaned)):
        if not in_dropout:
            # Check for sudden drop
            if diff.iloc[i] < -max_drop:
                in_dropout = True
                pre_dropout_level = rolling_med.iloc[max(0, i - 1)]
                cleaned.iloc[i] = np.nan
                n_removed += 1
        else:
            # Stay in dropout until temperature recovers to within 10% of pre-dropout level
            if cleaned.iloc[i] >= pre_dropout_level * 0.9:
                in_dropout = False
            else:
                cleaned.iloc[i] = np.nan
                n_removed += 1

    return cleaned, n_removed


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


def load_mat_all_instrument(config):
    """Load compiled MISO .mat file (date + temp arrays, MATLAB datenum)."""
    import scipy.io
    import datetime

    mat = scipy.io.loadmat(config["file"])
    dates_raw = mat["date"].flatten()
    temps_raw = mat["temp"]  # shape (N, 2) — two thermocouples

    # Drop rows where datenum is NaN
    valid = ~np.isnan(dates_raw)
    dates_raw = dates_raw[valid]
    temps_raw = temps_raw[valid]

    # Convert MATLAB datenum to datetime
    def matlab2datetime(datenum):
        return (datetime.datetime.fromordinal(int(datenum))
                + datetime.timedelta(days=datenum % 1)
                - datetime.timedelta(days=366))

    datetimes = pd.to_datetime([matlab2datetime(d) for d in dates_raw])

    # Use first thermocouple column as primary temperature
    out = pd.DataFrame({"temperature": temps_raw[:, 0]}, index=datetimes)
    out.index.name = "datetime"
    out = out.sort_index()

    # Subset to requested time window
    t_start = pd.Timestamp(config["time_start"])
    t_end = pd.Timestamp(config["time_end"])
    out = out.loc[t_start:t_end]

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
    if not config.get("skip_spike_filter", False):
        mad_thresh = config.get("mad_threshold", MAD_THRESHOLD)
        deployed_mask = out["deployed"]
        if deployed_mask.sum() > 10:
            temp_cleaned, n_spikes = remove_spikes_mad(
                out.loc[deployed_mask, "temperature"],
                window_hours=MAD_WINDOW_HOURS,
                threshold=mad_thresh,
            )
            out.loc[deployed_mask, "temperature"] = temp_cleaned

    # Remove duplicate timestamps before dropout filter
    out = out[~out.index.duplicated(keep="first")]

    # Dropout filter (rate-of-change)
    n_dropouts = 0
    max_drop = config.get("max_drop_per_hour")
    if max_drop is not None:
        deployed_mask = out["deployed"]
        if deployed_mask.sum() > 10:
            dep_notna = out.loc[deployed_mask, "temperature"].dropna()
            temp_cleaned, n_dropouts = remove_dropouts(dep_notna, max_drop_per_hour=max_drop)
            out.loc[temp_cleaned.index, "temperature"] = temp_cleaned

    out.attrs["instrument"] = config["instrument"]
    out.attrs["vent"] = config["vent"]
    out.attrs["field"] = config["field"]
    out.attrs["deployment"] = config["deployment"]
    out.attrs["n_capped"] = n_capped
    out.attrs["n_spikes"] = n_spikes
    out.attrs["n_dropouts"] = n_dropouts

    return out


def load_ooi_trhph_instrument(config):
    """Load OOI TRHPH NetCDF file(s) via h5py (vent_fluid_temperature)."""
    import h5py

    # Support single file or list of files
    files = config.get("files") or [config["file"]]

    all_time = []
    all_temp = []
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            all_time.append(f["time"][:])
            all_temp.append(f["vent_fluid_temperature"][:])

    time_raw = np.concatenate(all_time)
    temp_raw = np.concatenate(all_temp)

    # Convert OOI time (seconds since 1900-01-01) to datetime
    ooi_epoch = pd.Timestamp("1900-01-01")
    datetimes = pd.to_datetime(time_raw, unit="s", origin=ooi_epoch)

    out = pd.DataFrame({"temperature": temp_raw}, index=datetimes)
    out.index.name = "datetime"
    out = out.sort_index()

    # Subset to requested time window
    t_start = pd.Timestamp(config["time_start"])
    t_end = pd.Timestamp(config["time_end"])
    out = out.loc[t_start:t_end]

    # Cap unphysical values
    n_capped = int((out["temperature"] > PHYSICAL_MAX).sum())
    out.loc[out["temperature"] > PHYSICAL_MAX, "temperature"] = np.nan

    # Settling window QC — use per-instrument threshold if provided
    deploy_thresh = config.get("deploy_threshold", DEPLOYMENT_THRESHOLD)
    if deploy_thresh == 0:
        out["deployed"] = out["temperature"].notna()
    else:
        hot_mask = out["temperature"] > deploy_thresh
        if hot_mask.any():
            first_hot = out[hot_mask].index.min()
            stable_start = first_hot + pd.Timedelta(hours=SETTLING_HOURS)
            out["deployed"] = (out.index >= stable_start) & (out["temperature"] > 50)
        else:
            out["deployed"] = out["temperature"] > 50

    # MAD spike removal on deployed data
    n_spikes = 0
    if not config.get("skip_spike_filter", False):
        mad_thresh = config.get("mad_threshold", MAD_THRESHOLD)
        deployed_mask = out["deployed"]
        if deployed_mask.sum() > 10:
            temp_cleaned, n_spikes = remove_spikes_mad(
                out.loc[deployed_mask, "temperature"],
                window_hours=MAD_WINDOW_HOURS,
                threshold=mad_thresh,
            )
            out.loc[deployed_mask, "temperature"] = temp_cleaned

    # Remove duplicate timestamps before dropout filter
    out = out[~out.index.duplicated(keep="first")]

    # Dropout filter (rate-of-change)
    n_dropouts = 0
    max_drop = config.get("max_drop_per_hour")
    if max_drop is not None:
        deployed_mask = out["deployed"]
        if deployed_mask.sum() > 10:
            dep_notna = out.loc[deployed_mask, "temperature"].dropna()
            temp_cleaned, n_dropouts = remove_dropouts(dep_notna, max_drop_per_hour=max_drop)
            out.loc[temp_cleaned.index, "temperature"] = temp_cleaned

    out.attrs["instrument"] = config["instrument"]
    out.attrs["vent"] = config["vent"]
    out.attrs["field"] = config["field"]
    out.attrs["deployment"] = config["deployment"]
    out.attrs["n_capped"] = n_capped
    out.attrs["n_spikes"] = n_spikes
    out.attrs["n_dropouts"] = n_dropouts

    return out


def set_spine_width(ax, width=POSTER_SPINE_WIDTH):
    """Set the linewidth for all spines (plot bounding box) of an axes."""
    for spine in ax.spines.values():
        spine.set_linewidth(width)


def add_caption_justified(fig, caption_text, caption_width=0.85, fontsize=POSTER_CAPTION_SIZE, caption_left=0.05):
    """Add a left-aligned, justified caption below the figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to add caption to
    caption_text : str
        The caption text
    caption_width : float
        Width of caption in figure fraction (default 0.85 for ~85% of figure width)
    fontsize : int
        Caption font size
    """
    # Calculate approximate wrap width for line breaking
    caption_width_in = caption_width * fig.get_size_inches()[0]
    char_width_in = fontsize / 72 * 0.55  # approximate char width for sans-serif
    wrap_chars = int(caption_width_in / char_width_in)
    lines = textwrap.wrap(caption_text, width=wrap_chars)

    # Create caption axes at bottom
    caption_ax = fig.add_axes([caption_left, 0.02, caption_width, 0.15])
    caption_ax.axis('off')

    # Use renderer to measure actual text widths for true justification
    renderer = fig.canvas.get_renderer()
    ax_bbox = caption_ax.get_window_extent(renderer)

    # Measure line height
    sample = caption_ax.text(0, 0, "Tg", fontsize=fontsize, family='sans-serif',
                             transform=caption_ax.transAxes)
    sample_bbox = sample.get_window_extent(renderer)
    line_height = (sample_bbox.height * 1.35) / ax_bbox.height
    sample.remove()

    for i, line in enumerate(lines):
        y = 1.0 - i * line_height
        words = line.split()

        if i < len(lines) - 1 and len(words) > 1:
            # Measure each word's rendered width in axes coordinates
            word_widths = []
            for word in words:
                t = caption_ax.text(0, 0, word, fontsize=fontsize, family='sans-serif',
                                    transform=caption_ax.transAxes)
                wb = t.get_window_extent(renderer)
                word_widths.append(wb.width / ax_bbox.width)
                t.remove()

            total_word_width = sum(word_widths)
            remaining = 1.0 - total_word_width
            gap = remaining / (len(words) - 1)

            x = 0.0
            for j, word in enumerate(words):
                caption_ax.text(x, y, word, fontsize=fontsize, family='sans-serif',
                                transform=caption_ax.transAxes, va='top', ha='left')
                x += word_widths[j] + gap
        else:
            # Last line: left-aligned
            caption_ax.text(0.0, y, line, fontsize=fontsize, family='sans-serif',
                            transform=caption_ax.transAxes, va='top', ha='left')


def fig_historical_eruption(records, fig_path, eruption_date=None, bpr=None):
    """Figure: Historical vent temperatures around the 2011 eruption."""
    # Create figure — aspect ratio matches panel (a) of 2015 eruption figure
    fig = plt.figure(figsize=(10, 5.5), dpi=POSTER_DPI)
    ax = fig.add_axes([0.10, 0.30, 0.78, 0.58])  # [left, bottom, width, height] - plot area

    # Cut off data at local maximum (mid-July 2011) to exclude instrument recovery
    data_end = pd.Timestamp("2011-07-18")

    for rec in records:
        deployed = rec[rec["deployed"]]
        vent = rec.attrs["vent"]
        field = rec.attrs["field"]
        # Abbreviate International District
        field_abbr = "ID" if field == "International District" else field
        color = VENT_COLORS.get(vent, "#333333")
        label = f"{vent} ({field_abbr})"
        daily = deployed["temperature"].resample("D").mean()
        daily = daily[daily.index <= data_end]  # Trim to local max
        ax.plot(daily.index, daily.values, color=color, linewidth=POSTER_LINE_WIDTH, alpha=0.85, label=label)

    ax.set_xlabel("Date", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Temp (\u00b0C)", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
    ax.set_title("April 2011 Eruption Response", fontsize=POSTER_TITLE_SIZE, fontweight="bold", pad=21)
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

    # BPR on right axis
    ax2 = None
    if bpr is not None:
        xmin_ts = pd.Timestamp("2010-08-15")
        xmax_ts = pd.Timestamp("2011-08-01")
        ax2 = ax.twinx()
        bpr_visible = bpr.loc[xmin_ts:xmax_ts].dropna()
        if len(bpr_visible) > 0:
            ax2.plot(bpr_visible.index, bpr_visible.values,
                     color="#1E90FF", alpha=0.7, linewidth=POSTER_LINE_WIDTH + 0.5,
                     linestyle="-", label="Uplift (m)")
            ax2.set_ylabel("Uplift (m)", color="#1E90FF", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
            ax2.tick_params(axis="y", labelcolor="#1E90FF", labelsize=POSTER_TICK_SIZE)
            set_spine_width(ax2)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
    else:
        all_lines, all_labels = lines1, labels1
    ax.legend(all_lines, all_labels, loc="lower left", fontsize=POSTER_LEGEND_SIZE, frameon=True, framealpha=0.9)

    # Eruption annotation
    if eruption_date:
        ax.axvline(eruption_date, color="#CC0000", linestyle="--", linewidth=POSTER_ANNOT_LINE_WIDTH, alpha=0.8)
        ax.annotate("April 6, 2011\neruption", xy=(eruption_date, 328),
                    xytext=(5, -5), textcoords="offset points",
                    fontsize=POSTER_ANNOT_SIZE, color="#CC0000", va="top", fontweight="bold",
                    arrowprops=dict(arrowstyle='->', color='#CC0000', lw=1.5))

    # Figure caption - left-aligned, justified
    caption = (
        "Vent temperatures spanning the April 6, 2011 Axial Seamount eruption. "
        "Diva (red, International District) showed a stronger response (~70\u00b0C drop) "
        "than Casper (bluish green, Coquille, ~10\u00b0C). "
        "Blue dashed line: differential seafloor uplift (m) from bottom pressure recorders "
        "shows relative vertical displacement of caldera floor to top of caldera wall "
        "as a proxy for volcanic inflation and deflation (Chadwick et al., 2022)."
    )
    add_caption_justified(fig, caption, caption_width=0.85, fontsize=POSTER_CAPTION_SIZE)

    fig.savefig(fig_path, dpi=POSTER_DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {fig_path}")


# UNUSED: fig_2015_2019 - Removed per user request (keeping only fig_eruption_2015_vce for 2015 eruption)
# def fig_2015_2019(records, fig_path, eruption_date=None):
#     """Figure: Vixen and Trevi temperatures spanning 2015-2019 (includes April 2015 eruption)."""
#     # Create figure with space for caption below
#     fig = plt.figure(figsize=(10, 8), dpi=POSTER_DPI)
#     ax = fig.add_axes([0.1, 0.28, 0.85, 0.62])
#
#     # Find actual data range
#     all_starts = []
#     all_ends = []
#     for rec in records:
#         deployed = rec[rec["deployed"]]
#         if len(deployed) > 0:
#             all_starts.append(deployed.index.min())
#             all_ends.append(deployed.index.max())
#
#     for rec in records:
#         deployed = rec[rec["deployed"]]
#         if len(deployed) == 0:
#             continue
#
#         vent = rec.attrs["vent"]
#         field = rec.attrs["field"]
#         color = VENT_COLORS.get(vent, "#333333")
#         label = f"{vent} ({field})"
#         daily = deployed["temperature"].resample("D").mean()
#         ax.plot(daily.index, daily.values, color=color, linewidth=POSTER_LINE_WIDTH, alpha=0.85, label=label)
#
#     ax.set_xlabel("Date", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
#     ax.set_ylabel("Temperature (°C)", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
#     ax.set_title("Vent Temperatures After the April 2015 Eruption\nAxial Seamount", fontsize=POSTER_TITLE_SIZE, fontweight="bold")
#     ax.grid(True, alpha=0.3)
#     ax.tick_params(labelsize=POSTER_TICK_SIZE)
#     set_spine_width(ax)
#
#     # Set x-axis limits with balanced padding (based on actual data)
#     if all_starts:
#         xmin = min(all_starts) - pd.Timedelta(days=45)
#         xmax = max(all_ends) + pd.Timedelta(days=45)
#         ax.set_xlim(xmin, xmax)
#
#     # Clean date formatting - 4 month intervals for ~2 year span
#     ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
#     ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")
#
#     # Legend in upper right
#     ax.legend(loc="upper right", fontsize=POSTER_LEGEND_SIZE, frameon=True, framealpha=0.9)
#
#     # April 2015 eruption annotation
#     if eruption_date:
#         ax.axvline(eruption_date, color="#CC0000", linestyle="--", linewidth=POSTER_ANNOT_LINE_WIDTH, alpha=0.8)
#         ax.annotate("April 24, 2015\neruption", xy=(eruption_date, ax.get_ylim()[1]),
#                     xytext=(5, -5), textcoords="offset points",
#                     fontsize=POSTER_ANNOT_SIZE, color="#CC0000", va="top", fontweight="bold")
#
#     # Figure caption - left-aligned, justified
#     caption = (
#         "Vent temperatures following the April 24, 2015 Axial Seamount eruption. "
#         "Y-axis: temperature (°C); x-axis: date. Vixen (purple, Coquille field) and Trevi (orange, ASHES field). "
#         "Deployments began ~4 months post-eruption. Both vents show variable temperatures (~150–300°C)."
#     )
#     add_caption_justified(fig, caption, caption_width=0.85, fontsize=POSTER_CAPTION_SIZE)
#
#     fig.savefig(fig_path, dpi=POSTER_DPI, bbox_inches="tight", pad_inches=0.1)
#     plt.close(fig)
#     print(f"Saved: {fig_path}")


def load_bpr_historical():
    """Load and concatenate historical BPR data (2010-2017 combo + 2017-2025 NANO)."""
    # 2010-2017: multi-column CSV, each column is a deployment period
    df_combo = pd.read_csv(BPR_HIST_PATH)
    dt = pd.to_datetime(df_combo["Date/time"])

    # Stack deployment columns into a single series, keeping earliest non-NaN per timestamp
    pieces = []
    for col in ["2007-2010", "2010-2011", "2011-2013", "2013-2015", "2015-2017"]:
        s = pd.Series(df_combo[col].values, index=dt, name="uplift_m")
        pieces.append(s.dropna())
    bpr_hist = pd.concat(pieces).sort_index()
    bpr_hist = bpr_hist[~bpr_hist.index.duplicated(keep="first")]

    # Resample to daily means
    bpr_hist = bpr_hist.resample("D").mean().dropna()

    # 2017-2025 NANO daily means
    df_nano = pd.read_csv(BPR_NANO_PATH)
    dt_nano = pd.to_datetime(df_nano["Date"])
    bpr_nano = pd.Series(df_nano.iloc[:, 2].values, index=dt_nano, name="uplift_m").dropna()

    # Concatenate, prefer NANO where overlapping
    bpr_all = pd.concat([bpr_hist.loc[:bpr_nano.index.min() - pd.Timedelta(days=1)], bpr_nano])
    bpr_all = bpr_all.sort_index()
    bpr_all = bpr_all[~bpr_all.index.duplicated(keep="last")]

    return bpr_all


def fig_eruption_2015_vce(records, fig_path, eruption_date=None, bpr=None):
    """Figure: Vixen, Casper, and Escargot individual profiles around the April 2015 eruption.

    Three panels:
      (a) Vixen (Sep 2014 – Aug 10 2015)
      (b) Casper (full record)
      (c) Escargot (full record, crash period masked)
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), dpi=POSTER_DPI)
    fig.subplots_adjust(hspace=0.35, bottom=0.26, top=0.93, left=0.10, right=0.88)
    fig.suptitle("April 2015 Eruption Response", fontsize=POSTER_TITLE_SIZE,
                 fontweight="bold", y=0.97)

    # Find records by vent name
    vixen_rec = None
    casper_rec = None
    escargot_rec = None
    for rec in records:
        if rec.attrs["vent"] == "Vixen":
            vixen_rec = rec
        elif rec.attrs["vent"] == "Casper":
            casper_rec = rec
        elif rec.attrs["vent"] == "Escargot":
            escargot_rec = rec

    # ─── Helper: add BPR + eruption line to a panel ───
    def _add_bpr_and_eruption(ax, t_start, t_end, show_eruption=True,
                               show_legend=True, legend_loc="upper right",
                               annot_side="left", annot_y_frac=0.75):
        ax_r = None
        if bpr is not None:
            ax_r = ax.twinx()
            bpr_vis = bpr.loc[t_start:t_end].dropna()
            if len(bpr_vis) > 0:
                ax_r.plot(bpr_vis.index, bpr_vis.values,
                          color="#0868AC", alpha=0.5, linewidth=POSTER_LINE_WIDTH + 0.5,
                          linestyle="--", label="Uplift (m)")
                ax_r.set_ylabel("Uplift (m)", color="#0868AC",
                                fontsize=POSTER_LABEL_SIZE, fontweight="bold")
                ax_r.tick_params(axis="y", labelcolor="#0868AC", labelsize=POSTER_TICK_SIZE)
                set_spine_width(ax_r)
        if show_eruption and eruption_date and t_start <= eruption_date <= t_end:
            ax.axvline(eruption_date, color="#CC0000", linestyle="--",
                       linewidth=POSTER_ANNOT_LINE_WIDTH, alpha=0.8)
            ymin, ymax = ax.get_ylim()
            y_pos = ymin + (ymax - ymin) * annot_y_frac
            if annot_side == "right":
                ax.annotate("April 24, 2015\neruption", xy=(eruption_date, y_pos),
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=POSTER_ANNOT_SIZE, color="#CC0000",
                            va="center", ha="left", fontweight="bold",
                            arrowprops=dict(arrowstyle='->', color='#CC0000', lw=1.5))
            else:
                ax.annotate("April 24, 2015\neruption", xy=(eruption_date, y_pos),
                            xytext=(-5, 0), textcoords="offset points",
                            fontsize=POSTER_ANNOT_SIZE, color="#CC0000",
                            va="center", ha="right", fontweight="bold",
                            arrowprops=dict(arrowstyle='->', color='#CC0000', lw=1.5))
        if show_legend:
            lines1, labels1 = ax.get_legend_handles_labels()
            if ax_r is not None:
                lines2, labels2 = ax_r.get_legend_handles_labels()
                lines1 += lines2
                labels1 += labels2
            ax.legend(lines1, labels1, loc=legend_loc,
                      fontsize=POSTER_LEGEND_SIZE, frameon=True, framealpha=0.9)

    # ─── Panel (a): Vixen (Sep 2014 – Aug 10 2015) ───
    ax_a = axes[0]
    if vixen_rec is not None:
        t_start_v = vixen_rec[vixen_rec["deployed"]].index.min() - pd.Timedelta(days=7)
        t_end_v = pd.Timestamp("2015-08-10")
        deployed = vixen_rec[vixen_rec["deployed"]]
        temp_v = deployed["temperature"].loc[:t_end_v]

        color = VENT_COLORS.get("Vixen", "#D55E00")
        ax_a.plot(temp_v.index, temp_v.values, color=color,
                  linewidth=POSTER_LINE_WIDTH, alpha=0.85, label="Vixen (Coquille)")
        ax_a.set_ylabel("Temp (\u00b0C)", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
        ax_a.text(0.02, 0.08, "(a)", transform=ax_a.transAxes,
                  fontsize=POSTER_LEGEND_SIZE, fontweight="bold", va="bottom", ha="left",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8))
        ax_a.grid(True, alpha=0.3)
        ax_a.tick_params(labelsize=POSTER_TICK_SIZE)
        set_spine_width(ax_a)
        ax_a.set_xlim(t_start_v, t_end_v)
        ax_a.set_ylim(320, 332)
        ax_a.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax_a.xaxis.get_majorticklabels(), rotation=0, ha="center")

        ax_a.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        _add_bpr_and_eruption(ax_a, t_start_v, t_end_v, show_legend=False,
                              annot_side="left", annot_y_frac=0.15)

    # ─── Panel (b): Casper (full record) ───
    ax_b = axes[1]
    if casper_rec is not None:
        deployed = casper_rec[casper_rec["deployed"]]
        temp_c = deployed["temperature"]
        t_start_c = temp_c.index.min() - pd.Timedelta(days=7)
        t_end_c = temp_c.index.max() + pd.Timedelta(days=7)

        color = VENT_COLORS.get("Casper", "#009E73")
        ax_b.plot(temp_c.index, temp_c.values, color=color,
                  linewidth=POSTER_LINE_WIDTH, alpha=0.85, label="Casper (Coquille)")
        ax_b.set_ylabel("Temp (\u00b0C)", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
        ax_b.text(0.02, 0.08, "(b)", transform=ax_b.transAxes,
                  fontsize=POSTER_LEGEND_SIZE, fontweight="bold", va="bottom", ha="left",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8))
        ax_b.grid(True, alpha=0.3)
        ax_b.tick_params(labelsize=POSTER_TICK_SIZE)
        set_spine_width(ax_b)
        ax_b.set_xlim(t_start_c, t_end_c)
        ax_b.set_ylim(289, 302)
        ax_b.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax_b.xaxis.get_majorticklabels(), rotation=0, ha="center")

        ax_b.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

        _add_bpr_and_eruption(ax_b, t_start_c, t_end_c, show_legend=False,
                              annot_side="left", annot_y_frac=0.15)

    # ─── Panel (c): Escargot (full record, crash period masked) ───
    ax_c = axes[2]
    if escargot_rec is not None:
        deployed = escargot_rec[escargot_rec["deployed"]]
        daily = deployed["temperature"].resample("D").mean()
        # Mask crash period so line breaks naturally
        daily.loc["2014-11-24":"2015-07-14"] = np.nan
        t_start_e = daily.index.min() - pd.Timedelta(days=7)
        t_end_e = daily.index.max() + pd.Timedelta(days=7)

        color = VENT_COLORS.get("Escargot", "#CC79A7")
        ax_c.plot(daily.index, daily.values, color=color,
                  linewidth=POSTER_LINE_WIDTH, alpha=0.85, label="Escargot (Int'l District)")
        ax_c.set_xlabel("Date", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
        ax_c.set_ylabel("Temp (\u00b0C)", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
        ax_c.text(0.02, 0.08, "(c)", transform=ax_c.transAxes,
                  fontsize=POSTER_LEGEND_SIZE, fontweight="bold", va="bottom", ha="left",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8))
        ax_c.grid(True, alpha=0.3)
        ax_c.tick_params(labelsize=POSTER_TICK_SIZE)
        set_spine_width(ax_c)
        ax_c.set_xlim(t_start_e, t_end_e)
        ax_c.xaxis.set_major_locator(mdates.YearLocator())
        ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_c.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax_c.xaxis.get_majorticklabels(), rotation=0, ha="center")

        ax_c.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

        _add_bpr_and_eruption(ax_c, t_start_e, t_end_e, show_legend=False, annot_side="right")

    # Figure caption
    caption = (
        "Individual raw vent temperatures spanning the April 24, 2015 Axial Seamount eruption. "
        "$\\mathbf{(a)}$ Vixen (Coquille) shows increased variability and co-eruptive "
        "perturbation with ~2\u00b0C post-eruption decline. "
        "$\\mathbf{(b)}$ Casper (Coquille) shows ~5\u00b0C post-eruption drop. "
        "$\\mathbf{(c)}$ Escargot (Int'l District, 2014\u20132019) shows decline to ~255\u00b0C "
        "with multi-year recovery. Blue dashed line: differential seafloor uplift (m) from "
        "bottom pressure recorders shows relative vertical displacement of central caldera "
        "center BPR (MJ03F) to eastern caldera BPR (MJ03E) as a proxy for volcanic inflation "
        "and deflation."
    )
    add_caption_justified(fig, caption, caption_width=0.85, fontsize=POSTER_CAPTION_SIZE)

    fig.savefig(fig_path, dpi=POSTER_DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {fig_path}")


def fig_eruption_2015_vixen_casper(records, fig_path, eruption_date=None, bpr=None):
    """Figure: Vixen and Casper temperatures + uplift zoomed to the April 2015 eruption window."""
    fig = plt.figure(figsize=(10, 8), dpi=POSTER_DPI)
    ax = fig.add_axes([0.1, 0.28, 0.80, 0.62])

    # Time window: 2 months before eruption to end of June 2015
    t_start = pd.Timestamp("2015-02-24")
    t_end = pd.Timestamp("2015-07-01")

    for rec in records:
        vent = rec.attrs["vent"]
        if vent not in ("Vixen", "Casper"):
            continue
        deployed = rec[rec["deployed"]]
        if len(deployed) == 0:
            continue

        field = rec.attrs["field"]
        color = VENT_COLORS.get(vent, "#333333")
        label = f"{vent} ({field})"
        daily = deployed["temperature"].resample("D").mean()
        daily = daily.loc[t_start:t_end]
        ax.plot(daily.index, daily.values, color=color, linewidth=POSTER_LINE_WIDTH, alpha=0.85, label=label)

    ax.set_xlabel("Date", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Temperature (°C)", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
    ax.set_title("April 2015 Eruption — Vixen & Casper", fontsize=POSTER_TITLE_SIZE, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=POSTER_TICK_SIZE)
    set_spine_width(ax)

    ax.set_xlim(t_start, t_end)
    ax.set_ylim(bottom=290)

    # Monthly tick marks
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    # BPR on right axis
    ax2 = None
    if bpr is not None:
        ax2 = ax.twinx()
        bpr_visible = bpr.loc[t_start:t_end].dropna()
        if len(bpr_visible) > 0:
            ax2.plot(bpr_visible.index, bpr_visible.values,
                     color="#0868AC", alpha=0.5, linewidth=POSTER_LINE_WIDTH + 0.5,
                     linestyle="--", label="Uplift (m)")
            ax2.set_ylabel("Uplift (m)", color="#0868AC", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
            ax2.tick_params(axis="y", labelcolor="#0868AC", labelsize=POSTER_TICK_SIZE)
            set_spine_width(ax2)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
    else:
        all_lines, all_labels = lines1, labels1
    ax.legend(all_lines, all_labels, loc="upper right", fontsize=POSTER_LEGEND_SIZE, frameon=True, framealpha=0.9)

    # Eruption annotation
    if eruption_date:
        ax.axvline(eruption_date, color="#CC0000", linestyle="--", linewidth=POSTER_ANNOT_LINE_WIDTH, alpha=0.8)
        ymin, ymax = ax.get_ylim()
        y_pos = ymin + (ymax - ymin) * 0.25
        ax.annotate("April 24, 2015\neruption", xy=(eruption_date, y_pos),
                    xytext=(-5, 0), textcoords="offset points",
                    fontsize=POSTER_ANNOT_SIZE, color="#CC0000", va="center", ha="right", fontweight="bold",
                    arrowprops=dict(arrowstyle='->', color='#CC0000', lw=1.5))

    # Caption
    caption = (
        "Vent temperatures from Vixen and Casper (both Coquille vent field) spanning the "
        "April 24, 2015 Axial Seamount eruption. Y-axis: temperature (°C); right axis: "
        "seafloor uplift (m). Time window begins 2 months pre-eruption and ends before "
        "Vixen's sustained post-eruption temperature decline. BPR uplift (dashed blue) "
        "shows co-eruptive deflation and onset of re-inflation."
    )
    add_caption_justified(fig, caption, caption_width=0.85, fontsize=POSTER_CAPTION_SIZE)

    fig.savefig(fig_path, dpi=POSTER_DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {fig_path}")


def fig_single_vent_2015(rec, fig_path, eruption_date=None, bpr=None, t_end_override=None):
    """Figure: Single vent temperature + uplift around the 2015 eruption."""
    fig = plt.figure(figsize=(10, 8), dpi=POSTER_DPI)
    ax = fig.add_axes([0.1, 0.28, 0.80, 0.62])

    vent = rec.attrs["vent"]
    field = rec.attrs["field"]
    color = VENT_COLORS.get(vent, "#333333")

    deployed = rec[rec["deployed"]]
    daily = deployed["temperature"].resample("D").mean()

    if t_end_override is not None:
        daily = daily.loc[:t_end_override]

    t_start = daily.index.min() - pd.Timedelta(days=7)
    t_end = daily.index.max() + pd.Timedelta(days=7)

    ax.plot(daily.index, daily.values, color=color, linewidth=POSTER_LINE_WIDTH, alpha=0.85,
            label=f"{vent} ({field})")

    ax.set_xlabel("Date", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Temperature (\u00b0C)", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
    ax.set_title(f"{vent} ({field})", fontsize=POSTER_TITLE_SIZE, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=POSTER_TICK_SIZE)
    set_spine_width(ax)
    ax.set_xlim(t_start, t_end)

    # Adaptive tick spacing based on time span
    span_days = (t_end - t_start).days
    if span_days < 120:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    elif span_days < 400:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    # BPR on right axis
    ax2 = None
    if bpr is not None:
        ax2 = ax.twinx()
        bpr_visible = bpr.loc[t_start:t_end].dropna()
        if len(bpr_visible) > 0:
            ax2.plot(bpr_visible.index, bpr_visible.values,
                     color="#0868AC", alpha=0.5, linewidth=POSTER_LINE_WIDTH + 0.5,
                     linestyle="--", label="Uplift (m)")
            ax2.set_ylabel("Uplift (m)", color="#0868AC", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
            ax2.tick_params(axis="y", labelcolor="#0868AC", labelsize=POSTER_TICK_SIZE)
            set_spine_width(ax2)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
    else:
        all_lines, all_labels = lines1, labels1
    ax.legend(all_lines, all_labels, loc="upper right", fontsize=POSTER_LEGEND_SIZE, frameon=True, framealpha=0.9)

    # Eruption annotation
    if eruption_date and t_start <= eruption_date <= t_end:
        ax.axvline(eruption_date, color="#CC0000", linestyle="--", linewidth=POSTER_ANNOT_LINE_WIDTH, alpha=0.8)
        ymin, ymax = ax.get_ylim()
        y_pos = ymin + (ymax - ymin) * 0.25
        ax.annotate("April 24, 2015\neruption", xy=(eruption_date, y_pos),
                    xytext=(-5, 0), textcoords="offset points",
                    fontsize=POSTER_ANNOT_SIZE, color="#CC0000", va="center", ha="right", fontweight="bold",
                    arrowprops=dict(arrowstyle='->', color='#CC0000', lw=1.5))

    fig.savefig(fig_path, dpi=POSTER_DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {fig_path}")


def fig_escargot_multi(records, fig_path, eruption_date=None, bpr=None):
    """Figure: Escargot (top) + Vixen around eruption (bottom), both with BPR overlay."""
    # Find records
    escargot_rec = None
    vixen_rec = None
    for rec in records:
        if rec.attrs["vent"] == "Escargot":
            escargot_rec = rec
        elif rec.attrs["vent"] == "Vixen":
            vixen_rec = rec

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 12), dpi=POSTER_DPI)
    fig.subplots_adjust(hspace=0.35, bottom=0.12, top=0.94)

    # --- Panel (a): Escargot (full record) ---
    if escargot_rec is not None:
        deployed = escargot_rec[escargot_rec["deployed"]]
        daily = deployed["temperature"].resample("D").mean()
        t_start_e = daily.index.min() - pd.Timedelta(days=7)
        t_end_e = daily.index.max() + pd.Timedelta(days=7)

        color = VENT_COLORS.get("Escargot", "#CC79A7")
        ax_top.plot(daily.index, daily.values, color=color, linewidth=POSTER_LINE_WIDTH,
                    alpha=0.85, label="Escargot (Int'l District)")
        ax_top.set_ylabel("Temperature (\u00b0C)", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
        ax_top.set_title("(a) Escargot — International District", fontsize=POSTER_TITLE_SIZE, fontweight="bold")
        ax_top.grid(True, alpha=0.3)
        ax_top.tick_params(labelsize=POSTER_TICK_SIZE)
        set_spine_width(ax_top)
        ax_top.set_xlim(t_start_e, t_end_e)

        ax_top.xaxis.set_major_locator(mdates.MonthLocator())
        ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax_top.xaxis.get_majorticklabels(), rotation=0, ha="center")
        ax_top.set_xlabel("Date", fontsize=POSTER_LABEL_SIZE, fontweight="bold")

        # BPR overlay
        ax_top_r = None
        if bpr is not None:
            ax_top_r = ax_top.twinx()
            bpr_vis = bpr.loc[t_start_e:t_end_e].dropna()
            if len(bpr_vis) > 0:
                ax_top_r.plot(bpr_vis.index, bpr_vis.values,
                              color="#0868AC", alpha=0.5, linewidth=POSTER_LINE_WIDTH + 0.5,
                              linestyle="--", label="Uplift (m)")
                ax_top_r.set_ylabel("Uplift (m)", color="#0868AC", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
                ax_top_r.tick_params(axis="y", labelcolor="#0868AC", labelsize=POSTER_TICK_SIZE)
                set_spine_width(ax_top_r)

        # Legend
        lines1, labels1 = ax_top.get_legend_handles_labels()
        if ax_top_r is not None:
            lines2, labels2 = ax_top_r.get_legend_handles_labels()
            lines1 += lines2
            labels1 += labels2
        ax_top.legend(lines1, labels1, loc="upper right", fontsize=POSTER_LEGEND_SIZE, frameon=True, framealpha=0.9)

    # --- Panel (b): Vixen (2 months pre-eruption to Aug 10 2015) ---
    if vixen_rec is not None:
        t_start_v = pd.Timestamp("2015-02-24")
        t_end_v = pd.Timestamp("2015-08-10")

        deployed = vixen_rec[vixen_rec["deployed"]]
        daily = deployed["temperature"].resample("D").mean()
        daily = daily.loc[t_start_v:t_end_v]

        color = VENT_COLORS.get("Vixen", "#D55E00")
        ax_bot.plot(daily.index, daily.values, color=color, linewidth=POSTER_LINE_WIDTH,
                    alpha=0.85, label="Vixen (Coquille)")
        ax_bot.set_xlabel("Date", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
        ax_bot.set_ylabel("Temperature (\u00b0C)", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
        ax_bot.set_title("(b) Vixen — Coquille", fontsize=POSTER_TITLE_SIZE, fontweight="bold")
        ax_bot.grid(True, alpha=0.3)
        ax_bot.tick_params(labelsize=POSTER_TICK_SIZE)
        set_spine_width(ax_bot)
        ax_bot.set_xlim(t_start_v, t_end_v)

        ax_bot.xaxis.set_major_locator(mdates.MonthLocator())
        ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=0, ha="center")

        # BPR overlay
        ax_bot_r = None
        if bpr is not None:
            ax_bot_r = ax_bot.twinx()
            bpr_vis = bpr.loc[t_start_v:t_end_v].dropna()
            if len(bpr_vis) > 0:
                ax_bot_r.plot(bpr_vis.index, bpr_vis.values,
                              color="#0868AC", alpha=0.5, linewidth=POSTER_LINE_WIDTH + 0.5,
                              linestyle="--", label="Uplift (m)")
                ax_bot_r.set_ylabel("Uplift (m)", color="#0868AC", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
                ax_bot_r.tick_params(axis="y", labelcolor="#0868AC", labelsize=POSTER_TICK_SIZE)
                set_spine_width(ax_bot_r)

        # Eruption annotation
        if eruption_date and t_start_v <= eruption_date <= t_end_v:
            ax_bot.axvline(eruption_date, color="#CC0000", linestyle="--", linewidth=POSTER_ANNOT_LINE_WIDTH, alpha=0.8)
            ymin, ymax = ax_bot.get_ylim()
            y_pos = ymin + (ymax - ymin) * 0.25
            ax_bot.annotate("April 24, 2015\neruption", xy=(eruption_date, y_pos),
                            xytext=(-5, 0), textcoords="offset points",
                            fontsize=POSTER_ANNOT_SIZE, color="#CC0000", va="center", ha="right", fontweight="bold",
                            arrowprops=dict(arrowstyle='->', color='#CC0000', lw=1.5))

        # Legend
        lines1, labels1 = ax_bot.get_legend_handles_labels()
        if ax_bot_r is not None:
            lines2, labels2 = ax_bot_r.get_legend_handles_labels()
            lines1 += lines2
            labels1 += labels2
        ax_bot.legend(lines1, labels1, loc="upper right", fontsize=POSTER_LEGEND_SIZE, frameon=True, framealpha=0.9)

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
    """Figure 1: All vents as subplots, combining deployments on same panel."""
    plotable = [r for r in records if r[r["deployed"]].shape[0] > 0]

    # Group records by vent name
    from collections import defaultdict
    vent_groups = defaultdict(list)
    for rec in plotable:
        vent_groups[rec.attrs["vent"]].append(rec)

    # Sort vents: ASHES first, then International District
    # Get field for each vent (use first record's field)
    def vent_sort_key(vent_name):
        field = vent_groups[vent_name][0].attrs["field"]
        # ASHES = 0, International District = 1
        field_order = 0 if field == "ASHES" else 1
        return (field_order, vent_name)

    sorted_vents = sorted(vent_groups.keys(), key=vent_sort_key)

    n = len(sorted_vents)
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

    # Line styles for different deployments
    DEPLOY_STYLES = {
        "2022-2024": {"ls": "-", "lw": POSTER_LINE_WIDTH},
        "2024-2025": {"ls": "--", "lw": POSTER_LINE_WIDTH},
    }

    for ax, vent in zip(axes, sorted_vents):
        recs = vent_groups[vent]
        field = recs[0].attrs["field"]
        field_abbr = "ID" if field == "International District" else field
        color = VENT_COLORS.get(vent, "#333333")

        for rec in recs:
            deployed = rec[rec["deployed"]]
            dep = rec.attrs["deployment"]
            dep_abbr = dep.replace("2022-2024", "22–24").replace("2024-2025", "24–25")
            style = DEPLOY_STYLES.get(dep, {"ls": "-", "lw": POSTER_LINE_WIDTH})

            daily = deployed["temperature"].resample("D").mean()
            ax.plot(daily.index, daily.values, color=color,
                    linestyle=style["ls"], linewidth=style["lw"],
                    label=dep_abbr, alpha=0.85)

        ax.set_ylabel("°C", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
        ax.set_title(f"{vent} ({field_abbr})", fontsize=POSTER_LABEL_SIZE, fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=POSTER_TICK_SIZE)
        ax.set_xlim(xmin, xmax)
        set_spine_width(ax)

        # Add legend if multiple deployments
        if len(recs) > 1:
            ax.legend(loc="upper right", fontsize=POSTER_LEGEND_SIZE - 2, frameon=True, framealpha=0.9)

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
            ax.annotate("Chadwick\ncruise", xy=(deploy_change, ax.get_ylim()[1]),
                        xytext=(-5, -5), textcoords="offset points",
                        fontsize=POSTER_ANNOT_SIZE, color="#666666", va="top", ha="right")

    fig.suptitle("MISO Temperature Survey\nAll Recent Instruments (2022–2025)",
                 fontsize=POSTER_TITLE_SIZE, fontweight="bold", y=0.99)

    # Figure caption - left-aligned, justified
    caption = (
        "Daily mean vent temperatures from MISO deployments at Axial Seamount (2022–2025). "
        "Each panel shows one vent; solid lines = 2022–2024, dashed = 2024–2025. "
        "ASHES vents shown first, then International District (ID). "
        "Vertical dotted line marks Chadwick cruise (June 2024)."
    )

    # Calculate wrap width
    caption_width = 0.85
    caption_width_in = caption_width * fig.get_size_inches()[0]
    char_width_in = POSTER_CAPTION_SIZE / 72 * 0.50
    wrap_chars = int(caption_width_in / char_width_in)
    caption_wrapped = textwrap.fill(caption, width=wrap_chars)

    # Add caption at bottom with white background box
    caption_ax = fig.add_axes([0.075, 0.01, caption_width, 0.06])
    caption_ax.axis('off')
    caption_ax.text(0.0, 0.5, caption_wrapped, ha="left", va="center",
                    fontsize=POSTER_CAPTION_SIZE - 2, transform=caption_ax.transAxes,
                    family='sans-serif',
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

    # Figure caption - left-aligned, justified
    caption = (
        "Daily mean temperatures from high-temperature vents at Axial Seamount (2022–2025). "
        "Y-axis: temperature (°C). Inferno (ASHES) shows stable ~285–310°C across both deployments. "
        "Hell (ASHES) and El Guapo (International District) show greater variability, with El Guapo "
        "exhibiting dramatic swings (100–315°C). El Guapo Top is the hottest and most stable (~341°C). "
        "Vertical dashed line marks Chadwick cruise (June 2024)."
    )
    add_caption_justified(fig, caption, caption_width=0.85, fontsize=POSTER_CAPTION_SIZE)

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
        ("Hell", "2022-2024"):         {"color": "#E69F00", "ls": "-",  "lw": POSTER_LINE_WIDTH, "label": "Hell (ASHES, 22–24)"},
        ("El Guapo", "2022-2024"):     {"color": "#0072B2", "ls": "-",  "lw": POSTER_LINE_WIDTH, "label": "El Guapo (ID, 22–24)"},
        ("El Guapo (Top)", "2024-2025"): {"color": "#56B4E9", "ls": "--", "lw": POSTER_LINE_WIDTH, "label": "El Guapo Top (ID, 24–25)"},
    }

    # Single panel (with space for caption) — plot height matches Fig 1
    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=POSTER_DPI)
    fig.subplots_adjust(bottom=0.30, top=0.88, right=0.88)

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

    # BPR on right axis - show uplift relative to post-eruption minimum (April 2015)
    if bpr is not None:
        ax2 = ax1.twinx()
        # Reference to post-eruption minimum (April 29, 2015) for intuitive positive values
        post_eruption_min = bpr["differential_m"].loc['2015-04-01':'2015-06-01'].min()
        bpr_uplift_m = bpr["differential_m"] - post_eruption_min  # Now 0 = deflated, positive = inflated
        ax2.plot(bpr.index, bpr_uplift_m,
                 color="#0868AC", alpha=0.5, linewidth=POSTER_LINE_WIDTH + 0.5, linestyle="--",
                 label="Uplift")
        ax2.set_ylabel("Uplift since April 2015 (m)", color="#0868AC", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="#0868AC", labelsize=POSTER_TICK_SIZE)
        set_spine_width(ax2)

        # Constrain y-axis to visible BPR range (within the x-axis window)
        bpr_visible = bpr_uplift_m.loc[xmin:xmax].dropna()
        if len(bpr_visible) > 0:
            ypad = (bpr_visible.max() - bpr_visible.min()) * 0.05
            ax2.set_ylim(bpr_visible.min() - ypad, bpr_visible.max() + ypad)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
    else:
        all_lines, all_labels = ax1.get_legend_handles_labels()

    ax1.set_xlabel("Date", fontsize=POSTER_LABEL_SIZE, fontweight="bold")
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Chadwick cruise annotation on main panel (MISO servicing, June 2024)
    chadwick_cruise = pd.Timestamp("2024-06-26")
    if xmin <= chadwick_cruise <= xmax:
        ax1.axvline(chadwick_cruise, color="#666666", linestyle=":", linewidth=POSTER_ANNOT_LINE_WIDTH, alpha=0.7)
        ax1.annotate("Chadwick\ncruise", xy=(chadwick_cruise, ax1.get_ylim()[1]),
                     xytext=(-5, -5), textcoords="offset points",
                     fontsize=POSTER_ANNOT_SIZE, color="#666666", va="top", ha="right",
                     arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5))

    # Legend in lower right, to the right of deployment change line (vertical)
    ax1.legend(all_lines, all_labels,
               loc="lower left", bbox_to_anchor=(0.62, 0.02), ncol=1,
               fontsize=POSTER_LEGEND_SIZE - 2, frameon=True, framealpha=0.9)

    ax1.set_title("Recent System Changes",
                  fontsize=POSTER_TITLE_SIZE, fontweight="bold", pad=21)
    ax1.grid(True, alpha=0.3)
    set_spine_width(ax1)

    # Figure caption - justified at 22pt, shifted 1cm right to align with Fig 2 on poster
    caption = (
        "Daily mean high-temperature vent temps with seafloor uplift "
        "(right axis). Blue dashed line: differential seafloor uplift (m) from "
        "bottom pressure recorders shows relative vertical displacement of central caldera "
        "center BPR (MJ03F) to eastern caldera BPR (MJ03E) as a proxy for volcanic inflation "
        "and deflation. "
        "Vertical line marks Chadwick cruise "
        "(June 2024). BPR shows ~1.6 m re-inflation since 2015."
    )
    offset_1cm = (1.0 / 2.54) / fig.get_size_inches()[0]
    add_caption_justified(fig, caption, caption_width=0.85, fontsize=POSTER_CAPTION_SIZE,
                          caption_left=0.05 + offset_1cm)

    fig.savefig(fig_path, dpi=POSTER_DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {fig_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MISO survey analysis and figure generation")
    parser.add_argument("--figure", "-f", type=str, default=None,
                        help="Generate only a specific figure group: "
                             "survey, poster_bpr, eruption_2011, eruption_2015, all (default: all)")
    args = parser.parse_args()
    target = args.figure

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EXPLORATORY_DIR.mkdir(parents=True, exist_ok=True)

    # --- Current survey instruments (needed for survey, poster_bpr) ---
    records = None
    summary = None
    bpr = None
    tmpsf = None
    if target in (None, "all", "survey", "poster_bpr"):
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
        if BPR_PATH.exists():
            bpr = pd.read_parquet(BPR_PATH)
            print(f"\nBPR data loaded: {bpr.index.min()} to {bpr.index.max()}")

        # Load TMPSF
        if TMPSF_PATH.exists():
            tmpsf = pd.read_parquet(TMPSF_PATH)
            print(f"TMPSF data loaded: {tmpsf.index.min()} to {tmpsf.index.max()}")

        # Figures
        print("\nGenerating figures...")
        if target in (None, "all", "survey"):
            fig_survey_overview(records, EXPLORATORY_DIR / "survey_overview.png")
            fig_hightemp_comparison(records, summary, EXPLORATORY_DIR / "survey_hightemp_comparison.png")
        if target in (None, "all", "poster_bpr"):
            fig_poster_bpr(records, summary, bpr, FIG_DIR / "fig3_poster_temp_bpr_tmpsf.png", tmpsf=tmpsf)

    # --- Historical 2011 eruption ---
    if target in (None, "all", "eruption_2011"):
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
            bpr_2011 = None
            if BPR_HIST_PATH.exists():
                print("Loading historical BPR data for 2011 figure...")
                bpr_2011 = load_bpr_historical()
                print(f"  BPR range: {bpr_2011.index.min()} to {bpr_2011.index.max()}")
            fig_historical_eruption(historical_records, FIG_DIR / "fig1_eruption_2011_casper_diva.png",
                                    eruption_date=eruption_date, bpr=bpr_2011)

    # UNUSED: 2015-2019 figure (Vixen, Trevi) - Removed per user request
    # # Load and plot 2015-2019 data (Vixen, Trevi)
    # print("\nLoading 2015-2019 instruments (Vixen, Trevi)...")
    # records_2015 = []
    # for config in INSTRUMENTS_2015_2019:
    #     vent = config["vent"]
    #     inst = config["instrument"]
    #     print(f"Loading {vent} ({inst})...")
    #     try:
    #         rec = load_historical_instrument(config)
    #         records_2015.append(rec)
    #         deployed = rec[rec["deployed"]]
    #         n_dep = len(deployed)
    #         if n_dep > 0:
    #             temp = deployed["temperature"].dropna()
    #             print(f"  Deployed samples: {n_dep:,}  |  "
    #                   f"Temp: {temp.min():.1f}–{temp.max():.1f}°C  |  "
    #                   f"Capped: {rec.attrs['n_capped']}  Spikes: {rec.attrs['n_spikes']}")
    #         else:
    #             print(f"  No deployed samples (total rows: {len(rec)})")
    #     except Exception as e:
    #         print(f"  ERROR loading {vent} ({inst}): {e}")
    #
    # if records_2015:
    #     eruption_2015 = pd.Timestamp("2015-04-24")  # April 2015 eruption
    #     fig_2015_2019(records_2015, FIG_DIR / "eruption_2015_vixen_trevi.png",
    #                   eruption_date=eruption_2015)

    # --- 2015 eruption (Vixen, Casper, Escargot) ---
    if target in (None, "all", "eruption_2015"):
        print("\nLoading 2015 eruption instruments (Vixen, Casper, Escargot)...")
        records_2015_eruption = []
        for config in INSTRUMENTS_2015_ERUPTION:
            vent = config["vent"]
            inst = config["instrument"]
            fmt = config["format"]
            print(f"Loading {vent} ({inst})...")
            try:
                if fmt == "mat_all":
                    rec = load_mat_all_instrument(config)
                elif fmt == "ooi_trhph":
                    rec = load_ooi_trhph_instrument(config)
                else:
                    raise ValueError(f"Unknown format: {fmt}")
                records_2015_eruption.append(rec)
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

        if records_2015_eruption:
            eruption_2015 = pd.Timestamp("2015-04-24")
            # Load historical BPR
            bpr_hist = None
            if BPR_HIST_PATH.exists():
                print("Loading historical BPR data...")
                bpr_hist = load_bpr_historical()
                print(f"  BPR range: {bpr_hist.index.min()} to {bpr_hist.index.max()}")
            fig_eruption_2015_vce(records_2015_eruption, FIG_DIR / "fig2_eruption_2015_vixen_casper_escargot.png",
                                  eruption_date=eruption_2015, bpr=bpr_hist)
            fig_eruption_2015_vixen_casper(records_2015_eruption, EXPLORATORY_DIR / "eruption_2015_vixen_casper.png",
                                           eruption_date=eruption_2015, bpr=bpr_hist)

            # Individual vent plots
            for rec in records_2015_eruption:
                vent = rec.attrs["vent"]
                if vent == "Vixen":
                    fig_single_vent_2015(rec, EXPLORATORY_DIR / "eruption_2015_vixen.png",
                                         eruption_date=eruption_2015, bpr=bpr_hist,
                                         t_end_override=pd.Timestamp("2015-08-10"))
                elif vent == "Casper":
                    fig_single_vent_2015(rec, EXPLORATORY_DIR / "eruption_2015_casper.png",
                                         eruption_date=eruption_2015, bpr=bpr_hist)
                elif vent == "Escargot":
                    fig_single_vent_2015(rec, EXPLORATORY_DIR / "eruption_2015_escargot.png",
                                         eruption_date=eruption_2015, bpr=bpr_hist)
            fig_escargot_multi(records_2015_eruption, EXPLORATORY_DIR / "eruption_2015_escargot_multi.png",
                               eruption_date=eruption_2015, bpr=bpr_hist)

    print("\nDone!")


if __name__ == "__main__":
    main()
