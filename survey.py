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

# --- Paths ---
DATA_2022 = Path("/home/jovyan/my_data/axial/axial_miso/2022_2024_MISO")
DATA_2024 = Path("/home/jovyan/my_data/axial/axial_miso/2024_2025_MISO")
BPR_PATH = Path("/home/jovyan/repos/specKitScience/my-analysis_botpt/outputs/data/differential_uplift_daily.parquet")

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
        "field": "International District",
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
VENT_COLORS = {
    "Inferno": "#D62728",
    "Hell": "#FF7F0E",
    "El Guapo": "#2CA02C",
    "El Guapo (Top)": "#1F77B4",
    "Virgin": "#9467BD",
    "Trevi / Mkr156": "#8C564B",
    "Vixen / Mkr218": "#E377C2",
}


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
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), dpi=300, sharex=False)
    if n == 1:
        axes = [axes]

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

    axes[-1].set_xlabel("Date")
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

    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)

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
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def fig_poster_bpr(records, summary, bpr, fig_path):
    """Figure 3: Poster figure — high-temp vents + BPR overlay."""
    high_recs = [r for r, s in zip(records, summary.itertuples())
                 if s.Classification == "High-temp"]

    # Poster-specific styling: group by vent identity across deployments
    # Same base color for same vent, solid=2022-2024, dashed=2024-2025
    POSTER_STYLE = {
        ("Inferno", "2022-2024"):      {"color": "#D62728", "ls": "-",  "lw": 1.4, "label": "Inferno (ID, 2022–24)"},
        ("Inferno", "2024-2025"):      {"color": "#D62728", "ls": "--", "lw": 1.4, "label": "Inferno (ASHES, 2024–25)"},
        ("Hell", "2022-2024"):         {"color": "#FF7F0E", "ls": "-",  "lw": 1.4, "label": "Hell (ID, 2022–24)"},
        ("El Guapo", "2022-2024"):     {"color": "#2CA02C", "ls": "-",  "lw": 1.4, "label": "El Guapo (ID, 2022–24)"},
        ("El Guapo (Top)", "2024-2025"): {"color": "#2CA02C", "ls": "--", "lw": 1.4, "label": "El Guapo Top (ID, 2024–25)"},
    }

    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=300)

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

    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Temperature (°C)", fontsize=12)
    ax1.tick_params(labelsize=10)

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
        ax2.plot(bpr.index, bpr["differential_m"] * 100,
                 color="#0868AC", alpha=0.5, linewidth=1.5, linestyle="--",
                 label="Differential uplift")
        ax2.set_ylabel("Differential uplift (cm)", color="#0868AC", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="#0868AC", labelsize=10)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper center", bbox_to_anchor=(0.5, -0.12),
                   ncol=3, fontsize=9, frameon=True)
    else:
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
                   ncol=3, fontsize=9, frameon=True)

    ax1.set_title("Hydrothermal Vent Temperatures and Volcanic Deformation\nAxial Seamount (2022–2025)",
                  fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.subplots_adjust(bottom=0.2)
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

    # Figures
    print("\nGenerating figures...")
    fig_survey_overview(records, FIG_DIR / "survey_overview.png")
    fig_hightemp_comparison(records, summary, FIG_DIR / "survey_hightemp_comparison.png")
    fig_poster_bpr(records, summary, bpr, FIG_DIR / "poster_temp_bpr.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
