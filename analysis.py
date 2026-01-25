"""
MISO Hydrothermal Vent Temperature Analysis - Inferno Vent

Analyzes high-resolution temperature time series from the Inferno vent
at Axial Seamount's International District vent field.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
INFERNO_PATH = Path("/home/jovyan/my_data/axial/axial_miso/2022_2024_MISO/MISOTEMP_2017-002_Axial2022_inferno.csv")
BPR_PATH = Path("/home/jovyan/repos/specKitScience/my-analysis_botpt/outputs/data/differential_uplift_daily.parquet")
EQ_PATH = Path("/home/jovyan/repos/specKitScience/earthquakes_my-analysis/outputs/data/earthquake_daily.parquet")

OUTPUT_DIR = Path(__file__).parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures"

# Missing data flag
MISSING_FLAG = -888.88


def load_inferno_data(path: Path) -> pd.DataFrame:
    """
    Load and parse Inferno vent temperature data.

    Parameters
    ----------
    path : Path
        Path to the CSV file

    Returns
    -------
    pd.DataFrame
        Parsed temperature data with datetime index
    """
    df = pd.read_csv(
        path,
        encoding='utf-8-sig',  # Handle BOM
        parse_dates=['datetime'],
        date_format='%m/%d/%y %I:%M:%S %p'
    )

    # Replace missing data flag with NaN
    df['infernoTemp'] = df['infernoTemp'].replace(MISSING_FLAG, np.nan)
    df['infernoRef'] = df['infernoRef'].replace(MISSING_FLAG, np.nan)

    # Filter to valid vent temperatures (>50°C indicates deployed in vent)
    # Keep all data but flag pre-deployment
    df['deployed'] = df['infernoTemp'] > 50

    # Set datetime as index
    df = df.set_index('datetime')

    return df


def create_temperature_timeseries(df: pd.DataFrame, fig_path: Path):
    """Create temperature time series plot for Inferno vent."""
    # Filter to deployed data only
    deployed = df[df['deployed']].copy()

    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

    ax.plot(deployed.index, deployed['infernoTemp'],
            color='#D62728', linewidth=0.5, alpha=0.8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Inferno Vent Temperature - International District, Axial Seamount')

    ax.set_xlim(deployed.index.min(), deployed.index.max())
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


def create_temperature_bpr_overlay(df: pd.DataFrame, bpr: pd.DataFrame, fig_path: Path):
    """Create overlay of Inferno temperature and BPR differential uplift."""
    # Filter to deployed data and resample to daily
    deployed = df[df['deployed']].copy()
    temp_daily = deployed['infernoTemp'].resample('D').mean()

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

    # Temperature on left axis
    color1 = '#D62728'
    ax1.plot(temp_daily.index, temp_daily.values,
             color=color1, linewidth=0.5, alpha=0.8, label='Inferno temperature')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Find overlapping time range
    xmin = max(temp_daily.index.min(), bpr.index.min())
    xmax = min(temp_daily.index.max(), bpr.index.max())
    ax1.set_xlim(xmin, xmax)

    # BPR differential uplift on right axis
    ax2 = ax1.twinx()
    color2 = '#0868AC'
    ax2.plot(bpr.index, bpr['differential_m'] * 100,
             color=color2, alpha=0.8, linewidth=0.5, label='Differential uplift')
    ax2.set_ylabel('Differential uplift (cm)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title('Inferno Vent Temperature and Volcanic Deformation')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


def create_temperature_earthquake_overlay(df: pd.DataFrame, eq: pd.DataFrame, fig_path: Path):
    """Create overlay of Inferno temperature and earthquake counts."""
    # Filter to deployed data and resample to daily
    deployed = df[df['deployed']].copy()
    temp_daily = deployed['infernoTemp'].resample('D').mean()

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

    # Temperature on left axis
    color1 = '#D62728'
    ax1.plot(temp_daily.index, temp_daily.values,
             color=color1, linewidth=0.5, alpha=0.8, label='Inferno temperature')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Find overlapping time range
    xmin = max(temp_daily.index.min(), eq.index.min())
    xmax = min(temp_daily.index.max(), eq.index.max())
    ax1.set_xlim(xmin, xmax)

    # Earthquake counts on right axis
    ax2 = ax1.twinx()
    color2 = '#7A0177'
    ax2.fill_between(eq.index, eq['count'],
                     color=color2, alpha=0.3, label='Earthquakes/day')
    ax2.set_ylabel('Earthquakes per day', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(bottom=0)

    ax1.set_title('Inferno Vent Temperature and Seismicity')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


def main():
    print("Loading Inferno vent temperature data...")
    df = load_inferno_data(INFERNO_PATH)

    deployed = df[df['deployed']]
    print(f"Loaded {len(df):,} records")
    print(f"Deployed period: {deployed.index.min()} to {deployed.index.max()}")
    print(f"Temperature range: {deployed['infernoTemp'].min():.1f} to {deployed['infernoTemp'].max():.1f} °C")

    # Create output directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Save processed data
    df.to_parquet(DATA_DIR / "inferno_temperature.parquet")
    print(f"Saved: {DATA_DIR / 'inferno_temperature.parquet'}")

    # Daily averages
    daily = deployed[['infernoTemp', 'infernoRef']].resample('D').agg(['mean', 'min', 'max', 'std'])
    daily.columns = ['_'.join(col) for col in daily.columns]
    daily.to_parquet(DATA_DIR / "inferno_daily.parquet")
    print(f"Saved: {DATA_DIR / 'inferno_daily.parquet'}")

    # Create temperature time series figure
    create_temperature_timeseries(df, FIG_DIR / "inferno_temperature.png")

    # Load BPR data and create overlay
    if BPR_PATH.exists():
        print(f"Loading BPR data...")
        bpr = pd.read_parquet(BPR_PATH)
        create_temperature_bpr_overlay(df, bpr, FIG_DIR / "inferno_bpr_overlay.png")
    else:
        print(f"BPR data not found at {BPR_PATH}")

    # Load earthquake data and create overlay
    if EQ_PATH.exists():
        print(f"Loading earthquake data...")
        eq = pd.read_parquet(EQ_PATH)
        create_temperature_earthquake_overlay(df, eq, FIG_DIR / "inferno_earthquake_overlay.png")
    else:
        print(f"Earthquake data not found at {EQ_PATH}")

    print("\nDone!")


if __name__ == "__main__":
    main()
