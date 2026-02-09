"""
MISO Hydrothermal Vent Temperature Analysis

Analyzes high-resolution temperature time series from multiple vents
at Axial Seamount's International District vent field.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Paths
DATA_DIR_MISO = Path("/home/jovyan/my_data/axial/axial_miso/2022_2024_MISO")
BPR_PATH = Path("/home/jovyan/repos/specKitScience/my-analysis_botpt/outputs/data/differential_uplift_daily.parquet")
EQ_PATH = Path("/home/jovyan/repos/specKitScience/earthquakes_my-analysis/outputs/data/earthquake_daily.parquet")

OUTPUT_DIR = Path(__file__).parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory"

# Vent configurations
VENTS = {
    'inferno': {
        'file': 'MISOTEMP_2017-002_Axial2022_inferno.csv',
        'temp_col': 'infernoTemp',
        'ref_col': 'infernoRef',
        'color': '#D62728',
        'has_index': False,
    },
    'hell': {
        'file': 'MISOTEMP_2017-019_Axial2022_hel.csv',
        'temp_col': 'hell_temp1',
        'ref_col': 'hell_ref',
        'color': '#FF7F0E',
        'has_index': False,
    },
    'el_guapo': {
        'file': 'MISO_2017-006_Axial2022_ElGuapo.csv',
        'temp_col': 'Tem',
        'ref_col': 'refTemp',
        'color': '#2CA02C',
        'has_index': True,
    },
}

# Missing data flag
MISSING_FLAG = -888.88

# Quality control parameters
DEPLOYMENT_THRESHOLD = 200  # °C - temperature indicating sensor is in vent
SETTLING_HOURS = 24  # Hours to wait after first high temp
MAD_THRESHOLD = 5.0  # MAD multiplier for spike removal
MAD_WINDOW_HOURS = 24  # Rolling window for MAD calculation


def remove_spikes_mad(series: pd.Series, window_hours: int = 24, threshold: float = 5.0) -> pd.Series:
    """Remove spikes using rolling median and MAD (median absolute deviation).

    MAD is more robust to outliers than standard deviation.

    Parameters
    ----------
    series : pd.Series
        Temperature time series
    window_hours : int
        Rolling window size in hours (default: 24)
    threshold : float
        Number of scaled MADs for spike threshold (default: 5.0)

    Returns
    -------
    pd.Series
        Series with spikes replaced by NaN
    """
    cleaned = series.copy()

    # Calculate samples per hour (assume ~10 min sampling = 6 per hour)
    samples_per_hour = 6
    window = window_hours * samples_per_hour

    # Use median (robust to outliers)
    rolling_median = cleaned.rolling(window=window, center=True, min_periods=1).median()

    # Calculate MAD (median absolute deviation)
    deviation = (cleaned - rolling_median).abs()
    rolling_mad = deviation.rolling(window=window, center=True, min_periods=1).median()

    # Scale MAD to be comparable to std (for normal distribution, std ≈ 1.4826 * MAD)
    scaled_mad = 1.4826 * rolling_mad

    # Flag values more than threshold MADs from rolling median
    is_spike = deviation > (threshold * scaled_mad)

    n_spikes = is_spike.sum()
    if n_spikes > 0:
        cleaned[is_spike] = np.nan

    return cleaned, n_spikes


def load_vent_data(vent_name: str) -> pd.DataFrame:
    """
    Load and parse vent temperature data.

    Parameters
    ----------
    vent_name : str
        Name of the vent (inferno, hell, el_guapo)

    Returns
    -------
    pd.DataFrame
        Parsed temperature data with datetime index
    """
    config = VENTS[vent_name]
    path = DATA_DIR_MISO / config['file']

    if config['has_index']:
        # El Guapo has row index as first column
        df = pd.read_csv(
            path,
            encoding='utf-8-sig',
            index_col=0,
        )
        df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%y %I:%M:%S %p')
    else:
        df = pd.read_csv(
            path,
            encoding='utf-8-sig',
            parse_dates=['datetime'],
            date_format='%m/%d/%y %I:%M:%S %p'
        )

    temp_col = config['temp_col']
    ref_col = config['ref_col']

    # Replace missing data flag with NaN
    df[temp_col] = df[temp_col].replace(MISSING_FLAG, np.nan)
    if ref_col in df.columns:
        df[ref_col] = df[ref_col].replace(MISSING_FLAG, np.nan)

    # Standardize column names
    df = df.rename(columns={temp_col: 'temperature', ref_col: 'reference'})

    # Set datetime as index
    df = df.set_index('datetime')

    # Add vent name
    df['vent'] = vent_name

    # Quality control: settling window approach
    # Find first time temperature exceeds deployment threshold
    hot_mask = df['temperature'] > DEPLOYMENT_THRESHOLD
    if hot_mask.any():
        first_hot = df[hot_mask].index.min()
        stable_start = first_hot + pd.Timedelta(hours=SETTLING_HOURS)
        df['deployed'] = (df.index >= stable_start) & (df['temperature'] > 50)
    else:
        # Fallback to simple threshold if never gets hot
        df['deployed'] = df['temperature'] > 50

    # Apply MAD-based spike removal to deployed data
    deployed_mask = df['deployed']
    if deployed_mask.any():
        temp_cleaned, n_spikes = remove_spikes_mad(
            df.loc[deployed_mask, 'temperature'],
            window_hours=MAD_WINDOW_HOURS,
            threshold=MAD_THRESHOLD
        )
        df.loc[deployed_mask, 'temperature'] = temp_cleaned
        if n_spikes > 0:
            print(f"    Removed {n_spikes} spikes ({100*n_spikes/deployed_mask.sum():.2f}%)")

    return df


def create_temperature_timeseries(df: pd.DataFrame, vent_name: str, color: str, fig_path: Path):
    """Create temperature time series plot for a single vent."""
    deployed = df[df['deployed']].copy()

    fig, ax = plt.subplots(figsize=(6, 3), dpi=300)

    ax.plot(deployed.index, deployed['temperature'],
            color=color, linewidth=0.5, alpha=0.8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'{vent_name.replace("_", " ").title()} Vent Temperature - International District, Axial Seamount')

    ax.set_xlim(deployed.index.min(), deployed.index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


def create_multi_vent_comparison(vent_data: dict, fig_path: Path):
    """Create comparison plot of all vents."""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    for vent_name, df in vent_data.items():
        deployed = df[df['deployed']].copy()
        daily = deployed['temperature'].resample('D').mean()
        color = VENTS[vent_name]['color']
        label = vent_name.replace('_', ' ').title()
        ax.plot(daily.index, daily.values, color=color, linewidth=1, alpha=0.8, label=label)

    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Hydrothermal Vent Temperatures - International District, Axial Seamount')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


def create_multi_vent_bpr_overlay(vent_data: dict, bpr: pd.DataFrame, fig_path: Path):
    """Create overlay of all vent temperatures and BPR differential uplift."""
    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=300)

    # Plot each vent temperature
    for vent_name, df in vent_data.items():
        deployed = df[df['deployed']].copy()
        daily = deployed['temperature'].resample('D').mean()
        color = VENTS[vent_name]['color']
        label = vent_name.replace('_', ' ').title()
        ax1.plot(daily.index, daily.values, color=color, linewidth=0.8, alpha=0.8, label=label)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)')
    ax1.tick_params(axis='y')

    # Find common time range
    all_times = []
    for df in vent_data.values():
        deployed = df[df['deployed']]
        all_times.extend([deployed.index.min(), deployed.index.max()])
    xmin = max(min(all_times), bpr.index.min())
    xmax = min(max(all_times), bpr.index.max())
    ax1.set_xlim(xmin, xmax)

    # BPR differential uplift on right axis
    ax2 = ax1.twinx()
    ax2.plot(bpr.index, bpr['differential_m'] * 100,
             color='#0868AC', alpha=0.6, linewidth=1.5, linestyle='--', label='Differential uplift')
    ax2.set_ylabel('Differential uplift (cm)', color='#0868AC')
    ax2.tick_params(axis='y', labelcolor='#0868AC')

    ax1.set_title('Hydrothermal Vent Temperatures and Volcanic Deformation')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


def create_multi_vent_earthquake_overlay(vent_data: dict, eq: pd.DataFrame, fig_path: Path):
    """Create overlay of all vent temperatures and earthquake counts."""
    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=300)

    # Plot each vent temperature
    for vent_name, df in vent_data.items():
        deployed = df[df['deployed']].copy()
        daily = deployed['temperature'].resample('D').mean()
        color = VENTS[vent_name]['color']
        label = vent_name.replace('_', ' ').title()
        ax1.plot(daily.index, daily.values, color=color, linewidth=0.8, alpha=0.8, label=label)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)')
    ax1.tick_params(axis='y')

    # Find common time range
    all_times = []
    for df in vent_data.values():
        deployed = df[df['deployed']]
        all_times.extend([deployed.index.min(), deployed.index.max()])
    xmin = max(min(all_times), eq.index.min())
    xmax = min(max(all_times), eq.index.max())
    ax1.set_xlim(xmin, xmax)

    # Earthquake counts on right axis
    ax2 = ax1.twinx()
    ax2.fill_between(eq.index, eq['count'],
                     color='#7A0177', alpha=0.2, label='Earthquakes/day')
    ax2.set_ylabel('Earthquakes per day', color='#7A0177')
    ax2.tick_params(axis='y', labelcolor='#7A0177')
    ax2.set_ylim(bottom=0)

    ax1.set_title('Hydrothermal Vent Temperatures and Seismicity')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


def create_vent_bpr_overlay(df: pd.DataFrame, bpr: pd.DataFrame, vent_name: str, color: str, fig_path: Path):
    """Create overlay of vent temperature and BPR differential uplift."""
    deployed = df[df['deployed']].copy()
    temp_daily = deployed['temperature'].resample('D').mean()

    fig, ax1 = plt.subplots(figsize=(6, 3), dpi=300)

    # Temperature on left axis
    ax1.plot(temp_daily.index, temp_daily.values,
             color=color, linewidth=0.5, alpha=0.8, label=f'{vent_name.replace("_", " ").title()} temperature')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Find overlapping time range
    xmin = max(temp_daily.index.min(), bpr.index.min())
    xmax = min(temp_daily.index.max(), bpr.index.max())
    ax1.set_xlim(xmin, xmax)

    # BPR differential uplift on right axis
    ax2 = ax1.twinx()
    ax2.plot(bpr.index, bpr['differential_m'] * 100,
             color='#0868AC', alpha=0.8, linewidth=0.5, label='Differential uplift')
    ax2.set_ylabel('Differential uplift (cm)', color='#0868AC')
    ax2.tick_params(axis='y', labelcolor='#0868AC')

    ax1.set_title(f'{vent_name.replace("_", " ").title()} Vent Temperature and Volcanic Deformation')
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


def create_vent_earthquake_overlay(df: pd.DataFrame, eq: pd.DataFrame, vent_name: str, color: str, fig_path: Path):
    """Create overlay of vent temperature and earthquake counts."""
    deployed = df[df['deployed']].copy()
    temp_daily = deployed['temperature'].resample('D').mean()

    fig, ax1 = plt.subplots(figsize=(6, 3), dpi=300)

    # Temperature on left axis
    ax1.plot(temp_daily.index, temp_daily.values,
             color=color, linewidth=0.5, alpha=0.8, label=f'{vent_name.replace("_", " ").title()} temperature')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Find overlapping time range
    xmin = max(temp_daily.index.min(), eq.index.min())
    xmax = min(temp_daily.index.max(), eq.index.max())
    ax1.set_xlim(xmin, xmax)

    # Earthquake counts on right axis
    ax2 = ax1.twinx()
    ax2.fill_between(eq.index, eq['count'],
                     color='#7A0177', alpha=0.3, label='Earthquakes/day')
    ax2.set_ylabel('Earthquakes per day', color='#7A0177')
    ax2.tick_params(axis='y', labelcolor='#7A0177')
    ax2.set_ylim(bottom=0)

    ax1.set_title(f'{vent_name.replace("_", " ").title()} Vent Temperature and Seismicity')
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


def create_vent_bpr_normalized(df: pd.DataFrame, bpr: pd.DataFrame, vent_name: str, color: str, fig_path: Path):
    """Create normalized overlay of vent temperature and BPR differential uplift."""
    deployed = df[df['deployed']].copy()
    temp_daily = deployed['temperature'].resample('D').mean()

    # Find overlapping time range
    xmin = max(temp_daily.index.min(), bpr.index.min())
    xmax = min(temp_daily.index.max(), bpr.index.max())

    # Subset to common range
    temp_subset = temp_daily.loc[xmin:xmax].dropna()
    bpr_subset = bpr.loc[xmin:xmax, 'differential_m'].dropna()

    # Z-score normalization
    temp_z = (temp_subset - temp_subset.mean()) / temp_subset.std()
    bpr_z = (bpr_subset - bpr_subset.mean()) / bpr_subset.std()

    fig, ax = plt.subplots(figsize=(6, 3), dpi=300)

    ax.plot(temp_z.index, temp_z.values, color=color, linewidth=0.8, alpha=0.8,
            label=f'{vent_name.replace("_", " ").title()} temp')
    ax.plot(bpr_z.index, bpr_z.values, color='#0868AC', linewidth=0.8, alpha=0.8,
            label='Differential uplift')

    ax.set_xlabel('Date')
    ax.set_ylabel('Z-score (σ)')
    ax.set_title(f'{vent_name.replace("_", " ").title()} Temperature vs Deformation (Normalized)')
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


def create_vent_earthquake_normalized(df: pd.DataFrame, eq: pd.DataFrame, vent_name: str, color: str, fig_path: Path):
    """Create normalized overlay of vent temperature and earthquake counts."""
    deployed = df[df['deployed']].copy()
    temp_daily = deployed['temperature'].resample('D').mean()

    # Find overlapping time range
    xmin = max(temp_daily.index.min(), eq.index.min())
    xmax = min(temp_daily.index.max(), eq.index.max())

    # Subset to common range
    temp_subset = temp_daily.loc[xmin:xmax].dropna()
    eq_subset = eq.loc[xmin:xmax, 'count'].dropna()

    # Z-score normalization
    temp_z = (temp_subset - temp_subset.mean()) / temp_subset.std()
    eq_z = (eq_subset - eq_subset.mean()) / eq_subset.std()

    fig, ax = plt.subplots(figsize=(6, 3), dpi=300)

    ax.plot(temp_z.index, temp_z.values, color=color, linewidth=0.8, alpha=0.8,
            label=f'{vent_name.replace("_", " ").title()} temp')
    ax.fill_between(eq_z.index, eq_z.values, alpha=0.3, color='#7A0177',
                    label='Earthquakes/day')

    ax.set_xlabel('Date')
    ax.set_ylabel('Z-score (σ)')
    ax.set_title(f'{vent_name.replace("_", " ").title()} Temperature vs Seismicity (Normalized)')
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


def main():
    # Create output directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load BPR and earthquake data
    bpr = None
    eq = None
    if BPR_PATH.exists():
        print("Loading BPR data...")
        bpr = pd.read_parquet(BPR_PATH)
    if EQ_PATH.exists():
        print("Loading earthquake data...")
        eq = pd.read_parquet(EQ_PATH)

    # Load all vents and create individual figures
    vent_data = {}
    for vent_name in VENTS:
        print(f"\nLoading {vent_name} vent data...")
        df = load_vent_data(vent_name)
        vent_data[vent_name] = df

        deployed = df[df['deployed']]
        print(f"  Records: {len(df):,}")
        print(f"  Deployed: {deployed.index.min()} to {deployed.index.max()}")
        print(f"  Temperature: {deployed['temperature'].min():.1f} to {deployed['temperature'].max():.1f} °C")

        # Save individual vent data
        df.to_parquet(DATA_DIR / f"{vent_name}_temperature.parquet")
        print(f"  Saved: {DATA_DIR / f'{vent_name}_temperature.parquet'}")

        # Daily averages
        daily = deployed[['temperature', 'reference']].resample('D').agg(['mean', 'min', 'max', 'std'])
        daily.columns = ['_'.join(col) for col in daily.columns]
        daily.to_parquet(DATA_DIR / f"{vent_name}_daily.parquet")

        # Individual time series figure
        color = VENTS[vent_name]['color']
        create_temperature_timeseries(df, vent_name, color, FIG_DIR / f"{vent_name}_temperature.png")

        # BPR overlay
        if bpr is not None:
            create_vent_bpr_overlay(df, bpr, vent_name, color, FIG_DIR / f"{vent_name}_bpr_overlay.png")
            create_vent_bpr_normalized(df, bpr, vent_name, color, FIG_DIR / f"{vent_name}_bpr_normalized.png")

        # Earthquake overlay
        if eq is not None:
            create_vent_earthquake_overlay(df, eq, vent_name, color, FIG_DIR / f"{vent_name}_earthquake_overlay.png")
            create_vent_earthquake_normalized(df, eq, vent_name, color, FIG_DIR / f"{vent_name}_earthquake_normalized.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
