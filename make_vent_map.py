#!/usr/bin/env python3
"""
make_vent_map.py - Generate map of Axial Seamount caldera with MISO vent locations

Creates a publication-quality bathymetric map showing individual hydrothermal
vent locations where MISO temperature sensors are deployed.

Usage:
    uv run python make_vent_map.py
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from pathlib import Path

# Paths
BATHY_PATH = Path("/home/jovyan/my_data/axial/axial_bathy/MBARI_AxialSeamount_V2506_AUV_Summit_AUVOverShip_Topo1mSq.grd")
OUTPUT_DIR = Path("/home/jovyan/repos/specKitScience/miso_my-analysis/outputs/figures/poster")

# Vent field coordinates (from constitution)
VENT_FIELDS = {
    "ASHES": {"lon": -130.0137, "lat": 45.9336},
    "Int'l District": {"lon": -129.9790, "lat": 45.9263},
    "Coquille": {"lon": -129.9930, "lat": 45.9174},
    "Trevi": {"lon": -129.9838, "lat": 45.9463},
}

# Individual MISO vent locations
# Coordinates are approximate - placed within vent fields with small offsets
# At 46°N: 100m ≈ 0.0009° lat, 100m ≈ 0.0013° lon
VENTS = {
    # ASHES field vents - spread out for label clarity
    "Inferno": {
        "lon": -130.0155, "lat": 45.9350,
        "field": "ASHES", "deployment": "2022-2025",
        "temp": "~285-310°C", "type": "high-temp"
    },
    "Hell (ASHES)": {
        "lon": -130.0150, "lat": 45.9325,
        "field": "ASHES", "deployment": "2024-2025",
        "temp": "~57°C", "type": "low-temp"
    },
    "Virgin": {
        "lon": -130.0165, "lat": 45.9335,
        "field": "ASHES", "deployment": "2024-2025",
        "temp": "~50-290°C", "type": "intermittent"
    },
    "Trevi": {
        "lon": -129.9838, "lat": 45.9463,
        "field": "ASHES", "deployment": "2024-2025",
        "temp": "~135°C", "type": "intermittent"
    },
    "Vixen": {
        "lon": -130.0135, "lat": 45.9355,
        "field": "ASHES", "deployment": "2024-2025",
        "temp": "N/A", "type": "low-temp"
    },
    # CASM vent field (45°59.332'N, 130°1.632'W)
    "CASM": {
        "lon": -(130 + 1.632/60), "lat": 45 + 59.332/60,
        "field": "CASM", "deployment": "historical",
        "temp": "N/A", "type": "high-temp"
    },
    # International District vents - spread out for label clarity
    "Hell (ID)": {
        "lon": -129.9810, "lat": 45.9280,
        "field": "Int'l District", "deployment": "2022-2024",
        "temp": "~305°C", "type": "high-temp"
    },
    "El Guapo": {
        "lon": -129.9800, "lat": 45.9250,
        "field": "Int'l District", "deployment": "2022-2024",
        "temp": "~100-315°C", "type": "high-temp"
    },
    "El Guapo (Top)": {
        "lon": -129.9755, "lat": 45.9265,
        "field": "Int'l District", "deployment": "2024-2025",
        "temp": "~341°C", "type": "high-temp"
    },
}

# Vent type colors
VENT_TYPE_COLORS = {
    "high-temp": "#D55E00",      # Vermillion/red-orange
    "low-temp": "#0072B2",       # Blue
    "intermittent": "#CC79A7",   # Reddish purple
}

# Scale factors
LAT_PER_KM = 1 / 111.0
LON_PER_KM = 1 / 77.0


def load_bathymetry(path: Path, subsample: int = 1, extent: dict = None) -> tuple:
    """Load bathymetry data, optionally subsampling and clipping to extent."""
    print(f"Loading bathymetry from {path.name}...")
    ds = xr.open_dataset(path)

    if extent:
        ds = ds.sel(lon=slice(extent['lon_min'], extent['lon_max']),
                    lat=slice(extent['lat_min'], extent['lat_max']))
        print(f"  Clipped to extent: {extent['lon_min']:.3f} to {extent['lon_max']:.3f}°E, "
              f"{extent['lat_min']:.3f} to {extent['lat_max']:.3f}°N")

    if subsample > 1:
        ds = ds.isel(lon=slice(None, None, subsample), lat=slice(None, None, subsample))
        print(f"  Subsampled by {subsample}x")

    lon = ds.coords['lon'].values
    lat = ds.coords['lat'].values
    z = ds['z'].values

    print(f"  Grid size: {len(lon)} x {len(lat)}")
    print(f"  Depth range: {np.nanmin(z):.0f} to {np.nanmax(z):.0f} m")

    ds.close()
    return lon, lat, z


def plot_vent_map(lon, lat, z, output_path: Path):
    """Create shaded relief map with MISO vent locations."""

    print("Creating shaded relief map...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Create meshgrid for contours
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Create shaded relief
    ls = LightSource(azdeg=315, altdeg=45)
    z_min, z_max = np.nanpercentile(z, [2, 98])

    rgb = ls.shade(z, cmap=plt.cm.terrain, blend_mode='soft',
                   vmin=z_min, vmax=z_max)

    # Plot shaded relief
    ax.imshow(rgb, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
              origin='lower', aspect='equal')

    # Add depth contours
    print("  Adding depth contours...")
    contour_levels = np.arange(-2200, -1400, 50)
    cs = ax.contour(lon_grid, lat_grid, z, levels=contour_levels,
                    colors='black', linewidths=0.5, alpha=0.6)
    ax.clabel(cs, levels=contour_levels[::2], fontsize=7, fmt='%d m', inline=True)

    # Add individual vent locations
    print("  Adding MISO vent locations...")

    # Label offsets to avoid overlap (vent_name: (x_offset, y_offset))
    label_offsets = {
        "Inferno": (-70, 15),
        "Hell (ASHES)": (-95, -5),
        "Virgin": (-60, -30),
        "Trevi": (12, 5),
        "Vixen": (12, 8),
        "CASM": (12, 5),
        "Hell (ID)": (-70, 12),
        "El Guapo": (-75, -20),
        "El Guapo (Top)": (12, 5),
    }

    for name, info in VENTS.items():
        color = VENT_TYPE_COLORS[info['type']]

        # Plot marker
        ax.plot(info['lon'], info['lat'], 'o', markersize=10,
                color=color, markeredgecolor='white', markeredgewidth=1.5, zorder=8)

        # Add label
        offset = label_offsets.get(name, (10, 5))
        ax.annotate(name, (info['lon'], info['lat']),
                    xytext=offset, textcoords='offset points',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                             alpha=0.85, edgecolor='gray', linewidth=0.5),
                    zorder=11)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.terrain,
                                norm=plt.Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Depth (m)', fontsize=11)

    # Labels and title
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('MISO Hydrothermal Vent Locations\nAxial Seamount Caldera', fontsize=14, fontweight='bold')

    # Add scale bar (1 km)
    scale_lon = lon.min() + 0.005
    scale_lat = lat.min() + 0.005
    scale_length = 1.0 * LON_PER_KM
    ax.plot([scale_lon, scale_lon + scale_length], [scale_lat, scale_lat],
            'k-', linewidth=4)
    ax.text(scale_lon + scale_length/2, scale_lat + 0.003, '1 km',
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # Add legend for vent types
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=VENT_TYPE_COLORS['high-temp'],
               markersize=10, markeredgecolor='white', markeredgewidth=1.5,
               label='High-temp (>200°C)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=VENT_TYPE_COLORS['intermittent'],
               markersize=10, markeredgecolor='white', markeredgewidth=1.5,
               label='Intermittent'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=VENT_TYPE_COLORS['low-temp'],
               markersize=10, markeredgecolor='white', markeredgewidth=1.5,
               label='Low-temp (<100°C)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.95)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', color='white')

    plt.tight_layout()

    # Save
    output_file = output_path / "miso_vent_locations_map.png"
    plt.savefig(output_file, dpi=600, facecolor='white', bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")
    return output_file


def main():
    print("=" * 60)
    print("MISO Vent Location Map")
    print("=" * 60)

    # Print vent summary
    print("\nVents to plot:")
    for name, info in VENTS.items():
        print(f"  {name}: {info['field']}, {info['temp']}, {info['type']}")
    print()

    # Map extent centered on caldera (same as botpt project)
    center_lon = -130.008772
    center_lat = 45.95485
    half_width = 0.064
    half_height = 0.061

    extent = {
        'lon_min': center_lon - half_width,
        'lon_max': center_lon + half_width,
        'lat_min': center_lat - half_height,
        'lat_max': center_lat + half_height,
    }

    # Load bathymetry
    lon, lat, z = load_bathymetry(BATHY_PATH, subsample=1, extent=extent)

    # Create map
    output_file = plot_vent_map(lon, lat, z, OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
