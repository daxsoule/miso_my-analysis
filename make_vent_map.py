#!/usr/bin/env python3
"""
make_vent_map.py - Generate map of Axial Seamount caldera with MISO vent locations

Creates a publication-quality bathymetric map with:
- Main panel: caldera overview with vent field labels
- Inset panels: zoomed views of each vent field showing individual vents

Usage:
    uv run python make_vent_map.py
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch, ConnectionPatch
import matplotlib.patches as mpatches
from pathlib import Path

# Paths
BATHY_PATH = Path("/home/jovyan/my_data/axial/axial_bathy/MBARI_AxialSeamount_V2506_AUV_Summit_AUVOverShip_Topo1mSq.grd")
OUTPUT_DIR = Path("/home/jovyan/repos/specKitScience/miso_my-analysis/outputs/figures/poster")

# Vent field coordinates (from user-provided table)
# Format: degrees + minutes/60
VENT_FIELDS = {
    "ASHES": {"lon": -(130 + 0.8203/60), "lat": 45 + 56.0186/60, "depth": 1540},      # 45°56.0186'N, 130°00.8203'W
    "Coquille": {"lon": -(129 + 59.5793/60), "lat": 45 + 55.0448/60, "depth": 1538},  # 45°55.0448'N, 129°59.5793'W
    "Int'l District": {"lon": -(129 + 58.7394/60), "lat": 45 + 55.5786/60, "depth": 1522},  # 45°55.5786'N, 129°58.7394'W
    "Trevi": {"lon": -(129 + 59.023/60), "lat": 45 + 56.777/60, "depth": 1520},       # 45°56.777'N, 129°59.023'W
}

# Individual MISO vent locations (from constitution - precise coordinates)
VENTS = {
    # ASHES field vents
    "Inferno": {
        "lon": -130.013674, "lat": 45.933566,
        "field": "ASHES", "logger": "MISO 2023-005",
        "temp": "~285-310°C", "type": "high-temp"
    },
    "Hell": {
        "lon": -130.013943, "lat": 45.933307,
        "field": "ASHES", "logger": "MISO 2023-002",
        "temp": "~57°C", "type": "low-temp"
    },
    "Virgin": {
        "lon": -130.013237, "lat": 45.933624,
        "field": "ASHES", "logger": "MISO 2023-007",
        "temp": "~50-290°C", "type": "intermittent"
    },
    "Phoenix": {
        "lon": -130.0136515, "lat": 45.93327021,
        "field": "ASHES", "logger": None,
        "temp": "N/A", "type": "high-temp"
    },
    # Coquille field
    "Vixen/Mkr218": {
        "lon": -129.99295, "lat": 45.91733,
        "field": "Coquille", "logger": "MISO 2023-012",
        "temp": "N/A", "type": "low-temp"
    },
    # International District vents
    "El Guapo": {
        "lon": -129.979493, "lat": 45.926486,
        "field": "Int'l District", "logger": "MISO 2023-009",
        "temp": "~341°C", "type": "high-temp"
    },
    "Tiny Tower": {
        "lon": -129.979186, "lat": 45.926314,
        "field": "Int'l District", "logger": "MISO 2017-002",
        "temp": "N/A", "type": "high-temp"
    },
    "Castle": {
        "lon": -129.979996, "lat": 45.926218,
        "field": "Int'l District", "logger": "MISO 103",
        "temp": "N/A", "type": "high-temp"
    },
    # Trevi field
    "Trevi/Mkr156": {
        "lon": -129.983713, "lat": 45.946276,
        "field": "Trevi", "logger": "MISO 2023-010",
        "temp": "~135°C", "type": "intermittent"
    },
    # CASM field
    "T&S/Shepherd": {
        "lon": -130.027294, "lat": 45.989202,
        "field": "CASM", "logger": None,
        "temp": "N/A", "type": "high-temp"
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


def plot_shaded_relief(ax, lon, lat, z, extent=None, add_contours=True):
    """Plot shaded relief on an axis."""
    ls = LightSource(azdeg=315, altdeg=45)
    z_min, z_max = np.nanpercentile(z, [2, 98])

    rgb = ls.shade(z, cmap=plt.cm.terrain, blend_mode='soft',
                   vmin=z_min, vmax=z_max)

    ax.imshow(rgb, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
              origin='lower', aspect='equal')

    if add_contours:
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        contour_levels = np.arange(-2200, -1400, 50)
        cs = ax.contour(lon_grid, lat_grid, z, levels=contour_levels,
                        colors='black', linewidths=0.3, alpha=0.5)

    if extent:
        ax.set_xlim(extent['lon_min'], extent['lon_max'])
        ax.set_ylim(extent['lat_min'], extent['lat_max'])

    return z_min, z_max


def plot_vent_map(lon, lat, z, output_path: Path):
    """Create multi-panel map: main caldera view + inset panels for vent clusters."""

    print("Creating multi-panel vent map...")

    # Define vent field clusters for insets
    # ~100m buffer around vents in each cluster
    CLUSTERS = {
        "ASHES": {
            "vents": ["Inferno", "Hell", "Virgin", "Phoenix"],
            "center_lon": -130.0136, "center_lat": 45.9334,
            "half_size": 0.0012,  # ~100m
            "scale_m": 50,
        },
        "Int'l District": {
            "vents": ["El Guapo", "Tiny Tower", "Castle"],
            "center_lon": -129.9795, "center_lat": 45.9263,
            "half_size": 0.0012,
            "scale_m": 50,
        },
    }

    # Create figure with GridSpec
    fig = plt.figure(figsize=(14, 10))

    # Main map takes left 2/3, insets on right
    gs = fig.add_gridspec(3, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1, 1],
                          wspace=0.15, hspace=0.25)

    # Main caldera map (spans all rows on left)
    ax_main = fig.add_subplot(gs[:, 0])

    # Inset panels on right
    ax_ashes = fig.add_subplot(gs[0, 1:])
    ax_intl = fig.add_subplot(gs[1, 1:])
    ax_legend = fig.add_subplot(gs[2, 1:])
    ax_legend.axis('off')

    # --- Main Map ---
    print("  Plotting main caldera map...")
    z_min, z_max = plot_shaded_relief(ax_main, lon, lat, z, add_contours=True)

    # Plot vent field markers (larger, for overview)
    for field_name, field_info in VENT_FIELDS.items():
        ax_main.plot(field_info['lon'], field_info['lat'], 's', markersize=12,
                     color='yellow', markeredgecolor='black', markeredgewidth=1.5, zorder=8)

        # Label offset based on position
        if field_name == "ASHES":
            offset = (-70, 5)
        elif field_name == "Int'l District":
            offset = (10, -15)
        elif field_name == "Coquille":
            offset = (10, 5)
        elif field_name == "Trevi":
            offset = (10, 5)
        elif field_name == "CASM":
            offset = (10, 5)
        else:
            offset = (10, 5)

        ax_main.annotate(field_name, (field_info['lon'], field_info['lat']),
                         xytext=offset, textcoords='offset points',
                         fontsize=9, fontweight='bold', style='italic',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow',
                                  alpha=0.9, edgecolor='black', linewidth=0.5),
                         zorder=11)

    # Also plot isolated vents (not in tight clusters, with 2024 loggers)
    isolated_vents = ["Vixen/Mkr218", "Trevi/Mkr156"]
    for name in isolated_vents:
        if name in VENTS:
            info = VENTS[name]
            color = VENT_TYPE_COLORS[info['type']]
            ax_main.plot(info['lon'], info['lat'], 'o', markersize=8,
                        color=color, markeredgecolor='white', markeredgewidth=1, zorder=8)

    # Draw rectangles showing inset extents
    for cluster_name, cluster in CLUSTERS.items():
        rect = Rectangle(
            (cluster['center_lon'] - cluster['half_size'],
             cluster['center_lat'] - cluster['half_size']),
            2 * cluster['half_size'], 2 * cluster['half_size'],
            linewidth=2, edgecolor='red', facecolor='none', zorder=15
        )
        ax_main.add_patch(rect)

    # Scale bar (1 km)
    scale_lon = lon.min() + 0.005
    scale_lat = lat.min() + 0.005
    scale_length = 1.0 * LON_PER_KM
    ax_main.plot([scale_lon, scale_lon + scale_length], [scale_lat, scale_lat],
                 'k-', linewidth=4)
    ax_main.text(scale_lon + scale_length/2, scale_lat + 0.003, '1 km',
                 ha='center', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    ax_main.set_xlabel('Longitude', fontsize=10)
    ax_main.set_ylabel('Latitude', fontsize=10)
    ax_main.set_title('Axial Seamount Caldera\nVent Field Locations', fontsize=12, fontweight='bold')

    # --- ASHES Inset ---
    print("  Plotting ASHES inset...")
    cluster = CLUSTERS["ASHES"]
    extent_ashes = {
        'lon_min': cluster['center_lon'] - cluster['half_size'],
        'lon_max': cluster['center_lon'] + cluster['half_size'],
        'lat_min': cluster['center_lat'] - cluster['half_size'],
        'lat_max': cluster['center_lat'] + cluster['half_size'],
    }
    plot_shaded_relief(ax_ashes, lon, lat, z, extent=extent_ashes, add_contours=False)

    # Plot vents with labels and pointers
    for name in cluster['vents']:
        if name in VENTS:
            info = VENTS[name]
            color = VENT_TYPE_COLORS[info['type']]
            ax_ashes.plot(info['lon'], info['lat'], 'o', markersize=12,
                         color=color, markeredgecolor='white', markeredgewidth=2, zorder=8)

            # Add label with arrow pointer
            label_offsets_ashes = {
                "Inferno": (0.0004, 0.0004),
                "Hell": (-0.0006, -0.0004),
                "Virgin": (0.0004, -0.0002),
                "Phoenix": (-0.0006, 0.0002),
            }
            dx, dy = label_offsets_ashes.get(name, (0.0004, 0.0004))
            ax_ashes.annotate(
                name, (info['lon'], info['lat']),
                xytext=(info['lon'] + dx, info['lat'] + dy),
                textcoords='data',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         alpha=0.95, edgecolor='gray', linewidth=0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                zorder=11
            )

    # Scale bar (50m)
    scale_m = cluster['scale_m']
    scale_deg = scale_m / 77000  # approximate lon degrees for 50m
    scale_x = extent_ashes['lon_min'] + 0.0002
    scale_y = extent_ashes['lat_min'] + 0.0002
    ax_ashes.plot([scale_x, scale_x + scale_deg], [scale_y, scale_y], 'k-', linewidth=3)
    ax_ashes.text(scale_x + scale_deg/2, scale_y + 0.00015, f'{scale_m} m',
                  ha='center', fontsize=8, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9))

    ax_ashes.set_title('ASHES Vent Field', fontsize=11, fontweight='bold', color='red')
    ax_ashes.set_xticks([])
    ax_ashes.set_yticks([])
    for spine in ax_ashes.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2)

    # --- International District Inset ---
    print("  Plotting Int'l District inset...")
    cluster = CLUSTERS["Int'l District"]
    extent_intl = {
        'lon_min': cluster['center_lon'] - cluster['half_size'],
        'lon_max': cluster['center_lon'] + cluster['half_size'],
        'lat_min': cluster['center_lat'] - cluster['half_size'],
        'lat_max': cluster['center_lat'] + cluster['half_size'],
    }
    plot_shaded_relief(ax_intl, lon, lat, z, extent=extent_intl, add_contours=False)

    for name in cluster['vents']:
        if name in VENTS:
            info = VENTS[name]
            color = VENT_TYPE_COLORS[info['type']]
            ax_intl.plot(info['lon'], info['lat'], 'o', markersize=12,
                        color=color, markeredgecolor='white', markeredgewidth=2, zorder=8)

            label_offsets_intl = {
                "El Guapo": (0.0005, 0.0004),
                "Tiny Tower": (-0.0007, 0.0001),
                "Castle": (0.0004, -0.0004),
            }
            dx, dy = label_offsets_intl.get(name, (0.0004, 0.0004))
            ax_intl.annotate(
                name, (info['lon'], info['lat']),
                xytext=(info['lon'] + dx, info['lat'] + dy),
                textcoords='data',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         alpha=0.95, edgecolor='gray', linewidth=0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                zorder=11
            )

    # Scale bar
    scale_x = extent_intl['lon_min'] + 0.0002
    scale_y = extent_intl['lat_min'] + 0.0002
    ax_intl.plot([scale_x, scale_x + scale_deg], [scale_y, scale_y], 'k-', linewidth=3)
    ax_intl.text(scale_x + scale_deg/2, scale_y + 0.00015, f'{scale_m} m',
                 ha='center', fontsize=8, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9))

    ax_intl.set_title("Int'l District Vent Field", fontsize=11, fontweight='bold', color='red')
    ax_intl.set_xticks([])
    ax_intl.set_yticks([])
    for spine in ax_intl.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2)

    # --- Legend Panel ---
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow',
               markersize=12, markeredgecolor='black', markeredgewidth=1.5,
               label='Vent Field'),
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
        Line2D([0], [0], linestyle='', marker='s', color='w',
               markerfacecolor='none', markeredgecolor='red', markeredgewidth=2,
               markersize=12, label='Inset extent'),
    ]
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=10,
                     framealpha=0.95, ncol=1)
    ax_legend.set_title('Legend', fontsize=11, fontweight='bold')

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
