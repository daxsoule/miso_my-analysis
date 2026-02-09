#!/usr/bin/env python3
"""
make_intl_district_map.py - Generate ultra-high-resolution International District vent field map

Creates a detailed bathymetric map of the International District vent field
using 1cm LASS lidar data.

Usage:
    uv run python make_intl_district_map.py
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from datetime import date
from pathlib import Path
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Paths
BATHY_PATH = Path("/home/jovyan/my_data/axial/axial_bathy/MBARI_AxialSeamount_V2506_LASSlidar_IntlDist_Topo1cmSq.grd")
OUTPUT_DIR = Path(__file__).parent / "outputs" / "figures" / "poster" / "miso_maps"

# International District vent locations (picked from 1cm bathymetry with UTM 9N projection)
VENTS = {
    "El Guapo": {"lon": -129.979585, "lat": 45.926543, "type": "high-temp"},
    "Escargot": {"lon": -129.979223, "lat": 45.926365, "type": "high-temp"},
    "Castle": {"lon": -129.980102, "lat": 45.926212, "type": "high-temp"},
    "Diva": {"lon": -129.979105, "lat": 45.926377, "type": "high-temp"},
    "Flat Top": {"lon": -129.979836, "lat": 45.926141, "type": "high-temp"},
}

# Temperature classification colors (consistent with make_vent_map.py)
VENT_TYPE_COLORS = {
    "high-temp": "#D55E00",    # Vermillion
    "intermittent": "#CC79A7", # Reddish purple
    "low-temp": "#0072B2",     # Blue
    "no-data": "#999999",      # Gray
}

CAPTION_FONTSIZE = 18  # Match survey.py poster caption size


def draw_neatline(ax, n_segments=12, linewidth=6):
    """Draw an alternating black/white ladder border (neatline) around the axes."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]

    seg_w = width / n_segments
    seg_h = height / n_segments

    # Draw solid black border first (wider) as the base
    for edge in ['bottom', 'top', 'left', 'right']:
        if edge == 'bottom':
            xs, ys = [xlim[0], xlim[1]], [ylim[0], ylim[0]]
        elif edge == 'top':
            xs, ys = [xlim[0], xlim[1]], [ylim[1], ylim[1]]
        elif edge == 'left':
            xs, ys = [xlim[0], xlim[0]], [ylim[0], ylim[1]]
        else:
            xs, ys = [xlim[1], xlim[1]], [ylim[0], ylim[1]]
        ax.plot(xs, ys, color='black', linewidth=linewidth + 2,
                transform=ax.transData, clip_on=False, zorder=19,
                solid_capstyle='butt')

    # Overlay alternating black/white segments
    for i in range(n_segments):
        color = 'black' if i % 2 == 0 else 'white'
        # Bottom edge
        ax.plot([xlim[0] + i * seg_w, xlim[0] + (i + 1) * seg_w],
                [ylim[0], ylim[0]], color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20, solid_capstyle='butt')
        # Top edge
        ax.plot([xlim[0] + i * seg_w, xlim[0] + (i + 1) * seg_w],
                [ylim[1], ylim[1]], color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20, solid_capstyle='butt')
        # Left edge
        ax.plot([xlim[0], xlim[0]],
                [ylim[0] + i * seg_h, ylim[0] + (i + 1) * seg_h],
                color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20, solid_capstyle='butt')
        # Right edge
        ax.plot([xlim[1], xlim[1]],
                [ylim[0] + i * seg_h, ylim[0] + (i + 1) * seg_h],
                color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20, solid_capstyle='butt')


def load_bathymetry(path: Path, subsample: int = 1) -> tuple:
    """Load 1cm bathymetry data, optionally subsampling."""
    print(f"Loading bathymetry from {path.name}...")
    ds = xr.open_dataset(path)

    x = ds.coords['x'].values
    y = ds.coords['y'].values
    z = ds['z'].values

    if subsample > 1:
        x = x[::subsample]
        y = y[::subsample]
        z = z[::subsample, ::subsample]
        print(f"  Subsampled by {subsample}x")

    print(f"  Grid size: {len(x)} x {len(y)}")
    print(f"  Depth range: {np.nanmin(z):.1f} to {np.nanmax(z):.1f} m")

    ds.close()
    return x, y, z


def plot_intl_district_map(x, y, z, output_path: Path, subsample: int = 1):
    """Create shaded relief map of International District vent field with vent annotations."""

    print("Creating shaded relief map...")

    # UTM zone 9N for Axial Seamount (~130°W falls in zone 9: 132°W-126°W)
    utm9n = ccrs.UTM(zone=9, southern_hemisphere=False)
    data_crs = ccrs.PlateCarree()

    # Set up figure with UTM projection and space for caption below
    fig = plt.figure(figsize=(12, 14))
    ax = fig.add_axes([0.02, 0.22, 0.82, 0.72], projection=utm9n)

    # Create meshgrid for plotting
    x_grid, y_grid = np.meshgrid(x, y)

    # Create shaded relief with strong illumination for micro-topography
    ls = LightSource(azdeg=315, altdeg=35)
    z_min, z_max = np.nanpercentile(z, [1, 99])

    # Use hillshade blended with terrain colormap
    rgb = ls.shade(z, cmap=plt.cm.terrain, blend_mode='soft',
                   vmin=z_min, vmax=z_max, vert_exag=2)

    # Transform all four corners to UTM for correct extent
    corner_lons = np.array([x.min(), x.max(), x.max(), x.min()])
    corner_lats = np.array([y.min(), y.min(), y.max(), y.max()])
    corners_utm = utm9n.transform_points(data_crs, corner_lons, corner_lats)
    x_utm_min = corners_utm[:, 0].min()
    x_utm_max = corners_utm[:, 0].max()
    y_utm_min = corners_utm[:, 1].min()
    y_utm_max = corners_utm[:, 1].max()

    # Plot shaded relief in UTM coordinates
    ax.imshow(rgb, extent=[x_utm_min, x_utm_max, y_utm_min, y_utm_max],
              origin='lower', transform=utm9n)

    # Crop axes -- pull border in 20 meters on all sides
    margin = 20  # meters (UTM)
    vis_x_min = x_utm_min + margin
    vis_x_max = x_utm_max - margin
    vis_y_min = y_utm_min + margin
    vis_y_max = y_utm_max - margin
    ax.set_xlim(vis_x_min, vis_x_max)
    ax.set_ylim(vis_y_min, vis_y_max)

    # Transform grid to UTM for contours
    print("  Adding depth contours...")
    pts = utm9n.transform_points(data_crs, x_grid, y_grid)
    x_utm_grid = pts[:, :, 0]
    y_utm_grid = pts[:, :, 1]

    # Contour lines only (no labels -- depth info is in the colorbar)
    contour_levels = np.arange(-1530, -1498, 1)
    ax.contour(x_utm_grid, y_utm_grid, z, levels=contour_levels,
               colors='black', linewidths=0.3, alpha=0.4,
               transform=utm9n)

    # Add vent markers with labels
    print("  Adding vent locations...")
    for name, info in VENTS.items():
        color = VENT_TYPE_COLORS[info['type']]

        # Check if vent is within grid bounds (in lon/lat)
        if (x.min() <= info['lon'] <= x.max() and
            y.min() <= info['lat'] <= y.max()):

            # Plot vent marker with black outline for visibility
            ax.plot(info['lon'], info['lat'], 'o', markersize=2.5,
                    markerfacecolor=color, markeredgecolor='black',
                    markeredgewidth=1.5, zorder=10, transform=data_crs)

            # Position labels to avoid overlap
            label_offsets = {
                "El Guapo": (15, 12),
                "Escargot": (-85, -18),
                "Castle": (-75, 10),
                "Diva": (15, -18),
                "Flat Top": (-80, -18),
            }
            offset = label_offsets.get(name, (12, 5))

            ax.annotate(name, (info['lon'], info['lat']),
                        xycoords=data_crs._as_mpl_transform(ax),
                        xytext=offset, textcoords='offset points',
                        fontsize=11, fontweight='bold', color=color,
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.2, shrinkB=5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 alpha=0.9, edgecolor=color, linewidth=1.5),
                        zorder=12)
        else:
            print(f"    Warning: {name} is outside grid bounds")

    # Add colorbar with enough padding
    sm = plt.cm.ScalarMappable(cmap=plt.cm.terrain,
                                norm=plt.Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.04)
    cbar.set_label('Depth (m)', fontsize=11)

    # Title
    ax.set_title('International District\nVent Field',
                 fontsize=24, fontweight='bold', pad=20)

    # Lat/lon gridlines with labels
    gl = ax.gridlines(
        crs=data_crs,
        draw_labels=True,
        linewidth=0.5,
        color='white',
        alpha=0.3,
        linestyle='--',
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 9, 'rotation': 0}
    gl.ylabel_style = {'size': 9}
    gl.xpadding = 12
    gl.ypadding = 12
    gl.xlocator = mticker.FixedLocator(
        np.arange(-129.982, -129.977, 0.0005)
    )
    gl.ylocator = mticker.FixedLocator(
        np.arange(45.924, 45.929, 0.0005)
    )

    # Add scale bar (10 m) -- true meters via UTM
    vis_w = vis_x_max - vis_x_min
    vis_h = vis_y_max - vis_y_min
    scale_x = vis_x_min + vis_w * 0.05
    scale_y = vis_y_min + vis_h * 0.05
    ax.plot([scale_x, scale_x + 10], [scale_y, scale_y],
            'k-', linewidth=4, transform=utm9n)
    ax.text(scale_x + 5, scale_y + vis_h * 0.02, '10 m',
            ha='center', fontsize=10, fontweight='bold', transform=utm9n,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # North arrow (upper right, offset from colorbar)
    arrow_x = vis_x_max - vis_w * 0.15
    arrow_y = vis_y_max - vis_h * 0.04
    arrow_len = vis_h * 0.07
    ax.annotate('N', xy=(arrow_x, arrow_y),
                xytext=(arrow_x, arrow_y - arrow_len),
                fontsize=12, fontweight='bold', ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                transform=utm9n, zorder=15,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # Temperature classification legend
    legend_elements = [
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=VENT_TYPE_COLORS['high-temp'],
               markersize=12, markeredgecolor='black', markeredgewidth=1,
               label='High-temp (>200\u00b0C)'),
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=VENT_TYPE_COLORS['intermittent'],
               markersize=12, markeredgecolor='black', markeredgewidth=1,
               label='Intermittent'),
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=VENT_TYPE_COLORS['low-temp'],
               markersize=12, markeredgecolor='black', markeredgewidth=1,
               label='Low-temp (<100\u00b0C)'),
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=VENT_TYPE_COLORS['no-data'],
               markersize=12, markeredgecolor='black', markeredgewidth=1,
               label='No data'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.95, edgecolor='black')

    # Neatline (alternating black/white ladder border)
    draw_neatline(ax, n_segments=16, linewidth=8)

    # Date/projection stamp at bottom of map area
    stamp_text = f"Map updated: {date.today().strftime('%Y-%m-%d')}, WGS84, UTM 9N"
    ax.text(vis_x_max - vis_w * 0.02, vis_y_min + vis_h * 0.02, stamp_text,
            ha='right', va='bottom', fontsize=9, transform=utm9n, zorder=15,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # Figure caption -- left-aligned, poster font size, right edge aligned to map area
    import textwrap

    # Get colorbar right edge in figure fraction to set caption width
    fig.canvas.draw()
    map_right = cbar.ax.get_position().x1

    caption = (
        "International District vent field at Axial Seamount, mapped with 1 cm resolution "
        "LASS lidar bathymetry (MBARI, 2025). Shaded relief with 1-meter depth contours. "
        "Markers indicate vent locations colored by temperature classification from the "
        "2024\u20132025 MISO deployment (see legend)."
    )

    # Calculate wrap width: caption_ax spans 0 to map_right in figure fraction,
    # convert to inches, then estimate characters that fit at the caption font size
    caption_width_in = map_right * fig.get_size_inches()[0]
    char_width_in = CAPTION_FONTSIZE / 72 * 0.50  # approximate char width for sans-serif
    wrap_chars = int(caption_width_in / char_width_in)
    caption_wrapped = textwrap.fill(caption, width=wrap_chars)

    caption_ax = fig.add_axes([0.0, 0.18, map_right, 0.10])
    caption_ax.axis('off')
    caption_ax.text(0.0, 1.0, caption_wrapped, ha="left", va="top",
                    fontsize=CAPTION_FONTSIZE,
                    transform=caption_ax.transAxes,
                    family='sans-serif')

    # Save
    resolution_note = f"_{subsample}x" if subsample > 1 else ""
    output_file = output_path / f"intl_district_detail_map{resolution_note}.png"
    fig.savefig(output_file, dpi=600, facecolor='white', bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")
    return output_file


def main():
    print("=" * 60)
    print("International District Vent Field Detail Map")
    print("=" * 60)
    print("\n1cm resolution LASS lidar bathymetry")
    print("Vent coordinates from MISO project constitution\n")

    # Load full resolution (this is a large dataset)
    # For initial testing, subsample; for final output, use full res
    subsample = 1  # Full resolution

    x, y, z = load_bathymetry(BATHY_PATH, subsample=subsample)
    output_file = plot_intl_district_map(x, y, z, OUTPUT_DIR, subsample=subsample)

    print("\nDone!")


if __name__ == "__main__":
    main()
