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
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch, ConnectionPatch
import matplotlib.patches as mpatches
from datetime import date
from pathlib import Path
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# Paths
BATHY_PATH = Path("/home/jovyan/my_data/axial/axial_bathy/MBARI_AxialSeamount_V2506_AUV_Summit_AUVOverShip_Topo1mSq.grd")
LAVA_FLOW_2011_PATH = Path("/home/jovyan/my_data/axial/axial_bathy/2011_EruptionOutline/Axial-2011-lava-geo-v2.shp")
LAVA_FLOW_2015_PATH = Path("/home/jovyan/my_data/axial/axial_bathy/2015_eruptionOutline/JDF_AxialClague/Axial-2015-lava-geo-v2.shp")
OUTPUT_DIR = Path("/home/jovyan/repos/specKitScience/miso_my-analysis/outputs/figures/poster/miso_maps")

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
        ax.plot([xlim[0] + i * seg_w, xlim[0] + (i + 1) * seg_w],
                [ylim[0], ylim[0]], color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20, solid_capstyle='butt')
        ax.plot([xlim[0] + i * seg_w, xlim[0] + (i + 1) * seg_w],
                [ylim[1], ylim[1]], color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20, solid_capstyle='butt')
        ax.plot([xlim[0], xlim[0]],
                [ylim[0] + i * seg_h, ylim[0] + (i + 1) * seg_h],
                color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20, solid_capstyle='butt')
        ax.plot([xlim[1], xlim[1]],
                [ylim[0] + i * seg_h, ylim[0] + (i + 1) * seg_h],
                color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20, solid_capstyle='butt')


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


def plot_shaded_relief(ax, lon, lat, z, extent=None, add_contours=True,
                       contour_labels=True, utm_crs=None, data_crs=None,
                       contour_interval=50, contour_label_every=2):
    """Plot shaded relief on an axis.

    When utm_crs and data_crs are provided, transforms coordinates to UTM
    for proper projected display. Otherwise works in raw lon/lat.
    """
    ls = LightSource(azdeg=315, altdeg=45)
    z_min, z_max = np.nanpercentile(z, [2, 98])

    rgb = ls.shade(z, cmap=plt.cm.terrain, blend_mode='soft',
                   vmin=z_min, vmax=z_max)

    if utm_crs is not None and data_crs is not None:
        # Transform corner coordinates to UTM for imshow extent
        corner_lons = np.array([lon.min(), lon.max(), lon.max(), lon.min()])
        corner_lats = np.array([lat.min(), lat.min(), lat.max(), lat.max()])
        corners_utm = utm_crs.transform_points(data_crs, corner_lons, corner_lats)
        x_utm_min = corners_utm[:, 0].min()
        x_utm_max = corners_utm[:, 0].max()
        y_utm_min = corners_utm[:, 1].min()
        y_utm_max = corners_utm[:, 1].max()

        ax.imshow(rgb, extent=[x_utm_min, x_utm_max, y_utm_min, y_utm_max],
                  origin='lower', transform=utm_crs)

        if add_contours:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            pts = utm_crs.transform_points(data_crs, lon_grid, lat_grid)
            x_utm_grid = pts[:, :, 0]
            y_utm_grid = pts[:, :, 1]
            contour_levels = np.arange(-2200, -1400, contour_interval)
            cs = ax.contour(x_utm_grid, y_utm_grid, z, levels=contour_levels,
                            colors='black', linewidths=0.3, alpha=0.5,
                            transform=utm_crs)
            if contour_labels:
                ax.clabel(cs, levels=contour_levels[::contour_label_every],
                          fontsize=7, fmt='%d m', inline=True)

        if extent:
            ext_lons = np.array([extent['lon_min'], extent['lon_max']])
            ext_lats = np.array([extent['lat_min'], extent['lat_max']])
            ext_utm = utm_crs.transform_points(data_crs, ext_lons, ext_lats)
            ax.set_xlim(ext_utm[0, 0], ext_utm[1, 0])
            ax.set_ylim(ext_utm[0, 1], ext_utm[1, 1])
    else:
        ax.imshow(rgb, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                  origin='lower', aspect='equal')

        if add_contours:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            contour_levels = np.arange(-2200, -1400, contour_interval)
            cs = ax.contour(lon_grid, lat_grid, z, levels=contour_levels,
                            colors='black', linewidths=0.3, alpha=0.5)
            if contour_labels:
                ax.clabel(cs, levels=contour_levels[::contour_label_every],
                          fontsize=7, fmt='%d m', inline=True)

        if extent:
            ax.set_xlim(extent['lon_min'], extent['lon_max'])
            ax.set_ylim(extent['lat_min'], extent['lat_max'])

    return z_min, z_max


def plot_site_map(lon, lat, z, output_path: Path):
    """Create general site map showing all vent fields including CASM."""

    print("Creating site overview map...")

    # UTM zone 9N for Axial Seamount (~130°W falls in zone 9: 132°W–126°W)
    utm9n = ccrs.UTM(zone=9, southern_hemisphere=False)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_axes([0.08, 0.20, 0.84, 0.72], projection=utm9n)

    # Plot shaded relief with contours
    z_min, z_max = plot_shaded_relief(ax, lon, lat, z, add_contours=True,
                                       contour_labels=True,
                                       utm_crs=utm9n, data_crs=data_crs,
                                       contour_interval=20,
                                       contour_label_every=5)

    # 2011 lava flow - 12 individual features with white/light gray intensity by area
    lava_2011_gdf = gpd.read_file(LAVA_FLOW_2011_PATH)
    areas_2011 = lava_2011_gdf['Area'].values
    min_area_2011, max_area_2011 = areas_2011.min(), areas_2011.max()
    alphas_2011 = 0.25 + 0.35 * (areas_2011 - min_area_2011) / (max_area_2011 - min_area_2011)

    for idx, (geom, alpha) in enumerate(zip(lava_2011_gdf.geometry, alphas_2011)):
        if geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            ax.fill(lons, lats, transform=data_crs,
                   color='white', alpha=alpha, edgecolor='none', linewidth=0, zorder=3)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                coords = list(poly.exterior.coords)
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                ax.fill(lons, lats, transform=data_crs,
                       color='white', alpha=alpha, edgecolor='none', linewidth=0, zorder=3)

    flow_2011_union = lava_2011_gdf.union_all()
    if flow_2011_union.geom_type == 'Polygon':
        outline_coords = list(flow_2011_union.exterior.coords)
        ax.plot([c[0] for c in outline_coords], [c[1] for c in outline_coords],
               transform=data_crs, color='white', linewidth=2.5, linestyle='-', zorder=4)
    elif flow_2011_union.geom_type == 'MultiPolygon':
        for poly in flow_2011_union.geoms:
            outline_coords = list(poly.exterior.coords)
            ax.plot([c[0] for c in outline_coords], [c[1] for c in outline_coords],
                   transform=data_crs, color='white', linewidth=2.5, linestyle='-', zorder=4)

    # 2015 lava flow - 12 individual features with orange intensity by area
    lava_2015_gdf = gpd.read_file(LAVA_FLOW_2015_PATH)
    areas_2015 = lava_2015_gdf['Area'].values
    min_area_2015, max_area_2015 = areas_2015.min(), areas_2015.max()
    alphas_2015 = 0.3 + 0.4 * (areas_2015 - min_area_2015) / (max_area_2015 - min_area_2015)

    for idx, (geom, alpha) in enumerate(zip(lava_2015_gdf.geometry, alphas_2015)):
        if geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            ax.fill(lons, lats, transform=data_crs,
                   color='#D55E00', alpha=alpha, edgecolor='none', linewidth=0, zorder=5)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                coords = list(poly.exterior.coords)
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                ax.fill(lons, lats, transform=data_crs,
                       color='#D55E00', alpha=alpha, edgecolor='none', linewidth=0, zorder=5)

    flow_2015_union = lava_2015_gdf.union_all()
    if flow_2015_union.geom_type == 'Polygon':
        outline_coords = list(flow_2015_union.exterior.coords)
        ax.plot([c[0] for c in outline_coords], [c[1] for c in outline_coords],
               transform=data_crs, color='#D55E00', linewidth=2.5, linestyle='-', zorder=6)
    elif flow_2015_union.geom_type == 'MultiPolygon':
        for poly in flow_2015_union.geoms:
            outline_coords = list(poly.exterior.coords)
            ax.plot([c[0] for c in outline_coords], [c[1] for c in outline_coords],
                   transform=data_crs, color='#D55E00', linewidth=2.5, linestyle='-', zorder=6)

    # All vent fields including CASM
    all_vent_fields = {
        "ASHES": {"lon": -(130 + 0.8203/60), "lat": 45 + 56.0186/60},
        "Coquille": {"lon": -(129 + 59.5793/60), "lat": 45 + 55.0448/60},
        "Int'l District": {"lon": -(129 + 58.7394/60), "lat": 45 + 55.5786/60},
        "Trevi": {"lon": -(129 + 59.023/60), "lat": 45 + 56.777/60},
        "CASM": {"lon": -(130 + 1.632/60), "lat": 45 + 59.332/60},
    }

    # Label offsets for each field (in screen points) — tuned to avoid clipping
    label_offsets = {
        "ASHES": (-70, 10),
        "Coquille": (-80, 15),
        "Int'l District": (-100, -15),
        "Trevi": (20, 15),
        "CASM": (-70, 10),
    }

    # Plot vent field markers and labels with arrow pointers
    for field_name, field_info in all_vent_fields.items():
        ax.plot(field_info['lon'], field_info['lat'], 'o', markersize=8,
                markerfacecolor='white', markeredgecolor='black',
                markeredgewidth=1.5, zorder=10, transform=data_crs)

        offset = label_offsets.get(field_name, (20, 15))
        ax.annotate(field_name, (field_info['lon'], field_info['lat']),
                    xycoords=data_crs._as_mpl_transform(ax),
                    xytext=offset, textcoords='offset points',
                    fontsize=11, fontweight='bold', style='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             alpha=0.9, edgecolor='black', linewidth=1.5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2,
                                   shrinkB=5),
                    zorder=11)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.terrain,
                                norm=plt.Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Depth (m)', fontsize=11)

    # Scale bar (2 km) — neatline style: alternating black/white first km, solid second km
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    vis_w = xlim[1] - xlim[0]
    vis_h = ylim[1] - ylim[0]
    scale_x = xlim[0] + vis_w * 0.05
    scale_y = ylim[0] + vis_h * 0.05
    bar_lw = 6
    # First km: alternating black/white segments (10 x 100m)
    for i in range(10):
        seg_start = scale_x + i * 100
        color = 'black' if i % 2 == 0 else 'white'
        ax.plot([seg_start, seg_start + 100], [scale_y, scale_y],
                color=color, linewidth=bar_lw, solid_capstyle='butt',
                transform=utm9n, zorder=15)
    # Black outline around first km for white segments to read against map
    ax.plot([scale_x, scale_x + 1000], [scale_y, scale_y],
            color='black', linewidth=bar_lw + 2, solid_capstyle='butt',
            transform=utm9n, zorder=14)
    # Second km: single solid black bar
    ax.plot([scale_x + 1000, scale_x + 2000], [scale_y, scale_y],
            'k-', linewidth=bar_lw, solid_capstyle='butt',
            transform=utm9n, zorder=15)
    # Labels
    ax.text(scale_x, scale_y - vis_h * 0.015, '0',
            ha='center', fontsize=9, fontweight='bold', transform=utm9n)
    ax.text(scale_x + 1000, scale_y - vis_h * 0.015, '1',
            ha='center', fontsize=9, fontweight='bold', transform=utm9n)
    ax.text(scale_x + 2000, scale_y - vis_h * 0.015, '2 km',
            ha='center', fontsize=9, fontweight='bold', transform=utm9n)

    ax.set_title('Axial Seamount\nVent Fields', fontsize=36, fontweight='bold', pad=20)

    # Lat/lon gridlines with labels (replaces ax.grid + FuncFormatter)
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=0.5, color='white', alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 9, 'rotation': 0}
    gl.ylabel_style = {'size': 9}
    gl.xpadding = 12
    gl.ypadding = 12

    # North arrow (upper right) — positioned in UTM coordinates
    arrow_x = xlim[1] - vis_w * 0.08
    arrow_y = ylim[1] - vis_h * 0.04
    arrow_len = vis_h * 0.07
    ax.annotate('N', xy=(arrow_x, arrow_y),
                xytext=(arrow_x, arrow_y - arrow_len),
                fontsize=12, fontweight='bold', ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                transform=utm9n, zorder=15,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # Date/projection stamp at bottom-right of map area
    stamp_text = f"Map updated: {date.today().strftime('%Y-%m-%d')}, WGS84, UTM 9N"
    ax.text(xlim[1] - vis_w * 0.02, ylim[0] + vis_h * 0.02, stamp_text,
            ha='right', va='bottom', fontsize=9, transform=utm9n, zorder=15,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # Neatline (alternating black/white ladder border)
    draw_neatline(ax, n_segments=16, linewidth=6)

    # Figure caption — left-aligned, poster font size, right edge aligned to colorbar
    import textwrap

    # Get map and colorbar edges in figure fraction to set caption width
    fig.canvas.draw()
    map_left = ax.get_position().x0
    map_right = cbar.ax.get_position().x1

    caption = (
        "Overview map of the Axial Seamount caldera showing all five hydrothermal vent fields: "
        "ASHES, Coquille, International District, Trevi, and CASM. "
        "Bathymetry from 1-meter resolution AUV survey (MBARI, 2025) with 20-meter depth contours. "
        "Markers indicate vent field locations. "
        "The caldera floor ranges from ~1520 m (International District) to ~1580 m (CASM) depth. "
        "Coordinates in WGS84; UTM Zone 9N projection."
    )

    # Calculate wrap width: caption_ax spans 0 to map_right in figure fraction,
    # convert to inches, then estimate characters that fit at the caption font size
    CAPTION_FONTSIZE = 18
    caption_width_in = (map_right - map_left) * fig.get_size_inches()[0]
    char_width_in = CAPTION_FONTSIZE / 72 * 0.50  # approximate char width for sans-serif
    wrap_chars = int(caption_width_in / char_width_in)
    caption_wrapped = textwrap.fill(caption, width=wrap_chars)

    caption_ax = fig.add_axes([map_left, 0.02, map_right - map_left, 0.15])
    caption_ax.axis('off')
    caption_ax.text(0.0, 1.0, caption_wrapped, ha="left", va="top",
                    fontsize=CAPTION_FONTSIZE,
                    transform=caption_ax.transAxes,
                    family='sans-serif')

    output_file = output_path / "map_site_overview.png"
    fig.savefig(output_file, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    return output_file


def plot_ashes_map(lon, lat, z, output_path: Path):
    """Create detailed ASHES vent field map."""

    print("Creating ASHES detail map...")

    # ASHES extent (~150m around cluster center)
    center_lon, center_lat = -130.0136, 45.9335
    half_size = 0.002  # ~150m

    fig, ax = plt.subplots(figsize=(8, 8))

    extent = {
        'lon_min': center_lon - half_size,
        'lon_max': center_lon + half_size,
        'lat_min': center_lat - half_size,
        'lat_max': center_lat + half_size,
    }

    plot_shaded_relief(ax, lon, lat, z, extent=extent, add_contours=False)

    # ASHES vents
    ashes_vents = ["Inferno", "Hell", "Virgin", "Phoenix"]

    # Label positions (in data coordinates, offset from vent)
    label_positions = {
        "Inferno": (0.0007, 0.0007),
        "Hell": (-0.0012, -0.0005),
        "Virgin": (0.0007, -0.0003),
        "Phoenix": (-0.0012, 0.0003),
    }

    for name in ashes_vents:
        if name in VENTS:
            info = VENTS[name]
            color = VENT_TYPE_COLORS[info['type']]

            ax.plot(info['lon'], info['lat'], 'o', markersize=14,
                   color=color, markeredgecolor='white', markeredgewidth=2, zorder=8)

            dx, dy = label_positions.get(name, (0.0007, 0.0007))
            ax.annotate(
                name, (info['lon'], info['lat']),
                xytext=(info['lon'] + dx, info['lat'] + dy),
                textcoords='data',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         alpha=0.95, edgecolor='gray', linewidth=1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                               connectionstyle='arc3,rad=0.1'),
                zorder=11
            )

    # Scale bar (50m)
    scale_m = 50
    scale_deg = scale_m / 77000
    scale_x = extent['lon_min'] + 0.0003
    scale_y = extent['lat_min'] + 0.0003
    ax.plot([scale_x, scale_x + scale_deg], [scale_y, scale_y], 'k-', linewidth=4)
    ax.text(scale_x + scale_deg/2, scale_y + 0.00025, f'{scale_m} m',
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=VENT_TYPE_COLORS['high-temp'],
               markersize=12, markeredgecolor='white', markeredgewidth=1.5,
               label='High-temp (>200°C)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=VENT_TYPE_COLORS['intermittent'],
               markersize=12, markeredgecolor='white', markeredgewidth=1.5,
               label='Intermittent'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=VENT_TYPE_COLORS['low-temp'],
               markersize=12, markeredgecolor='white', markeredgewidth=1.5,
               label='Low-temp (<100°C)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)

    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('ASHES Vent Field\nMISO Sensor Locations', fontsize=14, fontweight='bold')

    # Format axis labels without scientific notation
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    plt.tight_layout()

    output_file = output_path / "map_ashes_detail.png"
    plt.savefig(output_file, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    return output_file


def plot_intl_district_map(lon, lat, z, output_path: Path):
    """Create detailed International District vent field map."""

    print("Creating Int'l District detail map...")

    # UTM zone 10N for Axial Seamount (~130°W, ~46°N)
    utm10n = ccrs.UTM(zone=10, southern_hemisphere=False)
    data_crs = ccrs.PlateCarree()

    # Int'l District extent (~150m around cluster center)
    center_lon, center_lat = -129.9795, 45.9263
    half_size = 0.002  # ~150m

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_axes([0.08, 0.20, 0.84, 0.72], projection=utm10n)

    extent = {
        'lon_min': center_lon - half_size,
        'lon_max': center_lon + half_size,
        'lat_min': center_lat - half_size,
        'lat_max': center_lat + half_size,
    }

    plot_shaded_relief(ax, lon, lat, z, extent=extent, add_contours=False,
                       utm_crs=utm10n, data_crs=data_crs)

    # Int'l District vents
    intl_vents = ["El Guapo", "Tiny Tower", "Castle"]

    # Label offsets in screen points
    label_offsets = {
        "El Guapo": (15, 12),
        "Tiny Tower": (-70, 5),
        "Castle": (15, -15),
    }

    for name in intl_vents:
        if name in VENTS:
            info = VENTS[name]
            color = VENT_TYPE_COLORS[info['type']]

            ax.plot(info['lon'], info['lat'], 'o', markersize=14,
                   color=color, markeredgecolor='white', markeredgewidth=2,
                   zorder=8, transform=data_crs)

            offset = label_offsets.get(name, (12, 5))
            ax.annotate(
                name, (info['lon'], info['lat']),
                xycoords=data_crs._as_mpl_transform(ax),
                xytext=offset, textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         alpha=0.95, edgecolor='gray', linewidth=1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                               connectionstyle='arc3,rad=0.1'),
                zorder=11
            )

    # Scale bar (50m) — true meters via UTM
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    vis_w = xlim[1] - xlim[0]
    vis_h = ylim[1] - ylim[0]
    scale_x = xlim[0] + vis_w * 0.05
    scale_y = ylim[0] + vis_h * 0.05
    ax.plot([scale_x, scale_x + 50], [scale_y, scale_y],
            'k-', linewidth=4, transform=utm10n)
    ax.text(scale_x + 25, scale_y + vis_h * 0.03, '50 m',
            ha='center', fontsize=10, fontweight='bold', transform=utm10n,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=VENT_TYPE_COLORS['high-temp'],
               markersize=12, markeredgecolor='white', markeredgewidth=1.5,
               label='High-temp (>200°C)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=VENT_TYPE_COLORS['intermittent'],
               markersize=12, markeredgecolor='white', markeredgewidth=1.5,
               label='Intermittent'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=VENT_TYPE_COLORS['low-temp'],
               markersize=12, markeredgecolor='white', markeredgewidth=1.5,
               label='Low-temp (<100°C)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)

    ax.set_title("International District Vent Field\nMISO Sensor Locations",
                 fontsize=14, fontweight='bold')

    # Lat/lon gridlines with labels (replaces FuncFormatter + set_xlabel/set_ylabel)
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=0.5, color='white', alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 9, 'rotation': 0}
    gl.ylabel_style = {'size': 9}
    gl.xpadding = 12
    gl.ypadding = 12

    # North arrow (upper right) — positioned in UTM coordinates
    arrow_x = xlim[1] - vis_w * 0.08
    arrow_y = ylim[1] - vis_h * 0.04
    arrow_len = vis_h * 0.07
    ax.annotate('N', xy=(arrow_x, arrow_y),
                xytext=(arrow_x, arrow_y - arrow_len),
                fontsize=12, fontweight='bold', ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                transform=utm10n, zorder=15,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # Neatline (alternating black/white ladder border)
    draw_neatline(ax, n_segments=12, linewidth=6)

    # Figure caption
    import textwrap
    caption = (
        "International District vent field at Axial Seamount. Bathymetry from "
        "1-meter resolution AUV survey (MBARI, 2025). Markers indicate vent "
        "locations colored by temperature classification from the 2024\u20132025 "
        "MISO deployment (see legend). Coordinates in WGS84; UTM Zone 10N projection."
    )
    caption_wrapped = textwrap.fill(caption, width=60)
    caption_ax = fig.add_axes([0.08, 0.02, 0.84, 0.16])
    caption_ax.axis('off')
    caption_ax.text(0.0, 1.0, caption_wrapped, ha="left", va="top",
                    fontsize=18, transform=caption_ax.transAxes,
                    family='sans-serif')

    output_file = output_path / "map_intl_district_detail.png"
    fig.savefig(output_file, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    return output_file


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
    print("MISO Vent Location Maps")
    print("=" * 60)

    # Print vent summary
    print("\nVents to plot:")
    for name, info in VENTS.items():
        print(f"  {name}: {info['field']}, {info['temp']}, {info['type']}")
    print()

    # Map extent centered on caldera — trimmed 2 km inward from original edges
    # At ~46°N: 2 km ≈ 0.026° lon, 0.018° lat
    center_lon = -130.008772
    center_lat = 45.95485
    half_width = 0.038    # was 0.064, minus ~2 km (0.026°)
    half_height = 0.043   # was 0.061, minus ~2 km (0.018°)

    extent = {
        'lon_min': center_lon - half_width,
        'lon_max': center_lon + half_width,
        'lat_min': center_lat - half_height,
        'lat_max': center_lat + half_height,
    }

    # Load bathymetry
    lon, lat, z = load_bathymetry(BATHY_PATH, subsample=1, extent=extent)

    # Toggle which maps to generate (set to True/False as needed)
    # ASHES detail is now produced by make_ashes_map.py at 1cm resolution
    GENERATE = {
        'site_overview': False,
        'intl_district': False,
    }

    print("\n--- Generating standalone maps ---\n")
    if GENERATE['site_overview']:
        plot_site_map(lon, lat, z, OUTPUT_DIR)
    if GENERATE['intl_district']:
        plot_intl_district_map(lon, lat, z, OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
