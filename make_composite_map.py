#!/usr/bin/env python3
"""
make_composite_map.py - Three-panel composite poster figure

Layout:
  +------------------------------------+----+
  | (a) Site     | (b) ASHES           | CB |
  | Overview     |  Detail             |    |
  |              |----------------------|    |
  |  (portrait)  | (c) Int'l           |    |
  |              |  District           |    |
  |              |  Detail             |    |
  +------------------------------------+----+
  | Three-part caption                      |
  +------------------------------------------+

Usage:
    uv run python make_composite_map.py
"""

import gc
import textwrap
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from datetime import date
from pathlib import Path
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# === Paths ===
BATHY_AUV_PATH = Path(
    "/home/jovyan/my_data/axial/axial_bathy/"
    "MBARI_AxialSeamount_V2506_AUV_Summit_AUVOverShip_Topo1mSq.grd"
)
BATHY_ASHES_PATH = Path(
    "/home/jovyan/my_data/axial/axial_bathy/"
    "MBARI_AxialSeamount_V2506_LASSlidar_Ashes_Topo1cmSq.grd"
)
BATHY_INTL_PATH = Path(
    "/home/jovyan/my_data/axial/axial_bathy/"
    "MBARI_AxialSeamount_V2506_LASSlidar_IntlDist_Topo1cmSq.grd"
)
LAVA_FLOW_2011_PATH = Path(
    "/home/jovyan/my_data/axial/axial_bathy/"
    "2011_EruptionOutline/Axial-2011-lava-geo-v2.shp"
)
LAVA_FLOW_2015_PATH = Path(
    "/home/jovyan/my_data/axial/axial_bathy/"
    "2015_eruptionOutline/JDF_AxialClague/Axial-2015-lava-geo-v2.shp"
)
OUTPUT_DIR = Path(__file__).parent / "outputs" / "figures" / "poster" / "miso_maps"

# === Vent field centers (site overview) ===
VENT_FIELDS = {
    "ASHES": {"lon": -(130 + 0.8203 / 60), "lat": 45 + 56.0186 / 60},
    "Coquille": {"lon": -(129 + 59.5793 / 60), "lat": 45 + 55.0448 / 60},
    "Int'l District": {"lon": -(129 + 58.7394 / 60), "lat": 45 + 55.5786 / 60},
    "Trevi": {"lon": -(129 + 59.023 / 60), "lat": 45 + 56.777 / 60},
    "CASM": {"lon": -(130 + 1.632 / 60), "lat": 45 + 59.332 / 60},
}

# === ASHES vents (1cm-picked coordinates) ===
ASHES_VENTS = {
    "Inferno": {"lon": -130.013865, "lat": 45.933519, "type": "high-temp"},
    "Hell": {"lon": -130.014140, "lat": 45.933272, "type": "low-temp"},
    "Virgin": {"lon": -130.013447, "lat": 45.933615, "type": "intermittent"},
    "Phoenix": {"lon": -130.013852, "lat": 45.933217, "type": "no-data"},
    "Mushroom": {"lon": -130.013757, "lat": 45.933563, "type": "no-data"},
}

# === Int'l District vents (1cm-picked coordinates) ===
INTL_VENTS = {
    "El Guapo": {"lon": -129.979585, "lat": 45.926543, "type": "high-temp"},
    "Escargot": {"lon": -129.979223, "lat": 45.926365, "type": "high-temp"},
    "Castle": {"lon": -129.980102, "lat": 45.926212, "type": "high-temp"},
    "Diva": {"lon": -129.979105, "lat": 45.926377, "type": "high-temp"},
    "Flat Top": {"lon": -129.979836, "lat": 45.926141, "type": "high-temp"},
}

# === Temperature classification colors ===
VENT_TYPE_COLORS = {
    "high-temp": "#D55E00",
    "intermittent": "#CC79A7",
    "low-temp": "#0072B2",
    "no-data": "#999999",
}

# === Font sizes (composite targets) ===
FS_PANEL_LABEL = 24   # Rubric: Title >= 24pt
FS_VENT_LABEL = 11    # Rubric: Feature labels >= 11pt
FS_LEGEND = 11        # Rubric: Feature labels >= 11pt
FS_SCALE_BAR = 11     # Rubric: Feature labels >= 11pt
FS_GRIDLINE = 11      # Rubric: Feature labels >= 11pt
FS_NORTH_ARROW = 11   # Rubric: Feature labels >= 11pt
FS_COLORBAR = 18      # Rubric: Axis/Caption >= 18pt
FS_CAPTION = 18       # Rubric: Axis/Caption >= 18pt
FS_DATE_STAMP = 11    # Rubric: Feature labels >= 11pt


# ─── Shared helpers ───────────────────────────────────────────────────────────

def draw_neatline(ax, n_segments=12, linewidth=5):
    """Draw an alternating black/white ladder border (neatline) around the axes."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]

    seg_w = width / n_segments
    seg_h = height / n_segments

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

    for i in range(n_segments):
        color = 'black' if i % 2 == 0 else 'white'
        ax.plot([xlim[0] + i * seg_w, xlim[0] + (i + 1) * seg_w],
                [ylim[0], ylim[0]], color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20,
                solid_capstyle='butt')
        ax.plot([xlim[0] + i * seg_w, xlim[0] + (i + 1) * seg_w],
                [ylim[1], ylim[1]], color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20,
                solid_capstyle='butt')
        ax.plot([xlim[0], xlim[0]],
                [ylim[0] + i * seg_h, ylim[0] + (i + 1) * seg_h],
                color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20,
                solid_capstyle='butt')
        ax.plot([xlim[1], xlim[1]],
                [ylim[0] + i * seg_h, ylim[0] + (i + 1) * seg_h],
                color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20,
                solid_capstyle='butt')


def add_date_stamp(ax, transform, vis_x_max, vis_y_min, vis_w, vis_h):
    """Add a date/projection stamp at the bottom-right of a map panel."""
    stamp = f"Map updated: {date.today().strftime('%Y-%m-%d')}, WGS84, UTM 9N"
    ax.text(vis_x_max - vis_w * 0.02, vis_y_min + vis_h * 0.02, stamp,
            ha='right', va='bottom', fontsize=FS_DATE_STAMP,
            transform=transform, zorder=15,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))


def add_panel_label(ax, text):
    """Add a combined panel label + title inside the upper-left of the axes."""
    ax.text(0.02, 0.97, text, transform=ax.transAxes,
            fontsize=FS_PANEL_LABEL, fontweight='bold', va='top', ha='left',
            zorder=15,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     alpha=0.92, edgecolor='black', linewidth=1))


# ─── Panel (a): Site Overview ─────────────────────────────────────────────────

def render_site_overview(fig, ax):
    """Render the caldera overview panel. Returns (z_min, z_max) for the shared colorbar."""
    print("Panel (a): Loading AUV 1m bathymetry...")
    ds = xr.open_dataset(BATHY_AUV_PATH)

    # Clip to caldera extent
    center_lon = -130.008772
    center_lat = 45.95485
    half_width = 0.038
    half_height = 0.043
    extent = {
        'lon_min': center_lon - half_width,
        'lon_max': center_lon + half_width,
        'lat_min': center_lat - half_height,
        'lat_max': center_lat + half_height,
    }
    ds = ds.sel(lon=slice(extent['lon_min'], extent['lon_max']),
                lat=slice(extent['lat_min'], extent['lat_max']))

    lon = ds.coords['lon'].values
    lat = ds.coords['lat'].values
    z = ds['z'].values
    ds.close()
    print(f"  Grid: {len(lon)} x {len(lat)}, depth {np.nanmin(z):.0f} to {np.nanmax(z):.0f} m")

    utm9n = ccrs.UTM(zone=9, southern_hemisphere=False)
    data_crs = ccrs.PlateCarree()

    # Shaded relief
    ls = LightSource(azdeg=315, altdeg=45)
    z_min, z_max = np.nanpercentile(z, [2, 98])
    rgb = ls.shade(z, cmap=plt.cm.terrain, blend_mode='soft',
                   vmin=z_min, vmax=z_max)

    # Transform to UTM
    corner_lons = np.array([lon.min(), lon.max(), lon.max(), lon.min()])
    corner_lats = np.array([lat.min(), lat.min(), lat.max(), lat.max()])
    corners_utm = utm9n.transform_points(data_crs, corner_lons, corner_lats)
    x_utm_min = corners_utm[:, 0].min()
    x_utm_max = corners_utm[:, 0].max()
    y_utm_min = corners_utm[:, 1].min()
    y_utm_max = corners_utm[:, 1].max()

    ax.imshow(rgb, extent=[x_utm_min, x_utm_max, y_utm_min, y_utm_max],
              origin='lower', transform=utm9n)

    # Contours (20m interval, label every 5th)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    pts = utm9n.transform_points(data_crs, lon_grid, lat_grid)
    x_utm_grid = pts[:, :, 0]
    y_utm_grid = pts[:, :, 1]
    contour_levels = np.arange(-2200, -1400, 20)
    cs = ax.contour(x_utm_grid, y_utm_grid, z, levels=contour_levels,
                    colors='black', linewidths=0.3, alpha=0.5, transform=utm9n)
    ax.clabel(cs, levels=contour_levels[::5], fontsize=7, fmt='%d m', inline=True)

    # Set extent in UTM
    ext_lons = np.array([extent['lon_min'], extent['lon_max']])
    ext_lats = np.array([extent['lat_min'], extent['lat_max']])
    ext_utm = utm9n.transform_points(data_crs, ext_lons, ext_lats)
    ax.set_xlim(ext_utm[0, 0], ext_utm[1, 0])
    ax.set_ylim(ext_utm[0, 1], ext_utm[1, 1])

    # 2011 lava flow - 12 individual features with white/light gray intensity by area
    lava_2011_gdf = gpd.read_file(LAVA_FLOW_2011_PATH)

    # Normalize areas for color intensity (0.25 to 0.6 alpha range)
    areas_2011 = lava_2011_gdf['Area'].values
    min_area_2011, max_area_2011 = areas_2011.min(), areas_2011.max()
    alphas_2011 = 0.25 + 0.35 * (areas_2011 - min_area_2011) / (max_area_2011 - min_area_2011)

    # Plot each 2011 flow feature with white/light color
    for idx, (geom, alpha) in enumerate(zip(lava_2011_gdf.geometry, alphas_2011)):
        if geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            ax.fill(lons, lats, transform=data_crs,
                   color='white', alpha=alpha, edgecolor='none',
                   linewidth=0, zorder=3)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                coords = list(poly.exterior.coords)
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                ax.fill(lons, lats, transform=data_crs,
                       color='white', alpha=alpha, edgecolor='none',
                       linewidth=0, zorder=3)

    # Add solid outline around entire 2011 flow field
    flow_2011_union = lava_2011_gdf.union_all()
    if flow_2011_union.geom_type == 'Polygon':
        outline_2011_coords = list(flow_2011_union.exterior.coords)
        outline_2011_lons = [c[0] for c in outline_2011_coords]
        outline_2011_lats = [c[1] for c in outline_2011_coords]
        ax.plot(outline_2011_lons, outline_2011_lats, transform=data_crs,
               color='white', linewidth=2.5, linestyle='-',
               zorder=4, label='2011 Lava Flow')
    elif flow_2011_union.geom_type == 'MultiPolygon':
        for poly in flow_2011_union.geoms:
            outline_2011_coords = list(poly.exterior.coords)
            outline_2011_lons = [c[0] for c in outline_2011_coords]
            outline_2011_lats = [c[1] for c in outline_2011_coords]
            ax.plot(outline_2011_lons, outline_2011_lats, transform=data_crs,
                   color='white', linewidth=2.5, linestyle='-', zorder=4)

    # 2015 lava flow - 12 individual features with color intensity by area
    lava_gdf = gpd.read_file(LAVA_FLOW_2015_PATH)

    # Normalize areas for color intensity (0.3 to 0.7 alpha range)
    areas = lava_gdf['Area'].values
    min_area, max_area = areas.min(), areas.max()
    alphas = 0.3 + 0.4 * (areas - min_area) / (max_area - min_area)

    # Plot each flow feature with intensity based on area
    for idx, (geom, alpha) in enumerate(zip(lava_gdf.geometry, alphas)):
        if geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            ax.fill(lons, lats, transform=data_crs,
                   color='#D55E00', alpha=alpha, edgecolor='none',
                   linewidth=0, zorder=5)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                coords = list(poly.exterior.coords)
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                ax.fill(lons, lats, transform=data_crs,
                       color='#D55E00', alpha=alpha, edgecolor='none',
                       linewidth=0, zorder=5)

    # Add solid outline around entire flow field
    flow_union = lava_gdf.union_all()
    if flow_union.geom_type == 'Polygon':
        outline_coords = list(flow_union.exterior.coords)
        outline_lons = [c[0] for c in outline_coords]
        outline_lats = [c[1] for c in outline_coords]
        ax.plot(outline_lons, outline_lats, transform=data_crs,
               color='#D55E00', linewidth=2.5, linestyle='-',
               zorder=6, label='2015 Lava Flow')
    elif flow_union.geom_type == 'MultiPolygon':
        for poly in flow_union.geoms:
            outline_coords = list(poly.exterior.coords)
            outline_lons = [c[0] for c in outline_coords]
            outline_lats = [c[1] for c in outline_coords]
            ax.plot(outline_lons, outline_lats, transform=data_crs,
                   color='#D55E00', linewidth=2.5, linestyle='-', zorder=6)

    # Vent field markers
    label_offsets = {
        "ASHES": (-70, 10),
        "Coquille": (-80, 15),
        "Int'l District": (-100, -15),
        "Trevi": (20, 15),
        "CASM": (-70, 10),
    }
    for field_name, field_info in VENT_FIELDS.items():
        ax.plot(field_info['lon'], field_info['lat'], 'o', markersize=8,
                markerfacecolor='white', markeredgecolor='black',
                markeredgewidth=1.5, zorder=10, transform=data_crs)
        offset = label_offsets.get(field_name, (20, 15))
        ax.annotate(field_name, (field_info['lon'], field_info['lat']),
                    xycoords=data_crs._as_mpl_transform(ax),
                    xytext=offset, textcoords='offset points',
                    fontsize=FS_VENT_LABEL, fontweight='bold', style='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             alpha=0.9, edgecolor='black', linewidth=1.5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2,
                                   shrinkB=5),
                    zorder=11)

    # Scale bar (2 km neatline-style)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    vis_w = xlim[1] - xlim[0]
    vis_h = ylim[1] - ylim[0]
    scale_x = xlim[0] + vis_w * 0.05
    scale_y = ylim[0] + vis_h * 0.05
    bar_lw = 6
    for i in range(10):
        seg_start = scale_x + i * 100
        c = 'black' if i % 2 == 0 else 'white'
        ax.plot([seg_start, seg_start + 100], [scale_y, scale_y],
                color=c, linewidth=bar_lw, solid_capstyle='butt',
                transform=utm9n, zorder=15)
    ax.plot([scale_x, scale_x + 1000], [scale_y, scale_y],
            color='black', linewidth=bar_lw + 2, solid_capstyle='butt',
            transform=utm9n, zorder=14)
    ax.plot([scale_x + 1000, scale_x + 2000], [scale_y, scale_y],
            'k-', linewidth=bar_lw, solid_capstyle='butt',
            transform=utm9n, zorder=15)
    ax.text(scale_x, scale_y - vis_h * 0.015, '0',
            ha='center', fontsize=FS_SCALE_BAR, fontweight='bold', transform=utm9n)
    ax.text(scale_x + 1000, scale_y - vis_h * 0.015, '1',
            ha='center', fontsize=FS_SCALE_BAR, fontweight='bold', transform=utm9n)
    ax.text(scale_x + 2000, scale_y - vis_h * 0.015, '2 km',
            ha='center', fontsize=FS_SCALE_BAR, fontweight='bold', transform=utm9n)

    # Panel label (inside axes, upper-left)
    ax.set_title('(a) Axial Seamount Caldera', fontsize=FS_PANEL_LABEL,
                 fontweight='bold', pad=10)

    # Gridlines
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=0.5, color='white', alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': FS_GRIDLINE, 'rotation': 0}
    gl.ylabel_style = {'size': FS_GRIDLINE}
    gl.xpadding = 12
    gl.ypadding = 12

    # North arrow
    arrow_x = xlim[1] - vis_w * 0.08
    arrow_y = ylim[1] - vis_h * 0.04
    arrow_len = vis_h * 0.07
    ax.annotate('N', xy=(arrow_x, arrow_y),
                xytext=(arrow_x, arrow_y - arrow_len),
                fontsize=FS_NORTH_ARROW, fontweight='bold', ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                transform=utm9n, zorder=15,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # Date stamp
    add_date_stamp(ax, utm9n, xlim[1], ylim[0], vis_w, vis_h)

    # Neatline
    draw_neatline(ax, n_segments=14, linewidth=5)

    # Free memory
    del lon, lat, z, rgb, lon_grid, lat_grid, pts, x_utm_grid, y_utm_grid
    gc.collect()
    print("  Panel (a) done.")
    return z_min, z_max


# ─── Panel (b): ASHES Detail ──────────────────────────────────────────────────

def render_ashes_detail(fig, ax):
    """Render the ASHES vent field detail panel. Returns visible dims (w, h) in meters."""
    print("Panel (b): Loading ASHES 1cm LASS bathymetry...")
    ds = xr.open_dataset(BATHY_ASHES_PATH)
    x = ds.coords['x'].values
    y = ds.coords['y'].values
    z = ds['z'].values
    ds.close()
    print(f"  Grid: {len(x)} x {len(y)}, depth {np.nanmin(z):.1f} to {np.nanmax(z):.1f} m")

    utm9n = ccrs.UTM(zone=9, southern_hemisphere=False)
    data_crs = ccrs.PlateCarree()

    # Shaded relief
    ls = LightSource(azdeg=315, altdeg=35)
    z_min, z_max = np.nanpercentile(z, [1, 99])
    rgb = ls.shade(z, cmap=plt.cm.terrain, blend_mode='soft',
                   vmin=z_min, vmax=z_max, vert_exag=2)

    # Transform corners to UTM
    corner_lons = np.array([x.min(), x.max(), x.max(), x.min()])
    corner_lats = np.array([y.min(), y.min(), y.max(), y.max()])
    corners_utm = utm9n.transform_points(data_crs, corner_lons, corner_lats)
    x_utm_min = corners_utm[:, 0].min()
    x_utm_max = corners_utm[:, 0].max()
    y_utm_min = corners_utm[:, 1].min()
    y_utm_max = corners_utm[:, 1].max()

    ax.imshow(rgb, extent=[x_utm_min, x_utm_max, y_utm_min, y_utm_max],
              origin='lower', transform=utm9n)

    # Crop with 10m margin
    margin = 10
    vis_x_min = x_utm_min + margin
    vis_x_max = x_utm_max - margin
    vis_y_min = y_utm_min + margin
    vis_y_max = y_utm_max - margin
    ax.set_xlim(vis_x_min, vis_x_max)
    ax.set_ylim(vis_y_min, vis_y_max)

    # Contours (1m interval)
    x_grid, y_grid = np.meshgrid(x, y)
    pts = utm9n.transform_points(data_crs, x_grid, y_grid)
    x_utm_grid = pts[:, :, 0]
    y_utm_grid = pts[:, :, 1]
    contour_levels = np.arange(-1550, -1530, 1)
    ax.contour(x_utm_grid, y_utm_grid, z, levels=contour_levels,
               colors='black', linewidths=0.3, alpha=0.4, transform=utm9n)

    # Vent markers
    label_offsets = {
        "Inferno": (0, -22),
        "Hell": (-55, -18),
        "Virgin": (12, 5),
        "Phoenix": (-75, -18),
        "Mushroom": (-18, 15),  # Moved left to avoid overlapping Virgin
    }
    for name, info in ASHES_VENTS.items():
        color = VENT_TYPE_COLORS[info['type']]
        if x.min() <= info['lon'] <= x.max() and y.min() <= info['lat'] <= y.max():
            ax.plot(info['lon'], info['lat'], 'o', markersize=2.5,
                    markerfacecolor=color, markeredgecolor='black',
                    markeredgewidth=1.5, zorder=10, transform=data_crs)
            offset = label_offsets.get(name, (12, 5))
            ax.annotate(name, (info['lon'], info['lat']),
                        xycoords=data_crs._as_mpl_transform(ax),
                        xytext=offset, textcoords='offset points',
                        fontsize=FS_VENT_LABEL, fontweight='bold', color=color,
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.2, shrinkB=5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 alpha=0.9, edgecolor=color, linewidth=1.5),
                        zorder=12)

    # Scale bar (10m plain black)
    vis_w = vis_x_max - vis_x_min
    vis_h = vis_y_max - vis_y_min
    scale_x = vis_x_min + vis_w * 0.05
    scale_y = vis_y_min + vis_h * 0.05
    ax.plot([scale_x, scale_x + 10], [scale_y, scale_y],
            'k-', linewidth=4, transform=utm9n)
    ax.text(scale_x + 5, scale_y + vis_h * 0.03, '10 m',
            ha='center', fontsize=FS_SCALE_BAR, fontweight='bold', transform=utm9n,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # Panel label (inside axes, upper-left)
    ax.set_title('(b) ASHES Vent Field', fontsize=FS_PANEL_LABEL,
                 fontweight='bold', pad=10)

    # Gridlines (lat/lon reference)
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=0.3, color='white', alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': FS_GRIDLINE, 'rotation': 0}
    gl.ylabel_style = {'size': FS_GRIDLINE}
    gl.xpadding = 8
    gl.ypadding = 8

    # Legend (temperature classification)
    legend_elements = [
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=VENT_TYPE_COLORS['high-temp'],
               markersize=6, markeredgecolor='black', markeredgewidth=0.8,
               label='High-temp (>200\u00b0C)'),
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=VENT_TYPE_COLORS['intermittent'],
               markersize=6, markeredgecolor='black', markeredgewidth=0.8,
               label='Intermittent'),
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=VENT_TYPE_COLORS['low-temp'],
               markersize=6, markeredgecolor='black', markeredgewidth=0.8,
               label='Low-temp (<100\u00b0C)'),
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=VENT_TYPE_COLORS['no-data'],
               markersize=6, markeredgecolor='black', markeredgewidth=0.8,
               label='No data'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=FS_LEGEND,
              framealpha=0.95, edgecolor='black',
              borderpad=0.4, labelspacing=0.3, handletextpad=0.4)

    # Neatline
    draw_neatline(ax, n_segments=12, linewidth=5)

    # Free memory — keep vis dimensions for panel (c) matching
    ashes_vis_dims = (vis_w, vis_h)
    del x, y, z, rgb, x_grid, y_grid, pts, x_utm_grid, y_utm_grid
    gc.collect()
    print("  Panel (b) done.")
    return ashes_vis_dims


# ─── Panel (c): Int'l District Detail ─────────────────────────────────────────

def render_intl_district_detail(fig, ax, target_dims=None):
    """Render the International District detail panel.

    If target_dims=(width, height) in meters is provided, the visible extent
    is center-cropped to those dimensions so both detail panels show the same
    physical area size.
    """
    print("Panel (c): Loading Int'l District 1cm LASS bathymetry...")
    ds = xr.open_dataset(BATHY_INTL_PATH)
    x = ds.coords['x'].values
    y = ds.coords['y'].values
    z = ds['z'].values
    ds.close()
    print(f"  Grid: {len(x)} x {len(y)}, depth {np.nanmin(z):.1f} to {np.nanmax(z):.1f} m")

    utm9n = ccrs.UTM(zone=9, southern_hemisphere=False)
    data_crs = ccrs.PlateCarree()

    # Shaded relief
    ls = LightSource(azdeg=315, altdeg=35)
    z_min, z_max = np.nanpercentile(z, [1, 99])
    rgb = ls.shade(z, cmap=plt.cm.terrain, blend_mode='soft',
                   vmin=z_min, vmax=z_max, vert_exag=2)

    # Transform corners to UTM
    corner_lons = np.array([x.min(), x.max(), x.max(), x.min()])
    corner_lats = np.array([y.min(), y.min(), y.max(), y.max()])
    corners_utm = utm9n.transform_points(data_crs, corner_lons, corner_lats)
    x_utm_min = corners_utm[:, 0].min()
    x_utm_max = corners_utm[:, 0].max()
    y_utm_min = corners_utm[:, 1].min()
    y_utm_max = corners_utm[:, 1].max()

    ax.imshow(rgb, extent=[x_utm_min, x_utm_max, y_utm_min, y_utm_max],
              origin='lower', transform=utm9n)

    # Crop — match ASHES visible extent if target_dims provided, else 20m margin
    # Then zoom in an additional 10m on each edge for tighter framing
    if target_dims:
        target_w, target_h = target_dims
        target_w -= 20   # 10m zoom-in from each side
        target_h -= 20
        center_x = (x_utm_min + x_utm_max) / 2
        center_y = (y_utm_min + y_utm_max) / 2
        vis_x_min = center_x - target_w / 2
        vis_x_max = center_x + target_w / 2
        vis_y_min = center_y - target_h / 2
        vis_y_max = center_y + target_h / 2
    else:
        margin = 20
        vis_x_min = x_utm_min + margin
        vis_x_max = x_utm_max - margin
        vis_y_min = y_utm_min + margin
        vis_y_max = y_utm_max - margin
    ax.set_xlim(vis_x_min, vis_x_max)
    ax.set_ylim(vis_y_min, vis_y_max)

    # Contours (1m interval)
    x_grid, y_grid = np.meshgrid(x, y)
    pts = utm9n.transform_points(data_crs, x_grid, y_grid)
    x_utm_grid = pts[:, :, 0]
    y_utm_grid = pts[:, :, 1]
    contour_levels = np.arange(-1530, -1498, 1)
    ax.contour(x_utm_grid, y_utm_grid, z, levels=contour_levels,
               colors='black', linewidths=0.3, alpha=0.4, transform=utm9n)

    # Vent markers
    label_offsets = {
        "El Guapo": (15, 12),
        "Escargot": (-85, -18),
        "Castle": (15, 15),
        "Diva": (15, -18),
        "Flat Top": (-80, -18),
    }
    for name, info in INTL_VENTS.items():
        color = VENT_TYPE_COLORS[info['type']]
        if x.min() <= info['lon'] <= x.max() and y.min() <= info['lat'] <= y.max():
            ax.plot(info['lon'], info['lat'], 'o', markersize=2.5,
                    markerfacecolor=color, markeredgecolor='black',
                    markeredgewidth=1.5, zorder=10, transform=data_crs)
            offset = label_offsets.get(name, (12, 5))
            ax.annotate(name, (info['lon'], info['lat']),
                        xycoords=data_crs._as_mpl_transform(ax),
                        xytext=offset, textcoords='offset points',
                        fontsize=FS_VENT_LABEL, fontweight='bold', color=color,
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.2, shrinkB=5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 alpha=0.9, edgecolor=color, linewidth=1.5),
                        zorder=12)

    # Scale bar (10m plain black)
    vis_w = vis_x_max - vis_x_min
    vis_h = vis_y_max - vis_y_min
    scale_x = vis_x_min + vis_w * 0.05
    scale_y = vis_y_min + vis_h * 0.05
    ax.plot([scale_x, scale_x + 10], [scale_y, scale_y],
            'k-', linewidth=4, transform=utm9n)
    ax.text(scale_x + 5, scale_y + vis_h * 0.03, '10 m',
            ha='center', fontsize=FS_SCALE_BAR, fontweight='bold', transform=utm9n,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # Panel label (inside axes, upper-left)
    ax.set_title('(c) International District', fontsize=FS_PANEL_LABEL,
                 fontweight='bold', pad=10)

    # Gridlines (lat/lon reference)
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=0.3, color='white', alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': FS_GRIDLINE, 'rotation': 0}
    gl.ylabel_style = {'size': FS_GRIDLINE}
    gl.xpadding = 8
    gl.ypadding = 8

    # Legend (matches panel (b))
    legend_elements = [
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=VENT_TYPE_COLORS['high-temp'],
               markersize=6, markeredgecolor='black', markeredgewidth=0.8,
               label='High-temp (>200\u00b0C)'),
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=VENT_TYPE_COLORS['intermittent'],
               markersize=6, markeredgecolor='black', markeredgewidth=0.8,
               label='Intermittent'),
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=VENT_TYPE_COLORS['low-temp'],
               markersize=6, markeredgecolor='black', markeredgewidth=0.8,
               label='Low-temp (<100\u00b0C)'),
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=VENT_TYPE_COLORS['no-data'],
               markersize=6, markeredgecolor='black', markeredgewidth=0.8,
               label='No data'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=FS_LEGEND,
              framealpha=0.95, edgecolor='black',
              borderpad=0.4, labelspacing=0.3, handletextpad=0.4)

    # Neatline
    draw_neatline(ax, n_segments=12, linewidth=5)

    # Free memory
    del x, y, z, rgb, x_grid, y_grid, pts, x_utm_grid, y_utm_grid
    gc.collect()
    print("  Panel (c) done.")


# ─── Composite assembly ───────────────────────────────────────────────────────

def make_composite_map():
    """Assemble the three-panel composite poster figure."""
    print("=" * 60)
    print("Composite Vent Field Map (3 panels)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    utm9n = ccrs.UTM(zone=9, southern_hemisphere=False)

    fig = plt.figure(figsize=(20, 14))

    # --- Axes positions (figure-fraction) ---
    # Titles outside panels.  (b) flush with (a) top, (c) flush with (a) bottom.
    # (c) title sits in the gap between (b) and (c).
    # Panel (a) enlarged and shifted right to tighten layout.
    #
    #   Panel (a): left=0.08, bottom=0.18, top=0.90  (height 0.72, width 0.40)  — title above
    #   Panel (b): left=0.44, bottom=0.59, top=0.90  (height 0.31, width 0.38)  — top flush w/ (a), title above
    #   Panel (c): left=0.44, bottom=0.18, top=0.49  (height 0.31, width 0.38)  — bottom flush w/ (a), title in gap
    #   Gap between (b) and (c): 0.59 - 0.49 = 0.10 (holds (c) title)

    ax1 = fig.add_axes([0.08, 0.18, 0.40, 0.72], projection=utm9n)   # (a) - wider, shifted right
    ax2 = fig.add_axes([0.44, 0.59, 0.38, 0.31], projection=utm9n)   # (b)
    ax3 = fig.add_axes([0.44, 0.18, 0.38, 0.31], projection=utm9n)   # (c)

    # Render panels sequentially (one dataset at a time for memory)
    z_min, z_max = render_site_overview(fig, ax1)
    ashes_dims = render_ashes_detail(fig, ax2)
    render_intl_district_detail(fig, ax3, target_dims=ashes_dims)

    # --- Dynamic colorbar placement ---
    # Draw to compute actual axes positions after Cartopy aspect-ratio adjustment
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox2 = ax2.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    bbox3 = ax3.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    right_edge = max(bbox2.x1, bbox3.x1)

    # Place colorbar with small breathing room (~2.5 mm)
    cax_left = right_edge + 0.012
    cax = fig.add_axes([cax_left, 0.18, 0.015, 0.72])

    sm = plt.cm.ScalarMappable(cmap=plt.cm.terrain,
                                norm=plt.Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Depth (m)', fontsize=FS_COLORBAR)
    cbar.ax.tick_params(labelsize=11)

    # --- Flowing caption (constrained to colorbar inner edge) ---
    caption_width = cax_left - 0.08   # from left margin (0.08) to colorbar inner edge
    caption_ax = fig.add_axes([0.08, 0.005, caption_width, 0.11])
    caption_ax.axis('off')

    caption_text = (
        "This figure shows the hydrothermal vent fields of Axial Seamount "
        "at multiple spatial scales. Panel (a) provides an overview of the "
        "caldera with five vent field locations plotted on 1 m AUV bathymetry "
        "(MBARI, 2025) with 20 m depth contours. Recent lava flows from the "
        "2011 (white) and 2015 (orange) eruptions are shown with color "
        "intensity indicating flow size; each eruption produced ~12 distinct "
        "flow lobes totaling 10\u201312 km² of new lava (Clague et al., 2017). "
        "Panel (b) shows a detailed view of the ASHES vent field mapped with "
        "1 cm LASS lidar bathymetry (MBARI, 2025) with 1 m depth contours. "
        "Panel (c) shows the International District vent field at the same "
        "scale, also mapped with 1 cm LASS lidar bathymetry (MBARI, 2025) "
        "with 1 m depth contours. Vent markers are colored by temperature "
        "classification from the 2024\u20132025 MISO deployment. Detail panels "
        "use locally enhanced color stretch; the colorbar shows the overview "
        "depth range."
    )

    # Compute wrap width dynamically from actual caption area - tighter packing
    # Figure is 20" wide; use 0.0065" per pt for tighter char width estimate
    caption_inches = caption_width * 20
    char_width = FS_CAPTION * 0.0065  # Tighter estimate to fit more words per line
    chars_per_line = int(caption_inches / char_width)
    wrapped_caption = textwrap.fill(caption_text, width=chars_per_line)
    caption_ax.text(0.0, 1.0, wrapped_caption, fontsize=FS_CAPTION, va='top',
                    transform=caption_ax.transAxes, family='sans-serif',
                    linespacing=1.4)

    # Save
    output_file = OUTPUT_DIR / "composite_vent_field_maps.png"
    print(f"\nSaving to {output_file}...")
    fig.savefig(output_file, dpi=600, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    return output_file


def main():
    make_composite_map()
    print("\nDone!")


if __name__ == "__main__":
    main()
