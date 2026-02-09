#!/usr/bin/env python3
"""
pick_coordinates.py - Interactive coordinate picker for vent field locations

Click on the map to get coordinates. Right-click to remove the last point.
Close the window when done - coordinates will be printed.

Usage:
    uv run python pick_coordinates.py
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from pathlib import Path

# Bathymetry path
BATHY_PATH = Path("/home/jovyan/my_data/axial/axial_bathy/MBARI_AxialSeamount_V2506_AUV_Summit_AUVOverShip_Topo1mSq.grd")

# Current vent field locations (for reference)
CURRENT_VENTS = {
    "ASHES": {"lon": -130.0137, "lat": 45.9336},
    "CASM": {"lon": -130.0272, "lat": 45.9889},
    "Coquille": {"lon": -129.9930, "lat": 45.9174},
    "International District": {"lon": -129.9790, "lat": 45.9263},
    "Trevi": {"lon": -129.9838, "lat": 45.9463},
}

# Store picked points
picked_points = []
point_markers = []
point_labels = []


def load_bathymetry():
    """Load caldera bathymetry."""
    print("Loading bathymetry...")
    ds = xr.open_dataset(BATHY_PATH)

    # Define caldera extent (same as make_caldera_map.py)
    center_lon = -130.008772
    center_lat = 45.95485
    half_width = 0.064
    half_height = 0.061

    ds = ds.sel(
        lon=slice(center_lon - half_width, center_lon + half_width),
        lat=slice(center_lat - half_height, center_lat + half_height)
    )

    lon = ds.coords['lon'].values
    lat = ds.coords['lat'].values
    z = ds['z'].values
    ds.close()

    return lon, lat, z


def on_click(event):
    """Handle mouse click events."""
    global picked_points, point_markers, point_labels

    if event.inaxes is None:
        return

    # Right-click to remove last point
    if event.button == 3:
        if picked_points:
            picked_points.pop()
            if point_markers:
                marker = point_markers.pop()
                marker.remove()
            if point_labels:
                label = point_labels.pop()
                label.remove()
            plt.draw()
            print("Removed last point")
        return

    # Left-click to add point
    if event.button == 1:
        lon, lat = event.xdata, event.ydata
        point_num = len(picked_points) + 1
        picked_points.append((lon, lat))

        # Add marker
        marker, = event.inaxes.plot(lon, lat, 'ro', markersize=12,
                                     markeredgecolor='white', markeredgewidth=2)
        point_markers.append(marker)

        # Add label
        label = event.inaxes.annotate(f'{point_num}', (lon, lat),
                                       xytext=(8, 8), textcoords='offset points',
                                       fontsize=12, fontweight='bold', color='red',
                                       bbox=dict(boxstyle='round,pad=0.3',
                                                facecolor='white', alpha=0.9))
        point_labels.append(label)

        plt.draw()
        print(f"Point {point_num}: lon={lon:.6f}, lat={lat:.6f}")


def main():
    print("=" * 60)
    print("Interactive Vent Field Coordinate Picker")
    print("=" * 60)
    print("\nControls:")
    print("  Left-click  : Add a point")
    print("  Right-click : Remove last point")
    print("  Close window: Print all coordinates")
    print("\nCurrent vent field locations shown as yellow circles.")
    print("=" * 60)

    lon, lat, z = load_bathymetry()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Create shaded relief
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    ls = LightSource(azdeg=315, altdeg=45)
    z_min, z_max = np.nanpercentile(z, [2, 98])
    rgb = ls.shade(z, cmap=plt.cm.terrain, blend_mode='soft',
                   vmin=z_min, vmax=z_max)

    ax.imshow(rgb, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
              origin='lower', aspect='equal')

    # Add contours
    contour_levels = np.arange(-2200, -1400, 50)
    cs = ax.contour(lon_grid, lat_grid, z, levels=contour_levels,
                    colors='black', linewidths=0.5, alpha=0.6)
    ax.clabel(cs, levels=contour_levels[::2], fontsize=7, fmt='%d m', inline=True)

    # Plot current vent field locations
    for name, info in CURRENT_VENTS.items():
        ax.plot(info['lon'], info['lat'], 'yo', markersize=10,
                markeredgecolor='black', markeredgewidth=1.5, alpha=0.7)
        ax.annotate(name, (info['lon'], info['lat']),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=9, style='italic',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow',
                             alpha=0.7, edgecolor='none'))

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Click to pick vent field coordinates\n(Yellow = current locations, Red = your picks)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', color='white')

    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.tight_layout()
    plt.show()

    # Print results when window closes
    print("\n" + "=" * 60)
    print("PICKED COORDINATES")
    print("=" * 60)
    if picked_points:
        for i, (plon, plat) in enumerate(picked_points, 1):
            print(f"Point {i}: lon={plon:.6f}, lat={plat:.6f}")

        print("\n# Python dict format:")
        print("VENTS = {")
        for i, (plon, plat) in enumerate(picked_points, 1):
            print(f'    "Point_{i}": {{"lon": {plon:.6f}, "lat": {plat:.6f}}},')
        print("}")
    else:
        print("No points picked.")


if __name__ == "__main__":
    main()
