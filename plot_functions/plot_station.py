# Functions to plot HadUK-Grid station data

import os
import sys
import numpy as np

import iris
import iris.util
import iris.analysis
import iris.coord_systems

import matplotlib

from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

sys.path.append("%s/.." % os.path.dirname(__file__))
from get_data.HadUKGrid.HUKG_monthly_load import dm_hukg


def get_land_mask():
    lm_plot = iris.load_cube(
        "%s/fixed_fields/land_mask/opfc_global_2019.nc" % os.getenv("DATADIR")
    )
    lm_plot = lm_plot.regrid(dm_hukg, iris.analysis.Linear())
    return lm_plot


def plotStationLocationsAxes(
    ax_map,
    meta,
    lMask=None,
    scolour="Red",
    ssize=100,
    zorder=1,
    sea_colour=(0.9, 0.9, 0.9, 1),
    land_colour=(0.4, 0.4, 0.4, 0.3),
):

    if lMask is None:
        lMask = get_land_mask()

    lons = lMask.coord("projection_x_coordinate").points
    lats = lMask.coord("projection_y_coordinate").points
    ax_map.set_ylim(min(lats), max(lats))
    ax_map.set_xlim(min(lons), max(lons))
    ax_map.set_axis_off()
    ax_map.set_aspect("equal", adjustable="box", anchor="C")
    ax_map.add_patch(
        Rectangle(
            (min(lons), min(lats)),
            max(lons) - min(lons),
            max(lats) - min(lats),
            facecolor=sea_colour,
            fill=True,
            zorder=zorder,
        )
    )

    # Plot the land mask
    mask_img = ax_map.pcolorfast(
        lMask.coord("projection_x_coordinate").points,
        lMask.coord("projection_y_coordinate").points,
        lMask.data,
        cmap=matplotlib.colors.ListedColormap(
            (
                (land_colour[0], land_colour[1], land_colour[2], 0),
                (land_colour[0], land_colour[1], land_colour[2], land_colour[3]),
            )
        ),
        vmin=0,
        vmax=1,
        alpha=1.0,
        zorder=zorder + 100,
    )

    # Add the stations
    for s_id in meta.keys():
        ax_map.add_patch(
            Circle(
                (meta[s_id]["X"], meta[s_id]["Y"]),
                radius=ssize,
                color=scolour,
                zorder=zorder + 200,
            )
        )

    return

# Plot observations
def plotObsAxes(
    ax_map,
    slons,slats,observations,
    lMask=None,
    cMap='RdBu',
    ssize=100,
    vmax=None,
    vmin=None,
    zorder=1,
    sea_colour=(0.9, 0.9, 0.9, 1),
    land_colour=(0.4, 0.4, 0.4, 0.3),
):

    if lMask is None:
        lMask = get_land_mask()

    lons = lMask.coord("projection_x_coordinate").points
    lats = lMask.coord("projection_y_coordinate").points
    ax_map.set_ylim(min(lats), max(lats))
    ax_map.set_xlim(min(lons), max(lons))
    ax_map.set_axis_off()
    ax_map.set_aspect("equal", adjustable="box", anchor="C")
    ax_map.add_patch(
        Rectangle(
            (min(lons), min(lats)),
            max(lons) - min(lons),
            max(lats) - min(lats),
            facecolor=sea_colour,
            fill=True,
            zorder=zorder,
        )
    )

    # Plot the land mask
    mask_img = ax_map.pcolorfast(
        lMask.coord("projection_x_coordinate").points,
        lMask.coord("projection_y_coordinate").points,
        lMask.data,
        cmap=matplotlib.colors.ListedColormap(
            (
                (land_colour[0], land_colour[1], land_colour[2], 0),
                (land_colour[0], land_colour[1], land_colour[2], land_colour[3]),
            )
        ),
        vmin=0,
        vmax=1,
        alpha=1.0,
        zorder=zorder + 100,
    )

    # Add the stations
    for s_idx in range(len(observations)):
        colour = max(0.01,min(0.999,(observations[s_idx]-vmin)/(vmax-vmin)))
        ax_map.add_patch(
            Circle(
                (slons[s_idx], slats[s_idx]),
                radius=ssize,
                color=cMap(colour),
                zorder=zorder + 200,
            )
        )

    return

def plotObsScatterAxes(
    ax, var_in, var_out, vMax=None, vMin=None, xlabel="", ylabel="",psize=1.0,
):
    if vMax is None:
        vMax = max(np.max(var_in.data), np.max(var_out.data))
    if vMin is None:
        vMin = min(np.min(var_in.data), np.min(var_out.data))
    ax.set_xlim(vMin, vMax)
    ax.set_ylim(vMin, vMax)
    ax.scatter(
        x=var_in,
        y=var_out,
        s=psize,
        c='blue',
        marker='.',
    )
    ax.add_line(
        Line2D(
            xdata=(vMin, vMax),
            ydata=(vMin, vMax),
            linestyle="solid",
            linewidth=0.5,
            color=(0.5, 0.5, 0.5, 1),
            zorder=100,
        )
    )
    ax.set(ylabel=ylabel, xlabel=xlabel)
    ax.grid(color="black", alpha=0.2, linestyle="-", linewidth=0.5)
