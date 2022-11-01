# Functions to plot HadUK-Grid monthly data for the UK region

import os
import numpy as np

import iris
import iris.util
import iris.analysis
import iris.coord_systems

import matplotlib

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import cmocean


def plotFieldAxes(
    ax_map,
    field,
    vMax=None,
    vMin=None,
    lMask=None,
    cMap=cmocean.cm.balance,
    plotCube=None,
):

    if plotCube is not None:
        field = field.regrid(plotCube, iris.analysis.Linear())
    if vMax is None:
        vMax = np.max(field.data)
    if vMin is None:
        vMin = np.min(field.data)
    if lMask is None:
        lMask = get_land_mask(plot_cube(resolution=0.1))

    lons = field.coord("projection_x_coordinate").points
    lats = field.coord("projection_y_coordinate").points
    ax_map.set_ylim(min(lats), max(lats))
    ax_map.set_xlim(min(lons), max(lons))
    ax_map.set_axis_off()
    ax_map.set_aspect("equal", adjustable="box", anchor="C")
    ax_map.add_patch(
        Rectangle(
            (min(lons), min(lats)),
            max(lons) - min(lons),
            max(lats) - min(lats),
            facecolor=(0.9, 0.9, 0.9, 1),
            fill=True,
            zorder=1,
        )
    )
    # Plot the field
    T_img = ax_map.pcolorfast(
        lons, lats, field.data, cmap=cMap, vmin=vMin, vmax=vMax, alpha=1.0, zorder=10,
    )

    # Overlay the land mask
    mask_img = ax_map.pcolorfast(
        lMask.coord("projection_x_coordinate").points,
        lMask.coord("projection_y_coordinate").points,
        lMask.data,
        cmap=matplotlib.colors.ListedColormap(
            ((0.4, 0.4, 0.4, 0), (0.4, 0.4, 0.4, 0.3))
        ),
        vmin=0,
        vmax=1,
        alpha=1.0,
        zorder=100,
    )
    return T_img


def plotField(
    field,
    opDir=".",
    fName="tst.png",
    vMax=None,
    vMin=None,
    lMask=None,
    cMap=cmocean.cm.balance,
    plotCube=None,
):

    fig = Figure(
        figsize=(10, 11),
        dpi=100,
        facecolor=(0.5, 0.5, 0.5, 1),
        edgecolor=None,
        linewidth=0.0,
        frameon=False,
        subplotpars=None,
        tight_layout=None,
    )
    canvas = FigureCanvas(fig)
    font = {
        "family": "sans-serif",
        "sans-serif": "Arial",
        "weight": "normal",
        "size": 20,
    }
    matplotlib.rc("font", **font)
    axb = fig.add_axes([0, 0, 1, 1])
    axb.add_patch(
        Rectangle((0, 1), 1, 1, facecolor=(0.6, 0.6, 0.6, 1), fill=True, zorder=1,)
    )

    # Axes for the map
    ax_map = fig.add_axes([0.025, 0.125, 0.95, 0.85])
    T_img = plotFieldAxes(
        ax_map, field, vMax=vMax, vMin=vMin, lMask=lMask, cMap=cMap, plotCube=plotCube,
    )

    # ColourBar
    ax_cb = fig.add_axes([0.125, 0.05, 0.75, 0.05])
    ax_cb.set_axis_off()
    cb = fig.colorbar(
        T_img, ax=ax_cb, location="bottom", orientation="horizontal", fraction=1.0
    )

    if not os.path.isdir(opDir):
        os.makedirs(opDir)

    # Output as png
    fig.savefig("%s/%s" % (opDir, fName))


def plotScatterAxes(
    ax, var_in, var_out, vMax=None, vMin=None, xlabel="", ylabel="", bins="log"
):
    if vMax is None:
        vMax = max(np.max(var_in.data), np.max(var_out.data))
    if vMin is None:
        vMin = min(np.min(var_in.data), np.min(var_out.data))
    ax.set_xlim(vMin, vMax)
    ax.set_ylim(vMin, vMax)
    ax.hexbin(
        x=var_in.data.flatten(),
        y=var_out.data.flatten(),
        cmap=cmocean.cm.ice_r,
        #        cmap=cmocean.tools.crop_by_percent(cmocean.cm.ice_r, 25, which="min"),
        bins=bins,
        gridsize=50,
        mincnt=1,
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
