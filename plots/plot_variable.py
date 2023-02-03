# Functions to plot 20CR monthly data for the UK region

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

# Make a dummy iris Cube for plotting.
# Makes a cube in equirectangular projection.
# Takes resolution, plot range, and pole location
#  (all in degrees) as arguments, returns an
#  iris cube.
def plot_cube(
    resolution=1,
    xmin=-3 - 16,
    xmax=-3 + 15,
    ymin=55 - 16,
    ymax=55 + 15,
    pole_latitude=90,
    pole_longitude=180,
    npg_longitude=0,
):

    cs = iris.coord_systems.RotatedGeogCS(pole_latitude, pole_longitude, npg_longitude)
    lat_values = np.arange(ymin, ymax + resolution, resolution)
    latitude = iris.coords.DimCoord(
        lat_values, standard_name="latitude", units="degrees_north", coord_system=cs
    )
    lon_values = np.arange(xmin, xmax + resolution, resolution)
    longitude = iris.coords.DimCoord(
        lon_values, standard_name="longitude", units="degrees_east", coord_system=cs
    )
    dummy_data = np.zeros((len(lat_values), len(lon_values)))
    plot_cube = iris.cube.Cube(
        dummy_data, dim_coords_and_dims=[(latitude, 0), (longitude, 1)]
    )
    return plot_cube


# def get_land_mask(grid_cube=None):
#    if grid_cube is None:
#        grid_cube = plot_cube()
#    lm = iris.load_cube("%s/20CR/version_3/fixed/land.nc" % os.getenv("SCRATCH"))
#    lm = iris.util.squeeze(lm)
#    coord_s = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
#    lm.coord("latitude").coord_system = coord_s
#    lm.coord("longitude").coord_system = coord_s
#    lm = lm.regrid(grid_cube, iris.analysis.Linear())
#    return lm


def get_land_mask(grid_cube=None):
    lm = iris.load_cube(
        "%s/fixed_fields/land_mask/opfc_global_2019.nc" % os.getenv("DATADIR")
    )
    lm = lm.regrid(grid_cube, iris.analysis.Linear())
    return lm


def plotFieldAxes(
    ax_map,
    field,
    vMax=None,
    vMin=None,
    lMask=None,
    cMap=cmocean.cm.balance,
    plotCube=None,
):

    if plotCube is None:
        plotCube = plot_cube()
    field = field.regrid(plotCube, iris.analysis.Linear())
    if vMax is None:
        vMax = np.max(field.data)
    if vMin is None:
        vMin = np.min(field.data)
    if lMask is None:
        lMask = get_land_mask(plot_cube(resolution=0.1))

    lons = plotCube.coord("longitude").points
    lats = plotCube.coord("latitude").points
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
        lons,
        lats,
        field.data,
        cmap=cMap,
        vmin=vMin,
        vmax=vMax,
        alpha=1.0,
        zorder=10,
    )

    # Overlay the land mask
    mask_img = ax_map.pcolorfast(
        lMask.coord("longitude").points,
        lMask.coord("latitude").points,
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
        Rectangle(
            (0, 1),
            1,
            1,
            facecolor=(0.6, 0.6, 0.6, 1),
            fill=True,
            zorder=1,
        )
    )

    # Axes for the map
    ax_map = fig.add_axes([0.025, 0.125, 0.95, 0.85])
    T_img = plotFieldAxes(
        ax_map,
        field,
        vMax=vMax,
        vMin=vMin,
        lMask=lMask,
        cMap=cMap,
        plotCube=plotCube,
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
        cmap=cmocean.tools.crop_by_percent(cmocean.cm.ice_r, 25, which="min"),
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
