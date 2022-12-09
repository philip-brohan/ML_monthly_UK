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
    scolour='Red',
    ssize=100,
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
            facecolor=(0.9, 0.9, 0.9, 1),
            fill=True,
            zorder=1,
        )
    )

    # Plot the land mask
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

# Add the stations
    for s_id in meta.keys():
        ax_map.add_patch(
            Circle((meta[s_id]['X'],meta[s_id]['Y']),radius=ssize,color=scolour,zorder=200))

    return 


