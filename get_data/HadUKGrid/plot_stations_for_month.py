#!/usr/bin/env python

# Plot the station locations, from Rainfall Rescue, for a selected month.


import os
import sys
import numpy as np
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

sys.path.append("%s/../.." % os.path.dirname(__file__))
from plot_functions.plot_station import plotStationLocationsAxes

from HUKG_monthly_load import load_station_metadata
from HUKG_monthly_load import load_rr_stations

import warnings

# warnings.filterwarnings("ignore", message=".*datum.*")
# warnings.filterwarnings("ignore", message=".*TransverseMercator.*")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Test year", type=int, required=False, default=1869)
parser.add_argument("--month", help="Test month", type=int, required=False, default=3)
parser.add_argument(
    "--src_id", help="Single selected station", type=str, required=False, default=None
)
parser.add_argument(
    "--size", help="station plot size", type=float, required=False, default=1
)
args = parser.parse_args()


# Load the station metadata
if args.src_id is not None:
    meta = load_station_metadata(args.srcid)
else:
    meta_full = load_station_metadata()
    monthly = load_rr_stations(args.year)
    meta = {}
    for src_id in monthly.keys():
        if monthly[src_id][args.month] is not None:
            meta[src_id] = meta_full[src_id]


# Make the plot
fig = Figure(
    figsize=(5, 5 * 1450 / 900),
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
axb.set_axis_off()
axb.add_patch(
    Rectangle(
        (0, 0),
        1,
        1,
        facecolor=(1.0, 1.0, 1.0, 1),
        fill=True,
        zorder=1,
    )
)
plotStationLocationsAxes(
    axb,
    meta,
    ssize=args.size * 1000,
)

fig.savefig("stations.png")
