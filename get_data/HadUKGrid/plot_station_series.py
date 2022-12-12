#!/usr/bin/env python

# Plot timeseries of RR station precip and HadUKGrid precip at the same location
#  (as anomalies).


import os
import sys
import numpy as np
import datetime
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.append("%s/../.." % os.path.dirname(__file__))
from plot_functions.plot_station import plotStationLocationsAxes

from HUKG_monthly_load import load_station_metadata
from HUKG_monthly_load import load_rr_stations
from HUKG_monthly_load import load_climatology
from HUKG_monthly_load import load_variable

import warnings

# warnings.filterwarnings("ignore", message=".*datum.*")
# warnings.filterwarnings("ignore", message=".*TransverseMercator.*")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--src_id", help="Station numeric ID", type=str, required=True, default=None
)
parser.add_argument(
    "--startyear", help="First year to plot", type=int, required=False, default=1677
)
parser.add_argument(
    "--endyear", help="Last year to plot", type=int, required=False, default=1960
)
args = parser.parse_args()


# Load the station metadata
meta = load_station_metadata(args.src_id)

# Grid location of station in HadUKGrid field
x_grid = int((meta["X"] + 199000) / 1000)
y_grid = int((meta["Y"] + 199000) / 1000)

# Get climatology for that grid location
clim = [0]
for month in range(1, 13):
    cf = load_climatology("monthly_rainfall", month)
    clim.append(cf.data[y_grid, x_grid])

# Load the station series
stn_series = {}
for year in range(args.startyear, args.endyear + 1):
    try:
        stn_y = load_rr_stations(year, srcid=args.src_id)
    except (FileNotFoundError, KeyError):
        for month in range(1, 13):
            stn_series["%04d%02d" % (year, month)] = np.nan
        continue
    for month in range(1, 13):
        if stn_y[month] is None:
            stn_series["%04d%02d" % (year, month)] = np.nan
        else:
            stn_series["%04d%02d" % (year, month)] = stn_y[month] - clim[month]

# Load the HadUKGrid series
huk_series = {}
for year in range(args.startyear, args.endyear + 1):
    for month in range(1, 13):
        try:
            huk_m = load_variable("monthly_rainfall", year, month).data[y_grid, x_grid]
        except FileNotFoundError:
            huk_m = np.nan
        if huk_m is None:
            huk_series["%04d%02d" % (year, month)] = np.nan
        else:
            huk_series["%04d%02d" % (year, month)] = huk_m - clim[month]


# Make plottable series and calculate limits
st_pl = []
ht_pl = []
dt_pl = []
for year in range(args.startyear, args.endyear + 1):
    for month in range(1, 13):
        dt_pl.append(datetime.date(year, month, 15))
        st_pl.append(stn_series["%04d%02d" % (year, month)])
        ht_pl.append(huk_series["%04d%02d" % (year, month)])

ymin = np.nanmin([np.nanmin(st_pl), np.nanmin(ht_pl)])
ymax = np.nanmax([np.nanmax(st_pl), np.nanmax(ht_pl)])
# print(st_pl)
# print(ht_pl)
# sys.exit(0)
yr = ymax - ymin
ymax += yr / 10
ymin -= yr / 10

# Make the plot
aspect = 3
fig = Figure(
    figsize=(5 * aspect, 5),
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
        facecolor=(0.95, 0.95, 0.95, 1),
        fill=True,
        zorder=1,
    )
)
ax_ts = fig.add_axes(
    [0.045, 0.08, 0.95, 0.9],
    xlim=(
        dt_pl[0] - datetime.timedelta(days=15),
        dt_pl[-1] + datetime.timedelta(days=15),
    ),
    ylim=(ymin, ymax),
)
ax_ts.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
ax_ts.add_line(
    Line2D(dt_pl, ht_pl, linewidth=1, color=(0, 0, 0, 1), alpha=1.0, zorder=50)
)
ax_ts.add_line(
    Line2D(dt_pl, st_pl, linewidth=1, color=(1, 0, 0, 1), alpha=1.0, zorder=60)
)

# Add thumbnail showing station location
axt = fig.add_axes([0.01, 0.75, 0.15 / aspect, 0.15 * 1450 / 900])
plotStationLocationsAxes(
    axt,
    {args.src_id: meta},
    ssize=25000,
    scolour="Red",
    sea_colour=(1.0, 1.0, 1.0, 1.0),
    land_colour=(0.0, 0.0, 0.0, 0.5),
    zorder=150,
)


fig.savefig("station_series.png")
