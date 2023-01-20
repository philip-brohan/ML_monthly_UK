#!/usr/bin/env python

# Plot timeseries of RR station precip and reconstructed precip at the same location
#  showing ensemble with and without assimilating station.


import os
import sys
import numpy as np
import glob
import csv
import datetime
from random import randint
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.append("%s/../../.." % os.path.dirname(__file__))
from plot_functions.plot_station import plotStationLocationsAxes

from get_data.HadUKGrid.HUKG_monthly_load import load_station_metadata

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
parser.add_argument(
    "--ipdir",
    help="input directory",
    type=str,
    required=False,
    default=None,
)
args = parser.parse_args()

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH

if args.ipdir is None:
    args.ipdir = "%s/RR_station_fits" % LSCRATCH

# Load the station metadata
meta = load_station_metadata(args.src_id)

# Load the fitted data
fitted = {}
for year in range(args.startyear, args.endyear + 1):
    fitted["%04d" % year] = {}
    for month in range(1, 13):
        fitted["%04d" % year]["%02d" % month] = {}
        ffiles = glob.glob("%s/%04d/%02d/*.csv" % (args.ipdir, year, month))
        for fi in ffiles:
            with open(fi, "r") as f:
                reader = csv.reader(f)
                for fl in reader:
                    if len(fl) == 0:
                        continue
                    sid = fl[0]
                    if sid not in fitted["%04d" % year]["%02d" % month].keys():
                        fitted["%04d" % year]["%02d" % month][sid] = {
                            "assimilated": [],
                            "obs": [],
                            "generated": [],
                        }
                    fitted["%04d" % year]["%02d" % month][sid]["assimilated"].append(
                        float(fl[1])
                    )
                    fitted["%04d" % year]["%02d" % month][sid]["obs"].append(
                        float(fl[2])
                    )
                    fitted["%04d" % year]["%02d" % month][sid]["generated"].append(
                        float(fl[3])
                    )


# Convert the fitted data into an apropriate format for plotting
def fitted_to_obs_series(ftd, sid):
    dt = []
    obs = []
    for year in range(args.startyear, args.endyear + 1):
        for month in range(1, 13):
            dt.append(datetime.date(year, month, 15))
            try:
                obs.append(ftd["%04d" % year]["%02d" % month][sid]["obs"][0])
            except KeyError:
                obs.append(np.nan)
    return (dt, obs)


def fitted_to_assimilated_series(ftd, sid):
    dt = []
    assim = []
    dt_mean = []
    assim_mean = []
    for year in range(args.startyear, args.endyear + 1):
        for month in range(1, 13):
            dt_mean.append(datetime.date(year, month, 15))
            try:
                n_cases = len(ftd["%04d" % year]["%02d" % month][sid]["generated"])
            except KeyError:
                assim_mean.append(np.nan)
                continue
            assimilated = [
                ftd["%04d" % year]["%02d" % month][sid]["generated"][i]
                for i in range(n_cases)
                if ftd["%04d" % year]["%02d" % month][sid]["assimilated"][i] == 1
            ]
            if len(assimilated) == 0:
                assim_mean.append(np.nan)
                continue
            assim_mean.append(np.mean(assimilated))
            for case in range(len(assimilated)):
                assim.append(assimilated[case])
                dt.append(datetime.date(year, month, randint(10, 20)))
    return (dt, assim, dt_mean, assim_mean)


def fitted_to_unassimilated_series(ftd, sid):
    dt = []
    assim = []
    dt_mean = []
    unassim_mean = []
    for year in range(args.startyear, args.endyear + 1):
        for month in range(1, 13):
            dt_mean.append(datetime.date(year, month, 15))
            try:
                n_cases = len(ftd["%04d" % year]["%02d" % month][sid]["generated"])
            except KeyError:
                unassim_mean.append(np.nan)
                continue
            unassimilated = [
                ftd["%04d" % year]["%02d" % month][sid]["generated"][i]
                for i in range(n_cases)
                if ftd["%04d" % year]["%02d" % month][sid]["assimilated"][i] == 0
            ]
            if len(unassimilated) == 0:
                unassim_mean.append(np.nan)
                continue
            unassim_mean.append(np.nanmean(unassimilated))
            for case in range(len(unassimilated)):
                assim.append(unassimilated[case])
                dt.append(datetime.date(year, month, randint(10, 20)))
    return (dt, assim, dt_mean, unassim_mean)


# Make the plot
aspect = 3
fsize = 5
fig = Figure(
    figsize=(fsize * aspect, fsize * 1.5),
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

# Main axes with observed assimilated and unassimilated time-series
op = fitted_to_obs_series(fitted, args.src_id)
ap = fitted_to_assimilated_series(fitted, args.src_id)
up = fitted_to_unassimilated_series(fitted, args.src_id)
ymin = np.nanmin([np.nanmin(op[1]), np.nanmin(ap[1]), np.nanmin(up[1])])
ymax = np.nanmax([np.nanmax(op[1]), np.nanmax(ap[1]), np.nanmax(up[1])])
yr = ymax - ymin
ymax += yr / 20
ymin -= yr / 20
# ymax = max(ymax, ymin * -1)
# ymin = min(ymin, ymax * -1)

ax_ts = fig.add_axes(
    [0.045, 0.08 / 1.5 + 0.33, 0.95, 0.9 / 1.5],
    xlim=(
        datetime.date(args.startyear, 1, 15) - datetime.timedelta(days=15),
        datetime.date(args.endyear, 12, 15) + datetime.timedelta(days=15),
    ),
    ylim=(ymin, ymax),
)
ax_ts.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
ax_ts.add_line(
    Line2D(op[0], op[1], linewidth=1, color=(0, 0, 0, 1), alpha=1.0, zorder=50)
)
ax_ts.add_line(
    Line2D(ap[2], ap[3], linewidth=1, color=(0, 0, 1, 1), alpha=1.0, zorder=60)
)
ax_ts.scatter(ap[0], ap[1], s=3, color=(0, 0, 1, 1), alpha=1.0, zorder=40)
ax_ts.add_line(
    Line2D(up[2], up[3], linewidth=1, color=(1, 0, 0, 1), alpha=1.0, zorder=60)
)
ax_ts.scatter(up[0], up[1], s=3, color=(1, 0, 0, 1), alpha=1.0, zorder=40)
ax_ts.set_xticklabels([])  # Share labels with the secondary axes

# Difference a pair of time-series with possibly different time axes
def diff_ts(t1, v1, t2, v2):
    for idx in range(len(v2)):
        try:
            i2 = t1.index(t2[idx])
        except ValueError:
            v2[idx] = np.nan
            continue
        v2[idx] -= v1[i2]
    return v2


# Secondary axes with difference time-series
df_am = diff_ts(op[0], op[1], ap[2], ap[3])
df_a = diff_ts(op[0], op[1], ap[0], ap[1])
df_um = diff_ts(op[0], op[1], up[2], up[3])
df_u = diff_ts(op[0], op[1], up[0], up[1])
yr = ymax - ymin
dymin = np.nanmin((np.nanmin(df_a), np.nanmin(df_u), -1 * yr / 10))
dymax = np.nanmax((np.nanmax(df_a), np.nanmax(df_u), -1 * yr / 10))
dyr = dymax - dymin
dymax += dyr / 20
dymin -= dyr / 20
dymax = max(dymax, dymin * -1)
dymin = min(dymin, dymax * -1)
ax_ss = fig.add_axes(
    [0.045, 0.08 / 1.5, 0.95, 0.45 / 1.5],
    xlim=(
        datetime.date(args.startyear, 1, 15) - datetime.timedelta(days=15),
        datetime.date(args.endyear, 12, 15) + datetime.timedelta(days=15),
    ),
    ylim=(dymin, dymax),
)
ax_ss.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)

ax_ss.add_line(
    Line2D(up[2], df_um, linewidth=1, color=(1, 0, 0, 1), alpha=1.0, zorder=50)
)
ax_ss.scatter(ap[0], df_a, s=3, color=(0, 0, 1, 1), alpha=1.0, zorder=40)
ax_ss.add_line(
    Line2D(ap[2], df_am, linewidth=1, color=(0, 0, 1, 1), alpha=1.0, zorder=50)
)
ax_ss.scatter(up[0], df_u, s=3, color=(1, 0, 1, 0), alpha=1.0, zorder=40)

# Add thumbnail showing station location
axt = fig.add_axes([0.046, 0.75 * 0.67 + 0.33, 0.15 / aspect, 0.10 * 1450 / 900])
plotStationLocationsAxes(
    axt,
    {args.src_id: meta},
    ssize=25000,
    scolour="Red",
    sea_colour=(1.0, 1.0, 1.0, 1.0),
    land_colour=(0.0, 0.0, 0.0, 0.5),
    zorder=150,
)


fig.savefig("station_error_series.png")
