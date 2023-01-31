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
parser.add_argument(
    "--psize", help="Scatter point size", type=float, required=False, default=3.0
)
parser.add_argument(
    "--offsets",
    help="Show version with station offsets",
    type=bool,
    required=False,
    default=False,
)
args = parser.parse_args()

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH

if args.ipdir is None:
    if args.offsets:
        args.ipdir = "%s/RR_station_fits_offsets" % LSCRATCH
    else:
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
                a_diffs = None
                u_diffs = None
                for fl in reader:
                    if len(fl) == 0:
                        continue
                    sid = fl[0]
                    if sid == args.src_id:
                        if sid not in fitted["%04d" % year]["%02d" % month].keys():
                            fitted["%04d" % year]["%02d" % month][sid] = {
                                "assimilated": [],
                                "obs": [],
                                "generated": [],
                                "a_diffs": [],
                                "u_diffs": [],
                            }
                        fitted["%04d" % year]["%02d" % month][sid][
                            "assimilated"
                        ].append(int(fl[1]))
                        fitted["%04d" % year]["%02d" % month][sid]["obs"].append(
                            float(fl[2])
                        )
                        fitted["%04d" % year]["%02d" % month][sid]["generated"].append(
                            float(fl[3])
                        )
                    else:
                        if a_diffs is None:
                            a_diffs = []
                            u_diffs = []
                        if int(fl[1]) == 1:
                            a_diffs.append((float(fl[3]) - float(fl[2])) ** 2)
                        else:
                            u_diffs.append((float(fl[3]) - float(fl[2])) ** 2)
                try:
                    fitted["%04d" % year]["%02d" % month][args.src_id][
                        "a_diffs"
                    ].append(np.sqrt(np.nanmean(a_diffs)))
                    fitted["%04d" % year]["%02d" % month][args.src_id][
                        "u_diffs"
                    ].append(np.sqrt(np.nanmean(u_diffs)))
                except Exception:
                    pass

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


def fitted_to_assimilated_series(ftd, sid, flag=1):
    dt = []
    assim = []
    a_diffs = []
    u_diffs = []
    dt_mean = []
    assim_mean = []
    ad_mean = []
    ud_mean = []
    for year in range(args.startyear, args.endyear + 1):
        for month in range(1, 13):
            dt_mean.append(datetime.date(year, month, 15))
            try:
                n_cases = len(ftd["%04d" % year]["%02d" % month][sid]["generated"])
            except KeyError:
                assim_mean.append(np.nan)
                ad_mean.append(np.nan)
                ud_mean.append(np.nan)
                continue
            assimilated = [
                ftd["%04d" % year]["%02d" % month][sid]["generated"][i]
                for i in range(n_cases)
                if ftd["%04d" % year]["%02d" % month][sid]["assimilated"][i] == flag
            ]
            ad = [
                ftd["%04d" % year]["%02d" % month][sid]["a_diffs"][i]
                for i in range(n_cases)
                if ftd["%04d" % year]["%02d" % month][sid]["assimilated"][i] == flag
            ]
            ud = [
                ftd["%04d" % year]["%02d" % month][sid]["u_diffs"][i]
                for i in range(n_cases)
                if ftd["%04d" % year]["%02d" % month][sid]["assimilated"][i] == flag
            ]
            if len(assimilated) == 0:
                assim_mean.append(np.nan)
                ad_mean.append(np.nan)
                ud_mean.append(np.nan)
                continue
            assim_mean.append(np.mean(assimilated))
            ad_mean.append(np.mean(ad))
            ud_mean.append(np.mean(ud))
            for case in range(len(assimilated)):
                assim.append(assimilated[case])
                a_diffs.append(ad[case])
                u_diffs.append(ud[case])
                dt.append(datetime.date(year, month, randint(10, 20)))
    return (dt, assim, dt_mean, assim_mean, a_diffs, u_diffs, ad_mean, ud_mean)


def fitted_to_unassimilated_series(ftd, sid):
    dt = []
    unassim = []
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
                unassim.append(unassimilated[case])
                dt.append(datetime.date(year, month, randint(10, 20)))
    return (dt, unassim, dt_mean, unassim_mean)


def difference_to_series(ftd):
    dt = []
    d_assim = []
    d_unassim = []
    for year in range(args.startyear, args.endyear + 1):
        for month in range(1, 13):
            dt.append(datetime.date(year, month, 15))
            try:
                d_assim.append(
                    np.sqrt(
                        ftd["%04d" % year]["%02d" % month]["diffs"]["a_diff"]
                        / ftd["%04d" % year]["%02d" % month]["diffs"]["a_count"]
                    )
                )
            except KeyError:
                d_assim.append(np.nan)
            try:
                d_unassim.append(
                    np.sqrt(
                        ftd["%04d" % year]["%02d" % month]["diffs"]["u_diff"]
                        / ftd["%04d" % year]["%02d" % month]["diffs"]["u_count"]
                    )
                )
            except KeyError:
                d_unassim.append(np.nan)
    return (dt, d_assim, d_unassim)


# Make the plot
aspect = 4
fsize = 5
fig = Figure(
    figsize=(fsize * aspect, fsize * 2.5),
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
ap = fitted_to_assimilated_series(fitted, args.src_id, flag=1)
up = fitted_to_assimilated_series(fitted, args.src_id, flag=0)
ymin = np.nanmin([np.nanmin(op[1]), np.nanmin(ap[1]), np.nanmin(up[1])])
ymax = np.nanmax([np.nanmax(op[1]), np.nanmax(ap[1]), np.nanmax(up[1])])
yr = ymax - ymin
ymax += yr / 20
ymin -= yr / 20
# ymax = max(ymax, ymin * -1)
# ymin = min(ymin, ymax * -1)

ax_ts = fig.add_axes(
    [0.045, 0.08 / 2.5 + 3 / 5, 0.95 * 0.75, 0.9 / 2.5],
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
ax_ts.scatter(ap[0], ap[1], s=args.psize, color=(0, 0, 1, 1), alpha=1.0, zorder=40)
ax_ts.add_line(
    Line2D(up[2], up[3], linewidth=1, color=(1, 0, 0, 1), alpha=1.0, zorder=60)
)
ax_ts.scatter(up[0], up[1], s=args.psize, color=(1, 0, 0, 1), alpha=1.0, zorder=40)
ax_ts.set_xticklabels([])  # Share labels with the bottom axes

# Add a histogram to the right
ax_tsh = fig.add_axes(
    [0.045 + 0.95 * 0.75 + 0.01, 0.08 / 2.5 + 3 / 5, 0.95 * 0.25 - 0.01, 0.9 / 2.5],
    ylim=(ymin, ymax),
)
ax_tsh.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
ax_tsh.set_xticklabels([])
ax_tsh.set_yticklabels([])
ax_tsh.hist(
    (op[1], ap[1], up[1]),
    bins=np.arange(ymin, ymax, (ymax - ymin) / 20),
    color=((0, 0, 0), (0, 0, 1), (1, 0, 0)),
    histtype="bar",
    density=True,
    orientation="horizontal",
    alpha=1.0,
    zorder=50,
)


# Difference a pair of time-series with possibly different time axes
def diff_ts(t1, v1, t2, v2):
    for idx in range(len(v2)):
        try:
            i2 = t1.index(datetime.date(t2[idx].year, t2[idx].month, 15))
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
dymax = np.nanmax((np.nanmax(df_a), np.nanmax(df_u), 1 * yr / 10))
dyr = dymax - dymin
dymax += dyr / 20
dymin -= dyr / 20
dymax = max(dymax, dymin * -1)
dymin = min(dymin, dymax * -1)
ax_ss = fig.add_axes(
    [0.045, 0.08 / 2.5 + 2 / 5, 0.95 * 0.75, 0.45 / 2.5],
    xlim=(
        datetime.date(args.startyear, 1, 15) - datetime.timedelta(days=15),
        datetime.date(args.endyear, 12, 15) + datetime.timedelta(days=15),
    ),
    ylim=(dymin, dymax),
)
ax_ss.set_xticklabels([])  # Share labels with the bottom axes
ax_ss.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)

ax_ss.add_line(
    Line2D(up[2], df_um, linewidth=1, color=(1, 0, 0, 1), alpha=1.0, zorder=50)
)
ax_ss.scatter(ap[0], df_a, s=args.psize, color=(0, 0, 1, 1), alpha=0.5, zorder=40)
ax_ss.add_line(
    Line2D(ap[2], df_am, linewidth=1, color=(0, 0, 1, 1), alpha=1.0, zorder=50)
)
ax_ss.scatter(up[0], df_u, s=args.psize, color=(1, 0, 0, 1), alpha=0.5, zorder=40)

# Add a histogram to the right
ax_ssh = fig.add_axes(
    [0.045 + 0.95 * 0.75 + 0.01, 0.08 / 2.5 + 2 / 5, 0.95 * 0.25 - 0.01, 0.45 / 2.5],
    ylim=(dymin, dymax),
)
ax_ssh.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
ax_ssh.set_xticklabels([])
ax_ssh.set_yticklabels([])
ax_ssh.hist(
    (df_a, df_u),
    bins=np.arange(dymin, dymax, (dymax - dymin) / 20),
    color=((0, 0, 1), (1, 0, 0)),
    histtype="bar",
    density=True,
    orientation="horizontal",
    alpha=1.0,
    zorder=50,
)

# Third axis with other station errors - doesn't matter whether selected station assimilated or not - so average that
all_assim_error = [(ap[6][i] + up[6][i]) / 2 for i in range(len(ap[6]))]
all_unassim_error = [(ap[7][i] + up[7][i]) / 2 for i in range(len(up[6]))]
dymax = np.nanmax((np.nanmax(ap[4] + up[4]), np.nanmax(ap[5] + up[5])))
dymax *= 1.1
dymin = 0
ax_se = fig.add_axes(
    [0.045, 0.08 / 2.5 + 1 / 5, 0.95 * 0.75, 0.45 / 2.5],
    xlim=(
        datetime.date(args.startyear, 1, 15) - datetime.timedelta(days=15),
        datetime.date(args.endyear, 12, 15) + datetime.timedelta(days=15),
    ),
    ylim=(dymin, dymax),
)
ax_se.set_xticklabels([])  # Share labels with the bottom axes
ax_se.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)

ax_se.add_line(
    Line2D(
        ap[2], all_assim_error, linewidth=1, color=(0, 0, 1, 1), alpha=1.0, zorder=50
    )
)
ax_se.scatter(ap[0], ap[4], s=args.psize, color=(0, 0, 1, 1), alpha=0.5, zorder=40)
ax_se.scatter(up[0], up[4], s=args.psize, color=(0, 0, 1, 1), alpha=0.5, zorder=40)
ax_se.add_line(
    Line2D(
        ap[2], all_unassim_error, linewidth=1, color=(1, 0, 0, 1), alpha=1.0, zorder=50
    )
)
ax_se.scatter(ap[0], ap[5], s=args.psize, color=(1, 0, 0, 1), alpha=0.5, zorder=40)
ax_se.scatter(up[0], up[5], s=args.psize, color=(1, 0, 0, 1), alpha=0.5, zorder=40)
ax_seh = fig.add_axes(
    [0.045 + 0.95 * 0.75 + 0.01, 0.08 / 2.5 + 1 / 5, 0.95 * 0.25 - 0.01, 0.45 / 2.5],
    ylim=(dymin, dymax),
)
ax_seh.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
ax_seh.set_xticklabels([])
ax_seh.set_yticklabels([])
ax_seh.hist(
    (ap[4] + up[4], ap[5] + up[5]),
    bins=np.arange(dymin, dymax, (dymax - dymin) / 20),
    color=((0, 0, 1), (1, 0, 0)),
    histtype="bar",
    density=True,
    orientation="horizontal",
    alpha=1.0,
    zorder=50,
)

# Final axis with difference of errors
diff_assim_error = [ap[7][i] - up[7][i] for i in range(len(ap[6]))]
diff_unassim_error = [ap[6][i] - up[6][i] for i in range(len(up[6]))]
dymax = np.nanmax((np.nanmax(diff_assim_error), np.nanmax(diff_unassim_error)))
dymin = np.nanmin((np.nanmin(diff_assim_error), np.nanmin(diff_unassim_error)))
dyr = dymax - dymin
dymax += dyr / 20
dymin -= dyr / 20
dymax = max(dymax, dymin * -1)
dymin = min(dymin, dymax * -1)
ax_sd = fig.add_axes(
    [0.045, 0.08 / 2.5 + 0 / 5, 0.95 * 0.75, 0.45 / 2.5],
    xlim=(
        datetime.date(args.startyear, 1, 15) - datetime.timedelta(days=15),
        datetime.date(args.endyear, 12, 15) + datetime.timedelta(days=15),
    ),
    ylim=(dymin, dymax),
)
ax_sd.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)

ax_sd.add_line(
    Line2D(
        ap[2], diff_assim_error, linewidth=1, color=(0, 0, 1, 1), alpha=1.0, zorder=50
    )
)
ax_sd.add_line(
    Line2D(
        ap[2], diff_unassim_error, linewidth=1, color=(1, 0, 0, 1), alpha=1.0, zorder=50
    )
)
ax_sdh = fig.add_axes(
    [0.045 + 0.95 * 0.75 + 0.01, 0.08 / 2.5 + 0 / 5, 0.95 * 0.25 - 0.01, 0.45 / 2.5],
    ylim=(dymin, dymax),
)
ax_sdh.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
ax_sdh.set_xticklabels([])
ax_sdh.set_yticklabels([])
ax_sdh.hist(
    (diff_assim_error, diff_unassim_error),
    bins=np.arange(dymin, dymax, (dymax - dymin) / 20),
    color=((0, 0, 1), (1, 0, 0)),
    histtype="bar",
    density=True,
    orientation="horizontal",
    alpha=1.0,
    zorder=50,
)

# Add thumbnail showing station location
axt = fig.add_axes(
    [1 - 0.15 / aspect - 0.006, 0.75 * 0.67 + 0.36, 0.15 / aspect, 0.10 * 1450 / 900]
)
plotStationLocationsAxes(
    axt,
    {args.src_id: meta},
    ssize=35000,
    scolour="Red",
    sea_colour=(0.8, 0.8, 0.8, 1.0),
    land_colour=(0.4, 0.4, 0.4, 1.0),
    zorder=150,
)

if args.offsets:
    fig.savefig("station_error_series_offsets.png")
else:
    fig.savefig("station_error_series.png")
