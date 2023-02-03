#!/usr/bin/env python

# Plot a time-series of UK-region T2m from
# HadUK-grid, 20CRv3 and the ML reanalysis fit.

import os
import numpy as np
import datetime
import pickle

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=990)
parser.add_argument("--startyear", type=int, required=False, default=1969)
parser.add_argument("--startmonth", type=int, required=False, default=1)
parser.add_argument("--endyear", type=int, required=False, default=1975)
parser.add_argument("--endmonth", type=int, required=False, default=12)
args = parser.parse_args()

# Load the data
dta = {}
for year in range(args.startyear, args.endyear + 1):
    for month in range(1, 13):
        if (year == args.startyear and month < args.startmonth) or (
            year == args.endyear and month == args.endmonth
        ):
            continue
        mfn = "%s/ML_monthly_UK/UK_averages/%04d/%04d/%02d.pkl" % (
            os.getenv("SCRATCH"),
            args.epoch,
            year,
            month,
        )
        if not os.path.exists(mfn):
            raise Exception("Missing data file %s" % mfn)
        with open(mfn, "rb") as infile:
            dta["%04d%02d" % (year, month)] = pickle.load(infile)

# Rearange the data into time-series
dt = []
hukg = []
v3 = []
fit = []
for member in range(80):
    v3.append([])
    fit.append([])
for key in dta:
    dt.append(datetime.date(int(key[:4]), int(key[-2:]), 15))
    hukg.append(dta[key]["T2m"]["HUKG"])
    for member in range(80):
        v3[member].append(dta[key]["T2m"]["20CR"][member])
    for member in range(80):
        fit[member].append(dta[key]["T2m"]["Fit"][member])

# Plot the resulting array as a set of line graphs
fig = Figure(
    figsize=(10, 5),
    dpi=300,
    facecolor=(0.5, 0.5, 0.5, 1),
    edgecolor=None,
    linewidth=0.0,
    frameon=False,
    subplotpars=None,
    tight_layout=None,
)
canvas = FigureCanvas(fig)
font = {"family": "sans-serif", "sans-serif": "Arial", "weight": "normal", "size": 16}
matplotlib.rc("font", **font)
axb = fig.add_axes([0, 0, 1, 1])
axb.add_patch(
    Rectangle(
        (0, 1),
        1,
        1,
        facecolor=(1.0, 1.0, 1.0, 1),
        fill=True,
        zorder=1,
    )
)

ax = fig.add_axes(
    [0.085, 0.07, 0.905, 0.9],
    xlim=(
        datetime.date(args.startyear, args.startmonth, 1),
        datetime.date(args.endyear, args.endmonth, 28),
    ),
    ylim=(0, 20),
)
ax.set_ylabel("T2m")

# HadUK-grid
ax.add_line(Line2D(dt, hukg, linewidth=1.0, color=(0, 0, 0, 1), alpha=1.0, zorder=100))

# 20CRv3
for m in range(80):
    ax.add_line(
        Line2D(dt, v3[m], linewidth=0.5, color=(1, 0, 0, 1), alpha=0.1, zorder=50)
    )

# Fit
for m in range(80):
    ax.add_line(
        Line2D(dt, fit[m], linewidth=0.5, color=(0, 0, 1, 1), alpha=0.1, zorder=60)
    )


fig.savefig("TMP2m_monthly.png")
