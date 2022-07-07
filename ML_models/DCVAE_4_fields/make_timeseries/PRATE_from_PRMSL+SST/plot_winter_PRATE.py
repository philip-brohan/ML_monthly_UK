#!/usr/bin/env python

# Plot a time-series of UK-region winter precipitation from
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
parser.add_argument("--endyear", type=int, required=False, default=1975)
args = parser.parse_args()

# Load the data
dta = {}
for year in range(args.startyear, args.endyear + 1):
    for month in [1, 2, 12]:
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
dt = list(range(args.startyear, args.endyear + 1))
ny = args.endyear - args.startyear + 1
hukg = [0] * ny
v3 = []
fit = []
for member in range(80):
    v3.append([0] * ny)
    fit.append([0] * ny)
for key in dta:
    year = int(key[:4])
    if (int(key[-2:])) == 12:
        year -= 1
    idx = year - args.startyear
    hukg[idx] += dta[key]["PRATE"]["HUKG"]
    for member in range(80):
        v3[member][idx] += dta[key]["PRATE"]["20CR"][member]
    for member in range(80):
        fit[member][idx] += dta[key]["PRATE"]["Fit"][member]

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
    Rectangle((0, 1), 1, 1, facecolor=(1.0, 1.0, 1.0, 1), fill=True, zorder=1,)
)

ax = fig.add_axes(
    [0.085, 0.07, 0.905, 0.9],
    xlim=(args.startyear - 0.5, args.endyear + 0.5),
    ylim=(0, 450),
)
ax.set_ylabel("Winter Precipitation")

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


fig.savefig("PRATE_winter.png")
