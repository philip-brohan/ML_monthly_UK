#!/usr/bin/env python

# Plot a time-series from HadUK-grid, 20CRv3 and the ML reanalysis fit.

import os
import sys
import numpy as np
import datetime
import pickle

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=100)
parser.add_argument("--startyear", type=int, required=False, default=1884)
parser.add_argument("--endyear", type=int, required=False, default=2014)
parser.add_argument(
    "--PRMSL", help="Fit to PRMSL?", dest="PRMSL", default=False, action="store_true"
)
parser.add_argument(
    "--SST", help="Fit to SST?", dest="SST", default=False, action="store_true"
)
parser.add_argument(
    "--TMP2m", help="Fit to TMP2m?", dest="TMP2m", default=False, action="store_true"
)
parser.add_argument(
    "--PRATE", help="Fit to PRATE?", dest="PRATE", default=False, action="store_true"
)
args = parser.parse_args()

cName = "constraints"
for constraint in ["PRMSL", "PRATE", "TMP2m", "SST"]:
    if vars(args)[constraint]:
        cName += "_%s" % constraint

# Load the data
dta = {}
for year in range(args.startyear, args.endyear + 1):
    for month in range(1, 13):
        mfn = ("%s/UK_averages/%s/%04d/%04d/%02d.pkl") % (
            LSCRATCH,
            cName,
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
orig = []
fit = []
for member in range(7):
    orig.append([0] * ny)
    fit.append([0] * ny)
for key in dta:
    year = int(key[:4])
    idx = year - args.startyear
    for member in range(7):
        orig[member][idx] += dta[key]["PRATE"]["Orig"][member]
    for member in range(7):
        fit[member][idx] += dta[key]["PRATE"]["Fit"][member]

# Plot the resulting array as a set of line graphs
fig = Figure(
    figsize=(10, 3),
    dpi=300,
    facecolor=(0.5, 0.5, 0.5, 1),
    edgecolor=None,
    linewidth=0.0,
    frameon=False,
    subplotpars=None,
    tight_layout=None,
)
canvas = FigureCanvas(fig)
font = {"family": "sans-serif", "sans-serif": "Arial", "weight": "normal", "size": 14}
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

ax = fig.add_axes(
    [0.085, 0.11, 0.91, 0.86],
    xlim=(args.startyear - 0.5, args.endyear + 0.5),
    ylim=(700, 1500),
)
ax.set_ylabel("Annual Precipitation")

# Orig
for m in range(7):
    ax.add_line(
        Line2D(dt, orig[m], linewidth=1, color=(0, 0, 0, 1), alpha=1, zorder=50)
    )

# Fit
for m in range(7):
    ax.add_line(
        Line2D(dt, fit[m], linewidth=0.5, color=(1, 0, 0, 1), alpha=0.5, zorder=60)
    )


fig.savefig("annual.png")
