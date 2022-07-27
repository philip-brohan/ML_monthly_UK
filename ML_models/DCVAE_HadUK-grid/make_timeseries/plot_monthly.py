#!/usr/bin/env python

# Plot a time-series of UK-region precipitation from
# HadUK-grid, 20CRv3 and the ML reanalysis fit.

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
parser.add_argument("--startyear", type=int, required=False, default=1995)
parser.add_argument("--startmonth", type=int, required=False, default=1)
parser.add_argument("--endyear", type=int, required=False, default=2015)
parser.add_argument("--endmonth", type=int, required=False, default=12)
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
        if (year == args.startyear and month < args.startmonth) or (
            year == args.endyear and month == args.endmonth
        ):
            continue
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
dt = []
hukg = []
v3 = []
fit = []
for member in range(7):
    v3.append([])
    fit.append([])
for key in dta:
    dt.append(datetime.date(int(key[:4]), int(key[-2:]), 15))
    for member in range(7):
        v3[member].append(dta[key]["PRATE"]["Orig"][member])
    for member in range(7):
        fit[member].append(dta[key]["PRATE"]["Fit"][member])

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
font = {"family": "sans-serif", "sans-serif": "Arial", "weight": "normal", "size": 14}
matplotlib.rc("font", **font)
axb = fig.add_axes([0, 0, 1, 1])
axb.add_patch(
    Rectangle((0, 1), 1, 1, facecolor=(1.0, 1.0, 1.0, 1), fill=True, zorder=1,)
)

ax = fig.add_axes(
    [0.08, 0.07, 0.91, 0.9],
    xlim=(
        datetime.date(args.startyear, args.startmonth, 1),
        datetime.date(args.endyear, args.endmonth, 28),
    ),
    ylim=(0, 250),
)
ax.set_ylabel("Precipitation")


# Original
for m in range(7):
    ax.add_line(
        Line2D(dt, v3[m], linewidth=1, color=(0, 0, 0, 1), alpha=1.0, zorder=50)
    )

# Fit
for m in range(7):
    ax.add_line(
        Line2D(dt, fit[m], linewidth=0.5, color=(1, 0, 0, 1), alpha=0.5, zorder=60)
    )


fig.savefig("monthly.png")
