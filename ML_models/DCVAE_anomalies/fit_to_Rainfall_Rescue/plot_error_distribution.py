#!/usr/bin/env python
# Plot a histogram of pre-calculated station errors

import os
import sys
import csv
import numpy as np
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--src_id", help="Station numeric ID", type=str, required=False, default=None
)
parser.add_argument(
    "--startyear", help="First year to plot", type=int, required=False, default=1677
)
parser.add_argument(
    "--endyear", help="Last year to plot", type=int, required=False, default=1960
)
parser.add_argument(
    "--top_n", help="List top n stations", type=int, required=False, default=10
)
parser.add_argument(
    "--nbins", help="No. of bins in histogram", type=int, required=False, default=250
)
args = parser.parse_args()

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH

# Load the errors
accum = []
topN = []  # keep track of the args.top_n errors
for year in range(args.startyear, args.endyear + 1):
    for month in range(1, 13):
        ofile = "%s/RR_station_error_probs/%04d/%02d.csv" % (LSCRATCH, year, month)
        with open(ofile, "r") as f:
            reader = csv.reader(f)
            for fl in reader:
                if len(fl) == 0:
                    continue
                if args.src_id is not None and fl[0] != args.src_id:
                    continue
                for i in range(1, 5):
                    fl[i] = float(fl[i])
                accum.append(fl[4])
                if len(topN) < args.top_n:
                    topN.append([year, month] + fl)
                    topN.sort(key=lambda x: abs(x[6]), reverse=True)
                if abs(fl[4]) > abs(topN[-1][6]):
                    topN[-1] = [year, month] + fl
                    topN.sort(key=lambda x: abs(x[6]), reverse=True)

fig = Figure(
    figsize=(10, 10),
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

xmax = np.nanmax(accum)
xmin = np.nanmin(accum)
xrange = xmax - xmin
xmax += xrange / 20
xmin -= xrange / 20
xmax = max(xmax, xmin * -1)
xmin = min(xmin, xmax * -1)
ax_tsh = fig.add_axes([0.1, 0.1, 0.88, 0.88], xlim=(xmin, xmax))
ax_tsh.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
ax_tsh.hist(
    accum,
    bins=args.nbins,
    color="red",
    histtype="bar",
    density=True,
    orientation="vertical",
    log=True,
    alpha=1.0,
    zorder=50,
)

fig.savefig("error_histogram.png")

for case in topN:
    print("%04d-%02d %s %6.2f %6.2f %6.2f %6.2f" % tuple(case))
