#!/usr/bin/env python

# Plot time-series of training progress

import os
import sys
import math
import numpy as np
import tensorflow as tf
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import glob
import pickle

sys.path.append("%s/." % os.path.dirname(__file__))
from localise import LSCRATCH

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--comparator", help="Comparison model name", type=str, required=False, default=None
)
parser.add_argument(
    "--ymax", help="Y range maximum", type=float, required=False, default=None
)
parser.add_argument(
    "--ymin", help="Y range minimum", type=float, required=False, default=None
)
args = parser.parse_args()


# Load the history
def loadHistory(LSC):
    fl = glob.glob("%s/models/Epoch_*/history.pkl" % LSC)
    fl.sort()
    c_epoch = int(fl[-1][-16:-12])
    fn = "%s/models/Epoch_%04d/history.pkl" % (LSC, c_epoch)
    with open(fn, "rb") as fh:
        history = pickle.load(fh)
    ymax = 0
    ymin = 1000000
    chts = {}
    s_epoch = c_epoch-len(history['PRMSL_train'])+1
    chts["epoch"] = list(range(s_epoch,c_epoch + 1))
    for key in history:
        chts[key] = [math.log(abs(t.numpy())) for t in history[key]]
        ymax = max(ymax, max(chts[key]))
        ymin = min(ymin, min(chts[key]))
    return (chts, ymax, ymin, c_epoch)


(hts, ymax, ymin, epoch) = loadHistory(LSCRATCH)

if args.comparator is not None:
    LSC = "%s/ML_monthly_UK/%s" % (os.getenv("SCRATCH"), args.comparator)
    (chts, cymax, cymin, cepoch) = loadHistory(LSC)
    epoch = max(epoch, cepoch)
    ymax = max(ymax, cymax)
    ymin = min(ymin, cymin)

if args.ymax is not None:
    ymax = args.ymax
if args.ymin is not None:
    ymin = args.ymin


fig = Figure(
    figsize=(15, 11),
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
    "size": 16,
}
matplotlib.rc("font", **font)
axb = fig.add_axes([0, 0, 1, 1])
axb.set_axis_off()
axb.add_patch(
    Rectangle((0, 0), 1, 1, facecolor=(0.95, 0.95, 0.95, 1), fill=True, zorder=1,)
)


def addLine(ax, dta, key, col, z):
    ax.add_line(
        Line2D(dta["epoch"], dta[key], linewidth=2, color=col, alpha=1.0, zorder=z,)
    )


# Top left - PRMSL
ax_prmsl = fig.add_axes(
    [0.055, 0.55, 0.27, 0.4], xlim=(-1, epoch + 1), ylim=(ymin, ymax)
)
ax_prmsl.set_ylabel("PRMSL")
ax_prmsl.set_xlabel("epoch")
addLine(ax_prmsl, hts, "PRMSL_train", (1, 0.5, 0.5, 1), 10)
addLine(ax_prmsl, hts, "PRMSL_test", (1, 0, 0, 1), 20)
if args.comparator is not None:
    addLine(ax_prmsl, chts, "PRMSL_train", (0.5, 0.5, 1, 1), 10)
    addLine(ax_prmsl, chts, "PRMSL_test", (0, 0, 1, 1), 20)

# Bottom left - SST
ax_sst = fig.add_axes([0.055, 0.06, 0.27, 0.4], xlim=(-1, epoch + 1), ylim=(ymin, ymax))
ax_sst.set_ylabel("SST")
ax_sst.set_xlabel("epoch")
addLine(ax_sst, hts, "SST_train", (1, 0.5, 0.5, 1), 10)
addLine(ax_sst, hts, "SST_test", (1, 0, 0, 1), 20)
if args.comparator is not None:
    addLine(ax_sst, chts, "SST_train", (0.5, 0.5, 1, 1), 10)
    addLine(ax_sst, chts, "SST_test", (0, 0, 1, 1), 20)

# Top centre - T2M
ax_t2m = fig.add_axes([0.385, 0.55, 0.27, 0.4], xlim=(-1, epoch + 1), ylim=(ymin, ymax))
ax_t2m.set_ylabel("T2M")
ax_t2m.set_xlabel("epoch")
addLine(ax_t2m, hts, "T2M_train", (1, 0.5, 0.5, 1), 10)
addLine(ax_t2m, hts, "T2M_test", (1, 0, 0, 1), 20)
if args.comparator is not None:
    addLine(ax_t2m, chts, "T2M_train", (0.5, 0.5, 1, 1), 10)
    addLine(ax_t2m, chts, "T2M_test", (0, 0, 1, 1), 20)

# Bottom centre - PRATE
ax_prate = fig.add_axes(
    [0.385, 0.06, 0.27, 0.4], xlim=(-1, epoch + 1), ylim=(ymin, ymax)
)
ax_prate.set_ylabel("PRATE")
ax_prate.set_xlabel("epoch")
addLine(ax_prate, hts, "PRATE_train", (1, 0.5, 0.5, 1), 10)
addLine(ax_prate, hts, "PRATE_test", (1, 0, 0, 1), 20)
if args.comparator is not None:
    addLine(ax_prate, chts, "PRATE_train", (0.5, 0.5, 1, 1), 10)
    addLine(ax_prate, chts, "PRATE_test", (0, 0, 1, 1), 20)

# Top right - logpz
ax_lpz = fig.add_axes([0.715, 0.55, 0.27, 0.4], xlim=(-1, epoch + 1), ylim=(ymin, ymax))
ax_lpz.set_ylabel("logpz")
ax_lpz.set_xlabel("epoch")
addLine(ax_lpz, hts, "logpz_train", (1, 0.5, 0.5, 1), 10)
addLine(ax_lpz, hts, "logpz_test", (1, 0, 0, 1), 20)
if args.comparator is not None:
    addLine(ax_lpz, chts, "logpz_train", (0.5, 0.5, 1, 1), 10)
    addLine(ax_lpz, chts, "logpz_test", (0, 0, 1, 1), 20)

# Bottom right - logqz_x
ax_lqz = fig.add_axes([0.715, 0.06, 0.27, 0.4], xlim=(-1, epoch + 1), ylim=(ymin, ymax))
ax_lqz.set_ylabel("logqz_x")
ax_lqz.set_xlabel("epoch")
addLine(ax_lqz, hts, "logqz_x_train", (1, 0.5, 0.5, 1), 10)
addLine(ax_lqz, hts, "logqz_x_test", (1, 0, 0, 1), 20)
if args.comparator is not None:
    addLine(ax_lqz, chts, "logqz_x_train", (0.5, 0.5, 1, 1), 10)
    addLine(ax_lqz, chts, "logqz_x_test", (0, 0, 1, 1), 20)


# Output as png
fig.savefig("training_progress.png")
