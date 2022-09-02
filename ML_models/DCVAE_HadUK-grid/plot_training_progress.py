#!/usr/bin/env python

# Plot time-series of training progress

import os
import sys
import math
import numpy as np
import tensorflow as tf

from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.framework import tensor_util

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.append("%s/." % os.path.dirname(__file__))
from localise import LSCRATCH

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--comparator", help="Comparison model name", type=str, required=False, default=None
)
parser.add_argument(
    "--rscale",
    help="Scale RMS losses in comparator",
    type=float,
    required=False,
    default=1.0,
)
parser.add_argument(
    "--ymax", help="Y range maximum", type=float, required=False, default=None
)
parser.add_argument(
    "--ymin", help="Y range minimum", type=float, required=False, default=None
)
args = parser.parse_args()


# Load the history
def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)


def loadHistory(LSC):
    history = {}
    summary_dir = "%s/models/Training_log" % LSC
    for filename in os.listdir(summary_dir):
        path = os.path.join(summary_dir, filename)
        for event in my_summary_iterator(path):
            for value in event.summary.value:
                t = tensor_util.MakeNdarray(value.tensor)
                if not value.tag in history.keys():
                    history[value.tag] = []
                if len(history[value.tag]) < event.step + 1:
                    history[value.tag].extend(
                        [0.0] * (event.step + 1 - len(history[value.tag]))
                    )
                history[value.tag][event.step] = t.item()

    ymax = 0
    ymin = 1000000
    hts = {}
    n_epochs = len(history["Train_loss"])
    hts["epoch"] = list(range(n_epochs))[1:]
    for key in history:
        hts[key] = [math.log(abs(t)) for t in history[key][1:]]
        ymax = max(ymax, max(hts[key]))
        ymin = min(ymin, min(hts[key]))

    return (hts, ymax, ymin, n_epochs)


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


def addLine(ax, dta, key, col, z, rscale=1):
    ax.add_line(
        Line2D(
            dta["epoch"],
            np.array(dta[key]) * rscale,
            linewidth=2,
            color=col,
            alpha=1.0,
            zorder=z,
        )
    )


# Top left - PRMSL
ax_prmsl = fig.add_axes(
    [0.055, 0.55, 0.27, 0.4], xlim=(-1, epoch + 1), ylim=(ymin, ymax)
)
ax_prmsl.set_ylabel("PRMSL")
ax_prmsl.set_xlabel("epoch")
ax_prmsl.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
addLine(ax_prmsl, hts, "Train_PRMSL", (1, 0.5, 0.5, 1), 10)
addLine(ax_prmsl, hts, "Test_PRMSL", (1, 0, 0, 1), 20)
if args.comparator is not None:
    addLine(ax_prmsl, chts, "PRMSL_train", (0.5, 0.5, 1, 1), 10, rscale=args.rscale)
    addLine(ax_prmsl, chts, "PRMSL_test", (0, 0, 1, 1), 20, rscale=args.rscale)

# Bottom left - SST
ax_sst = fig.add_axes([0.055, 0.06, 0.27, 0.4], xlim=(-1, epoch + 1), ylim=(ymin, ymax))
ax_sst.set_ylabel("SST")
ax_sst.set_xlabel("epoch")
ax_sst.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
addLine(ax_sst, hts, "Train_SST", (1, 0.5, 0.5, 1), 10)
addLine(ax_sst, hts, "Test_SST", (1, 0, 0, 1), 20)
if args.comparator is not None:
    addLine(ax_sst, chts, "SST_train", (0.5, 0.5, 1, 1), 10, rscale=args.rscale)
    addLine(ax_sst, chts, "SST_test", (0, 0, 1, 1), 20, rscale=args.rscale)

# Top centre - T2M
ax_t2m = fig.add_axes([0.385, 0.55, 0.27, 0.4], xlim=(-1, epoch + 1), ylim=(ymin, ymax))
ax_t2m.set_ylabel("T2M")
ax_t2m.set_xlabel("epoch")
ax_t2m.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
addLine(ax_t2m, hts, "Train_T2M", (1, 0.5, 0.5, 1), 10)
addLine(ax_t2m, hts, "Test_T2M", (1, 0, 0, 1), 20)
if args.comparator is not None:
    addLine(ax_t2m, chts, "T2M_train", (0.5, 0.5, 1, 1), 10, rscale=args.rscale)
    addLine(ax_t2m, chts, "T2M_test", (0, 0, 1, 1), 20, rscale=args.rscale)

# Bottom centre - PRATE
ax_prate = fig.add_axes(
    [0.385, 0.06, 0.27, 0.4], xlim=(-1, epoch + 1), ylim=(ymin, ymax)
)
ax_prate.set_ylabel("PRATE")
ax_prate.set_xlabel("epoch")
ax_prate.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
addLine(ax_prate, hts, "Train_PRATE", (1, 0.5, 0.5, 1), 10)
addLine(ax_prate, hts, "Test_PRATE", (1, 0, 0, 1), 20)
if args.comparator is not None:
    addLine(ax_prate, chts, "PRATE_train", (0.5, 0.5, 1, 1), 10, rscale=args.rscale)
    addLine(ax_prate, chts, "PRATE_test", (0, 0, 1, 1), 20, rscale=args.rscale)

# Top right - logpz
ax_lpz = fig.add_axes([0.715, 0.55, 0.27, 0.4], xlim=(-1, epoch + 1), ylim=(ymin, ymax))
ax_lpz.set_ylabel("logpz")
ax_lpz.set_xlabel("epoch")
ax_lpz.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
addLine(ax_lpz, hts, "Train_logpz", (1, 0.5, 0.5, 1), 10)
addLine(ax_lpz, hts, "Test_logpz", (1, 0, 0, 1), 20)
if args.comparator is not None:
    addLine(ax_lpz, chts, "logpz_train", (0.5, 0.5, 1, 1), 10)
    addLine(ax_lpz, chts, "logpz_test", (0, 0, 1, 1), 20)

# Bottom right - logqz_x
ax_lqz = fig.add_axes([0.715, 0.06, 0.27, 0.4], xlim=(-1, epoch + 1), ylim=(ymin, ymax))
ax_lqz.set_ylabel("logqz_x")
ax_lqz.set_xlabel("epoch")
ax_lqz.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
addLine(ax_lqz, hts, "Train_logqz_x", (1, 0.5, 0.5, 1), 10)
addLine(ax_lqz, hts, "Test_logqz_x", (1, 0, 0, 1), 20)
if args.comparator is not None:
    addLine(ax_lqz, chts, "logqz_x_train", (0.5, 0.5, 1, 1), 10)
    addLine(ax_lqz, chts, "logqz_x_test", (0, 0, 1, 1), 20)


# Output as png
fig.savefig("training_progress.png")
