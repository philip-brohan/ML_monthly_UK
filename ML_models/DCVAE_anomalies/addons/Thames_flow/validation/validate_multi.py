#!/usr/bin/env python

# Plot validation statistics for all the test cases

import os
import sys
import numpy as np
import tensorflow as tf
import datetime
from statistics import mean
import cmocean

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=250)
parser.add_argument(
    "--startyear", help="First year to plot", type=int, required=False, default=None
)
parser.add_argument(
    "--endyear", help="Last year to plot", type=int, required=False, default=None
)
parser.add_argument(
    "--training",
    help="Use training months (not test months)",
    dest="training",
    default=False,
    action="store_true",
)
args = parser.parse_args()

sys.path.append("%s/.." % os.path.dirname(__file__))
from generatorModel import GeneratorM
from makeDataset import getDataset

sys.path.append("%s/../../.." % os.path.dirname(__file__))
from localise import LSCRATCH
from localise import TSOURCE


# Set up the test data
purpose = "test"
if args.training:
    purpose = "training"
testData = getDataset(
    purpose=purpose, startyear=args.startyear, endyear=args.endyear, shuffle=False
)
testData = testData.batch(1)

generator = GeneratorM()
weights_dir = ("%s/flow_models/Epoch_%04d") % (
    LSCRATCH,
    args.epoch,
)
load_status = generator.load_weights("%s/ckpt" % weights_dir)
load_status.assert_existing_objects_matched()


# Get target and encoded statistics for one test case
def compute_stats(model, x):
    # get the date from the filename tensor
    fn = x[2].numpy()[0]
    year = int(fn[:4])
    month = int(fn[5:7])
    dtp = datetime.date(year, month, 15)
    # Pass the test field through the autoencoder
    generated = model.call(x[0], training=False)

    stats = {}
    stats["dtp"] = dtp
    stats["Target"] = np.argmax(x[1].numpy()) + np.random.random()
    stats["Model"] = np.argmax(generated[0].numpy()) + np.random.random()
    # print(generated[0])
    return stats


all_stats = {}
for case in testData:
    stats = compute_stats(generator, case)
    for key in stats.keys():
        if key in all_stats:
            all_stats[key].append(stats[key])
        else:
            all_stats[key] = [stats[key]]


# Make the plot
fig = Figure(
    figsize=(15, 5),
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
    "size": 8,
}
matplotlib.rc("font", **font)
axb = fig.add_axes([0, 0, 1, 1])
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


def plot_var(ts, t, m):
    ymin = min(min(t), min(m))
    ymax = max(max(t), max(m))
    ypad = (ymax - ymin) * 0.1
    if ypad == 0:
        ypad = 1
    ax_ts = fig.add_axes(
        [0.05, 0.05, 0.6, 0.9],
        xlim=(
            ts[0] - datetime.timedelta(days=15),
            ts[-1] + datetime.timedelta(days=15),
        ),
        ylim=(ymin - ypad, ymax + ypad),
    )
    ax_ts.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
    ax_ts.add_line(Line2D(ts, t, linewidth=2, color=(0, 0, 0, 1), alpha=1.0, zorder=50))
    ax_ts.add_line(Line2D(ts, m, linewidth=2, color=(1, 0, 0, 1), alpha=1.0, zorder=60))
    ax_sc = fig.add_axes(
        [0.69, 0.05, 0.3, 0.9],
        xlim=(ymin - ypad, ymax + ypad),
        ylim=(ymin - ypad, ymax + ypad),
    )
    ax_sc.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
    ax_sc.scatter(t, m, s=2, color=(1, 0, 0, 1), zorder=60)
    ax_sc.add_line(
        Line2D(
            (ymin - ypad, ymax + ypad),
            (ymin - ypad, ymax + ypad),
            linewidth=1,
            color=(0, 0, 0, 1),
            alpha=0.2,
            zorder=10,
        )
    )


# Top left - PRMSL
tsx = all_stats["dtp"]
ty = [x for x in all_stats["Target"]]
my = [x for x in all_stats["Model"]]
plot_var(tsx, ty, my)


fig.savefig("comparison.png")
