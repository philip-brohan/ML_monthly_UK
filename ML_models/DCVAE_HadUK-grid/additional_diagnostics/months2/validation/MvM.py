#!/usr/bin/env python

# Probability matrix for month estimation

import os
import sys

import tensorflow as tf
import numpy
import itertools

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.append("%s/../../.." % os.path.dirname(__file__))
from localise import LSCRATCH

sys.path.append("%s/.." % os.path.dirname(__file__))
from originalsDataset import getDataset
from generatorModel import NNG

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=100)
# Set nimages to a small number for fast testing
parser.add_argument(
    "--nimages",
    help="No of test cases to look at",
    type=int,
    required=False,
    default=None,
)
args = parser.parse_args()

# Instantiate the model
generator = NNG()
weights_dir = ("%s/additional_diagnostics/months2/models/Epoch_%04d") % (
    LSCRATCH,
    args.epoch,
)
load_status = generator.load_weights("%s/ckpt" % weights_dir)
load_status.assert_existing_objects_matched()

if args.nimages is not None:
    nTestImages = args.nimages
else:
    nTestImages = None

testData = getDataset(purpose="test", nImages=nTestImages, shuffle=False, cache=False)
testData = testData.batch(1)

count = numpy.zeros(13)
pmatrix = numpy.zeros((13, 13))
for testCase in testData:
    orig = testCase[1][0, :]
    original = numpy.where(orig == 1.0)[0][0] + 1
    generated = generator.call(testCase)
    dgProbabilities = generated[0, :]
    pmatrix[original, 1:] += dgProbabilities
    count[original] += 1
pmatrix /= count

# Plot a bar chart of generation probabilities for a single month
def plot1(ax, d):
    for td in range(1, 13):
        fc = "red"
        if td == d:
            fc = "blue"
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (td - 0.25, 0), 0.5, 1, fill=True, facecolor=(0, 0, 0, 0.1)
            )
        )
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (td - 0.25, 0), 0.5, pmatrix[d, td], fill=True, facecolor=fc
            )
        )


# Plot the histogram for all 12 target months
fig = Figure(
    figsize=(12, 12),
    dpi=100,
    facecolor="white",
    edgecolor="black",
    linewidth=0.0,
    frameon=False,
    subplotpars=None,
    tight_layout=None,
)
matplotlib.rcParams.update({"font.size": 22})
canvas = FigureCanvas(fig)
# Paint the background white
ax_full = fig.add_axes([0, 0, 1, 1])
ax_full.set_xlim([0, 1])
ax_full.set_ylim([0, 1])
ax_full.add_patch(
    matplotlib.patches.Rectangle((0, 0), 1, 1, fill=True, facecolor="white")
)

for td in range(1, 13):
    ax_digit = fig.add_axes([0.05, 0.04 + ((td - 1) * 1 / 12) * 0.95, 0.94, 0.9 / 12])
    ax_digit.set_xlim([0.5, 12.5])
    ax_digit.set_ylim([0, 1])
    ax_digit.spines["top"].set_visible(False)
    ax_digit.spines["right"].set_visible(False)
    if td != 1:
        ax_digit.get_xaxis().set_ticks([])
    else:
        ax_digit.get_xaxis().set_ticks(range(1, 13))
        # ax_digit.set_xlabel("Transcribed choice")
    ax_digit.get_yaxis().set_ticks([])
    ax_digit.set_ylabel("%2d  " % td, rotation=0, labelpad=10)
    plot1(ax_digit, td)

# Render the figure as a png
fig.savefig("MvM.png")
