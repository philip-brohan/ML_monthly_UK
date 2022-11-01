#!/usr/bin/env python

# plot histograms of 1 month of the normalised tensor data

import os
import sys
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np

import warnings

warnings.filterwarnings("ignore", message=".*partition.*")

sys.path.append("%s/../../.." % os.path.dirname(__file__))
from plot_functions.plot_variable import plotFieldAxes

sys.path.append("%s/." % os.path.dirname(__file__))
from tensor_utils import load_cList
from tensor_utils import cList_to_tensor
from tensor_utils import dm_HUKG
from tensor_utils import lm_TWCR
from tensor_utils import sCube

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument(
    "--opdir", help="Output directory", type=str, required=False, default="."
)
parser.add_argument(
    "--opfile", help="Output file name", type=str, required=False, default=None
)
args = parser.parse_args()
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

if args.opfile is None:
    args.opfile = "Hist.png"

# Load and normalise the data
cL = load_cList(args.year, args.month)
tL = cList_to_tensor(cL, extrapolate=False)

fig = Figure(
    figsize=(30, 22),
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


# Top left - PRMSL
var = sCube.copy()
var.data = tL.numpy()[:, :, 0]
ax_prmsl = fig.add_axes(
    [0.025 / 2 + 0.025, 0.125 / 2 + 0.5, 0.95 / 2 - 0.025, 0.85 / 2]
)
ax_prmsl.set_xlabel("PRMSL")
PRMSL_hist = ax_prmsl.hist(var.data.flatten(), bins=100, range=(0, 1), zorder=50)

# Bottom left - SST
var.data = tL.numpy()[:, :, 1]
var.data = np.ma.masked_where(lm_TWCR.data.mask, var.data, copy=False)
ax_sst = fig.add_axes([0.025 / 2 + 0.025, 0.125 / 2, 0.95 / 2 - 0.025, 0.85 / 2])
ax_sst.set_xlabel("SST")
SST_hist = ax_sst.hist(var.data.flatten().compressed(), bins=100, range=(0, 1))

# Top right - PRATE
var.data = tL.numpy()[:, :, 3]
var.data = np.ma.masked_where(dm_HUKG.data.mask, var.data, copy=False)
ax_prate = fig.add_axes(
    [0.025 / 2 + 0.5 + 0.025, 0.125 / 2 + 0.5, 0.95 / 2 - 0.025, 0.85 / 2]
)
ax_prate.set_xlabel("PRATE")
PRATE_hist = ax_prate.hist(var.data.flatten().compressed(), bins=100, range=(0, 1))

# Bottom left - T2m
var.data = tL.numpy()[:, :, 2]
var.data = np.ma.masked_where(dm_HUKG.data.mask, var.data, copy=False)
ax_tmp2m = fig.add_axes(
    [0.025 / 2 + 0.5 + 0.025, 0.125 / 2, 0.95 / 2 - 0.025, 0.85 / 2]
)
ax_tmp2m.set_xlabel("T2M")
T2M_hist = ax_tmp2m.hist(var.data.flatten().compressed(), bins=100, range=(0, 1))

if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

# Output as png
fig.savefig("%s/%s" % (args.opdir, args.opfile))
