#!/usr/bin/env python

# plot histograms of 1 month of the normalised tensor data

import os
import sys
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import iris
import iris.analysis
import numpy as np
import cmocean

import warnings

warnings.filterwarnings("ignore", message=".*partition.*")

sys.path.append("%s/../../.." % os.path.dirname(__file__))
from plot_functions.plot_variable import plotFieldAxes


sys.path.append("%s/." % os.path.dirname(__file__))
from tensor_utils import load_cList
from tensor_utils import cList_to_tensor
from tensor_utils import lm_plot
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
    args.opfile = "Month.png"

# Load and normalise the data
cL = load_cList(args.year, args.month)
tL = cList_to_tensor(cL, extrapolate=False)

fig = Figure(
    figsize=(30, 30*0.35),
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

xmargin=0.02
ymargin=0.02
width = (1-(xmargin*5))/4
height = (1-ymargin*2)
# Top left - PRMSL
var = sCube.copy()
var.data = tL.numpy()[:, :, 0]
ax_prmsl = fig.add_axes(
    [xmargin, ymargin, width, height]
)
ax_prmsl.set_xlabel("PRMSL")
ax_prmsl.set_axis_off()
PRMSL_img = plotFieldAxes(
    ax_prmsl,
    var,
    vMax=1.1,
    vMin=-0.1,
    lMask=lm_plot,
    cMap=cmocean.cm.diff,
)

# Bottom left - SST
var.data = tL.numpy()[:, :, 1]
var.data = np.ma.masked_where(var.data==0.5, var.data, copy=False)
ax_sst = fig.add_axes([xmargin*2+width,ymargin,width,height])
ax_sst.set_xlabel("SST")
ax_sst.set_axis_off()
SST_img = plotFieldAxes(
    ax_sst,
    var,
    vMax=0.8,
    vMin=0.2,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)

# Top right - PRATE
var.data = tL.numpy()[:, :, 3]
var.data = np.ma.masked_where(var.data==0.5, var.data, copy=False)
ax_prate = fig.add_axes(
    [xmargin*4+width*3,ymargin,width,height]
)
ax_prate.set_xlabel("PRATE")
ax_prate.set_axis_off()
PRATE_img = plotFieldAxes(
    ax_prate,
    var,
    vMax=1.0,
    vMin=0.0,
    lMask=lm_plot,
    cMap=cmocean.cm.tarn,
)

# Bottom left - T2m
var.data = tL.numpy()[:, :, 2]
var.data = np.ma.masked_where(var.data==0.5, var.data, copy=False)
ax_tmp2m = fig.add_axes(
    [xmargin*3+width*2,ymargin,width,height]
)
ax_tmp2m.set_xlabel("T2M")
ax_tmp2m.set_axis_off()
T2M_img = plotFieldAxes(
    ax_tmp2m,
    var,
    vMax=0.8,
    vMin=0.2,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)

if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

# Output as png
fig.savefig("%s/%s" % (args.opdir, args.opfile))
