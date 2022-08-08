#!/usr/bin/env python

# Test extrapolation - plot same field with various extrapolations

import os
import sys
import numpy as np
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import iris
import iris.analysis
import cmocean

import warnings

warnings.filterwarnings("ignore", message=".*invalid units.*")
warnings.filterwarnings("ignore", message=".*will ignore the.*")
warnings.filterwarnings("ignore", message=".*TransverseMercator*")

sys.path.append("%s/../../../get_data" % os.path.dirname(__file__))
from HUKG_monthly_load import load_cList
from HUKG_monthly_load import sCube
from HUKG_monthly_load import lm_plot

sys.path.append("%s/../plot_quad" % os.path.dirname(__file__))
from plot_variable import plotFieldAxes

sys.path.append("%s/." % os.path.dirname(__file__))
from tensor_utils import extrapolate_missing

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
    args.opfile = "Extrapolations.png" 

fig = Figure(
    figsize=(20, 22),
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
    Rectangle((0, 0), 1, 1, facecolor=(1.0, 1.0, 1.0, 1), fill=True, zorder=1,)
)

plotCube = sCube.copy()
lMask = lm_plot.copy()

qd = load_cList(args.year, args.month)

# Top left - no extrapolation
var = qd[2].copy()
var -= np.min(var.data)
dmin = 0
dmax = np.max(var.data)
var.data[np.where(var.data.mask==True)]=0
var.data = var.data.data
a1 = fig.add_axes([0.025 / 2, 0.125 / 2 + 0.5, 0.95 / 2, 0.85 / 2])
a1.set_axis_off()
p1_img = plotFieldAxes(
    a1,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.balance,
    plotCube=plotCube,
)
ax1_cb = fig.add_axes([0.125 / 2, 0.05 / 2 + 0.5, 0.75 / 2, 0.05 / 2])
ax1_cb.set_axis_off()
cb = fig.colorbar(
    p1_img, ax=ax1_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Bottom left - 20 iterations
v20 = extrapolate_missing(var,nsteps=20,scale=0.99)
a2 = fig.add_axes([0.025 / 2, 0.125 / 2, 0.95 / 2, 0.85 / 2])
a2.set_axis_off()
a2_img = plotFieldAxes(
    a2,
    v20,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.balance,
    plotCube=plotCube,
)
a2_cb = fig.add_axes([0.125 / 2, 0.05 / 2, 0.75 / 2, 0.05 / 2])
a2_cb.set_axis_off()
cb = fig.colorbar(
    a2_img, ax=a2_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Top right - 10 iterations
v10 = extrapolate_missing(var,nsteps=10,scale=0.99)
a3 = fig.add_axes([0.025 / 2 + 0.5, 0.125 / 2 + 0.5, 0.95 / 2, 0.85 / 2])
a3.set_axis_off()
a3_img = plotFieldAxes(
    a3,
    v10,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.balance,
    plotCube=plotCube,
)
a3_cb = fig.add_axes([0.125 / 2 + 0.5, 0.05 / 2 + 0.5, 0.75 / 2, 0.05 / 2])
a3_cb.set_axis_off()
cb = fig.colorbar(
    a3_img, ax=a3_cb, location="bottom", orientation="horizontal", fraction=1.0
)
# Bottom right - 30 iterations
v30 = extrapolate_missing(var,nsteps=100,scale=0.95)
a4 = fig.add_axes([0.025 / 2 + 0.5, 0.125 / 2, 0.95 / 2, 0.85 / 2])
a4.set_axis_off()
a4_img = plotFieldAxes(
    a4,
    v30,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.balance,
    plotCube=plotCube,
)
a4_cb = fig.add_axes([0.125 / 2 + 0.5, 0.05 / 2, 0.75 / 2, 0.05 / 2])
a4_cb.set_axis_off()
cb = fig.colorbar(
    a4_img, ax=a4_cb, location="bottom", orientation="horizontal", fraction=1.0
)

if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

# Output as png
fig.savefig("%s/%s" % (args.opdir, args.opfile))
