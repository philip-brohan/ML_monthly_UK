#!/usr/bin/env python

# plot monthly fields, for all four variables, for the UK region

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
from calendar import monthrange

import warnings

warnings.filterwarnings("ignore", message=".*invalid units.*")
warnings.filterwarnings("ignore", message=".*will ignore the.*")
warnings.filterwarnings("ignore", message=".*TransverseMercator*")

sys.path.append("%s/../../../get_data" % os.path.dirname(__file__))
from HUKG_monthly_load import load_cList
from HUKG_monthly_load import sCube
from HUKG_monthly_load import lm_plot

sys.path.append("%s/." % os.path.dirname(__file__))

sys.path.append("%s/." % os.path.dirname(__file__))
from plot_variable import plotFieldAxes

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
    args.opfile = "Quad_%04d-%02d.png" % (args.year, args.month)

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
axb.add_patch(
    Rectangle((0, 1), 1, 1, facecolor=(0.6, 0.6, 0.6, 1), fill=True, zorder=1,)
)

plotCube = sCube.copy()
lMask = lm_plot.copy()

qd = load_cList(args.year, args.month)

# Top left - PRMSL
var = qd[0].copy()
var /= 100
dmin = np.min(var.data)
dmax = np.max(var.data)
if (dmax - dmin) < 20:
    dd = dmax - dmin
    dmax += (20 - dd) / 2
    dmin -= (20 - dd) / 2
ax_prmsl = fig.add_axes([0.025 / 2, 0.125 / 2 + 0.5, 0.95 / 2, 0.85 / 2])
ax_prmsl.set_axis_off()
PRMSL_img = plotFieldAxes(
    ax_prmsl,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.diff,
    plotCube=plotCube,
)
ax_prmsl_cb = fig.add_axes([0.125 / 2, 0.05 / 2 + 0.5, 0.75 / 2, 0.05 / 2])
ax_prmsl_cb.set_axis_off()
cb = fig.colorbar(
    PRMSL_img, ax=ax_prmsl_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Bottom left - SST
var = qd[1].copy()
var -= 273.15
dmin = np.min(var.data)
dmax = np.max(var.data)
if (dmax - dmin) < 10:
    dd = dmax - dmin
    dmax += (10 - dd) / 2
    dmin -= (10 - dd) / 2
ax_sst = fig.add_axes([0.025 / 2, 0.125 / 2, 0.95 / 2, 0.85 / 2])
ax_sst.set_axis_off()
SST_img = plotFieldAxes(
    ax_sst,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.balance,
    plotCube=plotCube,
)
ax_sst_cb = fig.add_axes([0.125 / 2, 0.05 / 2, 0.75 / 2, 0.05 / 2])
ax_sst_cb.set_axis_off()
cb = fig.colorbar(
    SST_img, ax=ax_sst_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Top right - PRATE
var = qd[3].copy()
var /= 86400 * monthrange(args.year, args.month)[1]
var.data *= 1000  # Ugly, but makes colorbar legible
dmin = 0
dmax = max(np.max(var.data), 0.05)
ax_prate = fig.add_axes([0.025 / 2 + 0.5, 0.125 / 2 + 0.5, 0.95 / 2, 0.85 / 2])
ax_prate.set_axis_off()
PRATE_img = plotFieldAxes(
    ax_prate,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.rain,
    plotCube=plotCube,
)
ax_prate_cb = fig.add_axes([0.125 / 2 + 0.5, 0.05 / 2 + 0.5, 0.75 / 2, 0.05 / 2])
ax_prate_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_img, ax=ax_prate_cb, location="bottom", orientation="horizontal", fraction=1.0
)
# Bottom left - T2m
var = qd[2]
dmin = np.min(var.data)
dmax = np.max(var.data)
if (dmax - dmin) < 10:
    dd = dmax - dmin
    dmax += (10 - dd) / 2
    dmin -= (10 - dd) / 2
ax_tmp2m = fig.add_axes([0.025 / 2 + 0.5, 0.125 / 2, 0.95 / 2, 0.85 / 2])
ax_tmp2m.set_axis_off()
TMP2m_img = plotFieldAxes(
    ax_tmp2m,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.balance,
    plotCube=plotCube,
)
ax_tmp2m_cb = fig.add_axes([0.125 / 2 + 0.5, 0.05 / 2, 0.75 / 2, 0.05 / 2])
ax_tmp2m_cb.set_axis_off()
cb = fig.colorbar(
    TMP2m_img, ax=ax_tmp2m_cb, location="bottom", orientation="horizontal", fraction=1.0
)

if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

# Output as png
fig.savefig("%s/%s" % (args.opdir, args.opfile))
