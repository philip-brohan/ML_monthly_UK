#!/usr/bin/env python

# plot generated monthly fields for all four variables.

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import iris
import iris.analysis
import iris.util
import iris.coord_systems
import cmocean

import warnings

warnings.filterwarnings("ignore", message=".*invalid units.*")
# warnings.filterwarnings("ignore", message=".*will ignore the.*")

sys.path.append("%s/../../../get_data" % os.path.dirname(__file__))
from TWCR_monthly_load import unnormalise

sys.path.append("%s/../../../plots" % os.path.dirname(__file__))
from plot_variable import plotFieldAxes
from plot_variable import plot_cube
from plot_variable import get_land_mask

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=100)
args = parser.parse_args()

# Load the model specification
sys.path.append("%s/.." % os.path.dirname(__file__))
from autoencoderModel import DCVAE

autoencoder = DCVAE()
weights_dir = ("%s//ML_monthly_UK/models/DCVAE_4_fields/" + "Epoch_%04d") % (
    os.getenv("SCRATCH"),
    args.epoch,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()
eps = tf.random.normal(shape=(1, autoencoder.latent_dim))
generated = autoencoder.decode(eps)

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

plotCube = plot_cube()
lMask = get_land_mask(plot_cube(resolution=0.1))

# Top left - PRMSL
var = plotCube
var.data = np.squeeze(generated[0, :, :, 0].numpy())
var = unnormalise(var, "PRMSL") / 100
dmin = np.min(var.data)
dmax = np.max(var.data)
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
var.data = np.squeeze(generated[0, :, :, 1].numpy())
var = unnormalise(var, "TMPS") - 273.15
lm = iris.load_cube("%s/20CR/version_3/fixed/land.nc" % os.getenv("SCRATCH"))
lm = iris.util.squeeze(lm)
coord_s = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
lm.coord("latitude").coord_system = coord_s
lm.coord("longitude").coord_system = coord_s
lm = lm.regrid(plotCube, iris.analysis.Linear())
var.data = np.ma.masked_where(lm.data > 0.5, var.data, copy=True)
dmin = np.min(var.data)
dmax = np.max(var.data)
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
var.data = np.squeeze(generated[0, :, :, 3].numpy())
var = unnormalise(var, "PRATE") * 1000
dmax = np.max(var.data)
dmin = 0
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
var.data = np.squeeze(generated[0, :, :, 2].numpy())
var = unnormalise(var, "TMP2m") - 273.15
dmin = np.min(var.data)
dmax = np.max(var.data)
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

# Output as png
fig.savefig("%s/%s" % (".", "generated.png"))
