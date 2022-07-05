#!/usr/bin/env python

# Plot a validation figure for the autoencoder.

# For each variable:
#  1) Input field
#  2) Autoencoder output
#  3) scatter plot
#

import os
import sys
import numpy as np
import tensorflow as tf
import iris
import iris.fileformats
import iris.analysis
import cmocean

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=100)
parser.add_argument("--year", help="Test year", type=int, required=False, default=1969)
parser.add_argument("--month", help="Test month", type=int, required=False, default=3)
parser.add_argument(
    "--member", help="Test ensemble member", type=int, required=False, default=1
)
args = parser.parse_args()

sys.path.append("%s/../../../get_data" % os.path.dirname(__file__))
from TWCR_monthly_load import load_quad
from TWCR_monthly_load import get_range

sys.path.append("%s/../../make_tensors" % os.path.dirname(__file__))
from tensor_utils import quad_to_tensor
from tensor_utils import normalise
from tensor_utils import unnormalise

sys.path.append("%s/../../../plots" % os.path.dirname(__file__))
from plot_variable import plotFieldAxes
from plot_variable import plotScatterAxes
from plot_variable import plot_cube
from plot_variable import get_land_mask

# Load and standardise data
pc = plot_cube()
qd = load_quad(args.year, args.month, args.member)
ict = quad_to_tensor(qd, pc)
sst_mask = ict.numpy()[:, :, 1] == 0.0

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

# Get autoencoded tensors
encoded = autoencoder.call(tf.reshape(ict, [1, 32, 32, 4]))

# Make the plot
fig = Figure(
    figsize=(15, 22),
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

lMask = get_land_mask(plot_cube(resolution=0.1))


# Top left - PRMSL original
var = pc
var.data = np.squeeze(ict[:, :, 0].numpy())
var = unnormalise(var, "PRMSL") / 100
(dmin, dmax) = get_range("PRMSL", args.month, var)
dmin /= 100
dmax /= 100
ax_prmsl = fig.add_axes([0.025 / 3, 0.125 / 4 + 0.75, 0.95 / 3, 0.85 / 4])
ax_prmsl.set_axis_off()
PRMSL_img = plotFieldAxes(
    ax_prmsl, var, vMax=dmax, vMin=dmin, lMask=lMask, cMap=cmocean.cm.diff, plotCube=pc,
)
ax_prmsl_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.75, 0.75 / 3, 0.05 / 4])
ax_prmsl_cb.set_axis_off()
cb = fig.colorbar(
    PRMSL_img, ax=ax_prmsl_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Top centre - PRMSL encoded
var.data = np.squeeze(encoded[0, :, :, 0].numpy())
var = unnormalise(var, "PRMSL") / 100
ax_prmsl_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4 + 0.75, 0.95 / 3, 0.85 / 4])
ax_prmsl_e.set_axis_off()
PRMSL_e_img = plotFieldAxes(
    ax_prmsl_e,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.diff,
    plotCube=pc,
)
ax_prmsl_e_cb = fig.add_axes([0.125 / 3 + 1 / 3, 0.05 / 4 + 0.75, 0.75 / 3, 0.05 / 4])
ax_prmsl_e_cb.set_axis_off()
cb = fig.colorbar(
    PRMSL_e_img,
    ax=ax_prmsl_e_cb,
    location="bottom",
    orientation="horizontal",
    fraction=1.0,
)

# Top right - PRMSL scatter
varx = pc.copy()
varx.data = np.squeeze(ict[:, :, 0].numpy())
varx = unnormalise(varx, "PRMSL") / 100
vary = pc.copy()
vary.data = np.squeeze(encoded[0, :, :, 0].numpy())
vary = unnormalise(vary, "PRMSL") / 100
ax_prmsl_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4 + 0.75, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_prmsl_s, varx, vary, vMin=dmin, vMax=dmax)


# 2nd left - PRATE original
var = pc
var.data = np.squeeze(ict[:, :, 3].numpy())
var = unnormalise(var, "PRATE") * 1000
(dmin, dmax) = get_range("PRATE", args.month, var)
dmin = 0
dmax *= 1000
ax_prate = fig.add_axes([0.025 / 3, 0.125 / 4 + 0.5, 0.95 / 3, 0.85 / 4])
ax_prate.set_axis_off()
PRATE_img = plotFieldAxes(
    ax_prate, var, vMax=dmax, vMin=dmin, lMask=lMask, cMap=cmocean.cm.rain, plotCube=pc,
)
ax_prate_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.5, 0.75 / 3, 0.05 / 4])
ax_prate_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_img, ax=ax_prate_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd centre - PRATE encoded
var.data = np.squeeze(encoded[0, :, :, 3].numpy())
var = unnormalise(var, "PRATE") * 1000
ax_prate_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4 + 0.5, 0.95 / 3, 0.85 / 4])
ax_prate_e.set_axis_off()
PRATE_e_img = plotFieldAxes(
    ax_prate_e,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.rain,
    plotCube=pc,
)
ax_prate_e_cb = fig.add_axes([0.125 / 3 + 1 / 3, 0.05 / 4 + 0.5, 0.75 / 3, 0.05 / 4])
ax_prate_e_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_e_img,
    ax=ax_prate_e_cb,
    location="bottom",
    orientation="horizontal",
    fraction=1.0,
)

# 2nd right - PRATE scatter
varx = pc.copy()
varx.data = np.squeeze(ict[:, :, 3].numpy())
varx = unnormalise(varx, "PRATE") * 1000
vary = pc.copy()
vary.data = np.squeeze(encoded[0, :, :, 3].numpy())
vary = unnormalise(vary, "PRATE") * 1000
ax_prate_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4 + 0.5, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_prate_s, varx, vary, vMin=0.001, vMax=dmax)


# 3rd left - T2m original
var = pc
var.data = np.squeeze(ict[:, :, 2].numpy())
var = unnormalise(var, "TMP2m") - 273.15
(dmin, dmax) = get_range("TMP2m", args.month, var)
dmin -= 273.15 + 2
dmax -= 273.15 - 2
ax_t2m = fig.add_axes([0.025 / 3, 0.125 / 4 + 0.25, 0.95 / 3, 0.85 / 4])
ax_t2m.set_axis_off()
T2m_img = plotFieldAxes(
    ax_t2m,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.balance,
    plotCube=pc,
)
ax_t2m_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.25, 0.75 / 3, 0.05 / 4])
ax_t2m_cb.set_axis_off()
cb = fig.colorbar(
    T2m_img, ax=ax_t2m_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 3rd centre - T2m encoded
var.data = np.squeeze(encoded[0, :, :, 2].numpy())
var = unnormalise(var, "TMP2m") - 273.15
ax_t2m_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4 + 0.25, 0.95 / 3, 0.85 / 4])
ax_t2m_e.set_axis_off()
T2m_e_img = plotFieldAxes(
    ax_t2m_e,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.balance,
    plotCube=pc,
)
ax_t2m_e_cb = fig.add_axes([0.125 / 3 + 1 / 3, 0.05 / 4 + 0.25, 0.75 / 3, 0.05 / 4])
ax_t2m_e_cb.set_axis_off()
cb = fig.colorbar(
    T2m_e_img, ax=ax_t2m_e_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 3rd right - T2m scatter
varx = pc.copy()
varx.data = np.squeeze(ict[:, :, 2].numpy())
varx = unnormalise(varx, "TMP2m") - 273.15
vary = pc.copy()
vary.data = np.squeeze(encoded[0, :, :, 2].numpy())
vary = unnormalise(vary, "TMP2m") - 273.15
ax_t2m_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4 + 0.25, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_t2m_s, varx, vary, vMin=dmin, vMax=dmax)


# Bottom left - SST original
var = pc
var.data = np.squeeze(ict[:, :, 1].numpy())
var = unnormalise(var, "TMPS") - 273.15
(dmin, dmax) = get_range("TMPS", args.month, var)
var.data = np.ma.masked_where(sst_mask, var.data, copy=True)
dmin -= 273.15 + 2
dmax -= 273.15 - 2
ax_sst = fig.add_axes([0.025 / 3, 0.125 / 4, 0.95 / 3, 0.85 / 4])
ax_sst.set_axis_off()
SST_img = plotFieldAxes(
    ax_sst,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.balance,
    plotCube=pc,
)
ax_sst_cb = fig.add_axes([0.125 / 3, 0.05 / 4, 0.75 / 3, 0.05 / 4])
ax_sst_cb.set_axis_off()
cb = fig.colorbar(
    SST_img, ax=ax_sst_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd centre - SST encoded
var.data *= 0
var.data += encoded.numpy()[0, :, :, 1]
var = unnormalise(var, "TMPS") - 273.15
var.data = np.ma.masked_where(sst_mask, var.data, copy=True)
ax_sst_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4, 0.95 / 3, 0.85 / 4])
ax_sst_e.set_axis_off()
SST_e_img = plotFieldAxes(
    ax_sst_e,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.balance,
    plotCube=pc,
)
ax_sst_e_cb = fig.add_axes([0.125 / 3 + 1 / 3, 0.05 / 4, 0.75 / 3, 0.05 / 4])
ax_sst_e_cb.set_axis_off()
cb = fig.colorbar(
    SST_e_img, ax=ax_sst_e_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd right - SST scatter
varx = pc.copy()
varx.data = np.squeeze(ict[:, :, 1].numpy())
varx = unnormalise(varx, "TMPS") - 273.15
msk = np.ma.masked_where(sst_mask, varx.data, copy=False)
vary = pc.copy()
vary.data = np.squeeze(encoded[0, :, :, 1].numpy())
vary = unnormalise(vary, "TMPS") - 273.15
msk = np.ma.masked_where(sst_mask, vary.data, copy=False)
ax_sst_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_sst_s, varx, vary, vMin=dmin, vMax=dmax)


fig.savefig("comparison.png")
