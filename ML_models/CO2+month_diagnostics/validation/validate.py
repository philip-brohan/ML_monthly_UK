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

import warnings

warnings.filterwarnings("ignore", message=".*partition.*")
warnings.filterwarnings("ignore", message=".*TransverseMercator.*")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=99)
parser.add_argument("--year", help="Test year", type=int, required=False, default=1969)
parser.add_argument("--month", help="Test month", type=int, required=False, default=3)
parser.add_argument(
    "--member", help="Test ensemble member", type=int, required=False, default=1
)
args = parser.parse_args()

sys.path.append("%s/../../../get_data" % os.path.dirname(__file__))
from HUKG_monthly_load import load_cList
from HUKG_monthly_load import sCube
from HUKG_monthly_load import lm_plot
from HUKG_monthly_load import lm_20CR
from HUKG_monthly_load import dm_hukg
from TWCR_monthly_load import get_range

sys.path.append("%s/../make_tensors" % os.path.dirname(__file__))
from tensor_utils import cList_to_tensor
from tensor_utils import normalise
from tensor_utils import unnormalise

sys.path.append("%s/../plot_quad" % os.path.dirname(__file__))
from plot_variable import plotFieldAxes
from plot_variable import plotScatterAxes

# Load the model specification
sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH
from autoencoderModel import DCVAE
from makeDataset import normalise_co2
from makeDataset import normalise_month
from makeDataset import unnormalise_co2
from makeDataset import unnormalise_month

# Load and standardise data
qd = load_cList(args.year, args.month, args.member)
ict = cList_to_tensor(qd, lm_20CR.data.mask, dm_hukg.data.mask)
nco2 = normalise_co2("%04d" % args.year)
nmth = normalise_month("0000-%02d" % args.month)

autoencoder = DCVAE()
weights_dir = ("%s/models/Epoch_%04d") % (
    LSCRATCH,
    args.epoch,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()

# Get autoencoded tensors
t_input = (
    tf.reshape(ict, [1, 1440, 896, 4]),
    tf.reshape(tf.convert_to_tensor(nco2, np.float32), [1, 14]),
    tf.reshape(
        tf.convert_to_tensor(nmth, np.float32),
        [1, 12],
    ),
)

encoded = autoencoder.call(t_input)

# Make the plot
fig = Figure(
    figsize=(15, 23),
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
        (0, 1),
        1,
        1,
        facecolor=(0.6, 0.6, 0.6, 1),
        fill=True,
        zorder=1,
    )
)

# Top row - date, CO2 and month diagnostics

axb.text(0.03, 0.97, "%04d/%02d" % (args.year, args.month), fontsize=30, zorder=10)
axb.text(
    0.16,
    0.97,
    "CO2",
    fontsize=30,
    zorder=10,
)
ax_co2 = fig.add_axes([0.24, 0.965, 0.335, 0.028], xlim=(0, 15), ylim=(0, 1))
ax_co2.bar(
    [x - 0.2 for x in list(range(1, 15))],
    t_input[1][0, :].numpy(),
    width=0.4,
    color=(0, 0, 0, 1),
    tick_label="",
)
ax_co2.bar(
    [x + 0.2 for x in list(range(1, 15))],
    encoded[1][0, :].numpy(),
    width=0.4,
    color=(1, 0, 0, 1),
    tick_label="",
)
ax_co2.set_yticks(())
ax_co2.set_xticks(range(1, 15))

axb.text(
    0.61,
    0.97,
    "Month",
    fontsize=30,
    zorder=10,
)
ax_mnth = fig.add_axes([0.7, 0.965, 0.29, 0.028], xlim=(0, 13), ylim=(0, 1))
ax_mnth.bar(
    [x - 0.2 for x in list(range(1, 13))],
    t_input[2][0, :].numpy(),
    width=0.4,
    color=(0, 0, 0, 1),
    tick_label="",
)
ax_mnth.bar(
    [x + 0.2 for x in list(range(1, 13))],
    encoded[2][0, :].numpy(),
    width=0.4,
    color=(1, 0, 0, 1),
    tick_label="",
)
ax_mnth.set_yticks(())
ax_mnth.set_xticks(range(1, 13))


# Top left - PRMSL original
var = sCube.copy()
var.data = np.squeeze(ict[:, :, 0].numpy())
var = unnormalise(var, "PRMSL") / 100
(dmin, dmax) = get_range("PRMSL", args.month, var)
dmin /= 100
dmax /= 100
ax_prmsl = fig.add_axes([0.025 / 3, 0.12 / 4 + 0.72, 0.95 / 3, 0.81 / 4])
ax_prmsl.set_axis_off()
PRMSL_img = plotFieldAxes(
    ax_prmsl,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.diff,
    plotCube=sCube,
)
ax_prmsl_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.72, 0.75 / 3, 0.05 / 4])
ax_prmsl_cb.set_axis_off()
cb = fig.colorbar(
    PRMSL_img, ax=ax_prmsl_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Top centre - PRMSL encoded
var.data = np.squeeze(encoded[0][0, :, :, 0].numpy())
var = unnormalise(var, "PRMSL") / 100
ax_prmsl_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.12 / 4 + 0.72, 0.95 / 3, 0.81 / 4])
ax_prmsl_e.set_axis_off()
PRMSL_e_img = plotFieldAxes(
    ax_prmsl_e,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.diff,
    plotCube=sCube,
)
ax_prmsl_e_cb = fig.add_axes([0.125 / 3 + 1 / 3, 0.05 / 4 + 0.72, 0.75 / 3, 0.05 / 4])
ax_prmsl_e_cb.set_axis_off()
cb = fig.colorbar(
    PRMSL_e_img,
    ax=ax_prmsl_e_cb,
    location="bottom",
    orientation="horizontal",
    fraction=1.0,
)

# Top right - PRMSL scatter
varx = sCube.copy()
varx.data = np.squeeze(ict[:, :, 0].numpy())
varx = unnormalise(varx, "PRMSL") / 100
vary = sCube.copy()
vary.data = np.squeeze(encoded[0][0, :, :, 0].numpy())
vary = unnormalise(vary, "PRMSL") / 100
ax_prmsl_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.12 / 4 + 0.72, 0.95 / 3 - 0.06, 0.81 / 4]
)
plotScatterAxes(ax_prmsl_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


# 2nd left - PRATE original
var = sCube.copy()
var.data = np.squeeze(ict[:, :, 3].numpy())
var.data = np.ma.masked_where(dm_hukg.data.mask == True, var.data, copy=False)
var = unnormalise(var, "PRATE") * 1000
(dmin, dmax) = get_range("PRATE", args.month, var)
dmin = 0
dmax *= 1000
ax_prate = fig.add_axes([0.025 / 3, 0.12 / 4 + 0.48, 0.95 / 3, 0.81 / 4])
ax_prate.set_axis_off()
PRATE_img = plotFieldAxes(
    ax_prate,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.rain,
    plotCube=sCube,
)
ax_prate_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.48, 0.75 / 3, 0.05 / 4])
ax_prate_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_img, ax=ax_prate_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd centre - PRATE encoded
var.data = np.squeeze(encoded[0][0, :, :, 3].numpy())
var.data = np.ma.masked_where(dm_hukg.data.mask == True, var.data, copy=False)
var = unnormalise(var, "PRATE") * 1000
ax_prate_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.12 / 4 + 0.48, 0.95 / 3, 0.81 / 4])
ax_prate_e.set_axis_off()
PRATE_e_img = plotFieldAxes(
    ax_prate_e,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.rain,
    plotCube=sCube,
)
ax_prate_e_cb = fig.add_axes([0.125 / 3 + 1 / 3, 0.05 / 4 + 0.48, 0.75 / 3, 0.05 / 4])
ax_prate_e_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_e_img,
    ax=ax_prate_e_cb,
    location="bottom",
    orientation="horizontal",
    fraction=1.0,
)

# 2nd right - PRATE scatter
varx = sCube.copy()
varx.data = np.squeeze(ict[:, :, 3].numpy())
varx.data = np.ma.masked_where(dm_hukg.data.mask == True, varx.data, copy=False)
varx = unnormalise(varx, "PRATE") * 1000
vary = sCube.copy()
vary.data = np.squeeze(encoded[0][0, :, :, 3].numpy())
vary.data = np.ma.masked_where(dm_hukg.data.mask == True, vary.data, copy=False)
vary = unnormalise(vary, "PRATE") * 1000
ax_prate_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.12 / 4 + 0.48, 0.95 / 3 - 0.06, 0.81 / 4]
)
plotScatterAxes(ax_prate_s, varx, vary, vMin=0.001, vMax=dmax, bins=None)


# 3rd left - T2m original
var = sCube.copy()
var.data = np.squeeze(ict[:, :, 2].numpy())
var.data = np.ma.masked_where(dm_hukg.data.mask == True, var.data, copy=False)
var = unnormalise(var, "TMP2m") - 273.15
(dmin, dmax) = get_range("TMP2m", args.month, var)
dmin -= 273.15 + 2
dmax -= 273.15 - 2
ax_t2m = fig.add_axes([0.025 / 3, 0.12 / 4 + 0.24, 0.95 / 3, 0.81 / 4])
ax_t2m.set_axis_off()
T2m_img = plotFieldAxes(
    ax_t2m,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
    plotCube=sCube,
)
ax_t2m_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.24, 0.75 / 3, 0.05 / 4])
ax_t2m_cb.set_axis_off()
cb = fig.colorbar(
    T2m_img, ax=ax_t2m_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 3rd centre - T2m encoded
var = sCube.copy()
var.data = np.squeeze(encoded[0][0, :, :, 2].numpy())
var.data = np.ma.masked_where(dm_hukg.data.mask == True, var.data, copy=False)
var = unnormalise(var, "TMP2m") - 273.15
ax_t2m_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.12 / 4 + 0.24, 0.95 / 3, 0.81 / 4])
ax_t2m_e.set_axis_off()
T2m_e_img = plotFieldAxes(
    ax_t2m_e,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
    plotCube=sCube,
)
ax_t2m_e_cb = fig.add_axes([0.125 / 3 + 1 / 3, 0.05 / 4 + 0.24, 0.75 / 3, 0.05 / 4])
ax_t2m_e_cb.set_axis_off()
cb = fig.colorbar(
    T2m_e_img, ax=ax_t2m_e_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 3rd right - T2m scatter
varx = sCube.copy()
varx.data = np.squeeze(ict[:, :, 2].numpy())
varx.data = np.ma.masked_where(dm_hukg.data.mask == True, varx.data, copy=False)
varx = unnormalise(varx, "TMP2m") - 273.15
vary = sCube.copy()
vary.data = np.squeeze(encoded[0][0, :, :, 2].numpy())
vary.data = np.ma.masked_where(dm_hukg.data.mask == True, vary.data, copy=False)
vary = unnormalise(vary, "TMP2m") - 273.15
ax_t2m_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.12 / 4 + 0.24, 0.95 / 3 - 0.06, 0.81 / 4]
)
plotScatterAxes(ax_t2m_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


# Bottom left - SST original
var.data = np.squeeze(ict[:, :, 1].numpy())
var.data = np.ma.masked_where(lm_20CR.data.mask == True, var.data, copy=False)
var = unnormalise(var, "TMPS") - 273.15
(dmin, dmax) = get_range("TMPS", args.month, var)
dmin -= 273.15 + 2
dmax -= 273.15 - 2
ax_sst = fig.add_axes([0.025 / 3, 0.12 / 4, 0.95 / 3, 0.81 / 4])
ax_sst.set_axis_off()
SST_img = plotFieldAxes(
    ax_sst,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
    plotCube=sCube,
)
ax_sst_cb = fig.add_axes([0.125 / 3, 0.05 / 4, 0.75 / 3, 0.05 / 4])
ax_sst_cb.set_axis_off()
cb = fig.colorbar(
    SST_img, ax=ax_sst_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd centre - SST encoded
var.data = encoded[0].numpy()[0, :, :, 1]
var.data = np.ma.masked_where(lm_20CR.data.mask == True, var.data, copy=False)
var = unnormalise(var, "TMPS") - 273.15
ax_sst_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.12 / 4, 0.95 / 3, 0.81 / 4])
ax_sst_e.set_axis_off()
SST_e_img = plotFieldAxes(
    ax_sst_e,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
    plotCube=sCube,
)
ax_sst_e_cb = fig.add_axes([0.125 / 3 + 1 / 3, 0.05 / 4, 0.75 / 3, 0.05 / 4])
ax_sst_e_cb.set_axis_off()
cb = fig.colorbar(
    SST_e_img, ax=ax_sst_e_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd right - SST scatter
varx.data = np.squeeze(ict[:, :, 1].numpy())
varx.data = np.ma.masked_where(lm_20CR.data.mask == True, varx.data, copy=False)
varx = unnormalise(varx, "TMPS") - 273.15
vary.data = np.squeeze(encoded[0][0, :, :, 1].numpy())
vary.data = np.ma.masked_where(lm_20CR.data > 0, vary.data, copy=False)
vary = unnormalise(vary, "TMPS") - 273.15
ax_sst_s = fig.add_axes([0.025 / 3 + 2 / 3 + 0.06, 0.12 / 4, 0.95 / 3 - 0.06, 0.81 / 4])
plotScatterAxes(ax_sst_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


fig.savefig("comparison.png")
