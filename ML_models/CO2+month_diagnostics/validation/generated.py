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
warnings.filterwarnings("ignore", message=".*partition.*")
warnings.filterwarnings("ignore", message=".*TransverseMercator.*")

sys.path.append("%s/../../../get_data" % os.path.dirname(__file__))
from HUKG_monthly_load import sCube
from HUKG_monthly_load import lm_plot
from HUKG_monthly_load import lm_20CR
from HUKG_monthly_load import dm_hukg
from TWCR_monthly_load import get_range

sys.path.append("%s/../make_tensors" % os.path.dirname(__file__))
from tensor_utils import unnormalise

sys.path.append("%s/../plot_quad" % os.path.dirname(__file__))
from plot_variable import plotFieldAxes

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=770)
args = parser.parse_args()

# Load the model specification
sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH
from autoencoderModel import DCVAE
from makeDataset import unnormalise_co2
from makeDataset import unnormalise_month

autoencoder = DCVAE()
weights_dir = ("%s/models/Epoch_%04d") % (LSCRATCH, args.epoch,)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()
eps = tf.random.normal(shape=(1, autoencoder.latent_dim))
generated = autoencoder.decode(eps)
month = unnormalise_month(generated[2].numpy())
co2 = unnormalise_co2(generated[1].numpy())

fig = Figure(
    figsize=(20, 24),
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

# Top row - CO2 and month diagnostics

axb.text(
    0.1, 0.96, "CO2", fontsize=30, zorder=10,
)
ax_co2 = fig.add_axes([0.15, 0.955, 0.29, 0.028], xlim=(0, 15), ylim=(0, 1))
ax_co2.bar(
    list(range(1, 15)),
    generated[1][0, :].numpy(),
    width=0.8,
    color=(1, 0, 0, 1),
    tick_label="",
)
ax_co2.set_yticks(())

axb.text(
    0.5, 0.96, "Month", fontsize=30, zorder=10,
)
ax_mnth = fig.add_axes([0.57, 0.955, 0.29, 0.028], xlim=(0, 13), ylim=(0, 1))
ax_mnth.bar(
    list(range(1, 13)),
    generated[2][0, :].numpy(),
    width=0.8,
    color=(1, 0, 0, 1),
    tick_label="",
)
ax_mnth.set_yticks(())


# Top left - PRMSL
var = sCube.copy()
var.data = np.squeeze(generated[0][0, :, :, 0].numpy())
var = unnormalise(var, "PRMSL") / 100
(dmin, dmax) = get_range("PRMSL", month, var)
dmin /= 100
dmax /= 100
ax_prmsl = fig.add_axes([0.025 / 2, 0.11 / 2 + 0.46, 0.95 / 2, 0.78 / 2])
ax_prmsl.set_axis_off()
PRMSL_img = plotFieldAxes(
    ax_prmsl,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.diff,
    plotCube=None,
)
ax_prmsl_cb = fig.add_axes([0.125 / 2, 0.04 / 2 + 0.46, 0.75 / 2, 0.05 / 2])
ax_prmsl_cb.set_axis_off()
cb = fig.colorbar(
    PRMSL_img, ax=ax_prmsl_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Bottom left - SST
var = sCube.copy()
var.data = np.squeeze(generated[0][0, :, :, 1].numpy())
var.data = np.ma.masked_where(lm_20CR.data.mask == True, var.data, copy=True)
var = unnormalise(var, "TMPS") - 273.15
(dmin, dmax) = get_range("TMPS", month, var)
dmin -= 273.15 + 2
dmax -= 273.15 - 2
ax_sst = fig.add_axes([0.025 / 2, 0.11 / 2, 0.95 / 2, 0.78 / 2])
ax_sst.set_axis_off()
SST_img = plotFieldAxes(
    ax_sst,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
    plotCube=None,
)
ax_sst_cb = fig.add_axes([0.125 / 2, 0.04 / 2, 0.75 / 2, 0.04 / 2])
ax_sst_cb.set_axis_off()
cb = fig.colorbar(
    SST_img, ax=ax_sst_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Top right - PRATE
var = sCube.copy()
var.data = np.squeeze(generated[0][0, :, :, 3].numpy())
var.data = np.ma.masked_where(dm_hukg.data.mask == True, var.data, copy=True)
var = unnormalise(var, "PRATE") * 1000
(dmin, dmax) = get_range("PRATE", month, var)
dmin = 0
dmax *= 1000
ax_prate = fig.add_axes([0.025 / 2 + 0.5, 0.11 / 2 + 0.46, 0.95 / 2, 0.78 / 2])
ax_prate.set_axis_off()
PRATE_img = plotFieldAxes(
    ax_prate,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.rain,
    plotCube=None,
)
ax_prate_cb = fig.add_axes([0.125 / 2 + 0.5, 0.04 / 2 + 0.46, 0.75 / 2, 0.04 / 2])
ax_prate_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_img, ax=ax_prate_cb, location="bottom", orientation="horizontal", fraction=1.0
)
# Bottom left - T2m
var = sCube.copy()
var.data = np.squeeze(generated[0][0, :, :, 2].numpy())
var.data = np.ma.masked_where(dm_hukg.data.mask == True, var.data, copy=True)
var = unnormalise(var, "TMP2m") - 273.15
(dmin, dmax) = get_range("TMP2m", month, var)
dmin -= 273.15 + 2
dmax -= 273.15 - 2
ax_tmp2m = fig.add_axes([0.025 / 2 + 0.5, 0.11 / 2, 0.95 / 2, 0.78 / 2])
ax_tmp2m.set_axis_off()
TMP2m_img = plotFieldAxes(
    ax_tmp2m, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.balance,
)
ax_tmp2m_cb = fig.add_axes([0.125 / 2 + 0.5, 0.04 / 2, 0.75 / 2, 0.04 / 2])
ax_tmp2m_cb.set_axis_off()
cb = fig.colorbar(
    TMP2m_img, ax=ax_tmp2m_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Output as png
fig.savefig("%s/%s" % (".", "generated.png"))
