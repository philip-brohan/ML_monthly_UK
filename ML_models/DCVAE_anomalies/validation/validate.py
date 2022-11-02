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

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH

import warnings

warnings.filterwarnings("ignore", message=".*partition.*")
#warnings.filterwarnings("ignore", message=".*TransverseMercator.*")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=250)
parser.add_argument("--year", help="Test year", type=int, required=False, default=1969)
parser.add_argument("--month", help="Test month", type=int, required=False, default=3)
parser.add_argument(
    "--member", help="Test ensemble member", type=int, required=False, default=1
)
args = parser.parse_args()

sys.path.append("%s/../../.." % os.path.dirname(__file__))
from get_data.TWCR.TWCR_monthly_load import get_range
from plot_functions.plot_variable import plotFieldAxes
from plot_functions.plot_variable import plotScatterAxes

from autoencoderModel import DCVAE
from make_tensors.tensor_utils import cList_to_tensor
from make_tensors.tensor_utils import load_cList
from make_tensors.tensor_utils import sCube
from make_tensors.tensor_utils import lm_plot
from make_tensors.tensor_utils import normalise
from make_tensors.tensor_utils import unnormalise

# Load and standardise data
qd = load_cList(args.year, args.month, args.member)
ic_source = cList_to_tensor(qd, extrapolate=True)
ic_target = cList_to_tensor(qd, extrapolate=False)

autoencoder = DCVAE()
weights_dir = ("%s/models/Epoch_%04d") % (LSCRATCH, args.epoch,)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()

# Get autoencoded tensors
encoded = autoencoder.call(tf.reshape(ic_source, [1, 1440, 896, 4]),training=False)

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
axb.set_axis_off()
axb.add_patch(
    Rectangle((0, 0), 1, 1, facecolor=(1.0, 1.0, 1.0, 1), fill=True, zorder=1,)
)


# Top left - PRMSL original
varx = sCube.copy()
varx.data = np.squeeze(ic_target[:, :, 0].numpy())
varx = unnormalise(varx, "PRMSL") / 100
(dmin, dmax) = get_range("PRMSL", args.month, anomaly=True)
dmin /= 100
dmax /= 100
ax_prmsl = fig.add_axes([0.025 / 3, 0.125 / 4 + 0.75, 0.95 / 3, 0.85 / 4])
ax_prmsl.set_axis_off()
PRMSL_img = plotFieldAxes(
    ax_prmsl,
    varx,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.diff,
)
ax_prmsl_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.75, 0.75 / 3, 0.05 / 4])
ax_prmsl_cb.set_axis_off()
cb = fig.colorbar(
    PRMSL_img, ax=ax_prmsl_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Top centre - PRMSL encoded
vary = sCube.copy()
vary.data = np.squeeze(encoded[0, :, :, 0].numpy())
vary = unnormalise(vary, "PRMSL") / 100
ax_prmsl_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4 + 0.75, 0.95 / 3, 0.85 / 4])
ax_prmsl_e.set_axis_off()
PRMSL_e_img = plotFieldAxes(
    ax_prmsl_e,
    vary,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.diff,
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
ax_prmsl_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4 + 0.75, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_prmsl_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


# 2nd left - monthly_rainfall original
varx.data = np.squeeze(ic_target[:, :, 3].numpy())
varx.data = np.ma.masked_where(varx.data == 0.5, varx.data, copy=False)
varx = unnormalise(varx, "monthly_rainfall")
(dmin, dmax) = get_range("PRATE", args.month, anomaly=True)
dmin *= 86400*30 # Scale change from 20CR units to HadUKGrid
dmax *= 86400*30
ax_monthly_rainfall = fig.add_axes([0.025 / 3, 0.125 / 4 + 0.5, 0.95 / 3, 0.85 / 4])
ax_monthly_rainfall.set_axis_off()
monthly_rainfall_img = plotFieldAxes(
    ax_monthly_rainfall,
    varx,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.tarn,
)
ax_monthly_rainfall_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.5, 0.75 / 3, 0.05 / 4])
ax_monthly_rainfall_cb.set_axis_off()
cb = fig.colorbar(
    monthly_rainfall_img, ax=ax_monthly_rainfall_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd centre - monthly_rainfall encoded
vary.data = np.squeeze(encoded[0, :, :, 3].numpy())
vary.data = np.ma.masked_where(varx.data== 0.5, vary.data, copy=False)
vary = unnormalise(vary, "monthly_rainfall")
ax_monthly_rainfall_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4 + 0.5, 0.95 / 3, 0.85 / 4])
ax_monthly_rainfall_e.set_axis_off()
monthly_rainfall_e_img = plotFieldAxes(
    ax_monthly_rainfall_e,
    vary,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.tarn,
)
ax_monthly_rainfall_e_cb = fig.add_axes([0.125 / 3 + 1 / 3, 0.05 / 4 + 0.5, 0.75 / 3, 0.05 / 4])
ax_monthly_rainfall_e_cb.set_axis_off()
cb = fig.colorbar(
    monthly_rainfall_e_img,
    ax=ax_monthly_rainfall_e_cb,
    location="bottom",
    orientation="horizontal",
    fraction=1.0,
)

# 2nd right - monthly_rainfall scatter
ax_monthly_rainfall_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4 + 0.5, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_monthly_rainfall_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


# 3rd left - T2m original
varx.data = np.squeeze(ic_target[:, :, 2].numpy())
varx.data = np.ma.masked_where(varx.data == 0.5, varx.data, copy=False)
varx = unnormalise(varx, "monthly_meantemp")
(dmin, dmax) = get_range("TMP2m", args.month, anomaly=True)
dmin += 2
dmax -= 2
ax_t2m = fig.add_axes([0.025 / 3, 0.125 / 4 + 0.25, 0.95 / 3, 0.85 / 4])
ax_t2m.set_axis_off()
T2m_img = plotFieldAxes(
    ax_t2m,
    varx,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)
ax_t2m_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.25, 0.75 / 3, 0.05 / 4])
ax_t2m_cb.set_axis_off()
cb = fig.colorbar(
    T2m_img, ax=ax_t2m_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 3rd centre - T2m encoded
vary.data = np.squeeze(encoded[0, :, :, 2].numpy())
vary.data = np.ma.masked_where(varx.data == 0.5, vary.data, copy=False)
vary = unnormalise(vary, "monthly_meantemp")
ax_t2m_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4 + 0.25, 0.95 / 3, 0.85 / 4])
ax_t2m_e.set_axis_off()
T2m_e_img = plotFieldAxes(
    ax_t2m_e,
    vary,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)
ax_t2m_e_cb = fig.add_axes([0.125 / 3 + 1 / 3, 0.05 / 4 + 0.25, 0.75 / 3, 0.05 / 4])
ax_t2m_e_cb.set_axis_off()
cb = fig.colorbar(
    T2m_e_img, ax=ax_t2m_e_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 3rd right - T2m scatter
ax_t2m_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4 + 0.25, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_t2m_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


# Bottom left - SST original
varx.data = np.squeeze(ic_target[:, :, 1].numpy())
varx.data = np.ma.masked_where(varx.data==0.5, varx.data, copy=False)
varx = unnormalise(varx, "SST")
(dmin, dmax) = get_range("SST", args.month, anomaly=True)
dmin += 2
dmax -= 2
ax_sst = fig.add_axes([0.025 / 3, 0.125 / 4, 0.95 / 3, 0.85 / 4])
ax_sst.set_axis_off()
SST_img = plotFieldAxes(
    ax_sst,
    varx,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)
ax_sst_cb = fig.add_axes([0.125 / 3, 0.05 / 4, 0.75 / 3, 0.05 / 4])
ax_sst_cb.set_axis_off()
cb = fig.colorbar(
    SST_img, ax=ax_sst_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd centre - SST encoded
vary.data = encoded.numpy()[0, :, :, 1]
vary.data = np.ma.masked_where(varx.data == 0.5, vary.data, copy=False)
vary = unnormalise(vary, "SST")
ax_sst_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4, 0.95 / 3, 0.85 / 4])
ax_sst_e.set_axis_off()
SST_e_img = plotFieldAxes(
    ax_sst_e,
    vary,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)
ax_sst_e_cb = fig.add_axes([0.125 / 3 + 1 / 3, 0.05 / 4, 0.75 / 3, 0.05 / 4])
ax_sst_e_cb.set_axis_off()
cb = fig.colorbar(
    SST_e_img, ax=ax_sst_e_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd right - SST scatter
ax_sst_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_sst_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


fig.savefig("comparison.png")
