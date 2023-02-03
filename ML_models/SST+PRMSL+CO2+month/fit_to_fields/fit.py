#!/usr/bin/env python

# Find a point in latent space that maximises the fit to some given input fields,
#  and plot the fitted state.


import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import cmocean

import warnings

warnings.filterwarnings("ignore", message=".*partition.*")
warnings.filterwarnings("ignore", message=".*TransverseMercator.*")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=770)
parser.add_argument(
    "--year", help="Year to fit to", type=int, required=False, default=1969
)
parser.add_argument(
    "--month", help="Month to fit to", type=int, required=False, default=3
)
parser.add_argument(
    "--member", help="Member to fit to", type=int, required=False, default=1
)
parser.add_argument(
    "--PRMSL", help="Fit to PRMSL?", dest="PRMSL", default=False, action="store_true"
)
parser.add_argument(
    "--SST", help="Fit to SST?", dest="SST", default=False, action="store_true"
)
parser.add_argument(
    "--TMP2m", help="Fit to TMP2m?", dest="TMP2m", default=False, action="store_true"
)
parser.add_argument(
    "--PRATE", help="Fit to PRATE?", dest="PRATE", default=False, action="store_true"
)
parser.add_argument(
    "--iter",
    help="No. of iterations",
    type=int,
    required=False,
    default=1000,
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

# Load and standardise data
qd = load_cList(args.year, args.month, args.member)
ict = cList_to_tensor(qd, lm_20CR.data.mask, dm_hukg.data.mask)

# Load the model specification
sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH
from autoencoderModel import DCVAE
from makeDataset import normalise_co2
from makeDataset import normalise_month

autoencoder = DCVAE()
weights_dir = ("%s/models/Epoch_%04d") % (
    LSCRATCH,
    args.epoch,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()

# We are using the model in inference mode - (does this have any effect?)
autoencoder.decoder.trainable = False
for layer in autoencoder.decoder.layers:
    layer.trainable = False
autoencoder.decoder.compile()

latent = tf.Variable(tf.random.normal(shape=(1, autoencoder.latent_dim)))
target = tf.constant(tf.reshape(ict, [1, 1440, 896, 4]))
co2t = tf.reshape(
    tf.convert_to_tensor(normalise_co2("%04d" % args.year), np.float32), [1, 1]
)
cmt = tf.reshape(
    tf.convert_to_tensor(normalise_month("000-%02d" % args.month), np.float32), [1, 1]
)


def decodeFit():
    result = 0.0
    decoded = autoencoder.decode(tf.concat([latent, co2t, cmt], axis=1))
    if args.PRMSL:
        result = result + tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(decoded[:, :, :, 0], target[:, :, :, 0])
        )
    if args.SST:
        result = result + tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(
                tf.boolean_mask(
                    decoded[:, :, :, 1], np.invert(lm_20CR.data.mask), axis=1
                ),
                tf.boolean_mask(
                    target[:, :, :, 1], np.invert(lm_20CR.data.mask), axis=1
                ),
            )
        )
    if args.TMP2m:
        result = result + tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(
                tf.boolean_mask(
                    decoded[:, :, :, 2], np.invert(dm_hukg.data.mask), axis=1
                ),
                tf.boolean_mask(
                    target[:, :, :, 2], np.invert(dm_hukg.data.mask), axis=1
                ),
            )
        )
    if args.PRATE:
        result = result + tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(
                tf.boolean_mask(
                    decoded[:, :, :, 3], np.invert(dm_hukg.data.mask), axis=1
                ),
                tf.boolean_mask(
                    target[:, :, :, 3], np.invert(dm_hukg.data.mask), axis=1
                ),
            )
        )
    return result


if args.PRMSL or args.SST or args.TMP2m or args.PRATE:
    loss = tfp.math.minimize(
        decodeFit,
        trainable_variables=[latent],
        num_steps=args.iter,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
    )

encoded = autoencoder.decode(tf.concat([latent, co2t, cmt], axis=1))

# Make the plot - same as for validation script
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
    Rectangle(
        (0, 1),
        1,
        1,
        facecolor=(0.6, 0.6, 0.6, 1),
        fill=True,
        zorder=1,
    )
)


# Top left - PRMSL original
if args.PRMSL:
    ax_back = fig.add_axes([0.00, 0.75, 1.0, 0.25])
    ax_back.set_axis_off()
    ax_back.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=(0.0, 0.0, 0.0, 0.3),
            fill=True,
            zorder=1,
        )
    )
var = sCube.copy()
var.data = np.squeeze(ict[:, :, 0].numpy())
var = unnormalise(var, "PRMSL") / 100
(dmin, dmax) = get_range("PRMSL", args.month, var)
dmin /= 100
dmax /= 100
ax_prmsl = fig.add_axes([0.025 / 3, 0.125 / 4 + 0.75, 0.95 / 3, 0.85 / 4])
ax_prmsl.set_axis_off()
PRMSL_img = plotFieldAxes(
    ax_prmsl,
    var,
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
var.data = np.squeeze(encoded[0, :, :, 0].numpy())
var = unnormalise(var, "PRMSL") / 100
ax_prmsl_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4 + 0.75, 0.95 / 3, 0.85 / 4])
ax_prmsl_e.set_axis_off()
PRMSL_e_img = plotFieldAxes(
    ax_prmsl_e,
    var,
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
varx = sCube.copy()
varx.data = np.squeeze(ict[:, :, 0].numpy())
varx = unnormalise(varx, "PRMSL") / 100
vary = sCube.copy()
vary.data = np.squeeze(encoded[0, :, :, 0].numpy())
vary = unnormalise(vary, "PRMSL") / 100
ax_prmsl_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4 + 0.75, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_prmsl_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


# 2nd left - PRATE original
if args.PRATE:
    ax_back = fig.add_axes([0.00, 0.5, 1.0, 0.25])
    ax_back.set_axis_off()
    ax_back.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=(0.0, 0.0, 0.0, 0.3),
            fill=True,
            zorder=1,
        )
    )
var = sCube.copy()
var.data = np.squeeze(ict[:, :, 3].numpy())
var.data = np.ma.masked_where(dm_hukg.data == 0, var.data, copy=False)
var = unnormalise(var, "PRATE") * 1000
(dmin, dmax) = get_range("PRATE", args.month, var)
dmin = 0
dmax *= 1000
ax_prate = fig.add_axes([0.025 / 3, 0.125 / 4 + 0.5, 0.95 / 3, 0.85 / 4])
ax_prate.set_axis_off()
PRATE_img = plotFieldAxes(
    ax_prate,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.rain,
)
ax_prate_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.5, 0.75 / 3, 0.05 / 4])
ax_prate_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_img, ax=ax_prate_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd centre - PRATE encoded
var.data = np.squeeze(encoded[0, :, :, 3].numpy())
var.data = np.ma.masked_where(dm_hukg.data == 0, var.data, copy=False)
var = unnormalise(var, "PRATE") * 1000
ax_prate_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4 + 0.5, 0.95 / 3, 0.85 / 4])
ax_prate_e.set_axis_off()
PRATE_e_img = plotFieldAxes(
    ax_prate_e,
    var,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.rain,
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
varx = sCube.copy()
varx.data = np.squeeze(ict[:, :, 3].numpy())
varx.data = np.ma.masked_where(dm_hukg.data == 0, varx.data, copy=False)
varx = unnormalise(varx, "PRATE") * 1000
vary = sCube.copy()
vary.data = np.squeeze(encoded[0, :, :, 3].numpy())
vary.data = np.ma.masked_where(dm_hukg.data == 0, vary.data, copy=False)
vary = unnormalise(vary, "PRATE") * 1000
ax_prate_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4 + 0.5, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_prate_s, varx, vary, vMin=0.001, vMax=dmax, bins=None)


# 3rd left - T2m original
if args.TMP2m:
    ax_back = fig.add_axes([0.00, 0.25, 1.0, 0.25])
    ax_back.set_axis_off()
    ax_back.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=(0.0, 0.0, 0.0, 0.3),
            fill=True,
            zorder=1,
        )
    )
var = sCube.copy()
var.data = np.squeeze(ict[:, :, 2].numpy())
var.data = np.ma.masked_where(dm_hukg.data == 0, var.data, copy=False)
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
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)
ax_t2m_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.25, 0.75 / 3, 0.05 / 4])
ax_t2m_cb.set_axis_off()
cb = fig.colorbar(
    T2m_img, ax=ax_t2m_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 3rd centre - T2m encoded
var.data = np.squeeze(encoded[0, :, :, 2].numpy())
var.data = np.ma.masked_where(dm_hukg.data == 0, var.data, copy=False)
var = unnormalise(var, "TMP2m") - 273.15
ax_t2m_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4 + 0.25, 0.95 / 3, 0.85 / 4])
ax_t2m_e.set_axis_off()
T2m_e_img = plotFieldAxes(
    ax_t2m_e,
    var,
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
varx = sCube.copy()
varx.data = np.squeeze(ict[:, :, 2].numpy())
varx.data = np.ma.masked_where(dm_hukg.data == 0, varx.data, copy=False)
varx = unnormalise(varx, "TMP2m") - 273.15
vary = sCube.copy()
vary.data = np.squeeze(encoded[0, :, :, 2].numpy())
vary.data = np.ma.masked_where(dm_hukg.data == 0, vary.data, copy=False)
vary = unnormalise(vary, "TMP2m") - 273.15
ax_t2m_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4 + 0.25, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_t2m_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


# Bottom left - SST original
if args.SST:
    ax_back = fig.add_axes([0.00, 0.00, 1.0, 0.25])
    ax_back.set_axis_off()
    ax_back.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=(0.0, 0.0, 0.0, 0.3),
            fill=True,
            zorder=1,
        )
    )
var = sCube.copy()
var.data = np.squeeze(ict[:, :, 1].numpy())
var.data = np.ma.masked_where(lm_20CR.data > 0, var.data, copy=False)
var = unnormalise(var, "TMPS") - 273.15
(dmin, dmax) = get_range("TMPS", args.month, var)
dmin -= 273.15 + 2
dmax -= 273.15 - 2
ax_sst = fig.add_axes([0.025 / 3, 0.125 / 4, 0.95 / 3, 0.85 / 4])
ax_sst.set_axis_off()
SST_img = plotFieldAxes(
    ax_sst,
    var,
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
var.data = encoded.numpy()[0, :, :, 1]
var.data = np.ma.masked_where(lm_20CR.data > 0, var.data, copy=False)
var = unnormalise(var, "TMPS") - 273.15
ax_sst_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4, 0.95 / 3, 0.85 / 4])
ax_sst_e.set_axis_off()
SST_e_img = plotFieldAxes(
    ax_sst_e,
    var,
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
varx = sCube.copy()
varx.data = np.squeeze(ict[:, :, 1].numpy())
varx.data = np.ma.masked_where(lm_20CR.data > 0, varx.data, copy=False)
varx = unnormalise(varx, "TMPS") - 273.15
vary = sCube.copy()
vary.data = np.squeeze(encoded[0, :, :, 1].numpy())
vary.data = np.ma.masked_where(lm_20CR.data > 0, vary.data, copy=False)
vary = unnormalise(vary, "TMPS") - 273.15
ax_sst_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_sst_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


fig.savefig("fit.png")
