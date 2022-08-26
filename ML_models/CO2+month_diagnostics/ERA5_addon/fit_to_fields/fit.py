#!/usr/bin/env python

# Find a point in latent space that maximises the fit to some given input fields,
#  and plot the fitted state.
# This version fits to HadUK-Grid or ERA5 fields.


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
parser.add_argument(
    "--epoch_H", help="VAE_Epoch", type=int, required=False, default=299
)
parser.add_argument(
    "--epoch_E", help="Generator epoch", type=int, required=False, default=199
)
parser.add_argument(
    "--year", help="Year of data to fit to", type=int, required=False, default=1969
)
parser.add_argument(
    "--month", help="Month of data to fit to", type=int, required=False, default=3
)
parser.add_argument(
    "--member", help="Member to fit to", type=int, required=False, default=1
)
parser.add_argument(
    "--PRMSL_H",
    help="Fit to HadUK-Grid PRMSL?",
    dest="PRMSL_H",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--SST_H",
    help="Fit to HadUK-Grid SST?",
    dest="SST_H",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--TMP2m_H",
    help="Fit to HadUK-Grid TMP2m?",
    dest="TMP2m_H",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--PRATE_H",
    help="Fit to HadUK-Grid PRATE?",
    dest="PRATE_H",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--PRMSL_E",
    help="Fit to ERA5 PRMSL?",
    dest="PRMSL_E",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--SST_E", help="Fit to ERA5 SST?", dest="SST_E", default=False, action="store_true"
)
parser.add_argument(
    "--TMP2m_E",
    help="Fit to ERA5 TMP2m?",
    dest="TMP2m_E",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--PRATE_E",
    help="Fit to ERA5 PRATE?",
    dest="PRATE_E",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--CO2", help="Fit to CO2?", dest="CO2", default=False, action="store_true"
)
parser.add_argument(
    "--MNTH", help="Fit to Month?", dest="MNTH", default=False, action="store_true"
)
parser.add_argument(
    "--iter", help="No. of iterations", type=int, required=False, default=100,
)
args = parser.parse_args()

sys.path.append("%s/../../../../get_data" % os.path.dirname(__file__))
from HUKG_monthly_load import load_cList as H_load_cList
from ERA5_monthly_load import load_cList as E_load_cList
from HUKG_monthly_load import sCube
from HUKG_monthly_load import lm_plot
from HUKG_monthly_load import lm_20CR
from ERA5_monthly_load import lm_ERA5
from HUKG_monthly_load import dm_hukg
from TWCR_monthly_load import get_range

sys.path.append("%s/../../make_tensors" % os.path.dirname(__file__))
from tensor_utils import cList_to_tensor as H_cList_to_tensor
from tensor_utils import normalise
from tensor_utils import unnormalise

sys.path.append("%s/../make_tensors" % os.path.dirname(__file__))
from E_tensor_utils import cList_to_tensor as E_cList_to_tensor

sys.path.append("%s/../../plot_quad" % os.path.dirname(__file__))
from plot_variable import plotFieldAxes
from plot_variable import plotScatterAxes

# Load and standardise data - might be unavailable
try:
    qd_H = H_load_cList(args.year, args.month, args.member)
    ict_H = H_cList_to_tensor(qd_H, lm_20CR.data.mask, dm_hukg.data.mask)
except Exception:
    qd_H = None
    ict_H = None
    if args.PRMSL_H or args.SST_H or args.TMP2m_H or args.PRATE_H:
        raise Exception("HadUK-Grid data not available")
try:
    qd_E = E_load_cList(args.year, args.month)
    ict_E = E_cList_to_tensor(qd_E, lm_ERA5.data.mask, dm_hukg.data.mask)
except Exception:
    qd_E = None
    ict_E = None
    if args.PRMSL_E or args.SST_E or args.TMP2m_E or args.PRATE_E:
        raise Exception("ERA5 data not available")

# Load the model specifications
sys.path.append("%s/../.." % os.path.dirname(__file__))
from localise import LSCRATCH
from autoencoderModel import DCVAE
from autoencoderModel import PRMSL_scale
from autoencoderModel import SST_scale
from autoencoderModel import T2M_scale
from autoencoderModel import PRATE_scale
from autoencoderModel import CO2_scale
from autoencoderModel import MONTH_scale
from makeDataset import normalise_co2
from makeDataset import normalise_month

autoencoder = DCVAE()
weights_dir = ("%s/models/Epoch_%04d") % (LSCRATCH, args.epoch_H,)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
load_status.assert_existing_objects_matched()
autoencoder.trainable = False

sys.path.append("%s/.." % os.path.dirname(__file__))
from generatorModel import DCG

generator = DCG()
weights_dir = ("%s/models_ERA5_generator/Epoch_%04d") % (LSCRATCH, args.epoch_E,)
load_status = generator.load_weights("%s/ckpt" % weights_dir)
load_status.assert_existing_objects_matched()
generator.trainable = False

latent = tf.Variable(tf.random.normal(shape=(1, autoencoder.latent_dim)))
if ict_H is not None:
    target_H = tf.constant(tf.reshape(ict_H, [1, 1440, 896, 4]))
else:
    target_H = None
if ict_E is not None:
    target_E = tf.constant(tf.reshape(ict_E, [1, 1440, 896, 4]))
else:
    target_E = None
co2t = tf.reshape(
    tf.convert_to_tensor(normalise_co2("%04d" % args.year), np.float32), [1, 14]
)
cmt = tf.reshape(
    tf.convert_to_tensor(normalise_month("000-%02d" % args.month), np.float32), [1, 12]
)


def decodeFit():
    result = 0.0
    decoded = autoencoder.decode(latent)
    generated = generator.decode(latent)
    if args.PRMSL_H:
        result = (
            result
            + tf.reduce_mean(
                tf.keras.metrics.mean_squared_error(
                    decoded[0][:, :, :, 0], target_H[:, :, :, 0]
                )
            )
            * PRMSL_scale
        )
    if args.PRMSL_E:
        result = (
            result
            + tf.reduce_mean(
                tf.keras.metrics.mean_squared_error(
                    generated[:, :, :, 0], target_E[:, :, :, 0]
                )
            )
            * PRMSL_scale
        )
    if args.SST_H:
        result = (
            result
            + tf.reduce_mean(
                tf.keras.metrics.mean_squared_error(
                    tf.boolean_mask(
                        decoded[0][:, :, :, 1], np.invert(lm_20CR.data.mask), axis=1
                    ),
                    tf.boolean_mask(
                        target_H[:, :, :, 1], np.invert(lm_20CR.data.mask), axis=1
                    ),
                )
            )
            * SST_scale
        )
    if args.SST_E:
        result = (
            result
            + tf.reduce_mean(
                tf.keras.metrics.mean_squared_error(
                    tf.boolean_mask(
                        generated[:, :, :, 1], np.invert(lm_ERA5.data.mask), axis=1
                    ),
                    tf.boolean_mask(
                        target_E[:, :, :, 1], np.invert(lm_ERA5.data.mask), axis=1
                    ),
                )
            )
            * SST_scale
        )
    if args.TMP2m_H:
        result = (
            result
            + tf.reduce_mean(
                tf.keras.metrics.mean_squared_error(
                    tf.boolean_mask(
                        decoded[0][:, :, :, 2], np.invert(dm_hukg.data.mask), axis=1
                    ),
                    tf.boolean_mask(
                        target_H[:, :, :, 2], np.invert(dm_hukg.data.mask), axis=1
                    ),
                )
            )
            * T2M_scale
        )
    if args.TMP2m_E:
        result = (
            result
            + tf.reduce_mean(
                tf.keras.metrics.mean_squared_error(
                    tf.boolean_mask(
                        generated[:, :, :, 2], np.invert(dm_hukg.data.mask), axis=1
                    ),
                    tf.boolean_mask(
                        target_E[:, :, :, 2], np.invert(dm_hukg.data.mask), axis=1
                    ),
                )
            )
            * T2M_scale
        )
    if args.PRATE_H:
        result = (
            result
            + tf.reduce_mean(
                tf.keras.metrics.mean_squared_error(
                    tf.boolean_mask(
                        decoded[0][:, :, :, 3], np.invert(dm_hukg.data.mask), axis=1
                    ),
                    tf.boolean_mask(
                        target_H[:, :, :, 3], np.invert(dm_hukg.data.mask), axis=1
                    ),
                )
            )
            * PRATE_scale
        )
    if args.PRATE_E:
        result = (
            result
            + tf.reduce_mean(
                tf.keras.metrics.mean_squared_error(
                    tf.boolean_mask(
                        generated[:, :, :, 3], np.invert(dm_hukg.data.mask), axis=1
                    ),
                    tf.boolean_mask(
                        target_E[:, :, :, 3], np.invert(dm_hukg.data.mask), axis=1
                    ),
                )
            )
            * PRATE_scale
        )
    if args.CO2:
        result = result + (
            tf.keras.metrics.categorical_crossentropy(decoded[1], co2t) * CO2_scale
        )
    if args.MNTH:
        result = result + (
            tf.keras.metrics.categorical_crossentropy(decoded[2], cmt) * MONTH_scale
        )
    return result


if (
    args.PRMSL_H
    or args.SST_H
    or args.TMP2m_H
    or args.PRATE_H
    or args.PRMSL_E
    or args.SST_E
    or args.TMP2m_E
    or args.PRATE_E
    or args.CO2
    or args.MNTH
):
    loss = tfp.math.minimize(
        decodeFit,
        trainable_variables=[latent],
        num_steps=args.iter,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
    )

encoded = autoencoder.decode(latent)
generated = generator.decode(latent)

# Make the plot - same as for validation script
fig = Figure(
    figsize=(30, 23),
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
    Rectangle((0, 0), 1, 1, facecolor=(0.95, 0.95, 0.95, 1), fill=True, zorder=1,)
)

# Top row - date, CO2 and month diagnostics

axb.text(0.03 / 2, 0.97, "%04d/%02d" % (args.year, args.month), fontsize=30, zorder=10)

if args.CO2:
    ax_back = fig.add_axes([0.15 / 2, 0.96, 0.45 / 2, 0.04])
    ax_back.set_axis_off()
    ax_back.add_patch(
        Rectangle((0, 0), 1, 1, facecolor=(0.0, 0.0, 0.0, 0.3), fill=True, zorder=1,)
    )
axb.text(
    0.16 / 2, 0.97, "CO2", fontsize=30, zorder=10,
)
ax_co2 = fig.add_axes([0.24 / 2, 0.965, 0.335 / 2, 0.028], xlim=(0, 15), ylim=(0, 1))
ax_co2.bar(
    [x - 0.2 for x in list(range(1, 15))],
    co2t[0, :].numpy(),
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

if args.MNTH:
    ax_back = fig.add_axes([0.60 / 2, 0.96, 0.4 / 2, 0.04])
    ax_back.set_axis_off()
    ax_back.add_patch(
        Rectangle((0, 0), 1, 1, facecolor=(0.0, 0.0, 0.0, 0.3), fill=True, zorder=1,)
    )
axb.text(
    0.61 / 2, 0.97, "Month", fontsize=30, zorder=10,
)
ax_mnth = fig.add_axes([0.7 / 2, 0.965, 0.29 / 2, 0.028], xlim=(0, 13), ylim=(0, 1))
ax_mnth.bar(
    [x - 0.2 for x in list(range(1, 13))],
    cmt[0, :].numpy(),
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
if args.PRMSL_H:
    ax_back = fig.add_axes([0.00 / 2, 0.72, 1.0 / 2, 0.24])
    ax_back.set_axis_off()
    ax_back.add_patch(
        Rectangle((0, 0), 1, 1, facecolor=(0.0, 0.0, 0.0, 0.3), fill=True, zorder=1,)
    )
var = sCube.copy()
var.data = np.squeeze(ict_H[:, :, 0].numpy())
var = unnormalise(var, "PRMSL") / 100
(dmin, dmax) = get_range("PRMSL", args.month, var)
dmin /= 100
dmax /= 100
ax_prmsl = fig.add_axes([0.025 / 6, 0.12 / 4 + 0.72, 0.95 / 6, 0.81 / 4])
ax_prmsl.set_axis_off()
PRMSL_img = plotFieldAxes(
    ax_prmsl, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.diff,
)
ax_prmsl_cb = fig.add_axes([0.125 / 6, 0.05 / 4 + 0.72, 0.75 / 6, 0.05 / 4])
ax_prmsl_cb.set_axis_off()
cb = fig.colorbar(
    PRMSL_img, ax=ax_prmsl_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Top centre - PRMSL encoded
var.data = np.squeeze(encoded[0][0, :, :, 0].numpy())
var = unnormalise(var, "PRMSL") / 100
ax_prmsl_e = fig.add_axes([0.025 / 6 + 1 / 6, 0.12 / 4 + 0.72, 0.95 / 6, 0.81 / 4])
ax_prmsl_e.set_axis_off()
PRMSL_e_img = plotFieldAxes(
    ax_prmsl_e, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.diff,
)
ax_prmsl_e_cb = fig.add_axes([0.125 / 6 + 1 / 6, 0.05 / 4 + 0.72, 0.75 / 6, 0.05 / 4])
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
varx.data = np.squeeze(ict_H[:, :, 0].numpy())
varx = unnormalise(varx, "PRMSL") / 100
vary = sCube.copy()
vary.data = np.squeeze(encoded[0][0, :, :, 0].numpy())
vary = unnormalise(vary, "PRMSL") / 100
ax_prmsl_s = fig.add_axes(
    [0.025 / 6 + 2 / 6 + 0.06 / 2, 0.12 / 4 + 0.72, 0.95 / 6 - 0.06 / 2, 0.81 / 4]
)
plotScatterAxes(ax_prmsl_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)

# Top left 2 - PRMSL original ERA5
if args.PRMSL_E:
    ax_back = fig.add_axes([0.00 / 2 + 0.5, 0.72, 1.0 / 2, 0.24])
    ax_back.set_axis_off()
    ax_back.add_patch(
        Rectangle((0, 0), 1, 1, facecolor=(0.0, 0.0, 0.0, 0.3), fill=True, zorder=1,)
    )
var = sCube.copy()
var.data = np.squeeze(ict_E[:, :, 0].numpy())
var = unnormalise(var, "PRMSL") / 100
(dmin, dmax) = get_range("PRMSL", args.month, var)
dmin /= 100
dmax /= 100
ax_prmsl = fig.add_axes([0.025 / 6 + 0.5, 0.12 / 4 + 0.72, 0.95 / 6, 0.81 / 4])
ax_prmsl.set_axis_off()
PRMSL_img = plotFieldAxes(
    ax_prmsl, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.diff,
)
ax_prmsl_cb = fig.add_axes([0.125 / 6 + 0.5, 0.05 / 4 + 0.72, 0.75 / 6, 0.05 / 4])
ax_prmsl_cb.set_axis_off()
cb = fig.colorbar(
    PRMSL_img, ax=ax_prmsl_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Top centre 2 - PRMSL encoded ERA5
var.data = np.squeeze(generated[0, :, :, 0].numpy())
var = unnormalise(var, "PRMSL") / 100
ax_prmsl_e = fig.add_axes(
    [0.025 / 6 + 1 / 6 + 0.5, 0.12 / 4 + 0.72, 0.95 / 6, 0.81 / 4]
)
ax_prmsl_e.set_axis_off()
PRMSL_e_img = plotFieldAxes(
    ax_prmsl_e, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.diff,
)
ax_prmsl_e_cb = fig.add_axes(
    [0.125 / 6 + 1 / 6 + 0.5, 0.05 / 4 + 0.72, 0.75 / 6, 0.05 / 4]
)
ax_prmsl_e_cb.set_axis_off()
cb = fig.colorbar(
    PRMSL_e_img,
    ax=ax_prmsl_e_cb,
    location="bottom",
    orientation="horizontal",
    fraction=1.0,
)

# Top right 2 - PRMSL scatter ERA5
varx = sCube.copy()
varx.data = np.squeeze(ict_E[:, :, 0].numpy())
varx = unnormalise(varx, "PRMSL") / 100
vary = sCube.copy()
vary.data = np.squeeze(generated[0, :, :, 0].numpy())
vary = unnormalise(vary, "PRMSL") / 100
ax_prmsl_s = fig.add_axes(
    [0.025 / 6 + 2 / 6 + 0.06 / 2 + 0.5, 0.12 / 4 + 0.72, 0.95 / 6 - 0.06 / 2, 0.81 / 4]
)
plotScatterAxes(ax_prmsl_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


# 2nd left - PRATE original
if args.PRATE_H:
    ax_back = fig.add_axes([0.00 / 2, 0.48, 1.0 / 2, 0.24])
    ax_back.set_axis_off()
    ax_back.add_patch(
        Rectangle((0, 0), 1, 1, facecolor=(0.0, 0.0, 0.0, 0.3), fill=True, zorder=1,)
    )
var = sCube.copy()
var.data = np.squeeze(ict_H[:, :, 3].numpy())
var.data = np.ma.masked_where(dm_hukg.data == 0, var.data, copy=False)
var = unnormalise(var, "PRATE") * 1000
(dmin, dmax) = get_range("PRATE", args.month, var)
dmin = 0
dmax *= 1000
ax_prate = fig.add_axes([0.025 / 6, 0.12 / 4 + 0.48, 0.95 / 6, 0.81 / 4])
ax_prate.set_axis_off()
PRATE_img = plotFieldAxes(
    ax_prate, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.rain,
)
ax_prate_cb = fig.add_axes([0.125 / 6, 0.05 / 4 + 0.48, 0.75 / 6, 0.05 / 4])
ax_prate_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_img, ax=ax_prate_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd centre - PRATE encoded
var.data = np.squeeze(encoded[0][0, :, :, 3].numpy())
var.data = np.ma.masked_where(dm_hukg.data == 0, var.data, copy=False)
var = unnormalise(var, "PRATE") * 1000
ax_prate_e = fig.add_axes([0.025 / 6 + 1 / 6, 0.125 / 4 + 0.48, 0.95 / 6, 0.81 / 4])
ax_prate_e.set_axis_off()
PRATE_e_img = plotFieldAxes(
    ax_prate_e, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.rain,
)
ax_prate_e_cb = fig.add_axes([0.125 / 6 + 1 / 6, 0.05 / 4 + 0.48, 0.75 / 6, 0.05 / 4])
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
varx.data = np.squeeze(ict_H[:, :, 3].numpy())
varx.data = np.ma.masked_where(dm_hukg.data == 0, varx.data, copy=False)
varx = unnormalise(varx, "PRATE") * 1000
vary = sCube.copy()
vary.data = np.squeeze(encoded[0][0, :, :, 3].numpy())
vary.data = np.ma.masked_where(dm_hukg.data == 0, vary.data, copy=False)
vary = unnormalise(vary, "PRATE") * 1000
ax_prate_s = fig.add_axes(
    [0.025 / 6 + 2 / 6 + 0.06 / 2, 0.12 / 4 + 0.48, 0.95 / 6 - 0.06 / 2, 0.81 / 4]
)
plotScatterAxes(ax_prate_s, varx, vary, vMin=0.001, vMax=dmax, bins=None)


# 2nd left 2 - PRATE original ERA5
if args.PRATE_E:
    ax_back = fig.add_axes([0.00 / 2 + 0.5, 0.48, 1.0 / 2, 0.24])
    ax_back.set_axis_off()
    ax_back.add_patch(
        Rectangle((0, 0), 1, 1, facecolor=(0.0, 0.0, 0.0, 0.3), fill=True, zorder=1,)
    )
var = sCube.copy()
var.data = np.squeeze(ict_E[:, :, 3].numpy())
var.data = np.ma.masked_where(dm_hukg.data == 0, var.data, copy=False)
var = unnormalise(var, "PRATE") * 1000
(dmin, dmax) = get_range("PRATE", args.month, var)
dmin = 0
dmax *= 1000
ax_prate = fig.add_axes([0.025 / 6 + 0.5, 0.12 / 4 + 0.48, 0.95 / 6, 0.81 / 4])
ax_prate.set_axis_off()
PRATE_img = plotFieldAxes(
    ax_prate, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.rain,
)
ax_prate_cb = fig.add_axes([0.125 / 6 + 0.5, 0.05 / 4 + 0.48, 0.75 / 6, 0.05 / 4])
ax_prate_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_img, ax=ax_prate_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd centre 2 - PRATE encoded ERA5
var.data = np.squeeze(generated[0, :, :, 3].numpy())
var.data = np.ma.masked_where(dm_hukg.data == 0, var.data, copy=False)
var = unnormalise(var, "PRATE") * 1000
ax_prate_e = fig.add_axes(
    [0.025 / 6 + 1 / 6 + 0.5, 0.125 / 4 + 0.48, 0.95 / 6, 0.81 / 4]
)
ax_prate_e.set_axis_off()
PRATE_e_img = plotFieldAxes(
    ax_prate_e, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.rain,
)
ax_prate_e_cb = fig.add_axes(
    [0.125 / 6 + 1 / 6 + 0.5, 0.05 / 4 + 0.48, 0.75 / 6, 0.05 / 4]
)
ax_prate_e_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_e_img,
    ax=ax_prate_e_cb,
    location="bottom",
    orientation="horizontal",
    fraction=1.0,
)

# 2nd right 2 - PRATE scatter ERA5
varx = sCube.copy()
varx.data = np.squeeze(ict_E[:, :, 3].numpy())
varx.data = np.ma.masked_where(dm_hukg.data == 0, varx.data, copy=False)
varx = unnormalise(varx, "PRATE") * 1000
vary = sCube.copy()
vary.data = np.squeeze(generated[0, :, :, 3].numpy())
vary.data = np.ma.masked_where(dm_hukg.data == 0, vary.data, copy=False)
vary = unnormalise(vary, "PRATE") * 1000
ax_prate_s = fig.add_axes(
    [0.025 / 6 + 2 / 6 + 0.06 / 2 + 0.5, 0.12 / 4 + 0.48, 0.95 / 6 - 0.06 / 2, 0.81 / 4]
)
plotScatterAxes(ax_prate_s, varx, vary, vMin=0.001, vMax=dmax, bins=None)


# 3rd left - T2m original
if args.TMP2m_H:
    ax_back = fig.add_axes([0.00 / 2, 0.24, 1.0 / 2, 0.24])
    ax_back.set_axis_off()
    ax_back.add_patch(
        Rectangle((0, 0), 1, 1, facecolor=(0.0, 0.0, 0.0, 0.3), fill=True, zorder=1,)
    )
var = sCube.copy()
var.data = np.squeeze(ict_H[:, :, 2].numpy())
var.data = np.ma.masked_where(dm_hukg.data == 0, var.data, copy=False)
var = unnormalise(var, "TMP2m") - 273.15
(dmin, dmax) = get_range("TMP2m", args.month, var)
dmin -= 273.15 + 2
dmax -= 273.15 - 2
ax_t2m = fig.add_axes([0.025 / 6, 0.12 / 4 + 0.24, 0.95 / 6, 0.81 / 4])
ax_t2m.set_axis_off()
T2m_img = plotFieldAxes(
    ax_t2m, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.balance,
)
ax_t2m_cb = fig.add_axes([0.125 / 6, 0.05 / 4 + 0.24, 0.75 / 6, 0.05 / 4])
ax_t2m_cb.set_axis_off()
cb = fig.colorbar(
    T2m_img, ax=ax_t2m_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 3rd centre - T2m encoded
var.data = np.squeeze(encoded[0][0, :, :, 2].numpy())
var.data = np.ma.masked_where(dm_hukg.data == 0, var.data, copy=False)
var = unnormalise(var, "TMP2m") - 273.15
ax_t2m_e = fig.add_axes([0.025 / 6 + 1 / 6, 0.12 / 4 + 0.24, 0.95 / 6, 0.81 / 4])
ax_t2m_e.set_axis_off()
T2m_e_img = plotFieldAxes(
    ax_t2m_e, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.balance,
)
ax_t2m_e_cb = fig.add_axes([0.125 / 6 + 1 / 6, 0.05 / 4 + 0.24, 0.75 / 6, 0.05 / 4])
ax_t2m_e_cb.set_axis_off()
cb = fig.colorbar(
    T2m_e_img, ax=ax_t2m_e_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 3rd right - T2m scatter
varx = sCube.copy()
varx.data = np.squeeze(ict_H[:, :, 2].numpy())
varx.data = np.ma.masked_where(dm_hukg.data == 0, varx.data, copy=False)
varx = unnormalise(varx, "TMP2m") - 273.15
vary = sCube.copy()
vary.data = np.squeeze(encoded[0][0, :, :, 2].numpy())
vary.data = np.ma.masked_where(dm_hukg.data == 0, vary.data, copy=False)
vary = unnormalise(vary, "TMP2m") - 273.15
ax_t2m_s = fig.add_axes(
    [0.025 / 6 + 2 / 6 + 0.06 / 2, 0.12 / 4 + 0.24, 0.95 / 6 - 0.06 / 2, 0.81 / 4]
)
plotScatterAxes(ax_t2m_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


# 3rd left 2 - T2m original ERA5
if args.TMP2m_E:
    ax_back = fig.add_axes([0.00 / 2 + 0.5, 0.24, 1.0 / 2, 0.24])
    ax_back.set_axis_off()
    ax_back.add_patch(
        Rectangle((0, 0), 1, 1, facecolor=(0.0, 0.0, 0.0, 0.3), fill=True, zorder=1,)
    )
var = sCube.copy()
var.data = np.squeeze(ict_E[:, :, 2].numpy())
var.data = np.ma.masked_where(dm_hukg.data == 0, var.data, copy=False)
var = unnormalise(var, "TMP2m") - 273.15
(dmin, dmax) = get_range("TMP2m", args.month, var)
dmin -= 273.15 + 2
dmax -= 273.15 - 2
ax_t2m = fig.add_axes([0.025 / 6 + 0.5, 0.12 / 4 + 0.24, 0.95 / 6, 0.81 / 4])
ax_t2m.set_axis_off()
T2m_img = plotFieldAxes(
    ax_t2m, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.balance,
)
ax_t2m_cb = fig.add_axes([0.125 / 6 + 0.5, 0.05 / 4 + 0.24, 0.75 / 6, 0.05 / 4])
ax_t2m_cb.set_axis_off()
cb = fig.colorbar(
    T2m_img, ax=ax_t2m_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 3rd centre 2 - T2m encoded ERA5
var.data = np.squeeze(generated[0, :, :, 2].numpy())
var.data = np.ma.masked_where(dm_hukg.data == 0, var.data, copy=False)
var = unnormalise(var, "TMP2m") - 273.15
ax_t2m_e = fig.add_axes([0.025 / 6 + 1 / 6 + 0.5, 0.12 / 4 + 0.24, 0.95 / 6, 0.81 / 4])
ax_t2m_e.set_axis_off()
T2m_e_img = plotFieldAxes(
    ax_t2m_e, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.balance,
)
ax_t2m_e_cb = fig.add_axes(
    [0.125 / 6 + 1 / 6 + 0.5, 0.05 / 4 + 0.24, 0.75 / 6, 0.05 / 4]
)
ax_t2m_e_cb.set_axis_off()
cb = fig.colorbar(
    T2m_e_img, ax=ax_t2m_e_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 3rd right 2 - T2m scatter ERA5
varx = sCube.copy()
varx.data = np.squeeze(ict_E[:, :, 2].numpy())
varx.data = np.ma.masked_where(dm_hukg.data == 0, varx.data, copy=False)
varx = unnormalise(varx, "TMP2m") - 273.15
vary = sCube.copy()
vary.data = np.squeeze(generated[0, :, :, 2].numpy())
vary.data = np.ma.masked_where(dm_hukg.data == 0, vary.data, copy=False)
vary = unnormalise(vary, "TMP2m") - 273.15
ax_t2m_s = fig.add_axes(
    [0.025 / 6 + 2 / 6 + 0.06 / 2 + 0.5, 0.12 / 4 + 0.24, 0.95 / 6 - 0.06 / 2, 0.81 / 4]
)
plotScatterAxes(ax_t2m_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


# Bottom left - SST original
if args.SST_H:
    ax_back = fig.add_axes([0.00 / 2, 0.00, 1.0 / 2, 0.24])
    ax_back.set_axis_off()
    ax_back.add_patch(
        Rectangle((0, 0), 1, 1, facecolor=(0.0, 0.0, 0.0, 0.3), fill=True, zorder=1,)
    )
var = sCube.copy()
var.data = np.squeeze(ict_H[:, :, 1].numpy())
var.data = np.ma.masked_where(lm_20CR.data > 0, var.data, copy=False)
var = unnormalise(var, "TMPS") - 273.15
(dmin, dmax) = get_range("TMPS", args.month, var)
dmin -= 273.15 + 2
dmax -= 273.15 - 2
ax_sst = fig.add_axes([0.025 / 6, 0.12 / 4, 0.95 / 6, 0.81 / 4])
ax_sst.set_axis_off()
SST_img = plotFieldAxes(
    ax_sst, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.balance,
)
ax_sst_cb = fig.add_axes([0.125 / 6, 0.05 / 4, 0.75 / 6, 0.05 / 4])
ax_sst_cb.set_axis_off()
cb = fig.colorbar(
    SST_img, ax=ax_sst_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Bottom centre - SST encoded
var.data = encoded[0][0, :, :, 1].numpy()
var.data = np.ma.masked_where(lm_20CR.data > 0, var.data, copy=False)
var = unnormalise(var, "TMPS") - 273.15
ax_sst_e = fig.add_axes([0.025 / 6 + 1 / 6, 0.12 / 4, 0.95 / 6, 0.81 / 4])
ax_sst_e.set_axis_off()
SST_e_img = plotFieldAxes(
    ax_sst_e, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.balance,
)
ax_sst_e_cb = fig.add_axes([0.125 / 6 + 1 / 6, 0.05 / 4, 0.75 / 6, 0.05 / 4])
ax_sst_e_cb.set_axis_off()
cb = fig.colorbar(
    SST_e_img, ax=ax_sst_e_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Bottom right - SST scatter
varx = sCube.copy()
varx.data = np.squeeze(ict_H[:, :, 1].numpy())
varx.data = np.ma.masked_where(lm_20CR.data > 0, varx.data, copy=False)
varx = unnormalise(varx, "TMPS") - 273.15
vary = sCube.copy()
vary.data = np.squeeze(encoded[0][0, :, :, 1].numpy())
vary.data = np.ma.masked_where(lm_20CR.data > 0, vary.data, copy=False)
vary = unnormalise(vary, "TMPS") - 273.15
ax_sst_s = fig.add_axes(
    [0.025 / 6 + 2 / 6 + 0.06 / 2, 0.12 / 4, 0.95 / 6 - 0.06 / 2, 0.81 / 4]
)
plotScatterAxes(ax_sst_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


# Bottom left 2 - SST original ERA5
if args.SST_E:
    ax_back = fig.add_axes([0.00 / 2 + 0.5, 0.00, 1.0 / 2, 0.24])
    ax_back.set_axis_off()
    ax_back.add_patch(
        Rectangle((0, 0), 1, 1, facecolor=(0.0, 0.0, 0.0, 0.3), fill=True, zorder=1,)
    )
var = sCube.copy()
var.data = np.squeeze(ict_E[:, :, 1].numpy())
var.data = np.ma.masked_where(lm_ERA5.data.mask == True, var.data, copy=False)
var = unnormalise(var, "TMPS") - 273.15
(dmin, dmax) = get_range("TMPS", args.month, var)
dmin -= 273.15 + 2
dmax -= 273.15 - 2
ax_sst = fig.add_axes([0.025 / 6 + 0.5, 0.12 / 4, 0.95 / 6, 0.81 / 4])
ax_sst.set_axis_off()
SST_img = plotFieldAxes(
    ax_sst, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.balance,
)
ax_sst_cb = fig.add_axes([0.125 / 6 + 0.5, 0.05 / 4, 0.75 / 6, 0.05 / 4])
ax_sst_cb.set_axis_off()
cb = fig.colorbar(
    SST_img, ax=ax_sst_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Botom centre 2 - SST encoded ERA5
var.data = generated[0, :, :, 1].numpy()
var.data = np.ma.masked_where(lm_ERA5.data.mask == True, var.data, copy=False)
var = unnormalise(var, "TMPS") - 273.15
ax_sst_e = fig.add_axes([0.025 / 6 + 1 / 6 + 0.5, 0.12 / 4, 0.95 / 6, 0.81 / 4])
ax_sst_e.set_axis_off()
SST_e_img = plotFieldAxes(
    ax_sst_e, var, vMax=dmax, vMin=dmin, lMask=lm_plot, cMap=cmocean.cm.balance,
)
ax_sst_e_cb = fig.add_axes([0.125 / 6 + 1 / 6 + 0.5, 0.05 / 4, 0.75 / 6, 0.05 / 4])
ax_sst_e_cb.set_axis_off()
cb = fig.colorbar(
    SST_e_img, ax=ax_sst_e_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Bottom right 2 - SST scatter ERA5
varx = sCube.copy()
varx.data = np.squeeze(ict_E[:, :, 1].numpy())
varx.data = np.ma.masked_where(lm_ERA5.data.mask == True, varx.data, copy=False)
varx = unnormalise(varx, "TMPS") - 273.15
vary = sCube.copy()
vary.data = np.squeeze(generated[0, :, :, 1].numpy())
vary.data = np.ma.masked_where(lm_ERA5.data.mask == True, vary.data, copy=False)
vary = unnormalise(vary, "TMPS") - 273.15
ax_sst_s = fig.add_axes(
    [0.025 / 6 + 2 / 6 + 0.06 / 2 + 0.5, 0.12 / 4, 0.95 / 6 - 0.06 / 2, 0.81 / 4]
)
plotScatterAxes(ax_sst_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


fig.savefig("fit.png")
