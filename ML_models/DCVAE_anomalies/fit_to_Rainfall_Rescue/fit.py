#!/usr/bin/env python

# Find a point in latent space that maximises the fit to the Rainfall Rescue stations,
#  and plot the fitted state.


import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_addons.image import interpolate_bilinear

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import cmocean

import warnings

warnings.filterwarnings("ignore", message=".*partition.*")
warnings.filterwarnings("ignore", message=".*datum.*")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=250)
parser.add_argument(
    "--year", help="Year to fit to", type=int, required=False, default=1909
)
parser.add_argument(
    "--month", help="Month to fit to", type=int, required=False, default=3
)
parser.add_argument(
    "--iter",
    help="No. of iterations",
    type=int,
    required=False,
    default=100,
)
parser.add_argument(
    "--match",
    help="Station numeric IDs to fit to",
    type=str,
    nargs="*",
    action="extend",
)
parser.add_argument(
    "--skip",
    help="Station numeric IDs to leave out of the fit",
    type=str,
    nargs="*",
    action="extend",
)
args = parser.parse_args()

if args.match is not None and args.skip is not None:
    raise Exception("Can't use --match and --skip together")

sys.path.append("%s/../../.." % os.path.dirname(__file__))
from get_data.HadUKGrid.HUKG_monthly_load import load_station_metadata
from get_data.HadUKGrid.HUKG_monthly_load import load_rr_stations
from get_data.HadUKGrid.HUKG_monthly_load import load_climatology

# Load the station data
meta = load_station_metadata()
monthly = load_rr_stations(args.year, month=args.month)
if args.match is not None:
    for stn_id in monthly.keys():
        if stn_id not in args.match:
            del monthly[stn_id]
    if len(monthly.keys()) == 0:
        raise Exception("No stations match")
if args.skip is not None:
    for stn_id in monthly.keys():
        if stn_id in args.skip:
            del monthly[stn_id]
    if len(monthly.keys()) == 0:
        raise Exception("All stations skipped")

# Load the model specification
sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH
from autoencoderModel import DCVAE
from make_tensors.tensor_utils import load_cList
from make_tensors.tensor_utils import cList_to_tensor
from make_tensors.tensor_utils import normalise
from make_tensors.tensor_utils import unnormalise
from make_tensors.tensor_utils import lm_plot
from make_tensors.tensor_utils import lm_TWCR
from make_tensors.tensor_utils import dm_HUKG
from make_tensors.tensor_utils import sCube
from make_tensors.tensor_utils import nPar

sys.path.append("%s/../../.." % os.path.dirname(__file__))
from get_data.TWCR.TWCR_monthly_load import get_range
from plot_functions.plot_variable import plotFieldAxes
from plot_functions.plot_variable import plotScatterAxes


# convert a station location to a grid index
def xy_to_idx(x, y, cube=sCube):
    xg = cube.coord("projection_x_coordinate").bounds
    xi = np.max(np.where(xg[:, 0] < x))
    yg = cube.coord("projection_y_coordinate").bounds
    yi = np.max(np.where(yg[:, 0] < y))
    return (xi, yi)


# convert a station location to a grid fraction location (0-1)
def xy_to_gfl(x, y, cube=sCube):
    xg = cube.coord("projection_x_coordinate").bounds
    xf = (x - xg[0, 0]) / (xg[-1, 1] - xg[0, 0])
    yg = cube.coord("projection_y_coordinate").bounds
    yf = (y - yg[0, 0]) / (yg[-1, 1] - yg[0, 0])
    return (xf, yf)


# Anomalise and normalise the station data
clim = load_climatology("monthly_rainfall", args.month)
stn_ids = monthly.keys()
for stn_id in stn_ids:
    try:
        (meta[stn_id]["XI"], meta[stn_id]["YI"]) = xy_to_idx(
            meta[stn_id]["X"], meta[stn_id]["X"]
        )
        (meta[stn_id]["XF"], meta[stn_id]["YF"]) = xy_to_gfl(
            meta[stn_id]["X"], meta[stn_id]["X"]
        )
        monthly[stn_id] -= clim.data[meta[stn_id]["XI"], meta[stn_id]["YI"]]
        monthly[stn_id] -= nPar["monthly_rainfall"][0]
        monthly[stn_id] /= nPar["monthly_rainfall"][1] - nPar["monthly_rainfall"][0]
    except Exception:
        del monthly[stn_id]  # no location or bad obs


# Make a tensor with the station locations
# and another with the normalised station anomalies
nxp = len(sCube.coord("projection_x_coordinate").points)
nyp = len(sCube.coord("projection_y_coordinate").points)
s_x_coord = []
s_y_coord = []
s_n_anom = []
for stn_id in monthly.keys():
    s_x_coord.append(meta[stn_id]["XF"] * nxp)
    s_y_coord.append(meta[stn_id]["YF"] * nyp)
    s_n_anom.append(monthly[stn_id])
t_x_coord = tf.convert_to_tensor(s_x_coord, tf.float32)
t_y_coord = tf.convert_to_tensor(s_y_coord, tf.float32)
t_coords = tf.stack((t_y_coord, t_x_coord), axis=1)
t_obs = tf.convert_to_tensor(s_n_anom, tf.float32)

# Load the gridded data (if we have it)
qd = load_cList(args.year, args.month)
ict = cList_to_tensor(qd, extrapolate=False)


autoencoder = DCVAE()
weights_dir = ("%s/models/Epoch_%04d") % (
    LSCRATCH,
    args.epoch,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
load_status.assert_existing_objects_matched()

# We are using the model in inference mode - (does this have any effect?)
autoencoder.trainable = False

latent = tf.Variable(tf.random.normal(shape=(1, autoencoder.latent_dim)))
target = tf.constant(tf.reshape(ict, [1, 1440, 896, 4]))


def decodeFit():
    result = 0.0
    generated = autoencoder.generate(latent, training=False)
    at_obs = tf.squeeze(
        interpolate_bilinear(generated[0, :, :, 3], t_coords, indexing="ij")
    )
    result = tf.reduce_mean(tf.keras.metrics.mean_squared_error(t_obs, at_obs))
    return result


loss = tfp.math.minimize(
    decodeFit,
    trainable_variables=[latent],
    num_steps=args.iter,
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
)

generated = autoencoder.generate(latent, training=False)

# Make the plot - same as for validation script
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
    Rectangle(
        (0, 0),
        1,
        1,
        facecolor=(1.0, 1.0, 1.0, 1),
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
varx = sCube.copy()
varx.data = np.squeeze(ict[:, :, 0].numpy())
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

# Top centre - PRMSL generated
vary = sCube.copy()
vary.data = np.squeeze(generated[0, :, :, 0].numpy())
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
varx.data = np.squeeze(ict[:, :, 3].numpy())
varx.data = np.ma.masked_where(varx.data == 0.5, varx.data, copy=False)
varx = unnormalise(varx, "monthly_rainfall")
(dmin, dmax) = get_range("PRATE", args.month, anomaly=True)
dmin *= 86400 * 30
dmax *= 86400 * 30
ax_prate = fig.add_axes([0.025 / 3, 0.125 / 4 + 0.5, 0.95 / 3, 0.85 / 4])
ax_prate.set_axis_off()
PRATE_img = plotFieldAxes(
    ax_prate,
    varx,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.tarn,
)
ax_prate_cb = fig.add_axes([0.125 / 3, 0.05 / 4 + 0.5, 0.75 / 3, 0.05 / 4])
ax_prate_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_img, ax=ax_prate_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd centre - PRATE generated
vary.data = np.squeeze(generated[0, :, :, 3].numpy())
vary.data = np.ma.masked_where(varx.data == 0.5, vary.data, copy=False)
vary = unnormalise(vary, "monthly_rainfall")
ax_prate_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 4 + 0.5, 0.95 / 3, 0.85 / 4])
ax_prate_e.set_axis_off()
PRATE_e_img = plotFieldAxes(
    ax_prate_e,
    vary,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.tarn,
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
ax_prate_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 4 + 0.5, 0.95 / 3 - 0.06, 0.85 / 4]
)
plotScatterAxes(ax_prate_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)


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
varx.data = np.squeeze(ict[:, :, 2].numpy())
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

# 3rd centre - T2m generated
vary.data = np.squeeze(generated[0, :, :, 2].numpy())
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
varx.data = np.squeeze(ict[:, :, 1].numpy())
varx.data = np.ma.masked_where(lm_TWCR.data.mask, varx.data, copy=False)
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

# 2nd centre - SST generated
vary.data = generated.numpy()[0, :, :, 1]
vary.data = np.ma.masked_where(lm_TWCR.data.mask, vary.data, copy=False)
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


fig.savefig("fit.png")
