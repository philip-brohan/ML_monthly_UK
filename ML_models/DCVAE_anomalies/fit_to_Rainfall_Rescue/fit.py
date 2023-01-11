#!/usr/bin/env python

# Find a point in latent space that maximises the fit to the Rainfall Rescue stations,
#  and plot the fitted state.


import os
import sys
from random import shuffle
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
parser.add_argument(
    "--ssize", help="station plot size", type=float, required=False, default=5
)
parser.add_argument(
    "--psize", help="scatter plot point size", type=float, required=False, default=5
)
parser.add_argument(
    "--sdir",
    help="obs directory",
    type=str,
    required=False,
    default="monthly_rainfall_rainfall-rescue_v1.1.0",
)
parser.add_argument(
    "--decimate_to",
    help="No. of stations to keep",
    type=int,
    required=False,
    default=None,
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
from plot_functions.plot_station import plotObsAxes
from plot_functions.plot_station import plotObsScatterAxes


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


# convert a grid fraction location (0-1) to a station location
def gfl_to_xy(x, y, cube=sCube):
    xg = cube.coord("projection_x_coordinate").bounds
    xf = x * (xg[-1, 1] - xg[0, 0])
    xf += xg[0, 0]
    yg = cube.coord("projection_y_coordinate").bounds
    yf = y * (yg[-1, 1] - yg[0, 0])
    yf += yg[0, 0]
    return (xf, yf)


# Normalise a station value
def s_normalise(value):
    value -= nPar["monthly_rainfall"][0]
    value /= nPar["monthly_rainfall"][1] - nPar["monthly_rainfall"][0]
    return value


def s_unnormalise(value):
    value *= nPar["monthly_rainfall"][1] - nPar["monthly_rainfall"][0]
    value += nPar["monthly_rainfall"][0]
    return value


# Anomalise and normalise the station data
clim = load_climatology("monthly_rainfall", args.month)
stn_ids = list(monthly.keys())
for stn_id in stn_ids:
    try:
        (meta[stn_id]["XI"], meta[stn_id]["YI"]) = xy_to_idx(
            meta[stn_id]["X"], meta[stn_id]["Y"]
        )
        (meta[stn_id]["XF"], meta[stn_id]["YF"]) = xy_to_gfl(
            meta[stn_id]["X"], meta[stn_id]["Y"]
        )
        if clim.data.mask[meta[stn_id]["YI"], meta[stn_id]["XI"]]:
            raise Exception("Outside HadUKGrid")
        monthly[stn_id] -= clim.data[meta[stn_id]["YI"], meta[stn_id]["XI"]]
        monthly[stn_id] = s_normalise(monthly[stn_id])
    except Exception:
        del monthly[stn_id]  # no location or bad obs

if args.decimate_to is not None:
    all_keys = list(monthly.keys())
    if len(all_keys) > args.decimate_to:
        keep_keys = []
        lats = [meta[x]["Y"] for x in all_keys]
        min_l = min(lats)
        max_l = max(lats)
        for lr in range(1, args.decimate_to):
            lat_range = [
                min_l + (max_l - min_l) * x / args.decimate_to for x in (lr - 1, lr)
            ]
            range_keys = [
                x
                for x in all_keys
                if meta[x]["Y"] > lat_range[0]
                if meta[x]["Y"] <= lat_range[1]
            ]
            if len(range_keys) == 0:
                continue
            range_keys = sorted(range_keys)
            keep_keys.append(range_keys[0])
        for stn_id in all_keys:
            if stn_id not in keep_keys:
                del monthly[stn_id]

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
t_coords = tf.expand_dims(tf.stack((t_y_coord, t_x_coord), axis=1), axis=0)
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
        interpolate_bilinear(generated[0:1, :, :, 3:4], t_coords, indexing="ij")
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
generated_at_obs = tf.squeeze(
    interpolate_bilinear(generated[0:1, :, :, 3:4], t_coords, indexing="ij")
)
hukg_at_obs = tf.squeeze(
    interpolate_bilinear(
        tf.expand_dims(ict[:, :, 3:4], axis=0), t_coords, indexing="ij"
    )
)

# Make the plot - same as for validation script
fig = Figure(
    figsize=(20 * 1.54, 20),
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


# 2nd top - HadUKGrid
varx = sCube.copy()
varx.data = np.squeeze(ict[:, :, 3].numpy())
varx.data = np.ma.masked_where(varx.data == 0.5, varx.data, copy=False)
varx = unnormalise(varx, "monthly_rainfall")
(dmin, dmax) = get_range("PRATE", args.month, anomaly=True)
dmin *= 86400 * 30
dmax *= 86400 * 30
ax_prate = fig.add_axes([0.01 * 2 + 0.188, 0.05 + 0.01 + 0.465, 0.188, 0.465])
ax_prate.set_axis_off()
PRATE_img = plotFieldAxes(
    ax_prate,
    varx,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)

# 1st centre, PRATE observations
positions = t_coords.numpy()
n_stations = positions.shape[1]
s_lats = []
s_lons = []
s_anoms = []
for sidx in range(n_stations):
    (x, y) = gfl_to_xy(t_x_coord[sidx] / nxp, t_y_coord[sidx] / nyp)
    s_lons.append(x.numpy())
    s_lats.append(y.numpy())
    s_anoms.append(s_unnormalise(t_obs[sidx].numpy()))
ax_obs = fig.add_axes([0.01, 0.05 + 0.94 / 2 - 0.465 / 2, 0.188, 0.465])
ax_obs.set_axis_off()
PRATE_obs = plotObsAxes(
    ax_obs,
    s_lons,
    s_lats,
    s_anoms,
    vmax=dmax,
    vmin=dmin,
    ssize=args.ssize * 1000,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)

# 2nd bottom - ML generated
vary = sCube.copy()
vary.data = np.squeeze(generated[0, :, :, 3].numpy())
vary.data = np.ma.masked_where(varx.data == 0.5, vary.data, copy=False)
vary = unnormalise(vary, "monthly_rainfall")
ax_prate_e = fig.add_axes([0.01 * 2 + 0.188, 0.05, 0.188, 0.465])
ax_prate_e.set_axis_off()
PRATE_e_img = plotFieldAxes(
    ax_prate_e,
    vary,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)

# 3rd Centre - Field difference (ML-HadUKG)
vard = vary - varx
ax_prate_d = fig.add_axes(
    [0.01 * 3 + 0.188 * 2, 0.05 + 0.94 / 2 - 0.465 / 2, 0.188, 0.465]
)
ax_prate_d.set_axis_off()
PRATE_e_img = plotFieldAxes(
    ax_prate_d,
    vard,
    vMax=dmax,
    vMin=dmin,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)

# 4th Top - HadUKG obs difference (grid-obs)
dh_stn_diffs = []
for sidx in range(n_stations):
    dh_stn_diffs.append(s_unnormalise(hukg_at_obs[sidx].numpy()) - s_anoms[sidx])
ax_dhd = fig.add_axes([0.01 * 4 + 0.188 * 3, 0.05 + 0.01 + 0.465, 0.188, 0.465])
ax_dhd.set_axis_off()
PRATE_dhd = plotObsAxes(
    ax_dhd,
    s_lons,
    s_lats,
    dh_stn_diffs,
    vmax=dmax,
    vmin=dmin,
    ssize=args.ssize * 1000,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)

# 4th Bottom - ML obs difference (grid-obs)
ml_stn_diffs = []
for sidx in range(n_stations):
    ml_stn_diffs.append(s_unnormalise(generated_at_obs[sidx].numpy()) - s_anoms[sidx])
ax_mld = fig.add_axes([0.01 * 4 + 0.188 * 3, 0.05, 0.188, 0.465])
ax_mld.set_axis_off()
PRATE_mld = plotObsAxes(
    ax_mld,
    s_lons,
    s_lats,
    ml_stn_diffs,
    vmax=dmax,
    vmin=dmin,
    ssize=args.ssize * 1000,
    lMask=lm_plot,
    cMap=cmocean.cm.balance,
)


# Anomaly colourbar bottom left-ish
ax_prate_cb = fig.add_axes([0.02, 0.02, 0.188 * 2, 0.02])
ax_prate_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_img, ax=ax_prate_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Right centre - HUKG:ML field scatter
ax_f_s = fig.add_axes(
    [0.01 * 5 + 0.188 * 4 + 0.038, 0.05 + 0.0475 * 2 + 0.25, 0.15, 0.25]
)
plotScatterAxes(ax_f_s, varx, vary, vMin=dmin, vMax=dmax, bins=None)

# Right top - hadUKG:obs scatter
ax_h_s = fig.add_axes(
    [0.01 * 5 + 0.188 * 4 + 0.038, 0.05 + 0.0475 * 3 + 0.25 * 2, 0.15, 0.25]
)
dh_stn_anoms = [s_anoms[i] + dh_stn_diffs[i] for i in range(n_stations)]
plotObsScatterAxes(
    ax_h_s, s_anoms, dh_stn_anoms, vMin=dmin, vMax=dmax, psize=args.psize
)

# Right bottom - ML:obs scatter
ax_ml_s = fig.add_axes([0.01 * 5 + 0.188 * 4 + 0.038, 0.05 + 0.0475, 0.15, 0.25])
dh_ml_anoms = [s_anoms[i] + ml_stn_diffs[i] for i in range(n_stations)]
plotObsScatterAxes(
    ax_ml_s, s_anoms, dh_ml_anoms, vMin=dmin, vMax=dmax, psize=args.psize
)

fig.savefig("fit.png")
