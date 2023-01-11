#!/usr/bin/env python

# Find and store the station obs - fitted field values for a single fit to
# a selected month.

import os
import sys
from random import shuffle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_addons.image import interpolate_bilinear

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
    "--sdir",
    help="obs directory",
    type=str,
    required=False,
    default="monthly_rainfall_rainfall-rescue_v1.1.0",
)
parser.add_argument(
    "--opdir",
    help="output directory",
    type=str,
    required=False,
    default=None,
)
parser.add_argument(
    "--decimate_to",
    help="Fraction of stations to keep",
    type=float,
    required=False,
    default=0.5,
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

if args.opdir is None:
    args.opdir = "%s/RR_station_fits/%04d/%02d"

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

undecimated = monthly.copy()

if args.decimate_to is not None:
    all_keys = list(monthly.keys())
    args.decimate_to = int(len(all_keys) * args.decimate_to)
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

u_x_coord = []
u_y_coord = []
u_n_anom = []
for stn_id in undecimated.keys():
    u_x_coord.append(meta[stn_id]["XF"] * nxp)
    u_y_coord.append(meta[stn_id]["YF"] * nyp)
    u_n_anom.append(undecimated[stn_id])
t_ux_coord = tf.convert_to_tensor(u_x_coord, tf.float32)
t_uy_coord = tf.convert_to_tensor(u_y_coord, tf.float32)
u_coords = tf.expand_dims(tf.stack((t_uy_coord, t_ux_coord), axis=1), axis=0)
u_obs = tf.convert_to_tensor(u_n_anom, tf.float32)


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
    interpolate_bilinear(generated[0:1, :, :, 3:4], u_coords, indexing="ij")
)

# Output Station ID, whether assimilated, station ob and generated value at location of ob
positions = u_coords.numpy()
n_stations = positions.shape[1]
s_anoms = []
ml_anoms = []
keys = list(undecimated.keys())

if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

opfname = "%d.csv" % os.getpid()

with open("%s/%s" % (args.opdir, opfname), "w") as f:
    for sidx in range(n_stations):
        assimilated = 0
        if keys[sidx] in monthly:
            assimilated = 1
        f.write(
            "%s,%1d,%6.2f,%6.2f\n"
            % (
                keys[sidx],
                assimilated,
                s_unnormalise(u_obs[sidx].numpy()),
                s_unnormalise(generated_at_obs[sidx].numpy()),
            )
        )
