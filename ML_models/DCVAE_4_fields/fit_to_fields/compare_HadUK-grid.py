#!/usr/bin/env python

# Compare T2m and prate with HadUK-grid, for the SSST+PRMSL fit

import os
import sys
import numpy as np
import iris
import iris.analysis
from calendar import monthrange
import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import cmocean

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=990)
parser.add_argument(
    "--year", help="Year to fit to", type=int, required=False, default=1969
)
parser.add_argument(
    "--month", help="Month to fit to", type=int, required=False, default=3
)
parser.add_argument(
    "--member", help="Member to fit to", type=int, required=False, default=1
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
pc.coord("longitude").guess_bounds()
pc.coord("latitude").guess_bounds()
pch = plot_cube(0.01)
pch.coord("longitude").guess_bounds()
pch.coord("latitude").guess_bounds()
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

# We are using the model in inference mode - (does this have any effect?)
autoencoder.decoder.trainable = False
for layer in autoencoder.decoder.layers:
    layer.trainable = False
autoencoder.decoder.compile()

latent = tf.Variable(tf.random.normal(shape=(1, autoencoder.latent_dim)))
target = tf.constant(tf.reshape(ict, [1, 32, 32, 4]))

t2m_huk = iris.load_cube(
    "%s/haduk-grid/monthly_meantemp/%04d/%02d.nc"
    % (os.getenv("SCRATCH"), args.year, args.month)
)
prate_huk = iris.load_cube(
    "%s/haduk-grid/monthly_rainfall/%04d/%02d.nc"
    % (os.getenv("SCRATCH"), args.year, args.month)
)


def decodeFit():
    result = 0.0
    decoded = autoencoder.decode(latent)
    result = result + tf.reduce_mean(
        tf.keras.metrics.mean_squared_error(decoded[:, :, :, 0], target[:, :, :, 0])
    )
    result = result + tf.reduce_mean(
        tf.keras.metrics.mean_squared_error(
            tf.boolean_mask(decoded[:, :, :, 1], np.invert(sst_mask), axis=1),
            tf.boolean_mask(target[:, :, :, 1], np.invert(sst_mask), axis=1),
        )
    )
    return result


loss = tfp.math.minimize(
    decodeFit,
    trainable_variables=[latent],
    num_steps=1000,
    optimizer=tf.optimizers.Adam(learning_rate=0.05),
)

encoded = autoencoder.decode(latent)

# Make the plot - same as for validation script
fig = Figure(
    figsize=(15, 11),
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

lMask = get_land_mask(plot_cube(resolution=0.1))


# PRATE - get the HadUKG and encoded fields on the same grid
Ph = prate_huk.regrid(pch, iris.analysis.Nearest())
Ph = Ph.regrid(pc, iris.analysis.AreaWeighted())
Ph.data.mask[np.invert(sst_mask)] = True  # Apply 20CR land mask
Pe = pc.copy()
Pe.data = np.squeeze(encoded[0, :, :, 3].numpy())
Pe = unnormalise(Pe, "PRATE") * 86400 * monthrange(args.year, args.month)[1]
Pe.data = Pe.data + Ph.data * 0.0  # Apply HadUK data mask
Pe.data.mask[np.invert(sst_mask)] = True  # Apply 20CR land mask
dmin = 0.0
dmax = (max(np.max(Ph.data), np.max(Pe.data))) * 1.05


# 2nd left - PRATE hadUK-grid
ax_prate = fig.add_axes([0.025 / 3, 0.125 / 2 + 0.5, 0.95 / 3, 0.85 / 2])
ax_prate.set_axis_off()
PRATE_img = plotFieldAxes(
    ax_prate,
    Ph,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.rain,
    plotCube=pc,
)
ax_prate_cb = fig.add_axes([0.125 / 3, 0.05 / 2 + 0.5, 0.75 / 3, 0.05 / 2])
ax_prate_cb.set_axis_off()
cb = fig.colorbar(
    PRATE_img, ax=ax_prate_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# 2nd centre - PRATE encoded
ax_prate_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 2 + 0.5, 0.95 / 3, 0.85 / 2])
ax_prate_e.set_axis_off()
PRATE_e_img = plotFieldAxes(
    ax_prate_e,
    Pe,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.rain,
    plotCube=pc,
)
ax_prate_e_cb = fig.add_axes([0.125 / 3 + 1 / 3, 0.05 / 2 + 0.5, 0.75 / 3, 0.05 / 2])
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
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 2 + 0.5, 0.95 / 3 - 0.06, 0.85 / 2]
)
plotScatterAxes(ax_prate_s, Ph, Pe, vMin=0.001, vMax=dmax)


# T2m - get the HadUKG and encoded fields on the same grid
Th = t2m_huk.regrid(pch, iris.analysis.Nearest())
Th = Th.regrid(pc, iris.analysis.AreaWeighted())
Th.data.mask[np.invert(sst_mask)] = True  # Apply 20CR land mask
Te = pc.copy()
Te.data = np.squeeze(encoded[0, :, :, 2].numpy())
Te = unnormalise(Te, "TMP2m") - 273.15
Te.data = Te.data + Th.data * 0.0  # Apply HadUK data mask
Te.data.mask[np.invert(sst_mask)] = True  # Apply 20CR land mask
dmin = min(np.min(Th.data), np.min(Te.data)) - 1
dmax = max(np.max(Th.data), np.max(Te.data)) + 1


# Lower left - T2m observations
ax_t2m = fig.add_axes([0.025 / 3, 0.125 / 2, 0.95 / 3, 0.85 / 2])
ax_t2m.set_axis_off()
T2m_img = plotFieldAxes(
    ax_t2m,
    Th,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.balance,
    plotCube=pc,
)
ax_t2m_cb = fig.add_axes([0.125 / 3, 0.05 / 2, 0.75 / 3, 0.05 / 2])
ax_t2m_cb.set_axis_off()
cb = fig.colorbar(
    T2m_img, ax=ax_t2m_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Lower centre - T2m encoded
ax_t2m_e = fig.add_axes([0.025 / 3 + 1 / 3, 0.125 / 2, 0.95 / 3, 0.85 / 2])
ax_t2m_e.set_axis_off()
T2m_e_img = plotFieldAxes(
    ax_t2m_e,
    Te,
    vMax=dmax,
    vMin=dmin,
    lMask=lMask,
    cMap=cmocean.cm.balance,
    plotCube=pc,
)
ax_t2m_e_cb = fig.add_axes([0.125 / 3 + 1 / 3, 0.05 / 2, 0.75 / 3, 0.05 / 2])
ax_t2m_e_cb.set_axis_off()
cb = fig.colorbar(
    T2m_e_img, ax=ax_t2m_e_cb, location="bottom", orientation="horizontal", fraction=1.0
)

# Lower right - T2m scatter
ax_t2m_s = fig.add_axes(
    [0.025 / 3 + 2 / 3 + 0.06, 0.125 / 2, 0.95 / 3 - 0.06, 0.85 / 2]
)
plotScatterAxes(ax_t2m_s, Th, Te, vMin=dmin, vMax=dmax)


fig.savefig("compare_HadUK-grid.png")
