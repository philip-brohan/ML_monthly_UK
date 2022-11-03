#!/usr/bin/env python

# Plot fit statistics for all the test cases

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
from statistics import mean

import warnings

warnings.filterwarnings("ignore", message=".*datum.*")


import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# I don't need all the messages about a missing font
import logging

#logging.getLogger("matplotlib.font_manager").disabled = True


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=250)
parser.add_argument(
    "--startyear", help="First year to plot", type=int, required=False, default=None
)
parser.add_argument(
    "--endyear", help="Last year to plot", type=int, required=False, default=None
)
parser.add_argument(
    "--xpoint",
    help="Extract data at this x point",
    type=int,
    required=False,
    default=None,
)
parser.add_argument(
    "--ypoint",
    help="Extract data at this y point",
    type=int,
    required=False,
    default=None,
)
parser.add_argument(
    "--training",
    help="Use training months (not test months)",
    dest="training",
    default=False,
    action="store_true",
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
    default=100,
)
args = parser.parse_args()

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH
from localise import TSOURCE
from autoencoderModel import DCVAE
from makeDataset import getDataset
#from make_tensors.tensor_utils import load_cList
#from make_tensors.tensor_utils import cList_to_tensor
#from make_tensors.tensor_utils import normalise
#from make_tensors.tensor_utils import unnormalise
#from make_tensors.tensor_utils import lm_plot
from make_tensors.tensor_utils import lm_TWCR
from make_tensors.tensor_utils import dm_HUKG
from make_tensors.tensor_utils import sCube
from make_tensors.tensor_utils import nPar

# Set up the test data
purpose = "test"
if args.training:
    purpose = "training"
testData = getDataset(
    purpose=purpose, startyear=args.startyear, endyear=args.endyear, shuffle=False
)
testData = testData.batch(1)

model = DCVAE()
weights_dir = ("%s/models/Epoch_%04d") % (
    LSCRATCH,
    args.epoch,
)
load_status = model.load_weights("%s/ckpt" % weights_dir)
load_status.assert_existing_objects_matched()
latent = tf.Variable(tf.random.normal(shape=(1, model.latent_dim)))


def unnormalise(value, variable):
    if not variable in nPar:
        raise Exception("Unsupported variable " + variable)
    value *= nPar[variable][1] - nPar[variable][0]
    value += nPar[variable][0]
    return value


def field_to_scalar(field, variable, month, mask):
    f2 = unnormalise(field, variable)
    if args.xpoint is not None and args.ypoint is not None:
        return f2[args.ypoint, args.xpoint]
    if args.xpoint is None and args.ypoint is None:
        if mask is not None:
            return np.mean(f2[np.where(mask != 0.5)])
        return np.mean(f2)
    else:
        raise Exception("Must specify both xpoint and ypoint (or neither).")

def decodeFit():
    result = 0.0
    generated = model.generate(latent,training=False)
    if args.PRMSL:
        result = result + tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(generated[:, :, :, 0], target[:, :, :, 0])
        )
    if args.SST:
        result = result + tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(
                tf.boolean_mask(
                    generated[:, :, :, 1], np.invert(lm_TWCR.data.mask), axis=1
                ),
                tf.boolean_mask(
                    target[:, :, :, 1], np.invert(lm_TWCR.data.mask), axis=1
                ),
            )
        )
    if args.TMP2m:
        result = result + tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(
                tf.boolean_mask(
                    generated[:, :, :, 2], np.invert(dm_HUKG.data.mask), axis=1
                ),
                tf.boolean_mask(
                    target[:, :, :, 2], np.invert(dm_HUKG.data.mask), axis=1
                ),
            )
        )
    if args.PRATE:
        result = result + tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(
                tf.boolean_mask(
                    generated[:, :, :, 3], np.invert(dm_HUKG.data.mask), axis=1
                ),
                tf.boolean_mask(
                    target[:, :, :, 3], np.invert(dm_HUKG.data.mask), axis=1
                ),
            )
        )
    return result

# Get target and encoded statistics for one test case
def compute_stats(x):
    # get the date from the filename tensor
    fn = x[2].numpy()[0]
    year = int(fn[:4])
    month = int(fn[5:7])
    dtp = datetime.date(year, month, 15)
    # calculate the fit field
    global target
    target = x[1]
    global latent
    latent = tf.Variable(tf.random.normal(shape=(1, model.latent_dim)))
    if args.PRMSL or args.SST or args.TMP2m or args.PRATE:
        loss = tfp.math.minimize(
            decodeFit,
            trainable_variables=[latent],
            num_steps=args.iter,
            optimizer=tf.optimizers.Adam(learning_rate=0.1),
        )
    generated = model.generate(latent)
    stats = {}
    stats["dtp"] = dtp
    stats["PRMSL_target"] = field_to_scalar(
        x[0][0, :, :, 0].numpy(),
        "PRMSL",
        month,
        None,
    )
    stats["PRMSL_model"] = field_to_scalar(
        generated[0, :, :, 0].numpy(),
        "PRMSL",
        month,
        None,
    )
    vt = x[0][0, :, :, 1].numpy()
    mask = x[1][0, :, :, 1].numpy()
    stats["SST_target"] = field_to_scalar(
        vt, "SST", month, mask
    )
    vm = generated[0, :, :, 1].numpy()
    stats["SST_model"] = field_to_scalar(
        vm, "SST", month, mask
    )
    vt = x[0][0, :, :, 2].numpy()
    mask = x[1][0, :, :, 2].numpy()
    stats["T2M_target"] = field_to_scalar(
        vt, "monthly_meantemp", month, mask,
    )
    vm = generated[0, :, :, 2].numpy()
    stats["T2M_model"] = field_to_scalar(
        vm, "monthly_meantemp", month, mask
    )
    vt = x[0][0, :, :, 3].numpy()
    mask = x[1][0, :, :, 3].numpy()
    stats["PRATE_target"] = field_to_scalar(
        vt, "monthly_rainfall", month, mask
    )
    vm = generated[0, :, :, 3].numpy()
    stats["PRATE_model"] = field_to_scalar(
        vm, "monthly_rainfall", month, mask
    )
    return stats



all_stats = {}
for case in testData:
    stats = compute_stats(case)
    for key in stats.keys():
        if key in all_stats:
            all_stats[key].append(stats[key])
        else:
            all_stats[key] = [stats[key]]


# Plot sizes
tsh = 2
ssh = 2
tsw = 4
ssw = 2
hpad = 0.5
wpad = 0.5
fig_h = tsh * 2 + hpad * 3
fig_w = tsw * 2 + ssw * 2 + wpad * 5
# Make the plot
fig = Figure(
    figsize=(fig_w, fig_h),
    dpi=300,
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
    "size": 8,
}
matplotlib.rc("font", **font)
axb = fig.add_axes([0, 0, 1, 1])
axb.add_patch(
    Rectangle(
        (0, 0),
        1,
        1,
        facecolor=(0.95, 0.95, 0.95, 1),
        fill=True,
        zorder=1,
    )
)


def plot_var(ts, t, m, xp, yp, label):
    ymin = min(min(t), min(m))
    ymax = max(max(t), max(m))
    ypad = (ymax - ymin) * 0.1
    if ypad == 0:
        ypad = 1
    ax_ts = fig.add_axes(
        [
            (wpad / fig_w) * (2 * xp - 1) + (tsw + ssw) * (xp - 1) / fig_w,
            (hpad / fig_h) * yp + (tsh / fig_h) * (yp - 1),
            tsw / fig_w,
            tsh / fig_h,
        ],
        xlim=(
            ts[0] - datetime.timedelta(days=15),
            ts[-1] + datetime.timedelta(days=15),
        ),
        ylim=(ymin - ypad, ymax + ypad),
    )
    ax_ts.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
    ax_ts.text(
        ts[0] - datetime.timedelta(days=15),
        ymax + ypad,
        label,
        ha="left",
        va="top",
        bbox=dict(boxstyle="square,pad=0.5", fc=(1, 1, 1, 1)),
        zorder=100,
    )
    ax_ts.add_line(Line2D(ts, t, linewidth=1, color=(0, 0, 0, 1), alpha=1.0, zorder=50))
    ax_ts.add_line(Line2D(ts, m, linewidth=1, color=(1, 0, 0, 1), alpha=1.0, zorder=60))
    ax_sc = fig.add_axes(
        [
            (wpad / fig_w) * (2 * xp) + (tsw * xp + ssw * (xp - 1)) / fig_w,
            (hpad / fig_h) * yp + (tsh / fig_h) * (yp - 1),
            ssw / fig_w,
            tsh / fig_h,
        ],
        xlim=(ymin - ypad, ymax + ypad),
        ylim=(ymin - ypad, ymax + ypad),
    )
    ax_sc.grid(color=(0, 0, 0, 1), linestyle="-", linewidth=0.1)
    ax_sc.scatter(t, m, s=1, color=(1, 0, 0, 1), zorder=60)
    ax_sc.add_line(
        Line2D(
            (ymin - ypad, ymax + ypad),
            (ymin - ypad, ymax + ypad),
            linewidth=1,
            color=(0, 0, 0, 1),
            alpha=0.2,
            zorder=10,
        )
    )


# Top left - PRMSL
if args.PRMSL:
    ax_back = fig.add_axes([0.00, 0.5, 0.5, 0.5])
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
tsx = all_stats["dtp"]
ty = [x / 100 for x in all_stats["PRMSL_target"]]
my = [x / 100 for x in all_stats["PRMSL_model"]]
plot_var(tsx, ty, my, 1, 2, "PRMSL")

# Bottom left - SST
if args.SST:
    ax_back = fig.add_axes([0.00, 0.0, 0.5, 0.5])
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
if args.xpoint is None or lm_ERA5.data.mask[args.ypoint, args.xpoint] == False:
    offset = 0
    ty = [x - offset for x in all_stats["SST_target"]]
    my = [x - offset for x in all_stats["SST_model"]]
    plot_var(tsx, ty, my, 1, 1, "SST")

# Bottom right - TMP2m
if args.TMP2m:
    ax_back = fig.add_axes([0.5, 0.0, 0.5, 0.5])
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
offset = 0
ty = [x - offset for x in all_stats["T2M_target"]]
my = [x - offset for x in all_stats["T2M_model"]]
plot_var(tsx, ty, my, 2, 1, "T2M")

# Top right - PRATE
if args.PRATE:
    ax_back = fig.add_axes([0.5, 0.5, 0.5, 0.5])
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
ty = [x * 1 for x in all_stats["PRATE_target"]]
my = [x * 1 for x in all_stats["PRATE_model"]]
plot_var(tsx, ty, my, 2, 2, "PRATE")


fig.savefig("multi.png")
