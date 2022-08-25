#!/usr/bin/env python

# Plot validation statistics for all the test cases
# ERA5 addon version

import os
import sys
import numpy as np
import tensorflow as tf
import iris
import iris.fileformats
import iris.analysis
import datetime
from statistics import mean
import cmocean

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import warnings

warnings.filterwarnings("ignore", message=".*partition.*")
warnings.filterwarnings("ignore", message=".*TransverseMercator.*")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=99)
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
    "--annual",
    help="Make annual averages",
    dest="annual",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--anomalies",
    help="Make monthly anomalies",
    dest="anomalies",
    default=False,
    action="store_true",
)
args = parser.parse_args()

sys.path.append("%s/../../../../get_data" % os.path.dirname(__file__))
from ERA5_monthly_load import lm_ERA5
from HUKG_monthly_load import dm_hukg

sys.path.append("%s/../make_tensors" % os.path.dirname(__file__))
from E_tensor_utils import nPar

sys.path.append("%s/.." % os.path.dirname(__file__))
from generatorModel import DCG
from makeDataset import getPairedFileNames
from makeDataset import getDataset

# Load the model specification
sys.path.append("%s/../.." % os.path.dirname(__file__))
from localise import LSCRATCH
from localise import TSOURCE

# Set up the test data
testData = getDataset(purpose="test", startyear=args.startyear, endyear=args.endyear)
testData = testData.batch(1)
# Get the associated dates, so we can do time-series
dts = []
for fn in getPairedFileNames(
    purpose="test", startyear=args.startyear, endyear=args.endyear
)[2]:
    dtp = datetime.date(int(fn[-11:-7]), int(fn[-6:-4]), 15)
    dts.append(dtp)

generator = DCG()
weights_dir = ("%s/models_ERA5_generator/Epoch_%04d") % (LSCRATCH, args.epoch,)
load_status = generator.load_weights("%s/ckpt" % weights_dir)
load_status.assert_existing_objects_matched()


def field_to_scalar(field, mask):
    if args.xpoint is not None and args.ypoint is not None:
        return field[args.ypoint, args.xpoint]
    if args.xpoint is None and args.ypoint is None:
        if mask is not None:
            return np.mean(field[np.where(mask == False)])
        return np.mean(field)
    else:
        raise Exception("Must specify both xpoint and ypoint (or neither).")


# Get target and encoded statistics for one test case
def compute_stats(model, x):
    encoded = model.call(x)
    stats = {}
    stats["PRMSL_target"] = field_to_scalar(x[2][0, :, :, 0].numpy(), None)
    stats["PRMSL_model"] = field_to_scalar(encoded[0, :, :, 0].numpy(), None)
    vt = x[2][0, :, :, 1].numpy()
    stats["SST_target"] = field_to_scalar(vt, lm_ERA5.data.mask)
    vm = encoded[0, :, :, 1].numpy()
    stats["SST_model"] = field_to_scalar(vm, lm_ERA5.data.mask)
    vt = x[2][0, :, :, 2].numpy()
    stats["T2M_target"] = field_to_scalar(vt, dm_hukg.data.mask)
    vm = encoded[0, :, :, 2].numpy()
    stats["T2M_model"] = field_to_scalar(vm, dm_hukg.data.mask)
    vt = x[2][0, :, :, 3].numpy()
    stats["PRATE_target"] = field_to_scalar(vt, dm_hukg.data.mask)
    vm = encoded[0, :, :, 3].numpy()
    stats["PRATE_model"] = field_to_scalar(vm, dm_hukg.data.mask)
    return stats


all_stats = {}
for case in testData:
    stats = compute_stats(generator, case)
    for key in stats.keys():
        if key in all_stats:
            all_stats[key].append(stats[key])
        else:
            all_stats[key] = [stats[key]]


def to_annual_averages(ts, monthly):
    result = []
    current = []
    for idx in range(len(ts)):
        current.append(monthly[idx])
        if idx == len(ts) - 1 or (ts[idx].year != ts[idx + 1].year):
            result.append(mean(current))
            current = []
    return result


def to_monthly_anomalies(ts, monthly):
    climatology = []
    for m in range(13):
        climatology.append([])
    for idx in range(len(ts)):
        climatology[ts[idx].month].append(monthly[idx])
    for m in range(1, 13):
        climatology[m] = mean(climatology[m])
    result = []
    for idx in range(len(ts)):
        result.append(monthly[idx] - climatology[ts[idx].month])
    return result


# Plot sizes
tsh = 2
sh = 2
tsw = 4
ssw = 2
hpad = 0.5
wpad = 0.5
fig_h = tsh * 3 + hpad * 4
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
    Rectangle((0, 0), 1, 1, facecolor=(0.95, 0.95, 0.95, 1), fill=True, zorder=1,)
)


def unnormalise(value, variable):
    if not variable in nPar:
        raise Exception("Unsupported variable " + variable)
    value *= nPar[variable][1] - nPar[variable][0]
    value += nPar[variable][0]
    return value


def plot_var(ts, t, m, xp, yp, label):
    if args.anomalies:
        t = to_monthly_anomalies(ts, t)
        m = to_monthly_anomalies(ts, m)
    if args.annual:
        t = to_annual_averages(ts, t)
        m = to_annual_averages(ts, m)
        ts = list(range(dts[0].year, dts[-1].year + 1))
        ts = [datetime.date(y, 7, 1) for y in ts]
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
tsx = dts
ty = [unnormalise(x, "PRMSL") / 100 for x in all_stats["PRMSL_target"]]
my = [unnormalise(x, "PRMSL") / 100 for x in all_stats["PRMSL_model"]]
plot_var(tsx, ty, my, 1, 3, "PRMSL")

# Centre left - SST
if args.xpoint is None or lm_20CR.data.mask[args.ypoint, args.xpoint] == False:
    ty = [unnormalise(x, "TMPS") - 273.15 for x in all_stats["SST_target"]]
    my = [unnormalise(x, "TMPS") - 273.15 for x in all_stats["SST_model"]]
    plot_var(tsx, ty, my, 1, 2, "SST")

# Bottom left - T2M
if args.xpoint is None or dm_hukg.data.mask[args.ypoint, args.xpoint] == False:
    ty = [unnormalise(x, "TMP2m") - 273.15 for x in all_stats["T2M_target"]]
    my = [unnormalise(x, "TMP2m") - 273.15 for x in all_stats["T2M_model"]]
    plot_var(tsx, ty, my, 1, 1, "T2M")

# Top right - PRATE
if args.xpoint is None or dm_hukg.data.mask[args.ypoint, args.xpoint] == False:
    ty = [unnormalise(x, "PRATE") * 1000 for x in all_stats["PRATE_target"]]
    my = [unnormalise(x, "PRATE") * 1000 for x in all_stats["PRATE_model"]]
    plot_var(tsx, ty, my, 2, 3, "PRATE")


fig.savefig("multi.png")
