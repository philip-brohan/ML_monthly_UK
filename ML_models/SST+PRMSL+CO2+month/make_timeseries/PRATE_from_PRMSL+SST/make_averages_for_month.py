#!/usr/bin/env python

# Get UK-land averages from all data sources for a selected month
# Pickle them for plotting.

import os
import sys
import numpy as np
import iris
import iris.analysis
from calendar import monthrange
import pickle

import warnings

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
args = parser.parse_args()

sys.path.append("%s/../../../../get_data" % os.path.dirname(__file__))
from HUKG_monthly_load import load_cList
from HUKG_monthly_load import sCube
from HUKG_monthly_load import lm_20CR
from HUKG_monthly_load import dm_hukg


# Load the original data data
orig = []
for member in [1, 12, 24, 36, 48, 60, 72]:
    qd = load_cList(args.year, args.month, member)
    orig.append(qd)

# Load the fitted data
# and reduce to joint v3-land:HadUKg coverage on the 1 degree grid
def load_fitted(year, month, member, epoch):
    fn = (
        "%s/ML_monthly_UK/DCVAE+scalars/fitted/constraints_PRMSL_SST/"
        + "%04d/%04d/%02d/%02d.nc"
    ) % (
        os.getenv("SCRATCH"),
        epoch,
        year,
        month,
        member,
    )
    if not os.path.exists(fn):
        raise Exception("Missing data file %s" % fn)
    fitted = iris.load(fn)
    fitted.sort(key=lambda cube: cube.var_name)
    return fitted


ft = []
for member in [1, 12, 24, 36, 48, 60, 72]:
    qd = load_fitted(args.year, args.month, member, args.epoch)
    ft.append(qd)

# Package-up the averages
res = {
    "T2m": {
        "Orig": [],
        "Fit": [],
    },
    "PRATE": {
        "Orig": [],
        "Fit": [],
    },
}

seconds_in_month = 86400 * monthrange(args.year, args.month)[1]
for mi in range(len(ft)):
    res["T2m"]["Orig"].append(np.mean(orig[mi][2].data) - 273.15)
    res["T2m"]["Fit"].append(np.mean(ft[mi][3].data) - 273.15)
    res["PRATE"]["Orig"].append(np.mean(orig[mi][3].data) * seconds_in_month)
    res["PRATE"]["Fit"].append(np.mean(ft[mi][0].data) * seconds_in_month)

opfile = (
    "%s/ML_monthly_UK/DCVAE+scalars/UK_averages/PRMSL_SST/" + "%04d/%04d/%02d.pkl"
) % (
    os.getenv("SCRATCH"),
    args.epoch,
    args.year,
    args.month,
)

if not os.path.isdir(os.path.dirname(opfile)):
    os.makedirs(os.path.dirname(opfile))

pickle.dump(res, open(opfile, "wb"))
