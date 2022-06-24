#!/usr/bin/env python

# Read in the 4 monthly variables for a given month
# Regrid to the UK-region 1-degree grid
# Normalise: convert to the range ~0-1
# Convert into a TensorFlow tensor.
# Serialise and store on $SCRATCH.

import tensorflow as tf
import numpy as np
import iris
import iris.util
import iris.fileformats

# Going to do external parallelism - run this on one core
tf.config.threading.set_inter_op_parallelism_threads(1)
import dask

dask.config.set(scheduler="single-threaded")

import os
import sys

sys.path.append("%s/../../get_data/" % os.path.dirname(__file__))
from TWCR_monthly_load import load_monthly_member
from TWCR_monthly_load import normalise

sys.path.append("%s/../../plots" % os.path.dirname(__file__))
from plot_variable import plot_cube

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument(
    "--member", help="Ensemble member", default=1, type=int, required=False
)
parser.add_argument("--test", help="test data, not training", action="store_true")
parser.add_argument(
    "--opfile", help="tf data file name", default=None, type=str, required=False
)
args = parser.parse_args()
if args.opfile is None:
    purpose = "training"
    if args.test:
        purpose = "test"
    args.opfile = ("%s/ML_monthly_UK/datasets/%s/%04d-%02d_%02d.tfd") % (
        os.getenv("SCRATCH"),
        purpose,
        args.year,
        args.month,
        args.member,
    )

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

# Load and standardise data
pc = plot_cube()
v1 = load_monthly_member("PRMSL", args.year, args.month, args.member)
d1 = normalise(v1.regrid(pc, iris.analysis.Linear()), "PRMSL")
v1 = load_monthly_member("TMPS", args.year, args.month, args.member)
d2 = normalise(v1.regrid(pc, iris.analysis.Linear()), "TMPS")
lm = iris.load_cube("%s/20CR/version_3/fixed/land.nc" % os.getenv("SCRATCH"))
lm = iris.util.squeeze(lm)
coord_s = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
lm.coord("latitude").coord_system = coord_s
lm.coord("longitude").coord_system = coord_s
lm = lm.regrid(pc, iris.analysis.Linear())
d2.data[lm.data > 0.5] = 0.0
v1 = load_monthly_member("TMP2m", args.year, args.month, args.member)
d3 = normalise(v1.regrid(pc, iris.analysis.Linear()), "TMP2m")
v1 = load_monthly_member("PRATE", args.year, args.month, args.member)
d4 = normalise(v1.regrid(pc, iris.analysis.Linear()), "PRATE")
ic = np.stack((d1.data, d2.data, d3.data, d4.data), axis=2)

# Convert to Tensor
ict = tf.convert_to_tensor(ic.data, np.float32)

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile, sict)
