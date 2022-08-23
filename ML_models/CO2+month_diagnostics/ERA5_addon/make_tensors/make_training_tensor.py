#!/usr/bin/env python

# Read in the 4 monthly variables on the HadUK-Grid
# Normalise: convert to the range ~0-1
# Convert into a TensorFlow tensor.
# Serialise and store on $SCRATCH.
# This version uses ERA5 input data

import os
import sys
import tensorflow as tf
import numpy as np
import iris
import iris.util
import iris.fileformats

# Going to do external parallelism - run this on one core
tf.config.threading.set_inter_op_parallelism_threads(1)
import dask

dask.config.set(scheduler="single-threaded")

import warnings

warnings.filterwarnings("ignore", message=".*TransverseMercator.*")

sys.path.append("%s/../.." % os.path.dirname(__file__))
from localise import TSOURCE

sys.path.append("%s/../../../../get_data/" % os.path.dirname(__file__))
from ERA5_monthly_load import load_cList
from ERA5_monthly_load import lm_ERA5

sys.path.append("%s/." % os.path.dirname(__file__))
from tensor_utils import cList_to_tensor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument("--test", help="test data, not training", action="store_true")
parser.add_argument(
    "--opfile", help="tf data file name", default=None, type=str, required=False
)
args = parser.parse_args()
if args.opfile is None:
    purpose = "training"
    if args.test:
        purpose = "test"
    args.opfile = ("%s/datasets_ERA5/%s/%04d-%02d.tfd") % (
        TSOURCE,
        purpose,
        args.year,
        args.month,
    )

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

# Load and standardise data
qd = load_cList(args.year, args.month)
ict = cList_to_tensor(qd, lm_ERA5.data.mask)

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile, sict)
