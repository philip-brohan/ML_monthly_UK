#!/usr/bin/env python

# Read in the 4 monthly variables on the HadUK-Grid
# Normalise: convert to the range ~0-1
# Convert into a TensorFlow tensor.
# Serialise and store on $SCRATCH.

import os
import sys
import tensorflow as tf

# Going to do external parallelism - run this on one core
tf.config.threading.set_inter_op_parallelism_threads(1)
import dask

dask.config.set(scheduler="single-threaded")

import warnings

# warnings.filterwarnings("ignore", message=".*TransverseMercator.*")

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import TSOURCE

sys.path.append("%s/." % os.path.dirname(__file__))
from tensor_utils import load_cList
from tensor_utils import cList_to_tensor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument(
    "--member", help="Ensemble member", default=1, type=int, required=False
)
parser.add_argument("--test", help="test data, not training", action="store_true")
parser.add_argument(
    "--extrapolate", help="Fill in missing data with extrapolation", action="store_true"
)
parser.add_argument(
    "--opfile", help="tf data file name", default=None, type=str, required=False
)
args = parser.parse_args()
if args.opfile is None:
    purpose = "training"
    if args.test:
        purpose = "test"
    if args.extrapolate:
        purpose += "_source"
    else:
        purpose += "_target"
    args.opfile = ("%s/datasets/%s/%04d-%02d_%02d.tfd") % (
        TSOURCE,
        purpose,
        args.year,
        args.month,
        args.member,
    )

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

# Load and standardise data
qd = load_cList(args.year, args.month, args.member)
ict = cList_to_tensor(qd, extrapolate=args.extrapolate)

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile, sict)
