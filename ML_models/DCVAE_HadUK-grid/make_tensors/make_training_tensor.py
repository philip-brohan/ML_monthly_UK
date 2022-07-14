#!/usr/bin/env python

# Read in the 4 monthly variables on the HadUK-Grid
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

sys.path.append("%s/../../../get_data/" % os.path.dirname(__file__))
from HUKG_monthly_load import load_cList

sys.path.append("%s/." % os.path.dirname(__file__))
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
    "--opfile", help="tf data file name", default=None, type=str, required=False
)
args = parser.parse_args()
if args.opfile is None:
    purpose = "training"
    if args.test:
        purpose = "test"
    args.opfile = (
        "%s/ML_monthly_UK/DCVAE_HadUK-grid/datasets/%s/%04d-%02d_%02d.tfd"
    ) % (os.getenv("SCRATCH"), purpose, args.year, args.month, args.member,)

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

# Load and standardise data
qd = load_cList(args.year, args.month, args.member)
ict = cList_to_tensor(qd)

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile, sict)
