#!/usr/bin/env python

# Make and store the encoded latent vector for each input tensor

# This script leaks memory badly (tf bug?) so don't do everything at once -
#  run it decade by decade using --startyear and --endyear

import os
import sys
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epoch", help="Restart from epoch", type=int, required=False, default=299
)
parser.add_argument(
    "--startyear", help="First year to encode", type=int, required=False, default=None
)
parser.add_argument(
    "--endyear", help="Last year to encode", type=int, required=False, default=None
)
args = parser.parse_args()

# Load the data source and model specification
sys.path.append("%s/../.." % os.path.dirname(__file__))
from localise import LSCRATCH
from makeDataset import getDataset
from makeDataset import getFileNames
from autoencoderModel import DCVAE

# Set up the training data
trainingData = getDataset(
    purpose="training", startyear=args.startyear, endyear=args.endyear
).batch(1)
trainingFN = getFileNames(
    purpose="training", startyear=args.startyear, endyear=args.endyear
)

# Set up the test data
testData = getDataset(
    purpose="test", startyear=args.startyear, endyear=args.endyear
).batch(1)
testFN = getFileNames(purpose="test", startyear=args.startyear, endyear=args.endyear)

# Instantiate the model
autoencoder = DCVAE()
# load the weights
weights_dir = ("%s/models/Epoch_%04d") % (LSCRATCH, args.epoch,)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()
# Freeze the model - we're using it, not training it
autoencoder.trainable = False


def write_op(ict, purpose, inFN, group):
    opFN = "%s/latents/%s/%s/%s" % (LSCRATCH, purpose, group, inFN)
    if not os.path.isdir(os.path.dirname(opFN)):
        os.makedirs(os.path.dirname(opFN))
    sict = tf.io.serialize_tensor(ict)
    tf.io.write_file(opFN, sict)


# Encode each input
for idx, x in enumerate(trainingData):
    mean, logvar = autoencoder.encode(x)
    write_op(mean, "training", trainingFN[idx], "mean")
    write_op(logvar, "training", trainingFN[idx], "logvar")

for idx, x in enumerate(testData):
    mean, logvar = autoencoder.encode(x)
    write_op(mean, "test", testFN[idx], "mean")
    write_op(logvar, "test", testFN[idx], "logvar")
