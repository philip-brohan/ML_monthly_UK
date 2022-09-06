#!/usr/bin/env python

# Make and store the encoded latent vector for each input tensor

import os
import sys
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epoch", help="Training epoch to use", type=int, required=False, default=100
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
    purpose="training", startyear=args.startyear, endyear=args.endyear, shuffle=False,
).batch(1)

# Set up the test data
testData = getDataset(
    purpose="test", startyear=args.startyear, endyear=args.endyear, shuffle=False,
).batch(1)

# Instantiate the model
autoencoder = DCVAE()
# load the weights
weights_dir = ("%s/models/Epoch_%04d") % (LSCRATCH, args.epoch,)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()
# Freeze the model - we're using it, not training it
# Not really necessary, as we are not calling the training function, but
#  probably good practice.
autoencoder.trainable = False


def write_op(ict, purpose, inFNT, group):
    inFN = inFNT[0].numpy().decode("utf8")  # str from string tensor
    opFN = "%s/latents/%s/%s/%s" % (LSCRATCH, purpose, group, inFN)
    if not os.path.isdir(os.path.dirname(opFN)):
        os.makedirs(os.path.dirname(opFN))
    sict = tf.io.serialize_tensor(ict)
    tf.io.write_file(opFN, sict)


# Encode each input
for x in trainingData:
    mean, logvar = autoencoder.encode(x[0])
    write_op(mean, "training", x[2], "mean")
    write_op(logvar, "training", x[2], "logvar")

for x in testData:
    mean, logvar = autoencoder.encode(x[0])
    write_op(mean, "test", x[2], "mean")
    write_op(logvar, "test", x[2], "logvar")
