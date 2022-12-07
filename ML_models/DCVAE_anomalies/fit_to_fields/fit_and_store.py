#!/usr/bin/env python

# Find and store the latent space vectors from the test and training
#  cases - for future use.

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
from statistics import mean

import warnings

warnings.filterwarnings("ignore", message=".*datum.*")


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
    "--training",
    help="Use training months (not test months)",
    dest="training",
    default=False,
    action="store_true",
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

from make_tensors.tensor_utils import lm_TWCR

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


def decodeFit():
    result = 0.0
    generated = model.generate(latent, training=False)
    result = result + tf.reduce_mean(
        tf.keras.metrics.mean_squared_error(
            generated[:, :, :, 0], target[:, :, :, 0]
        )
    )
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
    result = result + tf.reduce_mean(
        tf.keras.metrics.mean_squared_error(
            tf.boolean_mask(
                generated[:, :, :, 2],
                target[:, :, :, 2] != 0.5,
            ),
            tf.boolean_mask(
                target[:, :, :, 2],
                target[:, :, :, 2] != 0.5,
            ),
        )
    )
    result = result + tf.reduce_mean(
        tf.keras.metrics.mean_squared_error(
            tf.boolean_mask(
                generated[:, :, :, 3],
                target[:, :, :, 3] != 0.5,
            ),
            tf.boolean_mask(
                target[:, :, :, 3],
                target[:, :, :, 3] != 0.5,
            ),
        )
    )
    return result


# Find latent space vector for one test case
def find_latent(x):
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
    loss = tfp.math.minimize(
        decodeFit,
        trainable_variables=[latent],
        num_steps=args.iter,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
    )
    return (year,month)


# Calculate and store the latent for each input case
opdir = "%s/latents/%s" % (TSOURCE, purpose)
if not os.path.isdir(opdir):
    os.makedirs(opdir)

for case in testData:
    (year,month) = find_latent(case)
    opfile = "%s/%04d-%02d.tfd" % (opdir,year,month)
    sict = tf.io.serialize_tensor(latent)
    tf.io.write_file(opfile, sict)

