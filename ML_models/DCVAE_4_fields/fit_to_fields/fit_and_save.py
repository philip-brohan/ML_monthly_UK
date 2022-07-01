#!/usr/bin/env python

# Find a point in latent space that maximises the fit to some given input fields,
#  and save the fitted state.


import os
import sys
import numpy as np
import iris
import tensorflow as tf
import tensorflow_probability as tfp

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=100)
parser.add_argument(
    "--year", help="Year to fit to", type=int, required=True,
)
parser.add_argument(
    "--month", help="Month to fit to", type=int, required=True,
)
parser.add_argument(
    "--member", help="Member to fit to", type=int, required=False, default=1
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
args = parser.parse_args()

opfile = "%s/ML_monthly_UK/fitted/constraints" % os.getenv("SCRATCH")
for constraint in ["PRMSL", "PRATE", "TMP2m", "SST"]:
    if vars(args)[constraint]:
        opfile += "_%s" % constraint
opfile += "/%04d/%04d/%02d/%02d.nc" % (args.epoch, args.year, args.month, args.member)

sys.path.append("%s/../../../get_data" % os.path.dirname(__file__))
from TWCR_monthly_load import load_quad

sys.path.append("%s/../../make_tensors" % os.path.dirname(__file__))
from tensor_utils import quad_to_tensor
from tensor_utils import tensor_to_quad

sys.path.append("%s/../../../plots" % os.path.dirname(__file__))
from plot_variable import plot_cube

# Load and standardise data
pc = plot_cube()
qd = load_quad(args.year, args.month, args.member)
ict = quad_to_tensor(qd, pc)
sst_mask = ict.numpy()[:, :, 1] == 0.0

# Load the model specification
sys.path.append("%s/.." % os.path.dirname(__file__))
from autoencoderModel import DCVAE

autoencoder = DCVAE()
weights_dir = ("%s//ML_monthly_UK/models/DCVAE_4_fields/" + "Epoch_%04d") % (
    os.getenv("SCRATCH"),
    args.epoch,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()

# We are using the model in inference mode - (does this have any effect?)
autoencoder.decoder.trainable = False
for layer in autoencoder.decoder.layers:
    layer.trainable = False
autoencoder.decoder.compile()

latent = tf.Variable(tf.random.normal(shape=(1, autoencoder.latent_dim)))
target = tf.constant(tf.reshape(ict, [1, 32, 32, 4]))


def decodeFit():
    result = 0.0
    decoded = autoencoder.decode(latent)
    if args.PRMSL:
        result = result + tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(decoded[:, :, :, 0], target[:, :, :, 0])
        )
    if args.SST:
        result = result + tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(
                tf.boolean_mask(decoded[:, :, :, 1], np.invert(sst_mask), axis=1),
                tf.boolean_mask(target[:, :, :, 1], np.invert(sst_mask), axis=1),
            )
        )
    if args.TMP2m:
        result = result + tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(decoded[:, :, :, 2], target[:, :, :, 2])
        )
    if args.PRATE:
        result = result + tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(decoded[:, :, :, 3], target[:, :, :, 3])
        )
    return result


if args.PRMSL or args.SST or args.TMP2m or args.PRATE:
    loss = tfp.math.minimize(
        decodeFit,
        trainable_variables=[latent],
        num_steps=1000,
        optimizer=tf.optimizers.Adam(learning_rate=0.05),
    )

encoded = autoencoder.decode(latent)

fitted = tensor_to_quad(encoded[0, :, :, :], pc)

if not os.path.isdir(os.path.dirname(opfile)):
    os.makedirs(os.path.dirname(opfile))

iris.save(fitted, opfile)
