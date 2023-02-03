#!/usr/bin/env python

# Convolutional Variational Autoencoder for HadUK-Grid monthly fields
# this version has the CO2 level and the month inserted into the latent space

import os
import sys
import time
import tensorflow as tf

import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epoch", help="Restart from epoch", type=int, required=False, default=0
)
args = parser.parse_args()

# Distribute across all GPUs
# strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.experimental.CentralStorageStrategy()
# Sadly, seems to be faster if you don't bother.
strategy = tf.distribute.get_strategy()

# Load the data source and model specification
sys.path.append("%s/." % os.path.dirname(__file__))
from localise import LSCRATCH
from makeDataset import getDataset
from autoencoderModel import DCVAE
from autoencoderModel import train_step
from autoencoderModel import compute_loss

# How many images to use?
nTrainingImages = 1491  # Max is 1491
nTestImages = 165  # Max is 165

# How many epochs to train for
nEpochs = 1000
# Length of an epoch - if None, same as nTrainingImages
nImagesInEpoch = None
nRepeatsPerEpoch = 5  # Show each image this many times

if nImagesInEpoch is None:
    nImagesInEpoch = nTrainingImages

# Dataset parameters
bufferSize = 1000  # Untested
batchSize = 32  # Arbitrary

# Function to store the model weights and the history state
history = {}
for loss in (
    "PRMSL_train",
    "PRMSL_test",
    "SST_train",
    "SST_test",
    "T2M_train",
    "T2M_test",
    "PRATE_train",
    "PRATE_test",
    "logpz_train",
    "logpz_test",
    "logqz_x_train",
    "logqz_x_test",
):
    history[loss] = []


def save_state(
    model,
    epoch,
    PRMSL_train,
    PRMSL_test,
    SST_train,
    SST_test,
    T2M_train,
    T2M_test,
    PRATE_train,
    PRATE_test,
    logpz_train,
    logpz_test,
    logqz_x_train,
    logqz_x_test,
):
    save_dir = ("%s/models/Epoch_%04d") % (
        LSCRATCH,
        epoch,
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model.save_weights("%s/ckpt" % save_dir)
    history["PRMSL_train"].append(PRMSL_train)
    history["PRMSL_test"].append(PRMSL_test)
    history["SST_train"].append(SST_train)
    history["SST_test"].append(SST_test)
    history["T2M_train"].append(T2M_train)
    history["T2M_test"].append(T2M_test)
    history["PRATE_train"].append(PRATE_train)
    history["PRATE_test"].append(PRATE_test)
    history["logpz_train"].append(logpz_train)
    history["logpz_test"].append(logpz_test)
    history["logqz_x_train"].append(logqz_x_train)
    history["logqz_x_test"].append(logqz_x_test)
    history_file = "%s/history.pkl" % save_dir
    pickle.dump(history, open(history_file, "wb"))


# @tf.function - will stop working if uncommented. No idea why.
def test_stats(autoencoder, tst_ds):
    rmse_PRMSL = tf.keras.metrics.Mean()
    rmse_SST = tf.keras.metrics.Mean()
    rmse_T2M = tf.keras.metrics.Mean()
    rmse_PRATE = tf.keras.metrics.Mean()
    logpz = tf.keras.metrics.Mean()
    logqz_x = tf.keras.metrics.Mean()
    for test_x in tst_ds:
        per_replica_loss = strategy.run(compute_loss, args=(autoencoder, test_x))
        vstack = strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None
        )
        rmse_PRMSL(vstack[0, :])
        rmse_SST(vstack[1, :])
        rmse_T2M(vstack[2, :])
        rmse_PRATE(vstack[3, :])
        logpz(vstack[4, :])
        logqz_x(vstack[5, :])
    return (rmse_PRMSL, rmse_SST, rmse_T2M, rmse_PRATE, logpz, logqz_x)


# Instantiate and run the model under the control of the distribution strategy
with strategy.scope():

    # Set up the training data
    trainingData = getDataset(purpose="training", nImages=nTrainingImages).repeat(5)
    trainingData = trainingData.shuffle(bufferSize).batch(batchSize)

    # Subset of the training data for metrics
    validationData = getDataset(purpose="training", nImages=nTestImages).batch(
        batchSize
    )

    # Set up the test data
    testData = getDataset(purpose="test", nImages=nTestImages)
    testData = testData.batch(batchSize)

    # Instantiate the model
    autoencoder = DCVAE()
    optimizer = tf.keras.optimizers.Adam(1e-3)
    # If we are doing a restart, load the weights
    if args.epoch > 0:
        weights_dir = ("%s/models/Epoch_%04d") % (
            LSCRATCH,
            args.epoch,
        )
        load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
        # Check the load worked
        load_status.assert_existing_objects_matched()

    # Train
    for epoch in range(args.epoch, nEpochs):
        start_time = time.time()
        for train_x in trainingData:
            per_replica_losses = strategy.run(
                train_step, args=(autoencoder, train_x, optimizer)
            )

        end_time = time.time()

        # Measure performance on training data
        (
            train_rmse_PRMSL,
            train_rmse_SST,
            train_rmse_T2M,
            train_rmse_PRATE,
            train_logpz,
            train_logqz_x,
        ) = test_stats(autoencoder, validationData)

        # Measure performance on test data
        (
            test_rmse_PRMSL,
            test_rmse_SST,
            test_rmse_T2M,
            test_rmse_PRATE,
            test_logpz,
            test_logqz_x,
        ) = test_stats(autoencoder, testData)

        # Save model state and validation statistics
        save_state(
            autoencoder,
            epoch,
            train_rmse_PRMSL.result(),
            test_rmse_PRMSL.result(),
            train_rmse_SST.result(),
            test_rmse_SST.result(),
            train_rmse_T2M.result(),
            test_rmse_T2M.result(),
            train_rmse_PRATE.result(),
            test_rmse_PRATE.result(),
            train_logpz.result(),
            test_logpz.result(),
            train_logqz_x.result(),
            test_logqz_x.result(),
        )
        val_time = time.time()

        # Report progress
        print("Epoch: {}".format(epoch))
        print(
            "PRMSL  : {:>6.1f}, {:>6.1f}".format(
                train_rmse_PRMSL.result(), test_rmse_PRMSL.result()
            )
        )
        print(
            "SST    : {:>6.1f}, {:>6.1f}".format(
                train_rmse_SST.result(), test_rmse_SST.result()
            )
        )
        print(
            "T2m    : {:>6.1f}, {:>6.1f}".format(
                train_rmse_T2M.result(), test_rmse_T2M.result()
            )
        )
        print(
            "PRATE  : {:>6.1f}, {:>6.1f}".format(
                train_rmse_PRATE.result(), test_rmse_PRATE.result()
            )
        )
        print(
            "logpz  : {:>6.1f}, {:>6.1f}".format(
                train_logpz.result(), test_logpz.result()
            )
        )
        print(
            "logqz_x: {:>6.1f}, {:>6.1f}".format(
                train_logqz_x.result(), test_logqz_x.result()
            )
        )
        print(
            "time: {} (+{})".format(
                int(end_time - start_time), int(val_time - end_time)
            )
        )
