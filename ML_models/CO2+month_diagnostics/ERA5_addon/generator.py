#!/usr/bin/env python

# Convolutional generator for ERA5 monthly fields from encoded HadUK-Grid state

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
from makeDataset import getDataset
from generatorModel import DCG
from generatorModel import train_step
from generatorModel import compute_loss

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH

# How many epochs to train for
nEpochs = 200
# Length of an epoch - if None, use all input data
nImagesInEpoch = None
nRepeatsPerEpoch = 2  # Show each input pair this many times

# How many cases to use for the validation data
# If None, use all the training cases
nValidationImages = None

# How many cases to use for the test data
# If None, use all the test cases
nTestImages = None

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
):
    save_dir = ("%s/models_ERA5_generator/Epoch_%04d") % (LSCRATCH, epoch,)
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
    history_file = "%s/history.pkl" % save_dir
    pickle.dump(history, open(history_file, "wb"))


def test_stats(generator, tst_ds):
    rmse_PRMSL = tf.keras.metrics.Mean()
    rmse_SST = tf.keras.metrics.Mean()
    rmse_T2M = tf.keras.metrics.Mean()
    rmse_PRATE = tf.keras.metrics.Mean()
    for test_x in tst_ds:
        per_replica_loss = strategy.run(compute_loss, args=(generator, test_x))
        vstack = strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None
        )
        rmse_PRMSL(vstack[0, :])
        rmse_SST(vstack[1, :])
        rmse_T2M(vstack[2, :])
        rmse_PRATE(vstack[3, :])
    return (
        rmse_PRMSL,
        rmse_SST,
        rmse_T2M,
        rmse_PRATE,
    )


# Instantiate and run the model under the control of the distribution strategy
with strategy.scope():

    # Set up the training data
    trainingData = getDataset(purpose="training", nImages=nImagesInEpoch).repeat(
        nRepeatsPerEpoch
    )
    trainingData = trainingData.shuffle(bufferSize).batch(batchSize)

    # Subset of the training data for metrics
    validationData = getDataset(purpose="training", nImages=nValidationImages).batch(
        batchSize
    )

    # Set up the test data
    testData = getDataset(purpose="test", nImages=nTestImages)
    testData = testData.batch(batchSize)

    # Instantiate the model
    generator = DCG()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    # If we are doing a restart, load the weights
    if args.epoch > 0:
        weights_dir = ("%s/models_ERA5_generator/Epoch_%04d") % (LSCRATCH, args.epoch,)
        load_status = generator.load_weights("%s/ckpt" % weights_dir)
        # Check the load worked
        load_status.assert_existing_objects_matched()

    # Train
    for epoch in range(args.epoch, nEpochs):
        start_time = time.time()
        for train_x in trainingData:
            per_replica_losses = strategy.run(
                train_step, args=(generator, train_x, optimizer)
            )

        end_time = time.time()

        # Measure performance on training data
        (
            train_rmse_PRMSL,
            train_rmse_SST,
            train_rmse_T2M,
            train_rmse_PRATE,
        ) = test_stats(generator, validationData)

        # Measure performance on test data
        (test_rmse_PRMSL, test_rmse_SST, test_rmse_T2M, test_rmse_PRATE,) = test_stats(
            generator, testData
        )

        # Save model state and validation statistics
        save_state(
            generator,
            epoch,
            train_rmse_PRMSL.result(),
            test_rmse_PRMSL.result(),
            train_rmse_SST.result(),
            test_rmse_SST.result(),
            train_rmse_T2M.result(),
            test_rmse_T2M.result(),
            train_rmse_PRATE.result(),
            test_rmse_PRATE.result(),
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
            "time: {} (+{})".format(
                int(end_time - start_time), int(val_time - end_time)
            )
        )
