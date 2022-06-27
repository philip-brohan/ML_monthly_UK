#!/usr/bin/env python

# Convolutional Variational Autoencoder for 20CRv3 UK monthly fields

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
# Doesn't yet work - need to change the training loop and datasets to be
#  strategy aware.
strategy = tf.distribute.get_strategy()

# Load the data source and model specification
sys.path.append("%s/." % os.path.dirname(__file__))
from makeDataset import getDataset
from autoencoderModel import DCVAE
from autoencoderModel import train_step
from autoencoderModel import compute_loss

# How many images to use?
nTrainingImages = 1491  # Max is 1491
nTestImages = 165  # Max is 165

# How many epochs to train for
nEpochs = 100
# Length of an epoch - if None, same as nTrainingImages
nImagesInEpoch = None

if nImagesInEpoch is None:
    nImagesInEpoch = nTrainingImages

# Dataset parameters
bufferSize = 1000  # Untested
batchSize = 32  # Arbitrary

# Set up the training data
trainingData = getDataset(purpose="training", nImages=nTrainingImages).repeat(2)
trainingData = trainingData.shuffle(bufferSize).batch(batchSize)

# Subset of the training data for metrics
validationData = getDataset(purpose="training", nImages=nTestImages).batch(batchSize)

# Set up the test data
testData = getDataset(purpose="test", nImages=nTestImages)
testData = testData.batch(batchSize)

# Instantiate the model
with strategy.scope():
    autoencoder = DCVAE()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    # If we are doing a restart, load the weights
    if args.epoch > 0:
        weights_dir = ("%s/Proxy_20CR/models/DCVAE_single_PUV/" + "Epoch_%04d") % (
            os.getenv("SCRATCH"),
            args.epoch,
        )
        load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
        # Check the load worked
        load_status.assert_existing_objects_matched()


# Save the model weights and the history state after every epoch
history = {}
history["loss"] = []
history["val_loss"] = []


def save_state(model, epoch, loss):
    save_dir = ("%s/ML_monthly_UK/models/DCVAE_4_fields/" + "Epoch_%04d") % (
        os.getenv("SCRATCH"),
        epoch,
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model.save_weights("%s/ckpt" % save_dir)
    history["loss"].append(loss)
    # history["val_loss"].append(logs["val_loss"])
    history_file = "%s/history.pkl" % save_dir
    pickle.dump(history, open(history_file, "wb"))


for epoch in range(nEpochs):
    start_time = time.time()
    for train_x in trainingData:
        train_step(autoencoder, train_x, optimizer)
    end_time = time.time()

    train_rmse_PRMSL = tf.keras.metrics.Mean()
    train_rmse_SST = tf.keras.metrics.Mean()
    train_rmse_T2m = tf.keras.metrics.Mean()
    train_rmse_PRATE = tf.keras.metrics.Mean()
    train_logpz = tf.keras.metrics.Mean()
    train_logqz_x = tf.keras.metrics.Mean()
    for test_x in validationData:
        (rmse_PRMSL, rmse_SST, rmse_T2m, rmse_PRATE, logpz, logqz_x) = compute_loss(
            autoencoder, test_x
        )
        train_rmse_PRMSL(rmse_PRMSL)
        train_rmse_SST(rmse_SST)
        train_rmse_T2m(rmse_T2m)
        train_rmse_PRATE(rmse_PRATE)
        train_logpz(logpz)
        train_logqz_x(logqz_x)
    test_rmse_PRMSL = tf.keras.metrics.Mean()
    test_rmse_SST = tf.keras.metrics.Mean()
    test_rmse_T2m = tf.keras.metrics.Mean()
    train_rmse_PRATE = tf.keras.metrics.Mean()
    test_logpz = tf.keras.metrics.Mean()
    test_logqz_x = tf.keras.metrics.Mean()
    for test_x in testData:
        (rmse_PRMSL, rmse_SST, rmse_T2m, rmse_PRATE, logpz, logqz_x) = compute_loss(
            autoencoder, test_x
        )
        test_rmse_PRMSL(rmse_PRMSL)
        test_rmse_SST(rmse_SST)
        test_rmse_T2m(rmse_T2m)
        test_rmse_PRATE(rmse_PRATE)
        test_logpz(logpz)
        test_logqz_x(logqz_x)
    print("Epoch: {}".format(epoch))
    print(
        "RMSE PRMSL: {}, {}".format(train_rmse_PRMSL.result(), test_rmse_PRMSL.result())
    )
    print("RMSE SST  : {}, {}".format(train_rmse_SST.result(), test_rmse_SST.result()))
    print("RMSE T2m  : {}, {}".format(train_rmse_T2m.result(), test_rmse_T2m.result()))
    print(
        "RMSE PRATE: {}, {}".format(train_rmse_PRATE.result(), test_rmse_PRATE.result())
    )
    print("logpz: {}, {}".format(train_logpz.result(), test_logpz.result()))
    print("logqz_x: {}, {}".format(train_logqz_x.result(), test_logqz_x.result()))
    print("time: {}".format(end_time - start_time))
    if epoch % 10 == 0:
        save_state(autoencoder, epoch, test_rmse_PRATE.result())
