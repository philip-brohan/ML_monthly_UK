#!/usr/bin/env python

# Convolutional Variational Autoencoder for HadUK-Grid monthly fields

import os
import sys
import time
import tensorflow as tf

import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epoch", help="Restart from epoch", type=int, required=False, default=1
)
args = parser.parse_args()

# Distribute across all GPUs
# strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.experimental.CentralStorageStrategy()
strategy = tf.distribute.get_strategy()

# Load the data path, data source, and model specification
sys.path.append("%s/." % os.path.dirname(__file__))
from localise import LSCRATCH
from makeDataset import getDataset
from autoencoderModel import DCVAE

# How many images to use?
nTrainingImages = None
nTestImages = None

# How many epochs to train for
nEpochs = 300
# Length of an epoch - if None, same as nTrainingImages
nImagesInEpoch = None
nRepeatsPerEpoch = 2  # Show each month this many times

if nImagesInEpoch is None:
    nImagesInEpoch = nTrainingImages

# Dataset parameters
bufferSize = 1000  # Untested
batchSize = 32  # Arbitrary


# Instantiate and run the model under the control of the distribution strategy
with strategy.scope():

    # Set up the training data
    trainingData = getDataset(purpose="training", nImages=nTrainingImages).repeat(
        nRepeatsPerEpoch
    )
    trainingData = trainingData.shuffle(bufferSize).batch(batchSize)
    trainingData = strategy.experimental_distribute_dataset(trainingData)

    # Subset of the training data for metrics
    validationData = getDataset(purpose="training", nImages=nTestImages).batch(
        batchSize
    )
    validationData = strategy.experimental_distribute_dataset(validationData)

    # Set up the test data
    testData = getDataset(purpose="test", nImages=nTestImages)
    testData = testData.batch(batchSize)
    testData = strategy.experimental_distribute_dataset(validationData)

    # Instantiate the model
    autoencoder = DCVAE()
    optimizer = tf.keras.optimizers.Adam(1e-3)
    # If we are doing a restart, load the weights
    if args.epoch > 1:
        weights_dir = ("%s/models/Epoch_%04d") % (LSCRATCH, args.epoch,)
        load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
        load_status.assert_existing_objects_matched()

    # Metrics for training and test loss
    # Each the mean over all the batches
    train_rmse_PRMSL = tf.keras.metrics.Mean()
    train_rmse_SST = tf.keras.metrics.Mean()
    train_rmse_T2M = tf.keras.metrics.Mean()
    train_rmse_PRATE = tf.keras.metrics.Mean()
    train_logpz = tf.keras.metrics.Mean()
    train_logqz_x = tf.keras.metrics.Mean()
    train_loss = tf.keras.metrics.Mean()
    test_rmse_PRMSL = tf.keras.metrics.Mean()
    test_rmse_SST = tf.keras.metrics.Mean()
    test_rmse_T2M = tf.keras.metrics.Mean()
    test_rmse_PRATE = tf.keras.metrics.Mean()
    test_logpz = tf.keras.metrics.Mean()
    test_logqz_x = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()

    # logfile to output the met
    log_FN = ("%s/models/Training_log") % LSCRATCH
    if not os.path.isdir(os.path.dirname(log_FN)):
        os.makedirs(os.path.dirname(log_FN))
    logfile_writer = tf.summary.create_file_writer(log_FN)

    # For each Epoch: train, save state, and report progress
    for epoch in range(args.epoch, nEpochs + 1):
        start_time = time.time()

        # Train on all batches in the training data
        for batch in trainingData:
            per_replica_op = strategy.run(
                autoencoder.train_on_batch, args=(batch, optimizer)
            )

        end_training_time = time.time()

        # Accumulate average losses over all batches in the validation data
        train_rmse_PRMSL.reset_states()
        train_rmse_SST.reset_states()
        train_rmse_T2M.reset_states()
        train_rmse_PRATE.reset_states()
        train_logpz.reset_states()
        train_logqz_x.reset_states()
        train_loss.reset_states()
        for batch in validationData:
            per_replica_op = strategy.run(autoencoder.compute_loss, args=(batch,))
            train_rmse_PRMSL.update_state(autoencoder.rmse_PRMSL)
            train_rmse_SST.update_state(autoencoder.rmse_SST)
            train_rmse_T2M.update_state(autoencoder.rmse_T2M)
            train_rmse_PRATE.update_state(autoencoder.rmse_PRATE)
            train_logpz.update_state(autoencoder.logpz)
            train_logqz_x.update_state(autoencoder.logqz_x)
            train_loss.update_state(autoencoder.loss)

        # Same, but for the test data
        test_rmse_PRMSL.reset_states()
        test_rmse_SST.reset_states()
        test_rmse_T2M.reset_states()
        test_rmse_PRATE.reset_states()
        test_logpz.reset_states()
        test_logqz_x.reset_states()
        for batch in testData:
            per_replica_op = strategy.run(autoencoder.compute_loss, args=(batch,))
            test_rmse_PRMSL.update_state(autoencoder.rmse_PRMSL)
            test_rmse_SST.update_state(autoencoder.rmse_SST)
            test_rmse_T2M.update_state(autoencoder.rmse_T2M)
            test_rmse_PRATE.update_state(autoencoder.rmse_PRATE)
            test_logpz.update_state(autoencoder.logpz)
            test_logqz_x.update_state(autoencoder.logqz_x)
            test_loss.update_state(autoencoder.loss)

        # Save model state and current metrics
        save_dir = ("%s/models/Epoch_%04d") % (LSCRATCH, epoch,)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        autoencoder.save_weights("%s/ckpt" % save_dir)
        with logfile_writer.as_default():
            tf.summary.scalar("Train_PRMSL", train_rmse_PRMSL.result(), step=epoch)
            tf.summary.scalar("Train_SST", train_rmse_SST.result(), step=epoch)
            tf.summary.scalar("Train_T2M", train_rmse_T2M.result(), step=epoch)
            tf.summary.scalar("Train_PRATE", train_rmse_PRATE.result(), step=epoch)
            tf.summary.scalar("Train_logpz", train_logpz.result(), step=epoch)
            tf.summary.scalar("Train_logqz_x", train_logqz_x.result(), step=epoch)
            tf.summary.scalar("Train_loss", train_loss.result(), step=epoch)
            tf.summary.scalar("Test_PRMSL", test_rmse_PRMSL.result(), step=epoch)
            tf.summary.scalar("Test_SST", test_rmse_SST.result(), step=epoch)
            tf.summary.scalar("Test_T2M", test_rmse_T2M.result(), step=epoch)
            tf.summary.scalar("Test_PRATE", test_rmse_PRATE.result(), step=epoch)
            tf.summary.scalar("Test_logpz", test_logpz.result(), step=epoch)
            tf.summary.scalar("Test_logqz_x", test_logqz_x.result(), step=epoch)
            tf.summary.scalar("Test_loss", test_loss.result(), step=epoch)

        end_monitoring_time = time.time()

        # Report progress
        print("Epoch: {}".format(epoch))
        print(
            "PRMSL  : {:>7.1f}, {:>7.1f}".format(
                train_rmse_PRMSL.result(), test_rmse_PRMSL.result()
            )
        )
        print(
            "SST    : {:>7.1f}, {:>7.1f}".format(
                train_rmse_SST.result(), test_rmse_SST.result()
            )
        )
        print(
            "T2m    : {:>7.1f}, {:>7.1f}".format(
                train_rmse_T2M.result(), test_rmse_T2M.result()
            )
        )
        print(
            "PRATE  : {:>7.1f}, {:>7.1f}".format(
                train_rmse_PRATE.result(), test_rmse_PRATE.result()
            )
        )
        print(
            "logpz  : {:>7.1f}, {:>7.1f}".format(
                train_logpz.result(), test_logpz.result()
            )
        )
        print(
            "logqz_x: {:>7.1f}, {:>7.1f}".format(
                train_logqz_x.result(), test_logqz_x.result()
            )
        )
        print(
            "loss   : {:>7.1f}, {:>7.1f}".format(
                train_loss.result(), test_loss.result()
            )
        )
        print(
            "time: {} (+{})".format(
                int(end_training_time - start_time),
                int(end_monitoring_time - end_training_time),
            )
        )
