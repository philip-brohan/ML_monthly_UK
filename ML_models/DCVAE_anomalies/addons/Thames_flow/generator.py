#!/usr/bin/env python

# Generator for Thames flow from DCVAE latent space

import os
import sys
import time
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epoch", help="Restart from epoch", type=int, required=False, default=1
)
args = parser.parse_args()

# Distribute across all GPUs
strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.experimental.CentralStorageStrategy()
# strategy = tf.distribute.get_strategy()

# Load the data path, data source, and model specification
sys.path.append("%s/." % os.path.dirname(__file__))
from makeDataset import getDataset
from generatorModel import GeneratorM

sys.path.append("%s/../.." % os.path.dirname(__file__))
from localise import LSCRATCH

# Can use less than all the data (for testing)
nTrainingImages = None
nTestImages = None

# How many epochs to train for
nEpochs = 250
# Length of an epoch - if None, same as nTrainingImages
nImagesInEpoch = None
nRepeatsPerEpoch = 2  # Show each month this many times

if nImagesInEpoch is None:
    nImagesInEpoch = nTrainingImages

# Dataset parameters
bufferSize = 1000  # Already shuffled data, so not so important
batchSize = 32  # Arbitrary

# Instantiate and run the model under the control of the distribution strategy
with strategy.scope():

    # Set up the training data
    trainingData = getDataset(
        purpose="training", nImages=nTrainingImages, shuffle=True, cache=False
    ).repeat(nRepeatsPerEpoch)
    trainingData = trainingData.shuffle(bufferSize).batch(batchSize)
    trainingData = strategy.experimental_distribute_dataset(trainingData)

    # Subset of the training data for metrics
    validationData = getDataset(
        purpose="training", nImages=nTestImages, shuffle=False, cache=False
    )
    validationData = validationData.batch(batchSize)
    validationData = strategy.experimental_distribute_dataset(validationData)

    # Set up the test data
    testData = getDataset(
        purpose="test", nImages=nTestImages, shuffle=False, cache=False
    )
    testData = testData.batch(batchSize)
    testData = strategy.experimental_distribute_dataset(testData)

    # Instantiate the model
    autoencoder = GeneratorM()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    # If we are doing a restart, load the weights
    if args.epoch > 1:
        weights_dir = ("%s/flow_models/Epoch_%04d") % (
            LSCRATCH,
            args.epoch,
        )
        load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
        load_status.assert_existing_objects_matched()

    # Metrics for training and test loss
    train_loss = tf.Variable(0.0, trainable=False)
    test_loss = tf.Variable(0.0, trainable=False)

    # logfile to output the metrics
    log_FN = ("%s/flow_models/Training_log") % LSCRATCH
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
        train_loss.assign(0.0)
        validation_batch_count = 0
        for batch in validationData:
            per_replica_losses = strategy.run(
                autoencoder.compute_loss, args=(batch, False)
            )
            batch_losses = strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None
            )
            train_loss.assign_add(batch_losses)
            validation_batch_count += 1

        # Same, but for the test data
        test_loss.assign(0.0)
        test_batch_count = 0
        for batch in testData:
            per_replica_losses = strategy.run(
                autoencoder.compute_loss, args=(batch, False)
            )
            batch_losses = strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None
            )
            test_loss.assign_add(batch_losses)
            test_batch_count += 1

        # Save model state and current metrics
        save_dir = ("%s/flow_models/Epoch_%04d") % (
            LSCRATCH,
            epoch,
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        autoencoder.save_weights("%s/ckpt" % save_dir)
        with logfile_writer.as_default():
            tf.summary.scalar(
                "Train_loss", train_loss / validation_batch_count, step=epoch
            )
            tf.summary.scalar("Test_loss", test_loss / test_batch_count, step=epoch)

        end_monitoring_time = time.time()

        # Report progress
        print("Epoch: {}".format(epoch))
        print(
            "loss   : {:>9.3f}, {:>9.3f}".format(
                train_loss.numpy() / validation_batch_count,
                test_loss.numpy() / test_batch_count,
            )
        )
        print(
            "time: {} (+{})".format(
                int(end_training_time - start_time),
                int(end_monitoring_time - end_training_time),
            )
        )
