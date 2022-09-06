# Make tf.data.Datasets using encoded versions of the HadUK-Grid quad as input.
# This one makes calendar month as the output.

import os
import sys
import tensorflow as tf
import numpy as np
import random

sys.path.append("%s/../.." % os.path.dirname(__file__))
from localise import TSOURCE


# Load a pre-calculated mean or logvar tensor from a file
def load_latent(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, np.float32)
    imt = tf.squeeze(imt)
    return imt


def normalise_month(file_name):
    month = int(file_name[5:7])
    mnth = np.repeat(np.float32(0), 12)
    mnth[month - 1] = 1
    return mnth


def unnormalise_month(mnth):
    return np.argmax(mnth) + 1


# Get a list of filenames containing latent-space mean tensors
def getFileNames(purpose, nImages=None, startyear=None, endyear=None):
    inFiles = sorted(os.listdir("%s/latents/%s/mean" % (TSOURCE, purpose)))
    if startyear is not None:
        inFiles = [fn for fn in inFiles if int(fn[:4]) >= startyear]
    if endyear is not None:
        inFiles = [fn for fn in inFiles if int(fn[:4]) <= endyear]
    if nImages is not None:
        if len(inFiles) >= nImages:
            inFiles = inFiles[0:nImages]
        else:
            raise ValueError(
                "Only %d images available, can't provide %d" % (len(inFiles), nImages)
            )
    return inFiles


# Get a dataset
def getDataset(
    purpose, nImages=None, startyear=None, endyear=None, shuffle=True, cache=False
):

    # Get a list of filenames containing tensors
    inFiles = getFileNames(
        purpose, nImages=nImages, startyear=startyear, endyear=endyear
    )
    if shuffle:
        random.shuffle(inFiles)

    # Turn the names into month targets
    tm_data = tf.data.Dataset.from_tensor_slices([normalise_month(x) for x in inFiles])

    # Create TensorFlow Dataset object from the mean file names
    inFiles = ["%s/latents/%s/mean/%s" % (TSOURCE, purpose, x) for x in inFiles]
    tmn_data = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles))

    # Turn the names into mean tensors
    tlm_data = tmn_data.map(
        load_latent, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # Get the logvar tensors similarly
    inFiles = [x.replace("mean", "logvar") for x in inFiles]
    tln_data = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles))
    tll_data = tln_data.map(
        load_latent, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Zip the latent-space together with the filenames
    # (so we can find the date and source of each data tensor if we need it).
    tz_data = tf.data.Dataset.zip((tlm_data, tll_data, tm_data, tmn_data))

    # Optimisation
    if cache:
        tz_data = tz_data.cache()  # Great, iff you have enough RAM for it
    tz_data = tz_data.prefetch(tf.data.experimental.AUTOTUNE)

    return tz_data
