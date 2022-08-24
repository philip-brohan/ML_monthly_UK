# Make tf.data.Datasets from RA5 fields and encoded versions of the HadUK-Grid quad.

import os
import sys
import tensorflow as tf
import numpy as np
import math
import glob

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import TSOURCE
from localise import LSCRATCH

# Load a pre-standardised 4-variable tensor from a file
def load_target(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, np.float32)
    imt = tf.reshape(imt, [1440, 896, 4])
    return imt


# Load a pre-calculated mean or logvar tensor from a file
def load_latent(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, np.float32)
    imt = tf.squeeze(imt)
    return imt


# Get a list of available target files
def getPairedFileNames(purpose, nImages=None, startyear=None, endyear=None):
    targetFiles = sorted(os.listdir("%s/datasets_ERA5/%s" % (TSOURCE, purpose)))
    if startyear is not None:
        targetFiles = [fn for fn in inFiles if int(fn[:4]) >= startyear]
    if endyear is not None:
        targetFiles = [fn for fn in inFiles if int(fn[:4]) <= endyear]
    targetList = []
    meanList = []
    for tf in targetFiles:
        mf = sorted(
            glob.glob("%s/latents/%s/mean/%s*" % (LSCRATCH, "training", tf[-14:-4]))
        )
        mf.extend(
            sorted(glob.glob("%s/latents/%s/mean/%s*" % (LSCRATCH, "test", tf[-14:-4])))
        )
        for fl in mf:
            if len(fl) != 0:  # Throw out cases where no match in glob
                meanList.append(fl)
                targetList.append("%s/datasets_ERA5/%s/%s" % (TSOURCE, purpose, tf))
    logvarList = [x.replace("mean", "logvar") for x in meanList]

    if nImages is not None:
        if len(meanList) >= nImages:
            meanList = meanList[0:nImages]
            logvarList = logvarList[0:nImages]
            targetList = targetList[0:nImages]
        else:
            raise ValueError(
                "Only %d images available, can't provide %d"
                % (len(sourceList), nImages)
            )
    return (meanList, logvarList, targetList)


# Get a dataset
def getDataset(purpose, nImages=None, startyear=None, endyear=None):

    # Get a list of filenames containing tensors
    inFiles = getPairedFileNames(
        purpose, nImages=nImages, startyear=startyear, endyear=endyear
    )

    tm_data = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles[0]))
    tm_data = tm_data.map(load_latent, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tl_data = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles[1]))
    tl_data = tl_data.map(load_latent, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tt_data = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles[2]))
    tt_data = tt_data.map(load_target, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Zip the source and target together
    tz_data = tf.data.Dataset.zip((tm_data, tl_data, tt_data))

    # Optimisation
    tz_data = tz_data.cache()
    tz_data = tz_data.prefetch(tf.data.experimental.AUTOTUNE)

    return tz_data
