# Make tf.data.Datasets from HadUK-Grid monthly UK fields

import os
import sys
import random
import tensorflow as tf
import numpy as np

sys.path.append("%s/." % os.path.dirname(__file__))
from localise import TSOURCE

# Load a pre-standardised 4-variable tensor from a file
def load_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, np.float32)
    imt = tf.reshape(imt, [1440, 896, 4])
    return imt


# Get a list of filenames containing tensors
def getFileNames(purpose, nImages=None, startyear=None, endyear=None):
    inFiles = sorted(os.listdir("%s/datasets/%s" % (TSOURCE, purpose)))
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

    # Create TensorFlow Dataset object from the file namelist
    tn_data = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles))

    # Convert from list of file names to Dataset of file contents
    inFiles = ["%s/datasets/%s/%s" % (TSOURCE, purpose, x) for x in inFiles]
    tr_data = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles))
    tr_data = tr_data.map(load_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Zip the data together with the filenames (so we can find the date and source of each
    #   data tensor if we need it).
    tz_data = tf.data.Dataset.zip((tr_data, tn_data))

    # Optimisation
    if cache:
        tz_data = tz_data.cache()  # Great, iff you have enough RAM for it
    tz_data = tz_data.prefetch(tf.data.experimental.AUTOTUNE)

    return tz_data
