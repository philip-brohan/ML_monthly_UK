# Make tf.data.Datasets from HadUK-Grid monthly latents and Thames flow data

import os
import sys
import random
import tensorflow as tf
import numpy as np

sys.path.append("%s/../.." % os.path.dirname(__file__))
from localise import TSOURCE

# Load the flow data
fdata = {}
with open(
    "%s/../../../../get_data/NRFA/normalised.txt" % os.path.dirname(__file__)
) as f:
    for line in f.readlines():
        if len(line) < 10:
            continue
        year = line[:4]
        month = line[5:7]
        flow = float(line[8:])
        if year not in fdata:
            fdata[year] = {}
        fdata[year][month] = flow


# Load a latent-space tensor from a file
def load_latent(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, np.float32)
    imt = tf.reshape(imt, [100])
    return imt


# Get a flow anomaly for a month given a latent-space tensor file name
def load_flow(file_name):
    fname = os.path.basename(file_name)
    year = fname[:4]
    month = fname[5:7]
    # return fdata[year][month]
    c2 = int(fdata[year][month] * 17) + 1
    c2c = np.repeat(np.float32(0), 20)
    c2c[c2] = 1.0
    return c2c


# Get a list of filenames containing tensors
def getFileNames(purpose, nImages=None, startyear=1885, endyear=2014):
    inFiles = sorted(os.listdir("%s/latents/%s" % (TSOURCE, purpose)))
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
    purpose, nImages=None, startyear=1885, endyear=2014, shuffle=True, cache=False
):

    # Get a list of filenames containing LS tensors
    inFiles = getFileNames(
        purpose, nImages=nImages, startyear=startyear, endyear=endyear
    )
    if shuffle:
        random.shuffle(inFiles)

    # Create TensorFlow Dataset object from the latent-space file names
    tn_data = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles))

    # Convert from list of file names to Dataset of source file contents
    fnFiles = ["%s/latents/%s/%s" % (TSOURCE, purpose, x) for x in inFiles]
    tl_data = tf.data.Dataset.from_tensor_slices(tf.constant(fnFiles))
    tl_data = tl_data.map(load_latent, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Create a matching dataset of flows
    tf_data = tf.data.Dataset.from_tensor_slices([load_flow(x) for x in inFiles])

    # Zip the data together with the filenames (so we can find the date and source of each
    #   data tensor if we need it).
    tz_data = tf.data.Dataset.zip((tl_data, tf_data, tn_data))

    # Optimisation
    if cache:
        tz_data = tz_data.cache()  # Great, iff you have enough RAM for it
    tz_data = tz_data.prefetch(tf.data.experimental.AUTOTUNE)

    return tz_data
