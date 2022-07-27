# Make tf.data.Datasets from HadUK-Grid monthly UK fields

import os
import sys
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


# Get a dataset
def getDataset(purpose, nImages=None):

    # Get a list of filenames containing tensors
    inFiles = os.listdir("%s/datasets/%s" % (TSOURCE, purpose))

    if nImages is not None:
        if len(inFiles) >= nImages:
            inFiles = inFiles[0:nImages]
        else:
            raise ValueError(
                "Only %d images available, can't provide %d" % (len(inFiles), nImages)
            )

    # Create TensorFlow Dataset object from the file namelist
    inFiles = ["%s/datasets/%s/%s" % (TSOURCE, purpose, x) for x in inFiles]
    tr_data = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles))

    # Convert the Dataset from file names to file contents
    tr_data = tr_data.map(load_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Optimisation
    tr_data = tr_data.cache()
    tr_data = tr_data.prefetch(tf.data.experimental.AUTOTUNE)

    return tr_data
