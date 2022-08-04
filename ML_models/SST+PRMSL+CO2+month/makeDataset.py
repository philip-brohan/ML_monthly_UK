# Make tf.data.Datasets from HadUK-Grid monthly UK fields
# The training dataset contains three components:
#  1) Weather fields (4 variables)
#  2) CO2 level
#  3) calendar month

import os
import sys
import tensorflow as tf
import numpy as np
import math

sys.path.append("%s/." % os.path.dirname(__file__))
from localise import TSOURCE

# Load a pre-standardised 4-variable tensor from a file
def load_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, np.float32)
    imt = tf.reshape(imt, [1440, 896, 4])
    return imt


# Make a scalar tensor containing CO2 level from the filename
CO2_since_1850 = (
    285.2,
    285.4,
    285.5,
    285.6,
    285.8,
    285.9,
    286.1,
    286.2,
    286.4,
    286.6,
    286.7,
    286.9,
    287.0,
    287.1,
    287.3,
    287.4,
    287.5,
    287.6,
    287.7,
    287.9,
    288.0,
    288.2,
    288.4,
    288.6,
    288.9,
    289.2,
    289.5,
    289.9,
    290.3,
    290.7,
    291.2,
    291.7,
    292.2,
    292.6,
    293.1,
    293.5,
    293.8,
    294.1,
    294.3,
    294.5,
    294.7,
    294.8,
    295.0,
    295.1,
    295.2,
    295.3,
    295.4,
    295.5,
    295.7,
    296.0,
    296.3,
    296.6,
    297.0,
    297.3,
    297.7,
    298.1,
    298.6,
    299.0,
    299.4,
    299.8,
    300.2,
    300.6,
    300.9,
    301.3,
    301.6,
    301.9,
    302.2,
    302.6,
    302.9,
    303.2,
    303.5,
    303.9,
    304.3,
    304.6,
    305.0,
    305.5,
    305.9,
    306.3,
    306.8,
    307.3,
    307.7,
    308.2,
    308.7,
    309.1,
    309.5,
    309.9,
    310.3,
    310.5,
    310.7,
    310.8,
    310.9,
    310.9,
    310.8,
    310.7,
    310.6,
    310.6,
    310.6,
    310.7,
    310.8,
    311.0,
    311.3,
    311.6,
    312.0,
    312.4,
    312.9,
    313.5,
    314.1,
    314.7,
    315.3,
    316.0,
    316.8,
    317.6,
    318.3,
    318.9,
    319.4,
    320.1,
    321.1,
    322.1,
    323.1,
    324.4,
    325.5,
    326.4,
    327.6,
    329.2,
    330.2,
    331.1,
    332.2,
    333.8,
    335.3,
    337.0,
    338.9,
    340.2,
    341.3,
    342.7,
    344.3,
    345.8,
    347.3,
    349.1,
    351.2,
    353.0,
    354.4,
    355.5,
    356.4,
    357.3,
    358.6,
    360.3,
    362.0,
    363.7,
    365.8,
    367.8,
    369.4,
    371.0,
    373.0,
    375.3,
    377.3,
    379.3,
    381.3,
    383.3,
    385.3,
    387.5,
    389.6,
    391.8,
    393.9,
    396.1,
    398.3,
    400.5,
    402.7,
    404.9,
    407.1,
    409.4,
    411.6,
    413.9,
    416.1,
    418.4,
    420.8,
    423.2,
)


def normalise_co2(file_name):
    year = int(file_name[:4])
    c2 = CO2_since_1850[year - 1850]
    c2 = (c2 - 250) / 150  # Normalise to ~0-1
    return c2


def normalise_month(file_name):
    month = int(file_name[5:7])
    month=math.sin(2*math.pi*(month)/12) # Normalise to 0-1 (periodic)
    return month


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

    tc_data = tf.data.Dataset.from_tensor_slices([normalise_co2(x) for x in inFiles])
    tm_data = tf.data.Dataset.from_tensor_slices([normalise_month(x) for x in inFiles])

    # Create TensorFlow Dataset object from the file namelist
    inFiles = ["%s/datasets/%s/%s" % (TSOURCE, purpose, x) for x in inFiles]
    tr_data = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles))

    # Convert the Dataset from file names to file contents
    tr_data = tr_data.map(load_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Zip the CO2 and fields together
    tz_data = tf.data.Dataset.zip((tr_data, tc_data, tm_data))

    # Optimisation
    tz_data = tz_data.cache()
    tz_data = tz_data.prefetch(tf.data.experimental.AUTOTUNE)

    return tz_data
