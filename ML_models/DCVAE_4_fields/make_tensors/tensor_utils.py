# Utility functions for creating and manipulating tensors

import os
import iris
import iris.util
import iris.coord_systems
import tensorflow as tf
import numpy as np


def quad_to_tensor(quad, plotCube, lm=None):
    d1 = normalise(quad[0].regrid(plotCube, iris.analysis.Linear()), "PRMSL")
    d2 = normalise(quad[1].regrid(plotCube, iris.analysis.Linear()), "TMPS")
    if lm is None:
        lm = iris.load_cube("%s/20CR/version_3/fixed/land.nc" % os.getenv("SCRATCH"))
        lm = iris.util.squeeze(lm)
        coord_s = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
        lm.coord("latitude").coord_system = coord_s
        lm.coord("longitude").coord_system = coord_s
    lm = lm.regrid(plotCube, iris.analysis.Linear())
    d2.data[lm.data > 0.5] = 0.0
    d3 = normalise(quad[2].regrid(plotCube, iris.analysis.Linear()), "TMP2m")
    d4 = normalise(quad[3].regrid(plotCube, iris.analysis.Linear()), "PRATE")
    ic = np.stack((d1.data, d2.data, d3.data, d4.data), axis=2)
    ict = tf.convert_to_tensor(ic.data, np.float32)
    return ict


def tensor_to_quad(tensor, plotCube):
    d1 = plotCube.copy()
    d1.data = np.squeeze(tensor[:, :, 0].numpy())
    d1 = unnormalise(d1, "PRMSL")
    d1.var_name = "PRMSL"
    d2 = plotCube.copy()
    d2.data = np.squeeze(tensor[:, :, 1].numpy())
    d2 = unnormalise(d2, "TMPS")
    sst_mask = tensor.numpy()[:, :, 1] == 0.0
    d2.data = np.ma.masked_where(sst_mask, d2.data, copy=True)
    d2.var_name = "SST"
    d3 = plotCube.copy()
    d3.data = np.squeeze(tensor[:, :, 2].numpy())
    d3 = unnormalise(d3, "TMP2m")
    d3.var_name = "TMP2m"
    d4 = plotCube.copy()
    d4.data = np.squeeze(tensor[:, :, 3].numpy())
    d4 = unnormalise(d4, "PRATE")
    d4.var_name = "PRATE"
    return [d1, d2, d3, d4]


nPar = {
    "PRMSL": (97700, 103300),
    "PRATE": (0, 0.00019),
    "TMP2m": (255, 303),
    "TMPS": (253, 305),
}


def normalise(cube, variable):
    cb = cube.copy()
    if not variable in nPar:
        raise Exception("Unsupported variable " + variable)
    cb.data -= nPar[variable][0]
    cb.data /= nPar[variable][1] - nPar[variable][0]
    return cb


def unnormalise(cube, variable):
    cb = cube.copy()
    if not variable in nPar:
        raise Exception("Unsupported variable " + variable)
    cb.data *= nPar[variable][1] - nPar[variable][0]
    cb.data += nPar[variable][0]
    return cb
