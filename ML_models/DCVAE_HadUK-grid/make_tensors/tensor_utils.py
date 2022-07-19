# Utility functions for creating and manipulating tensors

import os
import iris
import iris.util
import iris.coord_systems
import tensorflow as tf
import numpy as np


def cList_to_tensor(cL, sst_mask, hukg_mask):
    d1 = normalise(cL[0], "PRMSL")
    d2 = normalise(cL[1], "TMPS")
    d2.data[np.where(sst_mask == True)] = 0
    d3 = normalise(cL[2], "TMP2m")
    d3.data[np.where(hukg_mask == True)] = 0
    d3.data[d3.data>5] =0  # Kludge - mask varies
    d4 = normalise(cL[3], "PRATE")
    d4.data[np.where(hukg_mask == True)] = 0
    ic = np.stack((d1.data, d2.data, d3.data, d4.data), axis=2)
    ict = tf.convert_to_tensor(ic.data, np.float32)
    return ict


def tensor_to_cList(tensor, plotCube, sst_mask, hukg_mask):
    d1 = plotCube.copy()
    d1.data = np.squeeze(tensor[:, :, 0].numpy())
    d1 = unnormalise(d1, "PRMSL")
    d1.var_name = "PRMSL"
    d2 = plotCube.copy()
    d2.data = np.squeeze(tensor[:, :, 1].numpy())
    d2 = unnormalise(d2, "TMPS")
    d2.data = np.ma.masked_where(sst_mask, d2.data, copy=False)
    d2.var_name = "SST"
    d3 = plotCube.copy()
    d3.data = np.squeeze(tensor[:, :, 2].numpy())
    d3 = unnormalise(d3, "TMP2m")
    d3.data = np.ma.masked_where(hukg_mask, d3.data, copy=False)
    d3.var_name = "TMP2m"
    d4 = plotCube.copy()
    d4.data = np.squeeze(tensor[:, :, 3].numpy())
    d4 = unnormalise(d4, "PRATE")
    d4.data = np.ma.masked_where(hukg_mask, d4.data, copy=False)
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
