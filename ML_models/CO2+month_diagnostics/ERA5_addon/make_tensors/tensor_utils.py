# Utility functions for creating and manipulating tensors
# This version does the ERA5 addon-model tensors

import os
import sys
import iris
import iris.util
import iris.coord_systems
import tensorflow as tf
import numpy as np

# Want a smooth transition at the edge of the mask, so
#  set masked points near data points to close to the data value
def extrapolate_step(cb, scale=0.95):
    st = cb.data * 0
    st[:, 1:] = cb.data[:, :-1] * scale
    sb = cb.data * 0
    sb[:, :-1] = cb.data[:, 1:] * scale
    sa = np.maximum(st, sb)
    sl = cb.data * 0
    sl[1:, :] = cb.data[:-1, :] * scale
    sa = np.maximum(sa, sl)
    sr = cb.data * 0
    sr[:-1, :] = cb.data[1:, :] * scale
    sa = np.maximum(sa, sr)
    sp = np.where(cb.data == 0)
    result = cb.copy()
    result.data[sp] = sa[sp]
    return result


def extrapolate_missing(cb, nsteps=10, scale=0.95):
    cr = cb.copy()
    for step in range(nsteps):
        cr = extrapolate_step(cr, scale=scale)
    return cr


def cList_to_tensor(cL, sst_mask):
    d1 = normalise(cL[0], "PRMSL")
    d2 = normalise(cL[1], "TMPS")
    d2.data = d2.data.data
    d2.data[np.where(sst_mask == True)] = 0
    d2 = extrapolate_missing(d2, nsteps=100, scale=0.95)
    d3 = normalise(cL[2], "TMP2m")
    d4 = normalise(cL[3], "PRATE")
    ic = np.stack((d1.data, d2.data, d3.data, d4.data), axis=2)
    ict = tf.convert_to_tensor(ic.data, np.float32)
    return ict


def tensor_to_cList(tensor, plotCube, sst_mask):
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
