# Utility functions for creating and manipulating tensors

import os
import sys
import iris
import iris.cube
import iris.util
import iris.coords
import iris.coord_systems
import tensorflow as tf
import numpy as np

import warnings

warnings.filterwarnings("ignore", message=".*datum.*")
warnings.filterwarnings("ignore", message=".*frac.*")

# Define a standard-cube to work with
# Identical to that used in HadUK-Grid, except that the grid is trimmed to 896x1440
#  to be multiply divisible by 2 for easy use in a hierarchical CNN.
coord_s = iris.coord_systems.TransverseMercator(
    latitude_of_projection_origin=49.0,
    longitude_of_central_meridian=-2.0,
    false_easting=400000.0,
    false_northing=-100000.0,
    scale_factor_at_central_meridian=0.9996012717,
    ellipsoid=iris.coord_systems.GeogCS(
        semi_major_axis=6377563.396, semi_minor_axis=6356256.909
    ),
)
y_values = np.arange(-189500, 1250500, 1000)
y_coord = iris.coords.DimCoord(
    y_values,
    standard_name="projection_y_coordinate",
    units="metres",
    coord_system=coord_s,
)
y_coord.guess_bounds()
x_values = np.arange(-195500, 700500, 1000)
x_coord = iris.coords.DimCoord(
    x_values,
    standard_name="projection_x_coordinate",
    units="metres",
    coord_system=coord_s,
)
x_coord.guess_bounds()
dummy_data = np.zeros((len(y_values), len(x_values)))
sCube = iris.cube.Cube(dummy_data, dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])

# Need data from HadUK-Grid and from 20CR
sys.path.append("%s/../../.." % os.path.dirname(__file__))
from get_data.HadUKGrid import HUKG_monthly_load
from get_data.TWCR import TWCR_monthly_load

# Need the three masks (HadUK-Grid data, 20CR SST, and land-to-plot) on the standard cube
dm_HUKG = HUKG_monthly_load.dm_hukg.regrid(sCube, iris.analysis.Nearest())
lm_TWCR = TWCR_monthly_load.lm_TWCR.regrid(sCube, iris.analysis.Nearest())
lm_plot = iris.load_cube(
    "%s/fixed_fields/land_mask/opfc_global_2019.nc" % os.getenv("DATADIR")
)
lm_plot = lm_plot.regrid(sCube, iris.analysis.Linear())

# Load the data for 1 month, convert to 20CR equivalent units, and
#  regrid to the standard cube.
def load_cList(year, month, member=1, omit=[]):
    res = []
    # PRMSL
    try:
        prmsl = TWCR_monthly_load.load_monthly_member("PRMSL", year, month, member)
        clim = TWCR_monthly_load.load_climatology("PRMSL", month)
        prmsl.data -= clim.data
        prmsl = prmsl.regrid(sCube, iris.analysis.Linear())
    except Exception:
        if "PRMSL" in omit:
            prmsl = sCube.copy()
        else:
            raise
    res.append(prmsl)

    # SST
    try:
        sst = TWCR_monthly_load.load_monthly_member("SST", year, month, member)
        clim = TWCR_monthly_load.load_climatology("SST", month)
        sst.data -= clim.data
        sst = sst.regrid(sCube, iris.analysis.Linear())
    except Exception:
        if "SST" in omit:
            sst = sCube.copy()
        else:
            raise
    res.append(sst)

    # T2m
    try:
        t2m = HUKG_monthly_load.load_variable("monthly_meantemp", year, month)
        clim = HUKG_monthly_load.load_climatology("monthly_meantemp", month)
        t2m.data -= clim.data
        t2m = t2m.regrid(sCube, iris.analysis.Nearest())
    except IOError:
        if "monthly_meantemp" in omit:
            t2m = sCube.copy()
        else:
            raise
    res.append(t2m)

    # PRATE
    try:
        prate = HUKG_monthly_load.load_variable("monthly_rainfall", year, month)
        clim = HUKG_monthly_load.load_climatology(
            "monthly_rainfall",
            month,
        )
        prate.data -= clim.data
        prate = prate.regrid(sCube, iris.analysis.Nearest())
    except IOError:
        if "monthly_rainfall" in omit:
            prate = sCube.copy()
            prate.data[np.where(dm_HUKG.data.mask == False)] += 0.01
        else:
            raise
    res.append(prate)

    return res


# Need to normalise the data onto the range 0-1.

# Specify scale equivalents for 0 and 1 for each variable
nPar = {
    "PRMSL": (-1000, 1000),
    "monthly_rainfall": (-250, 250),
    "monthly_meantemp": (-5, 5),
    "SST": (-3, 3),
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


# Smooth the step change in missing data
def extrapolate_step(cb, cm, scale=1.0):
    ss = cb.data * 0
    sn = cb.data * 0
    count = cb.data * 0
    for ad in ([1, 0], [-1, 0], [0, 1], [0, -1]):
        sn = np.roll(cb.data, (ad[0], ad[1]), (0, 1))
        sn[0, :] = 0
        sn[-1, :] = 0
        sn[:, 0] = 0
        sn[:, -1] = 0
        ss[sn != 0] += sn[sn != 0]
        count[sn != 0] += 1
    ss[count != 0] /= count[count != 0]
    result = cb.copy()
    result.data[cm.data == 0] = ss[cm.data == 0]
    return result


def extrapolate_missing(cb, nsteps=10, scale=1.0):
    cr = cb.copy()
    for step in range(nsteps):
        cr = extrapolate_step(cr, cb, scale=scale)
    return cr


# Convert variable quad to normalised, (smoothed?) tensor
def cList_to_tensor(cL, extrapolate=False):
    d1 = normalise(cL[0], "PRMSL")
    d2 = normalise(cL[1], "SST")
    d2.data[np.where(lm_TWCR.data.mask == True)] = 0.5
    if extrapolate:
        d2 = extrapolate_missing(d2, nsteps=100, scale=1.0)
    d3 = normalise(cL[2], "monthly_meantemp")
    d3.data[np.where(dm_HUKG.data.mask == True)] = 0.5
    d3.data[d3.data > 5] = 0  # Kludge - mask varies
    if extrapolate:
        d3 = extrapolate_missing(d3, nsteps=100, scale=1.0)
    d4 = normalise(cL[3], "monthly_rainfall")
    d4.data[np.where(dm_HUKG.data.mask == True)] = 0.5
    if extrapolate:
        d4 = extrapolate_missing(d4, nsteps=100, scale=1.0)
    ic = np.stack((d1.data, d2.data, d3.data, d4.data), axis=2)
    ict = tf.convert_to_tensor(ic, np.float32)
    return ict


# Convert tensor to unnormalised, unsmoothed cubelist
def tensor_to_cList(
    tensor,
    plotCube,
):
    d1 = plotCube.copy()
    d1.data = np.squeeze(tensor[:, :, 0].numpy())
    d1 = unnormalise(d1, "PRMSL")
    d1.var_name = "PRMSL"
    d2 = plotCube.copy()
    d2.data = np.squeeze(tensor[:, :, 1].numpy())
    d2 = unnormalise(d2, "SST")
    d2.data = np.ma.masked_where(lm_TWCR.data.mask, d2.data, copy=False)
    d2.var_name = "SST"
    d3 = plotCube.copy()
    d3.data = np.squeeze(tensor[:, :, 2].numpy())
    d3 = unnormalise(d3, "monthly_meantemp")
    d3.data = np.ma.masked_where(dm_HUKG.data.mask, d3.data, copy=False)
    d3.var_name = "monthly_meantemp"
    d4 = plotCube.copy()
    d4.data = np.squeeze(tensor[:, :, 3].numpy())
    d4 = unnormalise(d4, "monthly_rainfall")
    d4.data = np.ma.masked_where(dm_HUKG.data.mask, d4.data, copy=False)
    d4.var_name = "monthly_rainfall"
    return [d1, d2, d3, d4]
