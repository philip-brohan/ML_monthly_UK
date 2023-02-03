# Functions to load ERA5 monthly data

import os
import iris
import iris.util
import iris.cube
import iris.analysis
import iris.coord_systems
import iris.fileformats
import numpy as np
from calendar import monthrange


# ERA5 data don't contain a coordinate system - need one to add
cs_ERA5 = iris.coord_systems.RotatedGeogCS(90, 180, 0)

# Land-mask for ERA5 SST grid
fname = "%s/ERA5/monthly/reanalysis/%04d/%s.nc" % (
    os.getenv("SCRATCH"),
    1959,
    "sea_surface_temperature",
)
if not os.path.isfile(fname):
    raise Exception("No data file %s" % fname)
ftt = iris.Constraint(time=lambda cell: cell.point.month == 1)
lm_ERA5 = iris.load_cube(fname, ftt)
lm_ERA5.coord("latitude").coord_system = cs_ERA5
lm_ERA5.coord("longitude").coord_system = cs_ERA5
lm_ERA5.data.data[np.where(lm_ERA5.data.mask == True)] = 0
lm_ERA5.data.data[np.where(lm_ERA5.data.mask == False)] = 1


def load_variable(variable, year, month):
    if variable == "cbrt_precipitation":
        varC = load_variable("total_precipitation", year, month)
        varC.data = np.cbrt(varC.data)
        return varC
    fname = "%s/ERA5/monthly/reanalysis/%04d/%s.nc" % (
        os.getenv("SCRATCH"),
        year,
        variable,
    )
    if not os.path.isfile(fname):
        raise Exception("No data file %s" % fname)
    ftt = iris.Constraint(time=lambda cell: cell.point.month == month)
    varC = iris.load_cube(fname, ftt)
    if len(varC.data.shape) == 3:
        varC = varC.extract(iris.Constraint(expver=1))
    varC.coord("latitude").coord_system = cs_ERA5
    varC.coord("longitude").coord_system = cs_ERA5
    return varC


def load_climatology(variable, month):
    fname = "%s/ERA5/monthly/climatology/%s_%02d.nc" % (
        os.getenv("SCRATCH"),
        variable,
        month,
    )
    if not os.path.isfile(fname):
        raise Exception("No climatology file %s" % fname)
    c = iris.load_cube(fname)
    c.long_name = variable
    c.coord("latitude").coord_system = cs_ERA5
    c.coord("longitude").coord_system = cs_ERA5
    return c


def load_sd_climatology(variable, month):
    fname = "%s/ERA5/monthly/sd_climatology/%s_%02d.nc" % (
        os.getenv("SCRATCH"),
        variable,
        month,
    )
    if not os.path.isfile(fname):
        raise Exception("No sd climatology file %s" % fname)
    c = iris.load_cube(fname)
    c.long_name = variable
    c.coord("latitude").coord_system = cs_ERA5
    c.coord("longitude").coord_system = cs_ERA5
    return c


def load_cList(year, month):
    res = []
    # PRMSL
    fname = "%s/ERA5/monthly/reanalysis/%04d/%s.nc" % (
        os.getenv("SCRATCH"),
        year,
        "mean_sea_level_pressure",
    )
    if not os.path.isfile(fname):
        raise Exception("No data file %s" % fname)
    ftt = iris.Constraint(time=lambda cell: cell.point.month == month)
    prmsl = iris.load_cube(fname, ftt)
    if len(prmsl.data.shape) == 3:
        prmsl = prmsl.extract(iris.Constraint(expver=1))
    prmsl.coord("latitude").coord_system = cs_ERA5
    prmsl.coord("longitude").coord_system = cs_ERA5
    prmsl = prmsl.regrid(sCube, iris.analysis.Nearest())
    res.append(prmsl)

    # SST
    fname = "%s/ERA5/monthly/reanalysis/%04d/%s.nc" % (
        os.getenv("SCRATCH"),
        year,
        "sea_surface_temperature",
    )
    if not os.path.isfile(fname):
        raise Exception("No data file %s" % fname)
    ftt = iris.Constraint(time=lambda cell: cell.point.month == month)
    sst = iris.load_cube(fname, ftt)
    if len(sst.data.shape) == 3:
        sst = sst.extract(iris.Constraint(expver=1))
    sst.coord("latitude").coord_system = cs_ERA5
    sst.coord("longitude").coord_system = cs_ERA5
    sst = sst.regrid(sCube, iris.analysis.Nearest())
    res.append(sst)
    # T2m
    fname = "%s/ERA5/monthly/reanalysis/%04d/%s.nc" % (
        os.getenv("SCRATCH"),
        year,
        "2m_temperature",
    )
    if not os.path.isfile(fname):
        raise Exception("No data file %s" % fname)
    ftt = iris.Constraint(time=lambda cell: cell.point.month == month)
    t2m = iris.load_cube(fname, ftt)
    if len(t2m.data.shape) == 3:
        t2m = t2m.extract(iris.Constraint(expver=1))
    t2m.coord("latitude").coord_system = cs_ERA5
    t2m.coord("longitude").coord_system = cs_ERA5
    t2m = t2m.regrid(sCube, iris.analysis.Nearest())
    res.append(t2m)
    # PRATE
    fname = "%s/ERA5/monthly/reanalysis/%04d/%s.nc" % (
        os.getenv("SCRATCH"),
        year,
        "total_precipitation",
    )
    if not os.path.isfile(fname):
        raise Exception("No data file %s" % fname)
    ftt = iris.Constraint(time=lambda cell: cell.point.month == month)
    prate = iris.load_cube(fname, ftt)
    if len(prate.data.shape) == 3:
        prate = prate.extract(iris.Constraint(expver=1))
    prate.coord("latitude").coord_system = cs_ERA5
    prate.coord("longitude").coord_system = cs_ERA5

    res.append(prate)
    return res
