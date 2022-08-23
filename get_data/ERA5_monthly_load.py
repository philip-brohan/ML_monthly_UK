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

# ERA5 data don't contain a coordinate system - need one to add
cs_ERA5 = iris.coord_systems.RotatedGeogCS(90, 180, 0)

# Also want a land mask:
lm_plot = iris.load_cube(
    "%s/fixed_fields/land_mask/opfc_global_2019.nc" % os.getenv("DATADIR")
)
lm_plot = lm_plot.regrid(sCube, iris.analysis.Linear())

# And a land-mask on the ERA5 SST grid
fname = "%s/ERA5/monthly/reanalysis/%04d/%s.nc" % (
    os.getenv("SCRATCH"),
    2014,
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
    prmsl.coord("latitude").coord_system = cs_ERA5
    prmsl.coord("longitude").coord_system = cs_ERA5
    prmsl = prmsl.regrid(sCube, iris.analysis.Linear())
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
    sst.coord("latitude").coord_system = cs_ERA5
    sst.coord("longitude").coord_system = cs_ERA5
    sst = sst.regrid(sCube, iris.analysis.Linear())
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
    t2m.coord("latitude").coord_system = cs_ERA5
    t2m.coord("longitude").coord_system = cs_ERA5
    t2m = t2m.regrid(sCube, iris.analysis.Linear())
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
    prate.coord("latitude").coord_system = cs_ERA5
    prate.coord("longitude").coord_system = cs_ERA5
    prate = prate.regrid(sCube, iris.analysis.Linear())
    prate *= 50000 # Empirical for range scaling - 1000 for correct area-weighted match.
    prate /= 86400 * monthrange(year, month)[1]

    res.append(prate)
    return res
