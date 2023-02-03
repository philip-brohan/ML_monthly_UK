# Functions to load 20CRv3 monthly data

import os
import iris
import iris.util
import iris.cube
import iris.time
import iris.analysis
import iris.coord_systems
import iris.fileformats
import datetime
import numpy as np
import warnings

warnings.filterwarnings("ignore", message=".*frac.*")

# Need to add coordinate system metadata so they work with cartopy
coord_s = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)

# Add a land-mask for TWCR SST grid
lm_TWCR = iris.load_cube("%s/20CR/version_3/fixed/land.nc" % os.getenv("SCRATCH"))
lm_TWCR = iris.util.squeeze(lm_TWCR)
lm_TWCR.coord("latitude").coord_system = coord_s
lm_TWCR.coord("longitude").coord_system = coord_s
lm_TWCR.data = np.ma.masked_where(lm_TWCR.data > 0.0, lm_TWCR.data, copy=False)
lm_TWCR.data.data[np.where(lm_TWCR.data.mask == True)] = 0
lm_TWCR.data.data[np.where(lm_TWCR.data.mask == False)] = 1


def load_quad(year, month, member):
    res = []
    for var in ("PRMSL", "TMPS", "TMP2m", "PRATE"):
        res.append(load_monthly_member(var, year, month, member))
    return res


def load_monthly_member(variable, year, month, member):
    if variable == "SST":
        ts = load_monthly_member("TMPS", year, month, member)
        return ts
    else:
        fname = "%s/20CR/version_3/monthly/members/%04d/%s.%04d.mnmean_mem%03d.nc" % (
            os.getenv("SCRATCH"),
            year,
            variable,
            year,
            member,
        )
        if not os.path.isfile(fname):
            raise Exception("No data file %s" % fname)
        ftt = iris.Constraint(time=lambda cell: cell.point.month == month)
        hslice = iris.load_cube(fname, ftt)
        if variable == "TMP2m":
            hslice = iris.util.squeeze(hslice)
        hslice.coord("latitude").coord_system = coord_s
        hslice.coord("longitude").coord_system = coord_s
        return hslice


def load_monthly_ensemble(variable, year, month):
    fname = "%s/20CR/version_3/monthly/members/%04d/%s.%04d.mnmean_mem*.nc" % (
        os.getenv("SCRATCH"),
        year,
        variable,
        year,
    )
    ftt = iris.Constraint(time=lambda cell: cell.point.month == month)
    hslice = iris.load(fname, ftt)
    for i, cb in enumerate(hslice):
        cb.add_aux_coord(
            iris.coords.AuxCoord(
                cb.attributes["realization"], standard_name="realization"
            )
        )
        cb = iris.util.new_axis(cb, "realization")
        del cb.attributes["realization"]
        del cb.attributes["history"]
        hslice[i] = cb
    hslice = hslice.concatenate_cube()
    hslice.coord("latitude").coord_system = coord_s
    hslice.coord("longitude").coord_system = coord_s
    return hslice


def load_climatology(variable, month):
    if variable == "SST":
        ts = load_climatology("TMPS", month)
        return ts
    fname = "%s/20CR/version_3/monthly/climatology/%s_%02d.nc" % (
        os.getenv("SCRATCH"),
        variable,
        month,
    )
    if not os.path.isfile(fname):
        raise Exception("No climatology file %s" % fname)
    return iris.load_cube(fname)


def load_sd_climatology(variable, month):
    if variable == "SST":
        ts = load_sd_climatology("TMPS", month)
        return ts
    fname = "%s/20CR/version_3/monthly/sd_climatology/%s_%02d.nc" % (
        os.getenv("SCRATCH"),
        variable,
        month,
    )
    if not os.path.isfile(fname):
        raise Exception("No sd climatology file %s" % fname)
    return iris.load_cube(fname)


def get_range(variable, month, cube=None, anomaly=False):
    clim = load_climatology(variable, month)
    if anomaly:
        clim.data *= 0
    sdc = load_sd_climatology(variable, month)
    if cube is not None:
        clim = clim.regrid(cube, iris.analysis.Nearest())
        sdc = sdc.regrid(cube, iris.analysis.Nearest())
    dmax = np.percentile(clim.data + (sdc.data * 2), 95)
    dmin = np.percentile(clim.data - (sdc.data * 2), 5)
    return (dmin, dmax)
