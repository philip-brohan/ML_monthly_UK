# Functions to load 20CRv3 monthly data

import os
import iris
import iris.util
import iris.cube
import iris.time
import iris.coord_systems
import iris.fileformats
import datetime
import numpy as np

# Need to add coordinate system metadata so they work with cartopy
coord_s = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)


def load_monthly_member(variable, year, month, member):
    if variable == "SST":
        ts = load_monthly_member("TMPS", year, month, member)
        lm = iris.load_cube("%s/20CR/version_3/fixed/land.nc" % os.getenv("SCRATCH"))
        lm = iris.util.squeeze(lm)
        lm.coord("latitude").coord_system = coord_s
        lm.coord("longitude").coord_system = coord_s
        ts = ts.regrid(lm, iris.analysis.Linear())
        msk = np.ma.masked_where(lm.data > 0.5, ts.data, copy=False)
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
    fname = "%s/20CR/version_3/monthly/climatology/%s_%02d.nc" % (
        os.getenv("SCRATCH"),
        variable,
        month,
    )
    if not os.path.isfile(fname):
        raise Exception("No climatology file %s" % fname)
    return iris.load_cube(fname)


def load_sd_climatology(variable, month):
    fname = "%s/20CR/version_3/monthly/sd_climatology/%s_%02d.nc" % (
        os.getenv("SCRATCH"),
        variable,
        month,
    )
    if not os.path.isfile(fname):
        raise Exception("No sd climatology file %s" % fname)
    return iris.load_cube(fname)


def get_range(variable, month):
    clim = load_climatology(variable, month)
    sdc = load_sd_climatology(variable, month)
    dmax = np.percentile(clim.data + (sdc.data * 2), 95)
    dmin = np.percentile(clim.data - (sdc.data * 2), 5)
    return (dmin, dmax)
