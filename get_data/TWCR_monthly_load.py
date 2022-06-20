# Functions to load 20CRv3 monthly data

import os
import iris
import iris.cube
import iris.time
import iris.coord_systems
import iris.fileformats
import datetime

# Need to add coordinate system metadata so they work with cartopy
coord_s = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)


def load_monthly_member(variable, year, month, member):
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
