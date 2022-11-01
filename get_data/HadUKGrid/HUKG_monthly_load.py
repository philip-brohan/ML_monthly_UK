# Functions to load HadUK-grid monthly data

import os
import iris
import iris.util
import iris.cube
import iris.analysis
import iris.coord_systems
import iris.fileformats
import numpy as np
from calendar import month_abbr


def load_variable(variable, year, month):
    if variable=='cbrt_rainfall':
        varC = load_variable('monthly_rainfall',year,month)
        varC.data = np.cbrt(varC.data)
        return varC
    varC = iris.load_cube(
        "%s/haduk-grid/%s/%04d/%02d.nc"
        % (os.getenv("SCRATCH"), variable, year, month)
    )
    return varC

def load_climatology(variable, month):
    varC = iris.load_cube(
        "%s/haduk-grid/%s_climatology/1961-1990/%s.nc"
        % (os.getenv("SCRATCH"), variable, month_abbr[month].lower())
    )
    return varC

# Specify a HadUK-Grid data mask
dm_hukg = load_variable('monthly_meantemp',2014,1)
dm_hukg.data.data[np.where(dm_hukg.data.mask == True)] = 0
dm_hukg.data.data[np.where(dm_hukg.data.mask == False)] = 1
