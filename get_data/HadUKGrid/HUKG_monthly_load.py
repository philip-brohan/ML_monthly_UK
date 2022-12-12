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
import csv

import warnings

warnings.filterwarnings("ignore", message=".*datum.*")


def load_variable(variable, year, month):
    if variable == "cbrt_rainfall":
        varC = load_variable("monthly_rainfall", year, month)
        varC.data = np.cbrt(varC.data)
        return varC
    varC = iris.load_cube(
        "%s/haduk-grid/%s/%04d/%02d.nc" % (os.getenv("SCRATCH"), variable, year, month)
    )
    return varC


def load_climatology(variable, month):
    varC = iris.load_cube(
        "%s/haduk-grid/%s_climatology/1961-1990/%s.nc"
        % (os.getenv("SCRATCH"), variable, month_abbr[month].lower())
    )
    return varC


# Specify a HadUK-Grid data mask
dm_hukg = load_variable("monthly_meantemp", 2014, 1)
dm_hukg.data.data[np.where(dm_hukg.data.mask == True)] = 0
dm_hukg.data.data[np.where(dm_hukg.data.mask == False)] = 1

# Load station data
def load_station_metadata(srcid=None):
    meta = {}
    with open(
        "%s/haduk-grid/station_data/metadata.csv" % os.getenv("SCRATCH"), newline=""
    ) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row["SRC_ID"] = "%06d" % int(row["SRC_ID"])
            for var in (
                "HIGH_PRCN_LAT",
                "HIGH_PRCN_LON",
                "ELEVATION",
                "LAT_WGS84",
                "LON_WGS84",
            ):
                if len(row[var]) > 0:
                    row[var] = float(row[var])
                else:
                    row[var] = None
            for var in ("EAST_GRID_REF", "NORTH_GRID_REF", "X", "Y"):
                if len(row[var]) > 0:
                    row[var] = int(row[var])
                else:
                    row[var] = None
            meta[row["SRC_ID"]] = row
    if srcid is not None:
        return meta[srcid]
    else:
        return meta


# Rainfall from Rainfall Rescue stations
def load_rr_stations(year, srcid=None, month=None):
    monthly = {}
    with open(
        "%s/haduk-grid/station_data/monthly_rainfall_rainfall-rescue_v1.1.0/monthly_rainfall_%04d.csv"
        % (os.getenv("SCRATCH"), year),
        newline="",
    ) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip headers
        for row in reader:
            row[0] = "%06d" % int(row[0])
            for var in range(1, 13):
                if len(row[var]) > 0:
                    row[var] = float(row[var])
                else:
                    row[var] = None
            if month is not None:
                monthly[row[0]] = row[month]
            else:
                monthly[row[0]] = row
    if srcid is not None:
        return monthly[srcid]
    else:
        return monthly
