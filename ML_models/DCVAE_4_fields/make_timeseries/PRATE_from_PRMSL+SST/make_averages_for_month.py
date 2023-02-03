#!/usr/bin/env python

# Get UK-land averages from all data sources for a selected month
# Pickle them for plotting.

import os
import sys
import numpy as np
import iris
import iris.analysis
from calendar import monthrange
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=990)
parser.add_argument(
    "--year", help="Year to fit to", type=int, required=False, default=1969
)
parser.add_argument(
    "--month", help="Month to fit to", type=int, required=False, default=3
)
args = parser.parse_args()

sys.path.append("%s/../../../../get_data" % os.path.dirname(__file__))
from TWCR_monthly_load import load_quad

sys.path.append("%s/../../../../plots" % os.path.dirname(__file__))
from plot_variable import plot_cube

# Cubes for regridding
pc = plot_cube()
pc.coord("longitude").guess_bounds()
pc.coord("latitude").guess_bounds()
pch = plot_cube(0.01)
pch.coord("longitude").guess_bounds()
pch.coord("latitude").guess_bounds()

# 20CRv3 land mask
lm = iris.load_cube("%s/20CR/version_3/fixed/land.nc" % os.getenv("SCRATCH"))
lm = iris.util.squeeze(lm)
coord_s = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
lm.coord("latitude").coord_system = coord_s
lm.coord("longitude").coord_system = coord_s
lm = lm.regrid(pc, iris.analysis.Linear())
sst_mask = lm.data < 0.5

# Load the HadUK-grid data
# and reduce to joint v3-land:HadUKg coverage on the 1 degree grid
t2m_huk = None
try:
    t2m_huk = iris.load_cube(
        "%s/haduk-grid/monthly_meantemp/%04d/%02d.nc"
        % (os.getenv("SCRATCH"), args.year, args.month)
    )
    t2m_huk = t2m_huk.regrid(pch, iris.analysis.Nearest())
    t2m_huk = t2m_huk.regrid(pc, iris.analysis.AreaWeighted())
    t2m_huk.data.mask[sst_mask] = True  # Apply 20CR land mask
except:
    print("No HadUK-grid T2m data for %04d-%02d" % (args.year, args.month))
prate_huk = None
try:
    prate_huk = iris.load_cube(
        "%s/haduk-grid/monthly_rainfall/%04d/%02d.nc"
        % (os.getenv("SCRATCH"), args.year, args.month)
    )
    prate_huk = prate_huk.regrid(pch, iris.analysis.Nearest())
    prate_huk = prate_huk.regrid(pc, iris.analysis.AreaWeighted())
    prate_huk.data.mask[sst_mask] = True  # Apply 20CR land mask
except:
    print("No HadUK-grid PRATE data for %04d-%02d" % (args.year, args.month))

# Load the 20CRv3 data
# and reduce to joint v3-land:HadUKg coverage on the 1 degree grid
v3 = []
for member in range(1, 81):
    qd = load_quad(args.year, args.month, member)
    for i in range(4):
        qd[i] = qd[i].regrid(pc, iris.analysis.Linear())
        qd[i].data = qd[i].data + prate_huk.data * 0.0  # Apply HadUK data mask
        qd[i].data.mask[sst_mask] = True  # Apply 20CR land mask
    v3.append(qd)

# Load the fitted data
# and reduce to joint v3-land:HadUKg coverage on the 1 degree grid
def load_fitted(year, month, member, epoch):
    fn = "%s/ML_monthly_UK/fitted/constraints_PRMSL_SST/%04d/%04d/%02d/%02d.nc" % (
        os.getenv("SCRATCH"),
        epoch,
        year,
        month,
        member,
    )
    if not os.path.exists(fn):
        raise Exception("Missing data file %s" % fn)
    fitted = iris.load(fn)
    fitted.sort(key=lambda cube: cube.var_name)
    return fitted


ft = []
for member in range(1, 81):
    qd = load_fitted(args.year, args.month, member, args.epoch)
    for i in range(4):
        qd[i].data = qd[i].data + prate_huk.data * 0.0  # Apply HadUK data mask
        qd[i].data.mask[sst_mask] = True  # Apply 20CR land mask
    ft.append(qd)

# Package-up the averages
res = {
    "T2m": {
        "HUKG": -999,
        "20CR": [],
        "Fit": [],
    },
    "PRATE": {
        "HUKG": -999,
        "20CR": [],
        "Fit": [],
    },
}
if t2m_huk is not None:
    res["T2m"]["HUKG"] = np.mean(t2m_huk.data)
if prate_huk is not None:
    res["PRATE"]["HUKG"] = np.mean(prate_huk.data)

seconds_in_month = 86400 * monthrange(args.year, args.month)[1]
for member in range(1, 81):
    res["T2m"]["20CR"].append(np.mean(v3[member - 1][2].data) - 273.15)
    res["T2m"]["Fit"].append(np.mean(ft[member - 1][3].data) - 273.15)
    res["PRATE"]["20CR"].append(np.mean(v3[member - 1][3].data) * seconds_in_month)
    res["PRATE"]["Fit"].append(np.mean(ft[member - 1][0].data) * seconds_in_month)

opfile = "%s/ML_monthly_UK/UK_averages/%04d/%04d/%02d.pkl" % (
    os.getenv("SCRATCH"),
    args.epoch,
    args.year,
    args.month,
)

if not os.path.isdir(os.path.dirname(opfile)):
    os.makedirs(os.path.dirname(opfile))

pickle.dump(res, open(opfile, "wb"))
