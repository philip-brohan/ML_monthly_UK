#!/usr/bin/env python

# We need to normalise the data - to map values on the range 0-1
# Estimate scale parameters corresponding to 0 and 1.
# (Max and min values of data, with a bit of wiggle room).

import os
import sys
import iris
import iris.analysis
import numpy as np
import argparse

sys.path.append("%s/." % os.path.dirname(__file__))
from TWCR_monthly_load import load_monthly_member

sys.path.append("%s/../plots" % os.path.dirname(__file__))
from plot_variable import plot_cube

parser = argparse.ArgumentParser()
parser.add_argument("--variable", help="Variable name", type=str, required=True)
parser.add_argument(
    "--opdir",
    help="Directory for output files",
    default="%s/20CR/version_3/monthly/climatology" % os.getenv("SCRATCH"),
)
args = parser.parse_args()
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir, exist_ok=True)

pc = plot_cube()
smax = -1000000.0
smin = 1000000.0
for year in range(1981, 2011):
    for month in range(1, 13):
        var = load_monthly_member(args.variable, year, month, 1)
        var = var.regrid(pc, iris.analysis.Linear())
        vmax = np.amax(var.data)
        vmin = np.amin(var.data)
        smax = max(smax, vmax)
        smin = min(smin, vmin)

print(smin, smax)
