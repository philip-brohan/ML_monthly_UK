#!/usr/bin/env python

# Make a climatology (1981-2010) for one month for one variable

import os
import sys
import iris
import argparse

sys.path.append("%s/." % os.path.dirname(__file__))
from TWCR_monthly_load import load_monthly_member

parser = argparse.ArgumentParser()
parser.add_argument("--variable", help="Variable name", type=str, required=True)
parser.add_argument("--month", help="Month", type=int, required=True)
parser.add_argument(
    "--opdir",
    help="Directory for output files",
    default="%s/20CR/version_3/monthly/climatology" % os.getenv("SCRATCH"),
)
args = parser.parse_args()
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir, exist_ok=True)

sum = None
count = 0
for year in range(1981, 2011):
    var = load_monthly_member(args.variable, year, args.month, 1)
    if sum is None:
        sum = var.copy()
    else:
        sum += var
    count += 1

sum /= count

iris.save(sum, "%s/%s_%02d.nc" % (args.opdir, args.variable, args.month))
