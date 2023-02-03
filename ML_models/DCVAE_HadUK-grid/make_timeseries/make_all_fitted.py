#!/usr/bin/env python

# Make a sequence of fitted fields

import os
import sys

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Model epoch", type=int, required=False, default=99)
parser.add_argument(
    "--startyear", help="Sequence start year", type=int, required=False, default=1884
)
parser.add_argument(
    "--endyear", help="Sequence end year", type=int, required=False, default=2014
)
parser.add_argument(
    "--PRMSL", help="Fit to PRMSL?", dest="PRMSL", default=False, action="store_true"
)
parser.add_argument(
    "--SST", help="Fit to SST?", dest="SST", default=False, action="store_true"
)
parser.add_argument(
    "--TMP2m", help="Fit to TMP2m?", dest="TMP2m", default=False, action="store_true"
)
parser.add_argument(
    "--PRATE", help="Fit to PRATE?", dest="PRATE", default=False, action="store_true"
)
parser.add_argument(
    "--iter",
    help="No. of iterations",
    type=int,
    required=False,
    default=100,
)
args = parser.parse_args()

cName = "constraints"
for constraint in ["PRMSL", "PRATE", "TMP2m", "SST"]:
    if vars(args)[constraint]:
        cName += "_%s" % constraint


def is_done(year, month, member):
    fn = ("%s/fitted/%s/%04d/%04d/%02d/%02d.nc") % (
        LSCRATCH,
        cName,
        args.epoch,
        year,
        month,
        member,
    )
    if os.path.exists(fn):
        return True
    return False


# Get path from cwd to this script
reldir = os.path.dirname("./" + os.path.relpath(__file__))

for year in range(args.startyear, args.endyear + 1):
    for month in range(1, 13):
        for member in [1, 12, 24, 36, 48, 60, 72]:
            if is_done(year, month, member):
                continue
            cmd = (
                "%s/../fit_to_fields/fit_and_save.py --year=%04d "
                + "--month=%d --member=%d --epoch=%d --iter=%d"
            ) % (reldir, year, month, member, args.epoch, args.iter)
            for constraint in ["PRMSL", "PRATE", "TMP2m", "SST"]:
                if vars(args)[constraint]:
                    cmd += " --%s" % constraint

            print(cmd)
