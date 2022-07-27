#!/usr/bin/env python

# Make 160 years of UK averages from all sources

import os
import sys

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epoch", help="Model epoch", type=int, required=False, default=100
)
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
args = parser.parse_args()

cName = "constraints"
for constraint in ["PRMSL", "PRATE", "TMP2m", "SST"]:
    if vars(args)[constraint]:
        cName += "_%s" % constraint


def is_done(year, month):
    fn = ("%s/UK_averages/%s/%04d/%04d/%02d.pkl") % (
        LSCRATCH,
        cName,
        epoch,
        year,
        month,
    )
    if os.path.exists(fn):
        return True
    return False


for year in range(args.startyear, args.endyear + 1):
    for month in range(1, 13):
        if is_done(year, month):
            continue
        cmd = ("./make_averages_for_month.py --year=%04d --month=%d --epoch=%d") % (
            year,
            month,
            epoch,
        )
        for constraint in ["PRMSL", "PRATE", "TMP2m", "SST"]:
            if vars(args)[constraint]:
                cmd += "--%s" % constraint
        print(cmd)
