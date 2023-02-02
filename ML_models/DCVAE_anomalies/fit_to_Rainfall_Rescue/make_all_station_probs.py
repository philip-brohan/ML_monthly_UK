#!/usr/bin/env python

# Make several years of station - fitted-field offset estimates
#  from already-calculated station errors

import os
import sys
import glob

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH


def is_done(year, month):
    fn = "%s/RR_station_error_probs/%04d/%02d.csv" % (LSCRATCH, year, month)
    if os.path.isfile(fn):
        return True
    else:
        return False


for year in range(1836, 1960):
    for month in range(1, 13):
        if is_done(year, month):
            continue
        cmd = "./estimate_error_probability.py --year=%04d --month=%02d" % (year, month)
        print(cmd)
