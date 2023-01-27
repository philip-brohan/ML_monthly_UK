#!/usr/bin/env python

# Make several years of station - fitted-field errors
# After allowing for pre-calculated station offsets

import os
import sys
import glob

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH


def count_done(year, month):
    files = glob.glob(
        "%s/RR_station_fits_offsets/%04d/%02d/*.csv" % (LSCRATCH, year, month)
    )
    return len(files)


for year in range(1836, 1891):
    for month in range(1, 13):
        nrun = 25 - count_done(year, month)
        for i in range(nrun):
            cmd = "./station_errors.py --year=%04d --month=%02d --offsets=True" % (
                year,
                month,
            )
            print(cmd)
