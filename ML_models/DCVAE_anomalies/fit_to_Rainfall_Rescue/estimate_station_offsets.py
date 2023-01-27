#!/usr/bin/env python

# Estimate the offset (obs-field), for all stations in a month, as the 3-year running mean
#  of the station obs-field_with_station_not_assimilated (precalculated)


import os
import sys
import numpy as np
import glob
import csv
import datetime
from dateutil import relativedelta

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--year", help="Year to estimate", type=int, required=False, default=1883
)
parser.add_argument(
    "--month", help="Month to estimate", type=int, required=False, default=3
)
args = parser.parse_args()

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import LSCRATCH

# Location of model fit files
ipdir = "%s/RR_station_fits" % LSCRATCH
# And the offsets we will calculate
opdir = "%s/RR_station_offsets/%04d" % (LSCRATCH, args.year)

# Load the fitted data for one month
def load_fitted(year, month):
    fitted = {}
    ffiles = glob.glob("%s/%04d/%02d/*.csv" % (ipdir, year, month))
    for fi in ffiles:
        with open(fi, "r") as f:
            reader = csv.reader(f)
            for fl in reader:
                if len(fl) == 0:
                    continue
                if int(fl[1]) == 1:  # Skip all the asssimilated stations
                    continue
                sid = fl[0]
                if sid not in fitted.keys():
                    fitted[sid] = []
                fitted[sid].append(float(fl[2]) - float(fl[3]))  # obs - generated
    return fitted


# Get the fitted data for +- 18 months around the present
fitted = {}
for m_offset in range(-18, 19):
    dto = datetime.date(args.year, args.month, 15) + relativedelta.relativedelta(
        months=m_offset
    )
    fitted["%04d%02d" % (dto.year, dto.month)] = load_fitted(dto.year, dto.month)

# Calculate the mean offset for each station
stations = fitted["%04d%02d" % (args.year, args.month)].keys()
offset = {}
for station in stations:
    offset[station] = 0
    count = 0
    for mnth in fitted.keys():
        if station in fitted[mnth].keys():
            offset[station] += np.nanmean(fitted[mnth][station])
            count += 1
    offset[station] /= count

if not os.path.isdir(opdir):
    os.makedirs(opdir)

opfname = "%02d.csv" % args.month

with open("%s/%s" % (opdir, opfname), "w") as f:
    for station in offset.keys():
        f.write("%s,%6.2f,\n" % (station, offset[station]))
