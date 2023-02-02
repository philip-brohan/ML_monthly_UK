#!/usr/bin/env python

# Estimate the distribution of offsets (mean and sd of obs-field) where the field does not assimilate
#  the station, and further estimate the probability this month's offset is not a sample from that
#  distribution (and so should fail QC). Distribution is estimated from offsets from the
#  surrounding +- 18 months.


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
opdir = "%s/RR_station_error_probs/%04d" % (LSCRATCH, args.year)

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

# Calculate the error probability for each station
c_mnth = "%04d%02d" % (args.year, args.month)
stations = fitted[c_mnth].keys()
offset = {}
for station in stations:
    current = np.nanmean(fitted[c_mnth][station])
    others = []
    for mnth in fitted.keys():
        if mnth == c_mnth:
            continue
        if station in fitted[mnth].keys():
            others.append(np.nanmean(fitted[mnth][station]))
    if len(others) < 10:  # Too little data to make comparison
        continue
    try:
        o_mean = np.nanmean(others)
        o_std = np.nanstd(others)
        o_z = (current - o_mean) / o_std
    except Exception:
        o_mean = np.nan
        o_std = np.nan
        o_z = np.nan
    offset[station] = (current, o_mean, o_std, o_z)

if not os.path.isdir(opdir):
    os.makedirs(opdir)

opfname = "%02d.csv" % args.month

with open("%s/%s" % (opdir, opfname), "w") as f:
    for station in offset.keys():
        f.write("%s," % station)
        f.write("%6.2f,%6.2f,%6.2f,%6.2f\n" % offset[station])
