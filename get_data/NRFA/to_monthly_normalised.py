#!/usr/bin/env python

# Convert the flow data to monthly anomalies on the range 0-1

import sys
import csv
from statistics import mean
import math

# Load the data
fdata = {}
with open("39001_gdf.csv", newline="") as csvfile:
    lrd = csv.reader(csvfile)
    for row in lrd:
        if len(row[0]) != 10:
            continue  # Skip headers
        dte = row[0]
        flow = float(row[1])
        if dte[4] == "-":  # F****ing excel and dates.
            year = dte[0:4]
            month = dte[5:7]
        else:
            year = dte[6:]
            month = dte[3:5]
        if not year in fdata:
            fdata[year] = {}
        if not month in fdata[year]:
            fdata[year][month] = []
        fdata[year][month].append(flow)

# Convert to monthly means
for year in range(1884, 2021):
    for month in range(1, 13):
        yr = "%04d" % year
        mn = "%02d" % month
        try:
            fdata[yr][mn] = mean(fdata[yr][mn])
        except:
            print(yr)
            print(mn)
            sys.exit(0)


# Calculate a climatology
clim = {}
for month in range(1, 13):
    mn = "%02d" % month
    clim[mn] = 0
    for year in range(1961, 1991):
        yr = "%04d" % year
    clim[mn] += fdata[yr][mn]
    clim[mn] /= 30

# Convert to anomalies and find range
max = -1000000
min = 1000000
for year in range(1884, 2021):
    for month in range(1, 13):
        yr = "%04d" % year
        mn = "%02d" % month
        fdata[yr][mn] /= clim[mn]
        fdata[yr][mn] = math.log(fdata[yr][mn])
        if fdata[yr][mn] > max:
            max = fdata[yr][mn]
        if fdata[yr][mn] < min:
            min = fdata[yr][mn]

# Scale to range 0-1 and print
for year in range(1884, 2021):
    for month in range(1, 13):
        yr = "%04d" % year
        mn = "%02d" % month
        fdata[yr][mn] -= min
        fdata[yr][mn] /= max - min
        print("%04d-%02d %3.2f" % (year, month, fdata[yr][mn]))
