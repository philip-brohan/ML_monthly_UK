#!/usr/bin/env python

# Make 160 years of UK averages from all sources

import os

epoch = 990


def is_done(year, month):
    fn = (
        "%s/ML_monthly_UK/DCVAE_HadUK-grid/UK_averages/SST/" + "%04d/%04d/%02d.pkl"
    ) % (os.getenv("SCRATCH"), epoch, year, month,)
    if os.path.exists(fn):
        return True
    return False


for year in range(1884, 2015):
    for month in range(1, 13):
        if is_done(year, month):
            continue
        cmd = (
            "./make_averages_for_month.py --year=%04d " + "--month=%d --epoch=%d"
        ) % (year, month, epoch,)
        print(cmd)
