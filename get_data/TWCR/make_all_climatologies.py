#!/usr/bin/env python

# Make climatologies and sd climatologies

import os
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()

for month in range(1, 13):
    for var in [
        "TMPS",
        "TMP2m",
        "PRMSL",
        "PRATE",
    ]:
        opfile = "%s/20CR/version_3/monthly/climatology/%s_%02d.nc" % (
            os.getenv("SCRATCH"),
            var,
            month,
        )
        if not os.path.isfile(opfile):
            print(
                ("./make_climatology_for_month.py --month=%d --variable=%s")
                % (
                    month,
                    var,
                )
            )
        opfile = "%s/20CR/version_3/monthly/sd_climatology/%s_%02d.nc" % (
            os.getenv("SCRATCH"),
            var,
            month,
        )
        if not os.path.isfile(opfile):
            print(
                ("./make_sd_climatology_for_month.py --month=%d --variable=%s")
                % (
                    month,
                    var,
                )
            )
