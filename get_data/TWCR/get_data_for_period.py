#!/usr/bin/env python

# Get monthly 20C$v3 members  data for several years, and store on SCRATCH.

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--startyear", type=int, required=True)
parser.add_argument("--endyear", type=int, required=True)
args = parser.parse_args()

for year in range(args.startyear, args.endyear + 1):
    for var in [
        "TMPS",
        "TMP2m",
        "PRMSL",
        "PRATE",
    ]:
        opfile = "%s/20CR/version_3/monthly/members/%04d/%s.%04d.mnmean_mem080.nc" % (
            os.getenv("SCRATCH"),
            year,
            var,
            year,
        )
        if not os.path.isfile(opfile):
            print(("./get_year_of_monthlies.py --year=%d --variable=%s") % (year, var,))
