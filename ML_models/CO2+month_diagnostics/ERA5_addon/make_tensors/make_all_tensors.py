#!/usr/bin/env python

# Make 50 years of monthly-data tensors
# From ERA5 data

import os
import sys

sys.path.append("%s/../.." % os.path.dirname(__file__))
from localise import TSOURCE


def is_done(year, month, purpose):
    fn = "%s/datasets_ERA5/%s/%04d-%02d.tfd" % (TSOURCE, purpose, year, month)
    if os.path.exists(fn):
        return True
    return False


count = 0
for year in range(1959, 2023):
    for month in range(1, 13):
        purpose = "training"
        count += 1
        if count % 10 == 0:
            purpose = "test"
        if is_done(year, month, purpose):
            continue
        cmd = "./make_training_tensor.py --year=%04d --month=%02d" % (
            year,
            month,
        )
        if purpose == "test":
            cmd += " --test"
        print(cmd)
