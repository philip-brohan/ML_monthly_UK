#!/usr/bin/env python

# Make 50 years of monthly-data tensors

import os
import sys

sys.path.append("%s/.." % os.path.dirname(__file__))
from localise import TSOURCE


def is_done(year, month, member, purpose):
    fn = "%s/datasets/%s/%04d-%02d_%02d.tfd" % (TSOURCE, purpose, year, month, member,)
    if os.path.exists(fn):
        return True
    return False


member = 1
count = 0
for year in range(1884, 2015):
    for month in range(1, 13):
        for memcnt in range(3):
            member += 7
            if member > 80:
                member -= 80
            purpose = "training"
            count += 1
            if count % 10 == 0:
                purpose = "test"
            if is_done(year, month, member, purpose):
                continue
            cmd = "./make_training_tensor.py --year=%04d --month=%02d --member=%d" % (
                year,
                month,
                member,
            )
            if purpose == "test":
                cmd += " --test"
            print(cmd)
