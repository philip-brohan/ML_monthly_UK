#!/usr/bin/env python

# Make 160 years of fitted fields

import os

epoch = 990


def is_done(year, month, member):
    fn = "%s/ML_monthly_UK/fitted/constraints_PRMSL_SST/%04d/%04d/%02d/%02d.nc" % (
        os.getenv("SCRATCH"),
        epoch,
        year,
        month,
        member,
    )
    if os.path.exists(fn):
        return True
    return False


for year in range(1850, 2015):
    for month in range(1, 13):
        for member in range(1, 81):
            if is_done(year, month, member):
                continue
            cmd = (
                "../../fit_to_fields/fit_and_save.py --year=%04d "
                + "--month=%d --member=%d --PRMSL --SST --epoch=%d"
            ) % (
                year,
                month,
                member,
                epoch,
            )
            print(cmd)
