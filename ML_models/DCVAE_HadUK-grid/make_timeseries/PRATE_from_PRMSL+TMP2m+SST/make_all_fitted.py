#!/usr/bin/env python

# Make 100+ years of fitted fields

import os

epoch = 770


def is_done(year, month, member):
    fn = (
        "%s/ML_monthly_UK/DCVAE_HadUK-grid/fitted/constraints_PRMSL_TMP2m_SST/"
        + "%04d/%04d/%02d/%02d.nc"
    ) % (os.getenv("SCRATCH"), epoch, year, month, member,)
    if os.path.exists(fn):
        return True
    return False


for year in range(1884, 2015):
    for month in range(1, 13):
        for member in [1, 12, 24, 36, 48, 60, 72]:
            if is_done(year, month, member):
                continue
            cmd = (
                "../../fit_to_fields/fit_and_save.py --year=%04d "
                + "--month=%d --member=%d --PRMSL --TMP2m --SST --epoch=%d"
            ) % (year, month, member, epoch,)
            print(cmd)
