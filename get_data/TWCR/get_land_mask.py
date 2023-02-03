#!/usr/bin/env python

# Retrieve V3 land mask

import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--opdir",
    help="Directory for output files",
    default="%s/20CR/version_3/fixed" % os.getenv("SCRATCH"),
)
args = parser.parse_args()
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir, exist_ok=True)


def remote_name():
    return (
        "https://downloads.psl.noaa.gov/Datasets/20thC_ReanV3/"
        + "timeInvariantSI/land.nc"
    )


def local_name():
    return "%s/land.nc" % args.opdir


if not os.path.isfile(local_name()):
    cmd = "curl -o %s %s" % (
        local_name(),
        remote_name(),
    )
    wg_retvalue = subprocess.call(cmd, shell=True)
    if wg_retvalue != 0:
        raise Exception("Failed to retrieve land mask")
