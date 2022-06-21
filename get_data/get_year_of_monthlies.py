#!/usr/bin/env python

# Retrieve V3 individual member monthly averages.
#  Every month in one year

import os
import subprocess
import tarfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--variable", help="Variable name", type=str, required=True)
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument(
    "--opdir",
    help="Directory for output files",
    default="%s/20CR/version_3/monthly/members" % os.getenv("SCRATCH"),
)
args = parser.parse_args()
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir, exist_ok=True)


def remote_name(variable, year):
    return (
        "https://portal.nersc.gov/archive/home/projects/incite11/"
        + "www/20C_Reanalysis_version_3/everymember_anal_netcdf/"
        + "mnmean/%s/%s_%04d_mnmean.tar" % (variable, variable, year)
    )


def local_name(variable, year):
    return "%s/%s_%04d_mnmean.tar" % (args.opdir, variable, year)


def unpack_local(variable, year):
    local_file = local_name(variable, year)
    tar = tarfile.open(local_file, "r")
    local_dir = os.path.dirname(local_file)
    tar.extractall(path=local_dir)
    tar.close()
    # Update the extracted file times
    #  To stop SCRATCH deleting them as too old
    nfiles = os.listdir("%s/%04d" % (local_dir, year))
    for nfile in nfiles:
        os.utime("%s/%04d/%s" % (local_dir, year, nfile))
    os.remove(local_file)


# Download the tar file
cmd = "wget -O %s %s" % (
    local_name(args.variable, args.year),
    remote_name(args.variable, args.year),
)
wg_retvalue = subprocess.call(cmd, shell=True)
if wg_retvalue != 0:
    raise Exception("Failed to retrieve data")
unpack_local(args.variable, args.year)
