#!/usr/bin/env python

# plot a single 20CRv3 monthly field for the UK region

import os
import sys
import cmocean

sys.path.append("%s/../get_data" % os.path.dirname(__file__))
from TWCR_monthly_load import load_monthly_member
from TWCR_monthly_load import get_range

sys.path.append("%s/." % os.path.dirname(__file__))

sys.path.append("%s/." % os.path.dirname(__file__))
from plot_variable import plotField

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument("--variable", help="Variable name", type=str, required=True)
parser.add_argument(
    "--opdir", help="Output directory", type=str, required=False, default="."
)
parser.add_argument(
    "--opfile", help="Output file name", type=str, required=False, default=None
)
args = parser.parse_args()
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

if args.opfile is None:
    args.opfile = "%s_%04d-%02d.png" % (args.variable, args.year, args.month)

if args.variable == "PRMSL":
    cMap = cmocean.cm.diff
elif args.variable == "TMPS":
    cMap = cmocean.cm.balance
elif args.variable == "TMP2m":
    cMap = cmocean.cm.balance
elif args.variable == "SST":
    cMap = cmocean.cm.balance
elif args.variable == "PRATE":
    cMap = cmocean.cm.rain
else:
    raise Exception("Unsupported variable " + args.variable)

var = load_monthly_member(args.variable, args.year, args.month, 1)
(dmin, dmax) = get_range(args.variable, args.month)
if args.variable == "PRATE":
    dmin = 0

plotField(var, opDir=args.opdir, fName=args.opfile, cMap=cMap, vMin=dmin, vMax=dmax)
