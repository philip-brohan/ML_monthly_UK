#!/usr/bin/env python

# Make encoded versions of all the HadUKGrid input tensors

for decade in range(1880, 2020, 10):
    cmd = "./make_latent.py --startyear=%04d --endyear=%04d --epoch=299" % (
        decade + 1,
        decade + 10,
    )
    print(cmd)
