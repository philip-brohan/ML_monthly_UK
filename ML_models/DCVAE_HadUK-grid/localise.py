# I want to experiment with model perturbations by copying this directory
# Each copy should save everything in a different location
# Localise that location to this script, so I don't have to
#  edit every script just to change the same file path.

import os

# Where to save output
LSCRATCH = "%s/ML_monthly_UK/DCVAE_HadUK-grid" % os.getenv("SCRATCH")

# Where to get input tensors
TSOURCE = "%s/ML_monthly_UK/DCVAE_HadUK-grid" % os.getenv("SCRATCH")
