#!/bin/bash

# Gather (most of) the figures, converting them into pdf format on the way

# Validation of model training (spatial)
convert ../../../ML_models/DCVAE_anomalies/validation/comparison.png figures/model_validation.pdf

# Validation of model training (time-series)
convert ../../../ML_models/DCVAE_anomalies/validation/multi.png figures/model_validation_multi.pdf
