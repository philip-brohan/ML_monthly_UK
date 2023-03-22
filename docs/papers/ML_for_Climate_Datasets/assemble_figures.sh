#!/bin/bash

# Gather (most of) the figures, converting them into pdf format on the way

# Training progress plot
convert ../../../ML_models/DCVAE_anomalies/Training_progress.png -crop 680x750+0+0 +repage figures/training_progress.pdf

# Validation of model training (spatial)
convert ../../../ML_models/DCVAE_anomalies/validation/comparison.png figures/model_validation.pdf

# Validation of model training (time-series)
convert ../../../ML_models/DCVAE_anomalies/validation/multi.png figures/model_validation_multi.pdf

# Assimilation of SST (time-series)
convert ../../../ML_models/DCVAE_anomalies/fit_to_fields/fit_to_SST.png figures/fit_to_SST.pdf

# Assimilation of SST+PRMSL (time-series)
convert ../../../ML_models/DCVAE_anomalies/fit_to_fields/fit_to_SST+PRMSL.png figures/fit_to_SST+PRMSL.pdf

# ML model structure
convert figures/ML_model_structure.png figures/ML_model_structure.pdf

# VAE structure
convert figures/VAE_structure.png figures/VAE_structure.pdf

# Extension structure
convert figures/Extension_structure.png figures/Extension_structure.pdf

# Perturbation process
convert figures/perturbation_method.png figures/perturbation_method.pdf

# Perturbation results
convert ../../../ML_models/DCVAE_anomalies/perturb_SST/perturbation_validation.png figures/perturbation_validation.pdf

# River flow prediction
convert figures/services_river_flow.png figures/services_river_flow.pdf
