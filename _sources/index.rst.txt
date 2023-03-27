Machine Learning for Climate Modelling
======================================

This web page is an accompaniment to the paper 'Machine Learning for Climate Modelling' (Brohan, 2023) - currently in final draft. It is an index to the `GitHub repository containing the source code of the scripts and the models used in that paper <https://github.com/philip-brohan/ML_monthly_UK>`_


Getting started
---------------

- Clone the `GitHub repository <https://github.com/philip-brohan/ML_monthly_UK>`_ or `download as a Zip file <https://github.com/philip-brohan/ML_monthly_UK/archive/refs/heads/main.zip>`_
- Make a `conda environment <https://docs.conda.io/en/latest/>`_ using the specifications in the `environjment directory <https://github.com/philip-brohan/ML_monthly_UK/tree/main/environments>`_. Use this environment for all subsequent steps.

Get the Training data
---------------------
- You need two groups of training data:
    - `Twentieth Century Reanalysis (v3) SST and MSLP <https://github.com/philip-brohan/ML_monthly_UK/tree/main/get_data/TWCR>`_
    - `HadUK-Grid T2m and Precipitation <https://github.com/philip-brohan/ML_monthly_UK/tree/main/get_data/HadUKGrid>`_. Note that the download script in this directory will only work within the Met Office, but you can get the same data from `The HadUK-Grid web-page <https://www.metoffice.gov.uk/research/climate/maps-and-data/data/haduk-grid/haduk-grid>`_.

The data download directories also contain scripts to make the climatologies needed, and library functions for easy access to the downloaded data.

Standardize and reformat the training data
------------------------------------------

The training data need to standardized to the range 0-1, converted into tensors, and split into training and test subsets.

- `Scripts to make the tensors from the raw data <https://github.com/philip-brohan/ML_monthly_UK/tree/main/ML_models/DCVAE_anomalies/make_tensors>`_ 
- `Functions to package the tensors into tf.data.Datasets <https://github.com/philip-brohan/ML_monthly_UK/blob/main/ML_models/DCVAE_anomalies/makeDataset.py>`_

HadUK-Grid includes valid data only over regions of the UK with a nearby observation - other regions of the rectangular domain contain missing data. SST has similar partial coverage. In the model output we deal with this by ignoring missing data points (setting values to 0 and not including those points in model error calculations). We can't do the same in the input because the model is convolutional, so instead missing data are set to match the nearest valid point. This means that there are separate input and output tensors (but they are only different for missing data points.)

Specify and train the VAE 
-------------------------

- `Model specification <https://github.com/philip-brohan/ML_monthly_UK/blob/main/ML_models/DCVAE_anomalies/autoencoderModel.py>`_ 
- `Model training script <https://github.com/philip-brohan/ML_monthly_UK/blob/main/ML_models/DCVAE_anomalies/autoencoder.py>`_ 
- `Validation scripts <https://github.com/philip-brohan/ML_monthly_UK/tree/main/ML_models/DCVAE_anomalies/validation>`_ 

Applying the model
------------------

- `Data Assimilation <https://github.com/philip-brohan/ML_monthly_UK/tree/main/ML_models/DCVAE_anomalies/fit_to_fields>`_ 
- `SST perturbation <https://github.com/philip-brohan/ML_monthly_UK/tree/main/ML_models/DCVAE_anomalies/perturb_SST>`_ 
- `Extend to predict river flow <https://github.com/philip-brohan/ML_monthly_UK/tree/main/ML_models/DCVAE_anomalies/addons/Thames_flow>`_ 
    - `Get the training data <https://github.com/philip-brohan/ML_monthly_UK/tree/main/get_data/NRFA>`_


This document is crown copyright (2023). It is published under the terms of the `Open Government Licence <https://www.nationalarchives.gov.uk/doc/open-government-licence/version/2/>`_. Source code included is published under the terms of the `BSD licence <https://opensource.org/licenses/BSD-2-Clause>`_.