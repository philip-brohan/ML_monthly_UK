#!/bin/bash

# We're going to work with haduk-grid fields.
# Copy the files onto $SCRATCH

mkdir -p $SCRATCH/haduk-grid

mkdir -p $SCRATCH/haduk-grid/monthly_meantemp
mkdir -p $SCRATCH/haduk-grid/monthly_rainfall

# Recent years
for year in 2019 2020
do
cp -rn /data/users/haduk/uk_climate_data/supported/haduk-grid/series_archive_provisional/grid/monthly_meantemp/$year $SCRATCH/haduk-grid/monthly_meantemp
cp -rn /data/users/haduk/uk_climate_data/supported/haduk-grid/series_archive_provisional/grid/monthly_rainfall/$year $SCRATCH/haduk-grid/monthly_rainfall
done

# Archive years
for year in {1862..2018}
do
cp -rn /data/users/haduk/uk_climate_data/supported/haduk-grid/v1.0.3.0/data/grid_archives/series_archive/grid/monthly_meantemp/$year $SCRATCH/haduk-grid/monthly_meantemp
cp -rn /data/users/haduk/uk_climate_data/supported/haduk-grid/v1.0.3.0/data/grid_archives/series_archive/grid/monthly_rainfall/$year $SCRATCH/haduk-grid/monthly_rainfall
done

# Climatology
mkdir -p $SCRATCH/haduk-grid/monthly_meantemp_climatology
cp -rn /data/users/haduk/uk_climate_data/supported/haduk-grid/v1.0.3.0/data/grid_archives/lta_archive_v1/grid/monthly_meantemp_climatology/1981-2010 $SCRATCH/haduk-grid/monthly_meantemp_climatology/
mkdir -p $SCRATCH/haduk-grid/monthly_rainfall_climatology
cp -rn /data/users/haduk/uk_climate_data/supported/haduk-grid/v1.0.3.0/data/grid_archives/lta_archive_v1/grid/monthly_rainfall_climatology/1981-2010 $SCRATCH/haduk-grid/monthly_rainfall_climatology/
