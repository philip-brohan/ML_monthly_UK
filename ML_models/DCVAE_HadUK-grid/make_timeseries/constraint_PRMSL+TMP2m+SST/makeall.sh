# Make time-series diagnostics

# This script is Met Office specific, as it uses spice_parallel to run
#  hundreds on single-processor jobs in parallel. Replace spice_parallel
#  with gnu parallel or a local tool if running elsewhere.

# Find state vector and fitted fields for each year+month+member
../make_all_fitted.py --epoch=100 --startyear=1884 --endyear=2014 --PRMSL --SST --TMP2m | spice_parallel --time=10

# Make monthly UK average for each year+month
../make_all_averages.py --epoch=100 --startyear=1884 --endyear=2014 --PRMSL --SST --TMP2m | spice_parallel --time=5 --batch=5

# Plot monthly UK average
../plot_monthly.py --epoch=100 --startyear=1995 --endyear=2014 --PRMSL --SST --TMP2m

# Plot annual average for full period
../plot_annual.py --epoch=100 --startyear=1995 --endyear=2014 --PRMSL --SST --TMP2m


