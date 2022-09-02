#!/usr/bin/env python

import iris.coord_systems
import cartopy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lat", help="Latitude", type=float, required=True)
parser.add_argument("--lon", help="Longitude", type=float, required=True)
args = parser.parse_args()

target = iris.coord_systems.TransverseMercator(
    latitude_of_projection_origin=49.0,
    longitude_of_central_meridian=-2.0,
    false_easting=400000.0,
    false_northing=-100000.0,
    scale_factor_at_central_meridian=0.9996012717,
    ellipsoid=iris.coord_systems.GeogCS(
        semi_major_axis=6377563.396, semi_minor_axis=6356256.909
    ),
)
target_c = target.as_cartopy_crs()

source = iris.coord_systems.RotatedGeogCS(90, 180, 0)
source_c = source.as_cartopy_crs()

t_coords = target_c.transform_point(args.lon, args.lat, source_c)

x_grid_point = int((t_coords[0] + 196000) / 1000)
y_grid_point = int((t_coords[1] + 190000) / 1000)

print("x grid point: %4d" % x_grid_point)
print("y grid point: %4d" % y_grid_point)
