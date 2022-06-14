from typing import Union

import numpy as np
import xarray as xr
from math import cos, asin, sqrt
from .errors import ChunkError, CoordinateError

data_types = Union[np.ndarray, xr.DataArray]
r = 6371

#return hav function
def hav(theta):
    return (1-cos(theta))/2

#return spherical distance using haversine formula
def haversine(lat1, lon1, lat2, lon2):
    if lat2 == np.nan or lon2 == np.nan:
        return np.nan
    havValue = hav(lat2 - lat1) + (1 - hav(lat1 - lat2) - hav(lat1 + lat2))*hav(lon2 - lon1)
    return 2*r*asin(sqrt(havValue))
    
#find interpolated value of lat, lon given data_in using inverse weighted distance
#return nan if any of four surrounding values are nan
def findValue(data_in, i, j, lat, lon):
    dist = np.asarray([0, 0, 0, 0])
    dist[0] = haversine(lat, lon, data_in.coords[data_in.dims[0]].values[i], data_in.coords[data_in.dims[1]].values[j])
    dist[1] = haversine(lat, lon, data_in.coords[data_in.dims[0]].values[i+1], data_in.coords[data_in.dims[1]].values[j])
    dist[2] = haversine(lat, lon, data_in.coords[data_in.dims[0]].values[i], data_in.coords[data_in.dims[1]].values[j+1])
    dist[3] = haversine(lat, lon, data_in.coords[data_in.dims[0]].values[i+1], data_in.coords[data_in.dims[1]].values[j+1])
    if np.nan in dist:
        return np.nan
    dist = 1/(dist*dist)
    dist = dist/np.sum(dist)
    dist[0] = dist[0]*data_in[data_in.dims[-3]].values[i][j]
    dist[1] = dist[0]*data_in[data_in.dims[-3]].values[i+1][j]
    dist[2] = dist[0]*data_in[data_in.dims[-3]].values[i][j+1]
    dist[3] = dist[0]*data_in[data_in.dims[-3]].values[i+1][j+1]
    return np.average(dist)


## todo: come up with an interpolation alg for 2d that can handle np.nan values

def interpolate(data_in: data_types,
                data_out: data_types = None,
                lat_in: data_types = None,
                lon_in: data_types = None,
                lat_out: data_types = None,
                lon_out: data_types = None,
                wrap_longitudes=False) -> data_types:

    data_out_given = True

    if not isinstance(data_in, xr):
        if lat_in is None or lon_in is None:
            raise CoordinateError(
                "Coordinates must be provided if xarray is given"
            )
        data_in = xr.DataArray(
                    data_in,
                    dims=['data', 'lat', 'lon'],
                    coords={
                        'lat': lat_in,
                        'lon': lon_in
                    }
        )
        data_in.coords[data_in.dims[-2]].values = lat_in
        data_in.coords[data_in.dims[-1]].values = lon_in
    else:
        lat_in = data_in.coords[data_in.dims[-2]]
        lon_in = data_in.coords[data_in.dims[-2]]
        if lat_in is None or lon_in is None:
            raise CoordinateError(
                "Coordinates must be provided in xarray"
            )

    if not isinstance(data_out, xr):
        data_out_given = False
        if lat_out is None or lon_out is None:
            raise CoordinateError(
                "Coordinates must be provided if xarray is not given"
            )
        data_out = xr.DataArray(
                data_out,
                dims = ['data', 'lat', 'lon'],
                coords={
                    'lat': lat_out,
                    'lon': lon_out
                }
        )
        data_out.coords[data_out.dims[-2]].values = lat_out
        data_out.coords[data_out.dims[-1]].values = lon_out
    else:
        lat_out = data_out.coords[data_out.dims[-2]]
        lon_out = data_out.coords[data_out.dims[-1]]
        if lat_out is None or lon_out is None:
            raise CoordinateError(
                "Coordinates must be provided in xarray"
            )

    # first step check for wrpped latitude and then use np.pad to wrap edges
    if wrap_longitudes:
        data_in = np.pad(data_in, ((0, 0), (1, 1)), mode='wrap')

    #iterate through in and out coords
    i_in, j_in, i_out, j_out = 0, 0, 0, 0

    while(i_in < len(lat_in) - 1 and j_in < len(lon_in) - 1 and i_out < len(lat_out) and j_out < len(lon_out)):

        # if coordinates need extrapolation
        if lat_out[i_out] < lat_in[0]:
            i_out += 1
            continue
        if lon_out[j_out] < lon_out[0]:
            j_out += 1
            continue
        if lon_out[j_out] > lon_out[len(lon_out) - 1] or lat_out[i_out] > lat_out[len(lat_out) - 1]:
            break

        # if coord equal
        if lat_in[i_in] == lat_out[i_out] and lon_in[j_in] == lon_out[j_out]:
            data_out[data_out.dims[-3]].values[i_out][j_out] = data_in[data_in.dims[-3]].values[i_in][j_in]
            i_out += 1
            j_out += 1
            continue

        # if ptr in right place (lower left of output coord)
        elif lat_in[i_in] <= lat_out[i_out] and lon_in[j_in] <= lon_out[j_out] and lat_in[i_in + 1] > lat_out[i_out] and lon_in[j_in+1] > lon_out[j_out]:
                data_out[i_out][j_out] = findValue(data_in, i_in, j_in, lat_out[i_out], lon_out[j_out])
                i_out += 1
                j_out += 1
                continue
        
        # if in coord too low
        if lat_in[i_in + 1] <= lat_out[i_out]:
            i_in += 1
        if lon_in[j_in + 1] <= lon_out[j_out]:
            j_in += 1

    if data_out_given:
        return data_out
        
    return data_out[data_out.dims[-3]].values
