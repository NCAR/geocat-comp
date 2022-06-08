from typing import Union

import numpy as np
import xarray as xr

data_types = Union[np.ndarray, xr.DataArray]

## todo: come up with an interpolation alg for 2d that can handle np.nan values


def interpolate(data_in: data_types,
                data_out: data_types = None,
                lat_in: data_types = None,
                lon_in: data_types = None,
                lat_out: data_types = None,
                lon_out: data_types = None,
                wrap_longitudes=False) -> data_types:
    # check for data_in is xarray and complain if xarray doesn't have coords
    # check for data_out at all, and check for coords, and complain if it
    # doesn't
    # check for data_in is numpy, and complain if we don't have lat_in and
    # lon_in, lat_out, and lon_out
    # check if the arrays are 1d or 2d, if they are 1d, make them 2d with
    # np.meshgrid

    # first step check for wrpped latitude and then use np.pad to wrap edges
    if wrap_longitudes:
        data_in = np.pad(data_in, ((0, 0), (1, 1)), mode='wrap')

    # for each output location, find the surrounding three points
    # if any point is np.nan, return np.nan
    # determine the weights for each point, delaunay?, inverse distance?, how?
    # calculate the output value
    # rinse and repeat

    # second step interpolate!
    out_shape = locations_out.shape

    return data_out
