from types import NoneType
import typing
import warnings

from dask.array.core import map_blocks
import numpy as np
import numpy.ma as ma
import xarray as xr
from .errors import ChunkError, CoordinateError

supported_types = typing.Union[xr.DataArray, np.ndarray]

# Helper Functions _<function_name>()
# These wrappers are executed within dask processes (if any), and could/should
# do anything that can benefit from parallel execution.

def _pre(data_in, icycx, msg_py, is_two_d):
    """
    Helper Function: Handling missing data functionality and 
    adding cyclic point if required

    Parameters
    ----------
    data_in : :class:`xarray.DataArray`
        The data on which to operate

    icycx : :class:`bool`
        Determines if cyclic point should be added or not.
        If true then add point, else do nothing.

    msg_py : :class:`int`, :class:`float`, Optional
        Provides an alternative to NaN
    
    is_two_d : :class:'bool'
        Informs if interpolation is 1-dimensional or 2-dimensional

    Returns
    -------
    data_in : :class:`xarray.DataArray`
       The data input with cyclic points added (if icycx is true) 
       and msg_py values replaced with np.nan
    
    Notes
    -------
    Adding cyclic point adapted from cartopy.add_cyclic_point - https://scitools.org.uk/cartopy/docs/latest/reference/generated/cartopy.util.add_cyclic_point.html
    
    """
    # replace msg_py with np.nan
    if msg_py != None :
        data_in = xr.DataArray(np.where(data_in.values == msg_py, np.nan, data_in.values),
                            dims = data_in.dims,
                            coords = data_in.coords)

    # add cyclic points and create new data array
    if icycx :
        lon = data_in.coords[data_in.dims[-1]].values
        delta_coord = np.diff(lon)
        new_coord = np.append(lon, lon[-1] + delta_coord[0])
        if is_two_d:
            temp = np.pad(data_in.values, ((0, 0), (0, 1,)), "wrap")
        else:
            temp = np.pad(data_in.values, (0, 1), "wrap")
        temp = xr.DataArray(temp,
                                dims = data_in.dims)
        if not is_two_d:
            data_in = temp.assign_coords({"x": new_coord})
        if is_two_d:
            data_in = temp.assign_coords({"x": data_in.coords[data_in.dims[-2]].values})
            data_in = data_in.assign_coords({"y": new_coord})

    return data_in

def _post(data_in, msg_py):
    """
    Helper Function: Handling missing data functionality 

    Parameters
    ----------
    data_in : :class:`xarray.DataArray`
        The data on which to operate

    msg_py : :class:`int`, :class:`float`, Optional
         Provides an alternative to NaN

    Returns
    -------
    data_in : :class:`xarray.DataArray`
       The data input with np.nan values replaced with msg_py
    """
    if msg_py != None:
        x = np.where(data_in.values == np.nan, msg_py, data_in.values)
        data_in = xr.DataArray(np.where(np.isnan(data_in.values), msg_py, data_in.values),
                            dims = data_in.dims,
                            coords = data_in.coords)
        
    return data_in


def interp_wrap(data_in: supported_types,
            lon_out: supported_types,
            lat_out: supported_types = None,
            lon_in: supported_types = None,
            lat_in: supported_types = None,
            icycx: bool = False,
            msg_py: np.number = None,
            assume_sorted: bool = False,
            method: str = "linear") -> supported_types:

    """ Multidimensional interpolation of variables

    Parameters
    ----------
    data_in : :class:`xarray.DataArray`, :class:`numpy.ndarray`
        - 1 or multi-dimensional array containing values to be interpolated.
        - If xarray.DataArray provided, must have associated coords
        - If 1d interpolation, 
            len(data_in[-1]) == len(lon_in) or associated coord
        - If 2d interpolation,
            len(data_in[-1][0]) == len(lon_in) or associated coord (data_in.coord[dims[-1]])
            len(data_in[-1]) == len(lat_in) or associated coord (data_in.coord[dims[-2]])

    lon_out : :class:`numpy.ndarray`
        - 1d array containing coordinate values for output data
        - Will perform 1d or 2d interpolation depending on presence of lat_out
        - Will not perform extrapolation, returns missing values if any surrounding points
        contain missing values

    lat_out : :class:'numpy.ndarray', Optional
        - 1d array contianing coordinate values for output data
        - Must be present if input is 2d
        - Will not perform extrapolation, returns missing values if any surrounding points
        contain missing values

    lon_in : :class:'numpy.ndarray', Optional
        - If data_in is not a DataArray, can provide input coordinates as np array
        - Must be same length as data_in
        - If 2d, must be same length as data_in[0]

    lat_in : :class:'numpy.ndarray', Optional
        - If data_in is not a DataArray, can provide input coordinates as np array
        - Must be present if input is 2d
        - Must be same length as data_in

    icycx: :class:'np.number', Optional
        - Set as true if lon values are cyclical but do not fully wrap around the globe
         (0, 1.5, 3, ..., 354, 355.5)
        - If true, all inputs (data_in, lon_in, lat_in) must be monotonically in/decreasing
        - Default is false

    msg_py : :class:'np.number', Optional
        - Provide a number to represent missing data
        - Alternative to using np.nan

    assume_sorted: :class:'bool', Optional
        - Set as true if array is sorted
        - Else xarray.interp will assume it is unsorted and sort the values
        - Default is false

    method: :class:'str', Optional
        - Provide specific method of interpolation
        - Default is "linear"

    Returns
    -------
    data_in : :class:`numpy.ndarray`, :class:`xarray.DataArray`
       - Returns same data type as input data_in
       - Shape will be the same as input array except for last two dimensions which will
       be equal to len(lat_out) x len(lon_out)

    Examples
    --------
    >>> import xarray as xr
    >>> import pandas as pd
    >>> import numpy as np
    >>> import geocat.comp
    >>> da = xr.DataArray(data = [[1, 2, 3, 4, 5, 99], [2, 4, 6, 8, 10, 12]],
                        dims = ("lat", "lon"),
                        coords={"lat": [0, 1], "lon": [0, 50, 100, 250, 300, 350]},
                        )                      
    >>> do = interp_wrap(da, lon_out=[0, 50, 360], lat_out=[0, 1], icycx=1, msg_py=99)
    >>> print(do)
    <xarray.DataArray (lat: 2, lon: 3)>
    array([[ 1.,  2., 99.],
       [ 2.,  4., 99.]])
    Coordinates:
      * lat      (lat) int64 0 1
      * lon      (lon) int64 0 50 360

    See Also
    --------
    https://docs.xarray.dev/en/stable/generated/xarray.DataArray.interp.html
    https://scitools.org.uk/cartopy/docs/latest/reference/generated/cartopy.util.add_cyclic_point.html
    https://www.ncl.ucar.edu/Document/Functions/Built-in/linint1.shtml
    https://www.ncl.ucar.edu/Document/Functions/Built-in/linint2.shtml

    """

    is_input_xr = True
    is_two_d = False

    if(lat_out != None):
        is_two_d = True

    # If the input is numpy.ndarray, convert it to xarray.DataArray
    if not isinstance(data_in, xr.DataArray):
        is_input_xr = False

        if (lon_in is None):
            raise CoordinateError(
                "Argument lon_in must be provided explicitly unless data_in is an xarray.DataArray."
            )
        
        if is_two_d:
            if lat_in is None:    
                raise CoordinateError(
                    "Argument y_in must be provided explicitly unless data_in is an xarray.DataArray"
                )
            else:
                data_in = xr.DataArray(data_in,
                                    dims = ["x", "y"],
                                    coords={
                                        "x": lat_in,
                                        "y": lon_in
                                    })
        else:
            data_in = xr.DataArray(data_in,
                                dims = ["x"],
                                coords = {
                                    "x": lon_in
                                })
                                
    # if input is xarray.DataArray then rename dims so that it is accepted by xarray.interp
    else:
        og_dims = data_in.dims
        dim_t = list(data_in.dims)
        if is_two_d:
            dim_t[-1] = "y"
            dim_t[-2] = "x"
            coords = {
                "x": data_in.coords[data_in.dims[-2]].values,
                "y": data_in.coords[data_in.dims[-1]].values
            }
        else:
            dim_t[-1] = "x"
            coords = {
                "x": data_in.coords[data_in.dims[-1]].values
            }
        data_in = xr.DataArray(data_in.values,
                        dims=dim_t,
                        coords=coords)

    """ CHUNKING TBD

    # If input data is already chunked
    if data_in.chunks is not None:

        # Ensure rightmost dimension of input is not chunked
        if list(data_in.chunks)[-1:] != [lon_in.shape]:
            raise Exception(
                "DataArray data_in must be unchunked along the last dimension")

        # If 2d ensure last two dimensions are not chunked
        if is_two_d:
            if list(data_in.chunks)[-2:] != [lat_in.shape]:
                raise Exception(
                    "DataArray data_in must be unchunked along the last two dimensions"
                )
    else:
        # Generate chunks of {'dim_0': 1, 'dim_1': 1, ..., 'dim_n': xi.shape}
        in_chunks = list(data_in.dims)
        if is_two_d:
            last_dim = -2
        else:
            last_dim = -1
        in_chunks[:last_dim] = [
            (k, 1) for (k, v) in zip(list(data_in.dims)[:last_dim],
                                    list(data_in.shape)[:last_dim])
        ]
        in_chunks[last_dim:] = [
        (k, v) for (k, v) in zip(list(data_in.dims)[last_dim:],
                                    list(data_in.shape)[last_dim:])
        ]
        in_chunks = dict(in_chunks)
        data_in = data_in.chunk(in_chunks)
    
    generate output shape and chunks
    t_chunks = in_chunks
    t_shape = list(data_in.values.shape)
    t_shape[-1] = len(lon_out)+1
    t_chunks.update({data_in.dims[-1]: len(lon_out)+1})
    if is_two_d:
        t_shape[-2] = len(lat_out)
        t_chunks.update({data_in.dims[-2]: len(lat_out)})
    
    template = xr.DataArray(np.empty(t_shape),
                            dims = data_in.dims
                            ).chunk(t_chunks)

    
    # pre work to be done function call
    data_in.map_blocks(pre, 
                    kwargs={
                        "icycx": icycx,
                        "msg_py": msg_py
                    }
                    ).compute()
    
    """
    
    data_in = _pre(data_in, icycx, msg_py, is_two_d)

    # interpolate
    if is_two_d:
        data_in = data_in.interp(y=lon_out, x=lat_out, assume_sorted=assume_sorted, method=method)
    else:
        data_in = data_in.interp(x=lon_out, assume_sorted=assume_sorted, method=method)

    data_in = _post(data_in, msg_py=msg_py)
    
    """ CHUNKING TBD
    # post work to be done function call
    data_in.map_blocks(post, 
                    kwargs={
                        "msg_py": msg_py
                    }
                    ).compute()
    """
    
    # if input was numpy.ndarray return np array of data
    if not is_input_xr:
        return data_in.values

    # if input was xarray.DataArray rename dims to original and return
    else:
        if is_two_d:
            coords = {
                og_dims[-2]: data_in.coords[data_in.dims[-2]].values,
                og_dims[-1]: data_in.coords[data_in.dims[-1]].values
            }
        else:
            coords = {
                og_dims[-1]: data_in.coords[data_in.dims[-1]].values
            }
        data_in = xr.DataArray(data_in.values,
                        dims=og_dims,
                        coords=coords)
        return data_in
