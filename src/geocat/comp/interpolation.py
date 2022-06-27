import typing
from .errors import ChunkError, CoordinateError

import cf_xarray
import metpy.interpolate
import numpy as np
import xarray as xr

supported_types = typing.Union[xr.DataArray, np.ndarray]

__pres_lev_mandatory__ = np.array([
    1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10,
    7, 5, 3, 2, 1
]).astype(np.float32)  # Mandatory pressure levels (mb)
__pres_lev_mandatory__ = __pres_lev_mandatory__ * 100.0  # Convert mb to Pa


def _func_interpolate(method='linear'):
    """Define interpolation function."""

    if method == 'linear':
        func_interpolate = metpy.interpolate.interpolate_1d
    elif method == 'log':
        func_interpolate = metpy.interpolate.log_interpolate_1d
    else:
        raise ValueError(f'Unknown interpolation method: {method}. '
                         f'Supported methods are: "log" and "linear".')

    return func_interpolate


def _pressure_from_hybrid(psfc, hya, hyb, p0=100000.):
    """Calculate pressure at the hybrid levels."""

    # p(k) = hya(k) * p0 + hyb(k) * psfc

    # This will be in Pa
    return hya * p0 + hyb * psfc


def _sigma_from_hybrid(psfc, hya, hyb, p0=100000.):
    """Calculate sigma at the hybrid levels."""

    # sig(k) = hya(k) * p0 / psfc + hyb(k)

    # This will be in Pa
    return hya * p0 / psfc + hyb


def _vertical_remap(func_interpolate, new_levels, xcoords, data, interp_axis=0):
    """Execute the defined interpolation function on data."""

    return func_interpolate(new_levels, xcoords, data, axis=interp_axis)


def interp_hybrid_to_pressure(data: xr.DataArray,
                              ps: xr.DataArray,
                              hyam: xr.DataArray,
                              hybm: xr.DataArray,
                              p0: float = 100000.,
                              new_levels: np.ndarray = __pres_lev_mandatory__,
                              lev_dim: str = None,
                              method: str = 'linear') -> xr.DataArray:
    """Interpolate data from hybrid-sigma levels to isobaric levels. Keeps
    attributes (i.e. meta information) of the input data in the output as
    default.

    Notes
    -----
    ACKNOWLEDGEMENT: We'd like to thank to [Brian Medeiros](https://github.com/brianpm),
    [Matthew Long](https://github.com/matt-long), and [Deepak Cherian](https://github.com/dcherian)
    at NCAR for their great contributions since the code implemented here is mostly
    based on their work.

    Parameters
    ----------
    data : :class:`xarray.DataArray`
        Multidimensional data array, which holds hybrid-sigma levels and has a `lev_dim` coordinate.

    ps : :class:`xarray.DataArray`
        A multi-dimensional array of surface pressures (Pa), same time/space shape as data.

    hyam, hybm : :class:`xarray.DataArray`
        One-dimensional arrays containing the hybrid A and B coefficients. Must have the same
        dimension size as the `lev_dim` dimension of data.

    p0 : :class:`float`, Optional
        Scalar numeric value equal to surface reference pressure (Pa). Defaults to 100000 Pa.

    new_levels : :class:`numpy.ndarray`, Optional
        A one-dimensional array of output pressure levels (Pa). If not given, the mandatory
        list of 21 pressure levels is used.

    lev_dim : :class:`str`, Optional
        String that is the name of level dimension in data. Defaults to "lev".

    method : :class:`str`, Optional
        String that is the interpolation method; can be either "linear" or "log". Defaults to "linear".

    Returns
    -------
    output : :class:`xarray.DataArray`
        Interpolated data with isobaric levels

    See Also
    --------
    Related NCL Functions:
    `vinth2p <https://www.ncl.ucar.edu/Document/Functions/Built-in/vinth2p.shtml>`_,
    `vinth2p_ecmwf <https://www.ncl.ucar.edu/Document/Functions/Built-in/vinth2p_ecmwf.shtml>`_
    """

    # Determine the level dimension and then the interpolation axis
    if lev_dim is None:
        try:
            lev_dim = data.cf["vertical"].name
        except Exception:
            raise ValueError(
                "Unable to determine vertical dimension name. Please specify the name via `lev_dim` argument.'"
            )

    try:
        func_interpolate = _func_interpolate(method)
    except ValueError as vexc:
        raise ValueError(vexc.args[0])

    interp_axis = data.dims.index(lev_dim)

    # Calculate pressure levels at the hybrid levels
    pressure = _pressure_from_hybrid(ps, hyam, hybm, p0)  # Pa

    # Make pressure shape same as data shape
    pressure = pressure.transpose(*data.dims)

    ###############################################################################
    # Workaround
    #
    # For the issue with metpy's xarray interface:
    #
    # `metpy.interpolate.interpolate_1d` had "no implementation found for
    # 'numpy.apply_along_axis'" issue for cases where the input is
    # xarray.Dataarray and has more than 3 dimensions (e.g. 4th dim of `time`).

    # Use dask.array.core.map_blocks instead of xarray.apply_ufunc and
    # auto-chunk input arrays to ensure using only Numpy interface of
    # `metpy.interpolate.interpolate_1d`.

    # # Apply vertical interpolation
    # # Apply Dask parallelization with xarray.apply_ufunc
    # output = xr.apply_ufunc(
    #     _vertical_remap,
    #     data,
    #     pressure,
    #     exclude_dims=set((lev_dim,)),  # Set dimensions allowed to change size
    #     input_core_dims=[[lev_dim], [lev_dim]],  # Set core dimensions
    #     output_core_dims=[["plev"]],  # Specify output dimensions
    #     vectorize=True,  # loop over non-core dims
    #     dask="parallelized",  # Dask parallelization
    #     output_dtypes=[data.dtype],
    #     dask_gufunc_kwargs={"output_sizes": {
    #         "plev": len(new_levels)
    #     }},
    # )

    # If an unchunked Xarray input is given, chunk it just with its dims
    if data.chunks is None:
        data_chunk = dict([
            (k, v) for (k, v) in zip(list(data.dims), list(data.shape))
        ])
        data = data.chunk(data_chunk)

    # Chunk pressure equal to data's chunks
    pressure = pressure.chunk(data.chunks)

    # Output data structure elements
    out_chunks = list(data.chunks)
    out_chunks[interp_axis] = (new_levels.size,)
    out_chunks = tuple(out_chunks)
    # ''' end of boilerplate

    from dask.array.core import map_blocks
    output = map_blocks(
        _vertical_remap,
        func_interpolate,
        new_levels,
        pressure.data,
        data.data,
        interp_axis,
        chunks=out_chunks,
        dtype=data.dtype,
        drop_axis=[interp_axis],
        new_axis=[interp_axis],
    )

    # End of Workaround
    ###############################################################################

    output = xr.DataArray(output, name=data.name, attrs=data.attrs)

    # Set output dims and coords
    dims = [
        data.dims[i] if i != interp_axis else "plev" for i in range(data.ndim)
    ]

    # Rename output dims. This is only needed with above workaround block
    dims_dict = {output.dims[i]: dims[i] for i in range(len(output.dims))}
    output = output.rename(dims_dict)

    coords = {}
    for (k, v) in data.coords.items():
        if k != lev_dim:
            coords.update({k: v})
        else:
            coords.update({"plev": new_levels})

    output = output.transpose(*dims).assign_coords(coords)

    return output


def interp_sigma_to_hybrid(data: xr.DataArray,
                           sig_coords: xr.DataArray,
                           ps: xr.DataArray,
                           hyam: xr.DataArray,
                           hybm: xr.DataArray,
                           p0: float = 100000.,
                           lev_dim: str = None,
                           method: str = 'linear') -> xr.DataArray:
    """Interpolate data from sigma to hybrid coordinates.  Keeps attributes
    (i.e. meta information) of the input data in the output as default.

    Parameters
    ----------
    data : :class:`xarray.DataArray`
        Multidimensional data array, which holds sigma levels and has a `lev_dim` coordinate.

    sig_coords : :class:`xarray.DataArray`
        A one-dimensional array of sigma coordinates of `lev_dim` of `data`.

    ps : :class:`xarray.DataArray`
        A multi-dimensional array of surface pressures (Pa), same time/space shape as data.

    hyam, hybm : :class:`xarray.DataArray`
        One-dimensional arrays containing the hybrid A and B coefficients. Must have the same
        dimension as the output hybrid levels.

    p0 : :class:`float`, Optional
        Scalar numeric value equal to surface reference pressure (Pa). Defaults to 100000 Pa.

    lev_dim : :class:`str`, Optional
        String that is the name of level dimension in data. Defaults to "lev".

    method : :class:`str`, Optional
        String that is the interpolation method; can be either "linear" or "log". Defaults to "linear".

    Returns
    -------
    output : :class:`xarray.DataArray`
        Interpolated data with hybrid levels

    See Also
    --------
    Related NCL Function:
    `sigma2hybrid <https://www.ncl.ucar.edu/Document/Functions/Built-in/sigma2hybrid.shtml>`_
    """

    # Determine the level dimension and then the interpolation axis
    if lev_dim is None:
        try:
            lev_dim = data.cf["vertical"].name
        except Exception:
            raise ValueError(
                "Unable to determine vertical dimension name. Please specify the name via `lev_dim` argument.'"
            )

    try:
        func_interpolate = _func_interpolate(method)
    except ValueError as vexc:
        raise ValueError(vexc.args[0])

    # Calculate sigma levels at the hybrid levels
    sigma = _sigma_from_hybrid(ps, hyam, hybm, p0)  # Pa

    non_lev_dims = list(data.dims)
    if (data.ndim > 1):
        non_lev_dims.remove(lev_dim)
        data_stacked = data.stack(combined=non_lev_dims).transpose()
        sigma_stacked = sigma.stack(combined=non_lev_dims).transpose()

        h_coords = sigma_stacked[0, :].copy()

        output = data_stacked[:, :len(hyam)].copy()

        for idx, (d, s) in enumerate(zip(data_stacked, sigma_stacked)):
            output[idx, :] = xr.DataArray(
                _vertical_remap(func_interpolate, s.data, sig_coords.data,
                                d.data))

        # Make output shape same as data shape
        output = output.unstack().transpose(*data.dims)
    else:
        h_coords = sigma

        output = data[:len(hyam)].copy()
        output[:len(hyam)] = xr.DataArray(
            _vertical_remap(func_interpolate, sigma.data, sig_coords.data,
                            data.data))

    # Set output dims and coords
    output = output.rename({lev_dim: 'hlev'})
    output = output.assign_coords({"hlev": h_coords.data})

    return output


def _pre(data_in, cyclic, missing_val, is_2D_coords):
    """Helper Function: Handling missing data functionality and adding cyclic
    point if required.

    Parameters
    ----------
    data_in : :class:`xarray.DataArray`
        The data on which to operate

    cyclic : :class:`bool`
        Determines if cyclic point should be added or not.
        If true then add point, else do nothing.

    missing_val : :class:`int`, :class:`float`, Optional
        Provides an alternative to NaN

    is_2D_coords : :class:'bool'
        Informs if interpolation is 1-dimensional or 2-dimensional

    Returns
    -------
    data_in : :class:`xarray.DataArray`
       The data input with cyclic points added (if icycx is true)
       and missing_val values replaced with np.nan

    Notes
    -------
    Adding cyclic point adapted from cartopy.add_cyclic_point - https://scitools.org.uk/cartopy/docs/latest/reference/generated/cartopy.util.add_cyclic_point.html
    """
    # replace msg_py with np.nan
    if missing_val is not None:
        data_in = xr.DataArray(np.where(data_in.values == missing_val, np.nan,
                                        data_in.values),
                               dims=data_in.dims,
                               coords=data_in.coords)

    # add cyclic points and create new data array
    if cyclic:
        lon = data_in.coords[data_in.dims[-1]].values
        delta_coord = np.diff(lon)
        new_coord = np.append(lon, lon[-1] + delta_coord[0])
        if is_2D_coords:
            temp = np.pad(data_in.values, ((0, 0), (
                0,
                1,
            )), "wrap")
        else:
            temp = np.pad(data_in.values, (0, 1), "wrap")
        temp = xr.DataArray(temp, dims=data_in.dims)
        if not is_2D_coords:
            data_in = temp.assign_coords({data_in.dims[-1]: new_coord})
        if is_2D_coords:
            data_in = temp.assign_coords(
                {data_in.dims[-2]: data_in.coords[data_in.dims[-2]].values})
            data_in = data_in.assign_coords({data_in.dims[-1]: new_coord})

    return data_in


def _post(data_in, missing_val):
    """Helper Function: Handling missing data functionality.

    Parameters
    ----------
    data_in : :class:`xarray.DataArray`
        The data on which to operate

    missing_val : :class:`int`, :class:`float`, Optional
         Provides an alternative to NaN

    Returns
    -------
    data_in : :class:`xarray.DataArray`
       The data input with np.nan values replaced with missing_val
    """
    if missing_val is not None:
        x = np.where(data_in.values == np.nan, missing_val, data_in.values)
        data_in = xr.DataArray(np.where(np.isnan(data_in.values), missing_val,
                                        data_in.values),
                               dims=data_in.dims,
                               coords=data_in.coords)

    return data_in


def interp_wrap(data_in: supported_types,
                lon_out: supported_types,
                lat_out: supported_types = None,
                lon_in: supported_types = None,
                lat_in: supported_types = None,
                cyclic: bool = False,
                missing_val: [np.float, np.int] = None,
                assume_sorted: bool = False,
                method: str = "linear") -> supported_types:
    """Multidimensional interpolation of variables.

    Parameters
    ----------
    data_in : :class:`xarray.DataArray`, :class:`numpy.ndarray`
        1 or multi-dimensional array containing values to be interpolated. If xarray.DataArray
        provided, must have associated coords. If 1d interpolation, last dimension must be
        same length as lon_in or associated coord. If 2d interpolation, last dimension must
        be same length as lon_in or associated coord and second to last must be same length
        as lat_in or associated coord

    lon_out : :class:`numpy.ndarray`
        1D array containing coordinate values for output data. Will perform 1D or 2D
        interpolation depending on presence of lat_out. Does not perform extrapolation,
        returns missing values if any surrounding points contain missing values.

    lat_out : :class:'numpy.ndarray', Optional
        1D array contianing coordinate values for output data. Must be present if input data is 2D
        Will not perform extrapolation, returns missing values if any surrounding points
        contain missing values

    lon_in : :class:'numpy.ndarray', Optional
        If data_in is not a DataArray, can provide input coordinates as np array. Must be same
        length as data_in. If 2D, must be same length as data_in[0].

    lat_in : :class:'numpy.ndarray', Optional
        If data_in is not a DataArray, can provide input coordinates as np array. Must be
        present if input is 2D. Must be same length as data_in.

    cyclic: :class:'np.number', Optional
        Set as true if lon values are cyclical but do not fully wrap around the globe
         (0, 1.5, 3, ..., 354, 355.5) If true, all inputs (data_in, lon_in, lat_in)
         must be monotonically in/decreasing. Default is false

    missing_val : :class:'np.number', Optional
        Provide a number to represent missing data. Alternative to using np.nan

    assume_sorted: :class:'bool', Optional
        Set as true if array is sorted. Else xarray.interp will assume it is unsorted
        and sort the values. Default is false

    method: :class:'str', Optional
        Provide specific method of interpolation. Default is "linear"
        “linear” or “nearest” for multidimensional array,
        “linear”, “nearest”, “zero”, “slinear”, “quadratic”, “cubic” for 1-dimensional array.

    Returns
    -------
    data_in : :class:`numpy.ndarray`, :class:`xarray.DataArray`
       Returns same data type as input data_in. Shape will be the same as input array except
       for last two dimensions which will be equal to len(lat_out) x len(lon_out)

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
    is_2D_coords = False

    if lat_out is not None:
        is_2D_coords = True

    # If the input is numpy.ndarray, convert it to xarray.DataArray
    if not isinstance(data_in, xr.DataArray):
        is_input_xr = False

        if lon_in is None:
            raise CoordinateError(
                "Argument lon_in must be provided explicitly unless data_in is an xarray.DataArray."
            )

        if is_2D_coords:
            if lat_in is None:
                raise CoordinateError(
                    "Argument lat_in must be provided explicitly unless data_in is an xarray.DataArray"
                )
            else:
                data_in = xr.DataArray(data_in,
                                       dims=["lat", "lon"],
                                       coords={
                                           "lat": lat_in,
                                           "lon": lon_in
                                       })
        else:
            data_in = xr.DataArray(data_in,
                                   dims=["lon"],
                                   coords={"lon": lon_in})

    if data_in.chunks is not None:

        # Ensure rightmost dimension of input is not chunked
        if list(data_in.chunks)[-1:] != [lon_in.shape]:
            raise ChunkError(
                "DataArray data_in must be unchunked along the last dimension")

        # If 2d ensure last two dimensions are not chunked
        if is_2D_coords:
            if list(data_in.chunks)[-2:] != [lat_in.shape]:
                raise ChunkError(
                    "DataArray data_in must be unchunked along the last two dimensions"
                )

    data_in = _pre(data_in, cyclic, missing_val, is_2D_coords)

    # interpolate
    if is_2D_coords:
        coords = {data_in.dims[-1]: lon_out, data_in.dims[-2]: lat_out}
    else:
        coords = {data_in.dims[-1]: lon_out}

    data_in = data_in.interp(coords, assume_sorted=assume_sorted, method=method)

    data_in = _post(data_in, missing_val=missing_val)

    # if input was numpy.ndarray return np array of data
    if not is_input_xr:
        return data_in.values

    return data_in
