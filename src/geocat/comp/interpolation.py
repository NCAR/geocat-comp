import cf_xarray
import metpy.interpolate
import numpy as np
import xarray as xr

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
                _vertical_remap(func_interpolate, s, sig_coords, d))

        # Make output shape same as data shape
        output = output.unstack().transpose(*data.dims)
    else:
        h_coords = sigma

        output = data[:len(hyam)].copy()
        output[:len(hyam)] = xr.DataArray(
            _vertical_remap(func_interpolate, sigma, sig_coords, data))

    # Set output dims and coords
    output = output.rename({lev_dim: 'hlev'})
    output = output.assign_coords({"hlev": h_coords.data})

    return output
