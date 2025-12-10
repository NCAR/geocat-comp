from __future__ import annotations

import typing
import warnings

import metpy.interpolate
import numpy as np
import xarray as xr

supported_types = typing.Union[xr.DataArray, np.ndarray]

__pres_lev_mandatory__ = np.array(
    [
        1000,
        925,
        850,
        700,
        500,
        400,
        300,
        250,
        200,
        150,
        100,
        70,
        50,
        30,
        20,
        10,
        7,
        5,
        3,
        2,
        1,
    ]
).astype(np.float32)  # Mandatory pressure levels (mb)
__pres_lev_mandatory__ = __pres_lev_mandatory__ * 100.0  # Convert mb to Pa


def _func_interpolate(method='linear'):
    """Define interpolation function."""

    if method == 'linear':
        func_interpolate = metpy.interpolate.interpolate_1d
    elif method == 'log':
        func_interpolate = metpy.interpolate.log_interpolate_1d
    else:
        raise ValueError(
            f'Unknown interpolation method: {method}. '
            f'Supported methods are: "log" and "linear".'
        )

    return func_interpolate


def _interpolate_mb(data, curr_levels, new_levels, axis, method='linear'):
    """Wrapper to call interpolation function for xarray map_blocks call."""
    if method == 'linear':
        ext_func = metpy.interpolate.interpolate_1d
    elif method == 'log':
        ext_func = metpy.interpolate.log_interpolate_1d
    else:
        raise ValueError(
            f'Unknown interpolation method: {method}. '
            f'Supported methods are: "log" and "linear".'
        )
    return ext_func(new_levels, curr_levels, data, axis=axis)


def _pre_interp_multidim(
    data_in: xr.DataArray,
    cyclic: bool,
    missing_val,
):
    """Helper Function: Handling missing data functionality and adding cyclic
    point if required.

    Parameters
    ----------
    data_in : :class:`xarray.DataArray`
        The data on which to operate

    cyclic : :class:`bool`
        Determines if cyclic point should be added or not.
        If true then add point, else do nothing.

    missing_val : int, float, optional
        Provides an alternative to NaN

    Returns
    -------
    data_in : :class:`xarray.DataArray`
       The data input with cyclic points added (if cyclic is true)
       and missing_val values replaced with np.nan

    Notes
    -------
    """
    # replace missing_val with np.nan
    if missing_val is not None:
        data_in = xr.DataArray(
            np.where(data_in.values == missing_val, np.nan, data_in.values),
            dims=data_in.dims,
            coords=data_in.coords,
        )

    # add cyclic points and create new data array
    if cyclic:
        padded_data = np.pad(data_in.values, ((0, 0), (1, 1)), mode='wrap')
        padded_longitudes = np.pad(
            data_in.coords[data_in.dims[-1]], (1, 1), mode='wrap'
        )
        padded_longitudes[0] -= 360
        padded_longitudes[-1] += 360

        data_in = xr.DataArray(
            padded_data,
            coords={
                data_in.dims[-2]: data_in.coords[data_in.dims[-2]].values,
                data_in.dims[-1]: padded_longitudes,
            },
            dims=data_in.dims,
        )

    return data_in


def _post_interp_multidim(data_in, missing_val):
    """Helper Function: Handling missing data functionality.

    Parameters
    ----------
    data_in : :class:`xarray.DataArray`
        The data on which to operate

    missing_val : int, float, optional
         Provides an alternative to NaN

    Returns
    -------
    data_in : :class:`xarray.DataArray`
       The data input with np.nan values replaced with missing_val
    """
    if missing_val is not None:
        data_in = xr.DataArray(
            np.where(np.isnan(data_in.values), missing_val, data_in.values),
            dims=data_in.dims,
            coords=data_in.coords,
        )

    return data_in


def _sigma_from_hybrid(psfc, hya, hyb, p0=100000.0):
    """Calculate sigma at the hybrid levels."""

    # sig(k) = hya(k) * p0 / psfc + hyb(k)

    # This will be in Pa
    return hya * p0 / psfc + hyb


def _vertical_remap(func_interpolate, new_levels, xcoords, data, interp_axis=0):
    """Execute the defined interpolation function on data."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", r"Interpolation point out of data bounds encountered"
        )
        return func_interpolate(new_levels, xcoords, data, axis=interp_axis)


def _temp_extrapolate(t_bot, lev, p_sfc, ps, phi_sfc):
    r"""This helper function extrapolates temperature below ground using the
    ECMWF formulation described in `Vertical Interpolation and Truncation of
    Model-Coordinate Data <https://dx.doi.org/10.5065/D6HX19NH>`__ by Trenberth,
    Berry, & Buja [NCAR/TN-396, 1993]. Specifically equation 16 is used:

    .. math::
        T = T_* \left( 1 + \alpha ln \frac{p}{p_s} + \frac{1}{2}\left( \alpha ln \frac{p}{p_s} \right)^2 + \frac{1}{6} \left( \alpha ln \frac{p}{p_s} \right)^3 \right)

    Parameters
    ----------
    t_bot: :class:`xarray.DataArray`
        The temperature at the lowest level of the model.

    lev: int
        The pressure levels of interest. Must be in the same units as ``ps`` and ``p_sfc``

    p_sfc: :class:`xarray.DataArray`
        The pressure at the lowest level of the model. Must be in the same units as ``lev`` and ``ps``

    ps: :class:`xarray.DataArray`
        An array of surface pressures. Must be in the same units as ``lev`` and ``p_sfc``

    phi_sfc: :class:`xarray.DataArray`
        The geopotential at the lowest level of the model.

    Returns
    -------
    result: :class:`xarray.DataArray`
        The extrapolated temperatures at the provided pressure levels.
    """
    R_d = 287.04  # dry air gas constant
    g_inv = 1 / 9.80616  # inverse of gravity
    alpha = 0.0065 * R_d * g_inv

    tstar = t_bot * (1 + alpha * (ps / p_sfc - 1))
    hgt = phi_sfc * g_inv
    t0 = tstar + 0.0065 * hgt
    tplat = xr.apply_ufunc(np.minimum, 298, t0, dask='parallelized')

    tprime0 = xr.where(
        (2000 <= hgt) & (hgt <= 2500),
        0.002 * ((2500 - hgt) * t0 + ((hgt - 2000) * tplat)),
        np.nan,
    )
    tprime0 = xr.where(2500 < hgt, tplat, tprime0)

    alnp = xr.where(
        hgt < 2000,
        alpha * np.log(lev / ps),
        R_d * (tprime0 - tstar) / phi_sfc * np.log(lev / ps),
    )
    alnp = xr.where(tprime0 < tstar, 0, alnp)

    return tstar * (1 + alnp + (0.5 * (alnp**2)) + (1 / 6 * (alnp**3)))


def _geo_height_extrapolate(t_bot, lev, p_sfc, ps, phi_sfc):
    r"""This helper function extrapolates geopotential height below ground using
    the ECMWF formulation described in `Vertical Interpolation and Truncation
    of Model-Coordinate Data <https://dx.doi.org/10.5065/D6HX19NH>`__ by
    Trenberth, Berry, & Buja [NCAR/TN-396, 1993]. Specifically equation 15 is
    used:

    .. math::
        \Phi = \Phi_s - R_d T_* ln \frac{p}{p_s} \left[ 1 + \frac{1}{2}\alpha ln\frac{p}{p_s} + \frac{1}{6} \left( \alpha ln \frac{p}{p_s} \right)^2 \right]

    Parameters
    ----------
    t_bot: :class:`xarray.DataArray`
        Temperature at the lowest (bottom) level of the model.

    lev: int
        The pressure level of interest. Must be in the same units as ``ps`` and ``p_sfc``

    p_sfc: :class:`xarray.DataArray`
        The pressure at the lowest level of the model. Must be in the same units as ``lev`` and ``ps``

    ps : :class:`xarray.DataArray`
        An array of surface pressures. Must be in the same units as ``lev`` and ``p_sfc``

    phi_sfc:
        The geopotential at the lowest level of the model.

    Returns
    -------
    result: :class:`xarray.DataArray`
        The extrapolated geopotential height in geopotential meters at the provided pressure levels.
    """
    R_d = 287.04  # dry air gas constant
    g_inv = 1 / 9.80616  # inverse of gravity
    alpha = 0.0065 * R_d * g_inv

    tstar = t_bot * (1 + alpha * (ps / p_sfc - 1))
    hgt = phi_sfc * g_inv
    t0 = tstar + 0.0065 * hgt

    alph = xr.where(
        (tstar <= 290.5) & (t0 > 290.5), R_d / phi_sfc * (290.5 - tstar), alpha
    )

    alph = xr.where((tstar > 290.5) & (t0 > 290.5), 0, alph)
    tstar = xr.where((tstar > 290.5) & (t0 > 290.5), 0.5 * (290.5 + tstar), tstar)

    tstar = xr.where((tstar < 255), 0.5 * (tstar + 255), tstar)

    alnp = alph * np.log(lev / ps)
    return hgt - R_d * tstar * g_inv * np.log(lev / ps) * (
        1 + 0.5 * alnp + 1 / 6 * alnp**2
    )


def _vertical_remap_extrap(
    new_levels, lev_dim, data, output, pressure, ps, variable, t_bot, phi_sfc
):
    """A helper function to call the appropriate extrapolation function based
    on the user's inputs.

    Parameters
    ----------
    new_levels: array-like
        The desired pressure levels for extrapolation in Pascals.

    lev_dim: str
        The name of the vertical dimension.

    data: :class:`xarray.DataArray`
        The data to extrapolate

    output: :class:`xarray.DataArray`
        An array to hold the output data

    pressure: :class:`xarray.DataArray`
        The pressure at the lowest level of the model. Must be in the same units as ``lev`` and ``ps``

    ps : :class:`xarray.DataArray`
        An array of surface pressures. Must be in the same units as ``lev`` and ``p_sfc``

    variable : str, optional
        String representing what variable is extrapolated below surface level.
        Temperature extrapolation = "temperature". Geopotential height
        extrapolation = "geopotential". All other variables = "other". If
        "other", the value of ``data`` at the lowest model level will be used
        as the below ground fill value. Required if extrapolate is True.

    t_bot: :class:`xarray.DataArray`
        Temperature at the lowest (bottom) level of the model.

    phi_sfc:
        The geopotential at the lowest level of the model.

    Returns
    -------
    output: :class:`xarray.DataArray`
        A DataArray containing the data after extrapolation.
    """

    sfc_index = pressure[lev_dim].argmax(dim=lev_dim)  # index of the model surface
    p_sfc = pressure.isel(
        {lev_dim: sfc_index}, drop=True
    )  # extract pressure at lowest level

    if variable == 'temperature':
        output = output.where(
            output.plev <= p_sfc,
            _temp_extrapolate(t_bot, output.plev, p_sfc, ps, phi_sfc),
        )
    elif variable == 'geopotential':
        output = output.where(
            output.plev <= p_sfc,
            _geo_height_extrapolate(t_bot, output.plev, p_sfc, ps, phi_sfc),
        )
    else:
        output = output.where(
            output.plev <= p_sfc, data.isel({lev_dim: sfc_index}, drop=True)
        )

    return output


def pressure_at_hybrid_levels(psfc, hya, hyb, p0=100000.0):
    """Pressure at the hybrid levels.

    .. math::

        p(k) = hya(k) * p0 + hyb(k) * psfc

    Parameters
    ----------
    psfc: :class:`xarray.DataArray`
        A multi-dimensional array of surface pressures (Pa)

    hya: :class:`xarray.DataArray`
        A one-dimensional array of the hybrid A coefficients (unitless)

    hyb: :class:`xarray.DataArray`
        A one-dimensional array of the hybrid B coefficients (unitless)

    p0 : float, optional
        Scalar numeric value equal to surface reference pressure (Pa). Defaults to 100000 Pa.

    Returns
    -------
    output : :class:`xarray.DataArray`
        Computed pressure at the provided hybrid levels (Pa)

    See Also
    --------
    Related NCL Functions:
    `pres_hybrid_ccm <https://www.ncl.ucar.edu/Document/Functions/Built-in/pres_hybrid_ccm.shtml>`__
    """

    # make sure hya and hyb dims align and are both DataArrays
    if isinstance(hya, xr.DataArray) and isinstance(hyb, xr.DataArray):
        # if not, assign hya dim to hyb
        if hya.dims != hyb.dims:
            hyb = hyb.rename({hyb.dims[0]: hya.dims[0]})
    # check if only one is a DataArray, convert the other
    elif isinstance(hya, xr.DataArray) and not isinstance(hyb, xr.DataArray):
        hyb = xr.DataArray(hyb, dims=hya.dims)
    elif not isinstance(hya, xr.DataArray) and isinstance(hyb, xr.DataArray):
        hya = xr.DataArray(hya, dims=hyb.dims)
    # if both not DataArrays, convert both w/ dim 'lev'
    else:
        # convert to xarray for rest of checks and calculation
        hya = xr.DataArray(hya, dims={'lev'})
        hyb = xr.DataArray(hyb, dims={'lev'})

    # check shape
    if hya.shape != hyb.shape:
        raise ValueError(f'dimension mismatch: hya: {hya.shape} hyb: {hyb.shape}')
    # check 1D
    if len(hya.shape) > 1:
        raise ValueError(f'hya and hyb must be 1-dimensional: {hya.shape}')

    # Results in Pa
    # p(k) = hya(k) * p0 + hyb(k) * psfc
    return hya * p0 + hyb * psfc


def delta_pressure_hybrid(
    ps: xr.DataArray | np.ndarray,
    hya: xr.DataArray | np.ndarray,
    hyb: xr.DataArray | np.ndarray,
    p0: float = 100000.0,
) -> xr.DataArray | np.ndarray:
    """Calculates pressure layer thickness of a hybrid coordinate system

    Parameters
    ----------
    ps: :class:`xarray.DataArray`, :class:`numpy.ndarray`
        A multi-dimensional array of surface pressures (Pa)

    hya: :class:`xarray.DataArray`, :class:`numpy.ndarray`
        A one-dimensional array of the hybrid A coefficients (unitless)

    hyb: :class:`xarray.DataArray`, :class:`numpy.ndarray`
        A one-dimensional array of the hybrid B coefficients. Must be same type and length as ``hya`` (unitless)

    p0 : float, optional
        Scalar numeric value equal to surface reference pressure (Pa). Defaults to 100000 Pa.

    Returns
    -------
    output : :class:`xarray.DataArray`, :class:`numpy.ndarray`
        Computed pressure layer thicknesses. Will be same type as ``ps`` (Pa)

    See Also
    --------
    Related NCL Functions:
    `dpres_hybrid_ccm <https://www.ncl.ucar.edu/Document/Functions/Built-in/dpres_hybrid_ccm.shtml>`__
    """

    # save type of ps to use as output type later
    out_type = type(ps)

    # Validate inputs
    # type check
    if not {type(ps), type(hya), type(hyb)}.issubset({xr.DataArray, np.ndarray}):
        raise TypeError("Inputs must be xarray DataArrays or numpy arrays")
    if type(p0) not in {float, int}:
        raise TypeError(f"p0 must be a scalar numeric value, received {type(p0)}")
    if type(hya) is not type(hyb):
        raise TypeError(
            f"hya and hyb must be the same type. hya: {type(hya)} hyb: {type(hyb)}"
        )

    # check shape
    if hya.shape != hyb.shape:
        raise ValueError(f'dimension mismatch: hya: {hya.shape} hyb: {hyb.shape}')
    # check 1D
    if len(hya.shape) > 1:
        raise ValueError(f'hya and hyb must be 1-dimensional: {hya.shape}')

    # make sure working w/ all np or all xr
    if isinstance(ps, np.ndarray) and not isinstance(
        hya, np.ndarray
    ):  # not sure anybody would do this, but checking anyway
        hya = hya.values
        hyb = hyb.values
    elif isinstance(ps, xr.DataArray) and not isinstance(hya, xr.DataArray):
        hya = xr.DataArray(hya)
        hyb = xr.DataArray(hyb)

    # broadcast any np inputs or drop xarray lev dims
    if isinstance(hya, np.ndarray):  # hya and hyb already validated to be same type
        hya = np.expand_dims(hya, axis=(1, 2))
        hyb = np.expand_dims(hyb, axis=(1, 2))
    elif isinstance(hya, xr.DataArray):
        # drop lev coords for delta calculation, or xr will try to align
        if len(hya.coords) > 0:
            hya = hya.drop(list(hya.coords)[0])
        if len(hyb.coords) > 0:
            hyb = hyb.drop(list(hyb.coords)[0])
        # make sure hya and hyb dims align, favor hya dim name
        if set(hya.dims) != set(hyb.dims):
            hyb = hyb.rename({hyb.dims[0]: hya.dims[0]})

    pa = p0 * hya[:-1] + hyb[:-1] * ps
    pb = p0 * hya[1:] + hyb[1:] * ps

    dph = abs(pa - pb)

    # if output is xarray, set attributes
    if out_type == xr.DataArray and not isinstance(dph, xr.DataArray):
        # shouldn't happen, but convert to dataarray just in case
        dph = xr.DataArray(dph)
    if out_type == xr.DataArray:
        dph = dph.drop_attrs()
        dph.name = "dph"
        dph.attrs['long_name'] = "pressure layer thickness"
        dph.attrs['units'] = 'Pa'

    return dph


def interp_hybrid_to_pressure(
    data: xr.DataArray,
    ps: xr.DataArray,
    hyam: xr.DataArray,
    hybm: xr.DataArray,
    p0: float = 100000.0,
    new_levels: np.ndarray = __pres_lev_mandatory__,
    lev_dim: str = None,
    method: str = 'linear',
    extrapolate: bool = False,
    variable: str = None,
    t_bot: xr.DataArray = None,
    phi_sfc: xr.DataArray = None,
) -> xr.DataArray:
    """Interpolate and extrapolate data from hybrid-sigma levels to isobaric
    levels. Keeps attributes (i.e. metadata) of the input data in the output as
    default.

    Notes
    -----
    Atmosphere hybrid-sigma pressure coordinates are commonly defined in two different
    ways as described below and in CF Conventions. This particular function expects the
    first formulation. However, with some minor adjustments on the user side it can
    support datasets leveraging the second formulation as well.  In this case, you can
    set the input parameters p0=1 and hyam=ap to adapt the function to meet your needs.

    Formulation 1: p(n,k,j,i) = a(k)*p0 + b(k)*ps(n,j,i)
    Formulation 2: p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)

    ACKNOWLEDGEMENT: We'd like to thank to `Brian Medeiros <https://github.com/brianpm>`__,
    `Matthew Long <https://github.com/matt-long>`__, and `Deepak Cherian <https://github.com/dcherian>`__
    at NSF NCAR for their great contributions since the code implemented here is mostly
    based on their work.

    Parameters
    ----------
    data : :class:`xarray.DataArray`
        Multidimensional data array of hybrid-sigma levels and has a ``lev_dim`` coordinate.

    ps : :class:`xarray.DataArray`
        A multi-dimensional array of surface pressures (Pa), same time/space shape as data.

    hyam, hybm : :class:`xarray.DataArray`
        One-dimensional arrays containing the hybrid A and B coefficients. Must have the same
        dimension size as the ``lev_dim`` dimension of data.

    p0 : float, optional
        Scalar numeric value equal to surface reference pressure (Pa). Defaults to 100000 Pa.

    new_levels : ndarray, optional
        A one-dimensional array of output pressure levels (Pa). If not given, the mandatory
        list of 21 pressure levels is used.

    lev_dim : str, optional
        String that is the name of level dimension in data. Defaults to "lev".

    method : str, optional
        String that is the interpolation method; can be either "linear" or "log". Defaults to "linear".

    extrapolate : bool, optional
        If True, below ground extrapolation for ``variable`` will be done using
        an `ECMWF formulation <https://dx.doi.org/10.5065/D6HX19NH>`__. Defaults
        to False.

    variable : str, optional
        String representing what variable is extrapolated below surface level.
        Temperature extrapolation = "temperature". Geopotential height
        extrapolation = "geopotential". All other variables = "other". If
        "other", the value of ``data`` at the lowest model level will be used
        as the below ground fill value. Required if extrapolate is True.

    t_bot : :class:`xarray.DataArray`, optional
        Temperature in Kelvin at the lowest layer of the model. Not necessarily
        the same as surface temperature. Required if ``extrapolate`` is True
        and ``variable`` is not ``'other'``

    phi_sfc: :class:`xarray.DataArray`, optional
        Geopotential in J/kg at the lowest layer of the model. Not necessarily
        the same as surface geopotential. Required if ``extrapolate`` is True
        and ``variable`` is not ``'other'``.

    Returns
    -------
    output : :class:`xarray.DataArray`
        Interpolated data with isobaric levels

    See Also
    --------
    Related NCL Functions:
    `vinth2p <https://www.ncl.ucar.edu/Document/Functions/Built-in/vinth2p.shtml>`__,
    `vinth2p_ecmwf <https://www.ncl.ucar.edu/Document/Functions/Built-in/vinth2p_ecmwf.shtml>`__
    """

    # check input types
    in_types = []
    in_pint = False
    in_dask = False
    for i in [data, ps, hyam, hybm, new_levels]:
        it = type(i)
        in_types.append(it)

        if isinstance(i, xr.DataArray):
            if hasattr(i.data, '__module__'):
                if i.data.__module__ == 'pint':
                    in_pint = True
                if i.data.__module__ == 'dask.array.core':
                    in_dask = True

    # Check inputs
    if extrapolate and (variable is None):
        raise ValueError("If `extrapolate` is True, `variable` must be provided.")

    if variable in ['geopotential', 'temperature'] and (
        t_bot is None or phi_sfc is None
    ):
        raise ValueError(
            "If `variable` is 'geopotential' or 'temperature', both `t_bot` and `phi_sfc` must be provided"
        )

    if variable not in ['geopotential', 'temperature', 'other', None]:
        raise ValueError(
            "The value of `variable` is "
            + variable
            + ", but the accepted values are 'temperature', 'geopotential', 'other', or None."
        )

    # Determine the level dimension and then the interpolation axis
    if lev_dim is None:
        try:
            lev_dim = data.cf["vertical"].name
        except Exception:
            raise ValueError(
                "Unable to determine vertical dimension name. Please specify the name via `lev_dim` argument."
            )

    try:
        func_interpolate = _func_interpolate(method)
    except ValueError as vexc:
        raise ValueError(vexc.args[0])

    interp_axis = data.dims.index(lev_dim)

    # Calculate pressure levels at the hybrid levels
    pressure = pressure_at_hybrid_levels(ps, hyam, hybm, p0)  # Pa

    # Make pressure shape same as data shape
    pressure = pressure.transpose(*data.dims)

    # choose how to call function based on input types
    output = None
    # try xr.map_blocks first if chunked input
    if isinstance(data, xr.DataArray):
        # check for chunking along lev_dim in chunksizes
        if lev_dim in data.chunksizes:
            # check chunks along lev_dim
            if len(data.chunksizes[lev_dim]) == 1:
                # if there's not chunking in the lev dim, try to proceed with xr.map_blocks
                try:
                    output = xr.map_blocks(
                        _interpolate_mb,
                        data,
                        args=(pressure, new_levels, interp_axis, method),
                    )
                # The base Exception is included here because xarray can raise it specifically here
                except (NotImplementedError, ValueError, Exception):
                    # make sure output is None to trigger dask run
                    output = None
            else:
                # warn user about chunking in lev_dim
                warnings.warn(
                    f"WARNING: Chunking along {lev_dim} is not recommended for performance reasons."
                )

    # if xr.map_blocks won't work, but there's a dask input, use dask map_blocks
    if in_dask and output is None:
        from dask.array.core import map_blocks

        # Chunk pressure equal to data's chunks
        pressure = pressure.chunk(data.chunksizes)

        # Output data structure elements
        out_chunks = list(data.chunks)
        out_chunks[interp_axis] = (new_levels.size,)
        out_chunks = tuple(out_chunks)

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
    # if no chunked/dask inputs, just call the function directly
    else:
        output = func_interpolate(
            new_levels, pressure.data, data.data, axis=interp_axis
        )

    output = xr.DataArray(output, name=data.name, attrs=data.attrs)

    # Check if we've gotten a pint array back from metpy w/o pint in args
    if hasattr(output.data, '__module__'):
        if output.data.__module__ == 'pint' and not in_pint:
            output.data = output.data.to('pascal').magnitude

    # Set output dims and coords
    dims = [data.dims[i] if i != interp_axis else "plev" for i in range(data.ndim)]

    # Rename output dims. This is only needed with above workaround block
    dims_dict = {output.dims[i]: dims[i] for i in range(len(output.dims))}
    output = output.rename(dims_dict)

    coords = {}
    for k, v in data.coords.items():
        if k != lev_dim:
            coords.update({k: v})
        else:
            coords.update({"plev": new_levels})

    output = output.transpose(*dims).assign_coords(coords)

    if extrapolate:
        output = _vertical_remap_extrap(
            new_levels, lev_dim, data, output, pressure, ps, variable, t_bot, phi_sfc
        )

        # Check again if we got pint back
        if hasattr(output.data, '__module__'):
            if output.data.__module__ == 'pint' and not in_pint:
                output.data = output.data.to('pascal').magnitude

    return output


def interp_sigma_to_hybrid(
    data: xr.DataArray,
    sig_coords: xr.DataArray,
    ps: xr.DataArray,
    hyam: xr.DataArray,
    hybm: xr.DataArray,
    p0: float = 100000.0,
    lev_dim: str = None,
    method: str = 'linear',
) -> xr.DataArray:
    """Interpolate data from sigma to hybrid coordinates.  Keeps the attributes
    (i.e. meta information) of the input data in the output as default.

    Parameters
    ----------
    data : :class:`xarray.DataArray`
        Multidimensional data array, which holds sigma levels and has a ``lev_dim`` coordinate.

    sig_coords : :class:`xarray.DataArray`
        A one-dimensional array of sigma coordinates of ``lev_dim`` of ``data``.

    ps : :class:`xarray.DataArray`
        A multi-dimensional array of surface pressures (Pa), same time/space shape as data.

    hyam, hybm : :class:`xarray.DataArray`
        One-dimensional arrays containing the hybrid A and B coefficients. Must have the same
        dimension as the output hybrid levels.

    p0 : float, optional
        Scalar numeric value equal to surface reference pressure (Pa). Defaults to 100000 Pa.

    lev_dim : str, optional
        String that is the name of level dimension in data. Defaults to "lev".

    method : str, optional
        String that is the interpolation method; can be either "linear" or "log". Defaults to "linear".

    Returns
    -------
    output : :class:`xarray.DataArray`
        Interpolated data with hybrid levels

    See Also
    --------
    Related NCL Function:
    `sigma2hybrid <https://www.ncl.ucar.edu/Document/Functions/Built-in/sigma2hybrid.shtml>`__
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
    if data.ndim > 1:
        non_lev_dims.remove(lev_dim)
        data_stacked = data.stack(combined=non_lev_dims).transpose()
        sigma_stacked = sigma.stack(combined=non_lev_dims).transpose()

        h_coords = sigma_stacked[0, :].copy()

        output = data_stacked[:, : len(hyam)].copy()

        for idx, (d, s) in enumerate(zip(data_stacked, sigma_stacked)):
            output[idx, :] = xr.DataArray(
                _vertical_remap(func_interpolate, s.data, sig_coords.data, d.data),
                dims=[lev_dim],
            )

        # Make output shape same as data shape
        output = output.unstack().transpose(*data.dims)
    else:
        h_coords = sigma

        output = data[: len(hyam)].copy()
        output[: len(hyam)] = xr.DataArray(
            _vertical_remap(func_interpolate, sigma.data, sig_coords.data, data.data),
            dims=[lev_dim],
        )

    # Set output dims and coords
    output = output.rename({lev_dim: 'hlev'})
    output = output.assign_coords({"hlev": h_coords.data})

    return output


def interp_multidim(
    data_in: supported_types,
    lat_out: np.ndarray,
    lon_out: np.ndarray,
    lat_in: np.ndarray = None,
    lon_in: np.ndarray = None,
    cyclic: bool = False,
    missing_val: np.number = None,
    method: str = "linear",
    fill_value: typing.Union[str, np.number] = np.nan,
) -> supported_types:
    """Multidimensional interpolation of variables. Uses ``xarray.interp`` to
    perform interpolation. Will not perform extrapolation by default, returns
    missing values if any surrounding points contain missing values.

    .. warning::
        The output data type may be promoted to that of the coordinate data.

    Parameters
    ----------
    data_in : :class:`xarray.DataArray`, ndarray
        Data array with data to be interpolated and associated coords. If
        it is a np array, then ``lat_in`` and ``lon_in`` must be provided. Length must
        be coordinated with given coordinates.

    lat_out: ndarray
        List of latitude coordinates to be interpolated to.

    lon_out: ndarray
        List of longitude coordinates to be interpolated to.

    lat_in: ndarray
        List of latitude coordinates corresponding to ``data_in``. Must be
        given if ``data_in`` is not an xarray.

    lon_in: ndarray
        List of longitude coordinates corresponding to ``data_in``. Must be
        given if ``data_in`` is not an xarray.

    cyclic: bool, optional
        Set as true if lon values are cyclical but do not fully wrap around
        the globe
        (0, 1.5, 3, ..., 354, 355.5) Default is false

    missing_val : :class:`np.number`, optional
        Provide a number to represent missing data. Alternative to using ``np.nan``

    method: str, optional
        Provide specific method of interpolation. Default is "linear"
        “linear” or “nearest” for multidimensional array

    fill_value: str, optional
        Set as 'extrapolate' to allow extrapolation of data. Default is
        no extrapolation.

    Returns
    -------
    data_out : ndarray, :class:`xarray.DataArray`
       Returns the same type of object as input ``data_in``. However, the type of
       the data in the array may be promoted to that of the coordinates. Shape
       will be the same as input array except for last two dimensions which will
       be equal to the coordinates given in ``data_out``.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> import geocat.comp
    >>> data = np.asarray([[1, 2, 3, 4, 5, 99],
    ...                   [2, 4, 6, 8, 10, 12]])
    >>> lat_in = [0, 1]
    >>> lon_in = [0, 50, 100, 250, 300, 350]
    >>> data_in = xr.DataArray(data,
    ...                        dims=['lat', 'lon'],
    ...                        coords={'lat':lat_in,
    ...                                'lon': lon_in})
    >>> data_out = xr.DataArray(dims=['lat', 'lon'],
    ...                         coords={'lat': [0, 1],
    ...                                 'lon': [0, 50, 360]})
    >>> do = interp_multidim(data_in,
    ...                      [0, 1],
    ...                      [0, 50, 360],
    ...                      cyclic=True,
    ...                      missing_val=99)
    >>> print(do)
    <xarray.DataArray (lat: 2, lon: 3)>
    array([[ 1.,  2., 99.],
       [ 2.,  4., 99.]])
    Coordinates:
      * lat      (lat) int64 0 1
      * lon      (lon) int64 0 50 360

    See Also
    --------
    Related External Functions:
    `xarray.DataArray.interp <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.interp.html>`__,
    `cartopy.util.add_cyclic_point <https://cartopy.readthedocs.io/stable/reference/generated/cartopy.util.add_cyclic_point.html#cartopy.util.add_cyclic_point>`__

    Related NCL Function:
    `NCL linint2 <https://www.ncl.ucar.edu/Document/Functions/Built-in/linint2.shtml>`__
    """
    # check for xarray/numpy
    if not isinstance(data_in, xr.DataArray):
        if lat_in is None or lon_in is None:
            raise ValueError(
                "Argument lat_in and lon_in must be provided if data_in is not an xarray"
            )
        data_in = xr.DataArray(
            data_in, dims=['lat', 'lon'], coords={'lat': lat_in, 'lon': lon_in}
        )

    output_coords = {
        data_in.dims[-1]: lon_out,
        data_in.dims[-2]: lat_out,
    }

    data_in_modified = _pre_interp_multidim(data_in, cyclic, missing_val)
    data_out = data_in_modified.interp(
        output_coords, method=method, kwargs={'fill_value': fill_value}
    )
    data_out_modified = _post_interp_multidim(data_out, missing_val=missing_val)

    return data_out_modified
