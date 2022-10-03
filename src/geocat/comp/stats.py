import numpy as np
import xskillscore as xs
import xskillscore.core.np_deterministic as xs_internal
import xarray as xr
import warnings


def pearson_r(a,
              b,
              dim=None,
              weights=None,
              skipna=False,
              keep_attrs=False,
              axis=None):
    """This function wraps the function of the same name from `xskillscore <htt
    ps://xskillscore.readthedocs.io/en/stable/api/xskillscore.pearson_r.html#xs
    killscore.pearson_r>`_. The difference between the xskillscore version and
    this one is that the GeoCAT version allows for array-like inputs rather
    than only supporting `xarray.DataArrays`. The parameters work the same way
    as in the xskillscore version, with an added parameter `axis` (see below).

    Parameters
    ----------
    a : :class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`numpy.ndarray`
        Arrays over which to apply the function.
    b : :class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`numpy.ndarray`
        Arrays over which to apply the function.
    dim : :class:`str`, :class:`list`, Optional
        The dimension(s) to apply the correlation along. Note that this dimension will be reduced as a result. Defaults to None reducing all dimensions. Only used when `a` and `b` are of type `xarray.DataArray` or `xarray.Dataset`.
    weights : :class:`xarray.DataArray`, :class:`numpy.ndarray`, Optional
        Weights matching dimensions of `dim` to apply during the function.
    skipna : :class:`bool`, Optional
        If True, skip NaNs when computing function.
    keep_attrs : :class:`bool`, Optional
        If True, the attributes (attrs) will be copied from the first input to the new one. If False (default), the new object will be returned without attributes. Only used when `a` and `b` are of types `xarray.Dataset` or `xarray.DataArray`.
    axis : :class:`int`
        The axis to apply the correlation along. Only used when `a` and `b` are of type `np.ndarray` or are array-like.

    Returns
    -------
    r : :class:`xarray.DataArray`, :class:`numpy.ndarray`
        Pearson's correlation coefficient
    """
    # if a and v are not xr.DataArrays
    if not isinstance(a, xr.DataArray) and not isinstance(b, xr.DataArray):
        if (dim is not None) and (axis is not None):
            warnings.warn(
                "The `dim` keyword is unused with non xarray.DataArray inputs")
        if axis is None:  # squash array to 1D for element wise computation
            axis = 0
            a = np.ravel(a)
            b = np.ravel(b)
            if weights is not None:
                weights = np.ravel(weights)
        try:
            return xs_internal._pearson_r(a, b, weights, axis, skipna)
        except ValueError:
            print('Data along `axis` must have the same dimension as `weights`')
    else:  # if a and b are xr.DataArrays
        if (dim is not None) and (axis is not None):
            warnings.warn(
                "The `axis` keyword is unused with xarray.DataArray inputs")
        try:
            return xs.pearson_r(a, b, dim, weights, skipna, keep_attrs)
        except ValueError:
            print('Data along `dim` must have the same dimension as `weights`')
