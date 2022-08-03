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
    a : :class:`xarray.DataArray`, :class:`numpy.ndarray`
        See xskillscore `documentation <https://xskillscore.readthedocs.io/en/stable/api/xskillscore.pearson_r.html#xskillscore.pearson_r>`_.
    b : :class:`xarray.DataArray`, :class:`numpy.ndarray`
        See xskillscore `documentation <https://xskillscore.readthedocs.io/en/stable/api/xskillscore.pearson_r.html#xskillscore.pearson_r>`_.
    dim : :class:`str`, :class:`list`, Optional
        Used when `a` and `b` are of type `xarray.DataArray`. Is ignored otherwise.
        See xskillscore `documentation <https://xskillscore.readthedocs.io/en/stable/api/xskillscore.pearson_r.html#xskillscore.pearson_r>`_.
    weights : :class:`xarray.DataArray`, :class:`numpy.ndarray`, Optional
        Must have the same dimensions as `dim`.
        See xskillscore `documentation <https://xskillscore.readthedocs.io/en/stable/api/xskillscore.pearson_r.html#xskillscore.pearson_r>`_.
    skipna : :class:`bool`, Optional
        See xskillscore `documentation <https://xskillscore.readthedocs.io/en/stable/api/xskillscore.pearson_r.html#xskillscore.pearson_r>`_.
    keep_attrs : :class:`bool`, Optional
        See xskillscore `documentation <https://xskillscore.readthedocs.io/en/stable/api/xskillscore.pearson_r.html#xskillscore.pearson_r>`_.
    axis : :class:`int`
        The axis to apply the correlation along. Used when `a` and `b` are of type `np.ndarray` or are array-like. Is ignored otherwise.

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
