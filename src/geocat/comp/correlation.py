import numpy as np
import xskillscore as xs
import xskillscore.core.np_deterministic as xs_internal
import xarray as xr
import warnings

def pearson_r(a, b, dim=None, weights=None, skipna=False, keep_attrs=False, axis=None):
    if not isinstance(a, xr.DataArray) and not isinstance(b, xr.DataArray):  # if a and v are not xr.DataArrays
        if (dim is not None) and (axis is not None):
            warnings.warn("The `dim` keyword is unused with non xarray.DataArray inputs")
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
            warnings.warn("The `axis` keyword is unused with xarray.DataArray inputs")
        try:
            return xs.pearson_r(a, b, dim, weights, skipna, keep_attrs)
        except ValueError:
            print('Data along `dim` must have the same dimension as `weights`')