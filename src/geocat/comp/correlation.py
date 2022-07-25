import xskillscore as xs
import xskillscore.core.np_deterministic as xs_internal
import xarray as xr

def pearson_r(a, b, dim=None, weights=None, skipna=False, keep_attrs=False, axis=None):
    if not isinstance(a, xr.DataArray) and not isinstance(b, xr.DataArray):
        return xs_internal._pearson_r(a, b, weights, -1, skipna)
    return xs.pearson_r(a, b, dim, weights, skipna, keep_attrs)