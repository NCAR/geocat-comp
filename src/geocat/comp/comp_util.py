import copy

import numpy as np
import xarray as xr


# remove xarray has implemented _is_duck_array() OBE
def _is_duck_array(value):
    """Returns True when ``value`` is array-like."""
    if isinstance(value, np.ndarray):
        return True
    return (hasattr(value, "ndim") and hasattr(value, "shape") and
            hasattr(value, "dtype") and hasattr(value, "__array_function__") and
            hasattr(value, "__array_ufunc__"))


# todo nan_average supports median/mean/w/e for averages between two array


# todo xr to nc4 and back
def write_xarray_to_netcdf(data: xr.Dataset,
                           variable_name: str = 'variable') -> None:
    sdata = copy.deepcopy(data)
    sdata = sdata.rename(variable_name)
    sdata.to_netcdf(str(variable_name + '.nc4'))


def read_xarray_from_netcdf(file_name: str) -> xr.Dataset:
    if file_name[-4:] != '.nc4':
        file_name = str(file_name + '.nc4')
    return xr.load_dataset(file_name)
