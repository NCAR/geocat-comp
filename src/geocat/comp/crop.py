import numpy as np
import xarray as xr


def max_daylight(jday, lat):
    """Computes maximum number of daylight hours as describe in the Food and
     Agriculture Organization (FAO) Irrigation and Drainage Paper 56 entitled:
     Crop evapotranspiration - Guidelines for computing crop water
     requirement. Specifically, see equation 34 of Chapter 3.

    Note for abs(lat) > 55 the eqns have limited validity.

    Parameters
    ----------
    jday : numpy.ndarray, xr.DataArray, list, float
        Day of year

    lat : numpy.ndarray, xr.DataArray, list, float
        Latitude in degrees

    Returns
    -------
    sunmax : numpy.ndarray, xr.DataArray, float
        Calculated maximum sunlight
    """

    x_out = False
    if isinstance(jday, xr.DataArray):
        x_out = True
        save_dims = jday.dims
        save_coords = jday.coords
        save_attrs = jday.attrs

    # convert inputs to numpy arrays for function call
    jday = np.asarray(jday)
