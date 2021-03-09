import numpy as np
import xarray as xr
import warnings


def max_daylight(jday, lat):
    """Computes maximum number of daylight hours as describe in the Food and
     Agriculture Organization (FAO) Irrigation and Drainage Paper 56 entitled:
     Crop evapotranspiration - Guidelines for computing crop water
     requirement. Specifically, see equation 34 of Chapter 3.

    Note for abs(lat) > 55 the eqns have limited validity.

    Parameters
    ----------
    jday : numpy.ndarray, xr.DataArray, list, float
        Day of year. Must be 1D

    lat : numpy.ndarray, xr.DataArray, list, float
        Latitude in degrees. Must be 1D

    Returns
    -------
    sunmax : numpy.ndarray, xr.DataArray, float
        Calculated maximum sunlight in hours/day
    """

    x_out = False
    if isinstance(jday, xr.DataArray):
        x_out = True
        save_dims = jday.dims
        save_coords = jday.coords
        save_attrs = jday.attrs

    # convert inputs to numpy arrays for function call
    jday = np.asarray(jday)
    lat = np.asarray(lat)

    # check to ensure dimension of lat is not greater than two
    if lat.ndim > 1 or jday.ndim > 1:
        raise ValueError('max_daylight: inputs must have at most one dimension')

    # check if latitude is outside of acceptable ranges
    # warn if more than abs(55), error if more than abs(66)
    if (abs(lat) > 55).all():
        warnings.warn("WARNING: max_daylight has limited validity for abs(lat) > 55 ")
    elif (abs(lat) > 66).all():
        warnings.warn('WARNING: max_daylight: calculation not possible for abs(lat) > 66 for all values of jday, '
                      'errors may occur')

    # define constants
    pi = np.pi
    rad = pi / 180
    pi2yr = 2 * pi / 365
    latrad = lat * rad
    con = 24 / pi

    # Equation 24 from FAO56
    sdec = 0.409 * np.sin(pi2yr * jday - 1.39)

    # Equation 25 from FAO56
    ws = np.arccos(np.outer(-np.tan(latrad), np.tan(sdec)))

    # Equation 34 from FAO56
    dlm = con*ws

    # handle metadata if xarray output
    if x_out:
        print('meta')
        dlm = xr.DataArray(dlm, coords=[jday,lat], dims=["doy", "lat"])
        dlm.attrs['long_name'] = "maximum daylight: FAO_56"
        dlm.attrs['units'] = "hours/day"
        dlm.attrs['url'] = "http://www.fao.org/docrep/X0490E/x0490e07.htm"
        dlm.attrs['info'] = "FAO 56; EQN 34; max_daylight"

    return dlm


print(max_daylight(246, -20))
# print(max_daylight([15, 180, 246, 306], [-20, 0, 45]))