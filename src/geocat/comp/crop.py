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
    lat = np.asarray(lat)

    # check to ensure dimension of lat is not greater than two
    if lat.ndim > 2:
        raise ValueError('Number of dimensions of lat must be two or less')

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
    dlm = np.transpose(ws * con)

    # handle metadata if xarray output
    if x_out:
        print('meta')


    return dlm


# print(max_daylight(246, -20))
# print(max_daylight([15, 180, 246, 306], [-20, 0, 45]))
print(max_daylight([15, 180], [[20, 0, 45], [10, -5, -10], [55, -55, 9]]))
