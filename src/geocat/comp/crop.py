import warnings

import numpy as np
import xarray as xr

from .comp_util import _is_duck_array


def max_daylight(jday, lat):
    """Computes maximum number of daylight hours as described in the Food and
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

    Examples
    --------
    >>> from geocat.comp import max_daylight
    >>> import numpy as np
    >>> jday = np.array([100, 123, 246])
    >>> lat = np.array([10, 20])
    >>> max_daylight(jday, lat)
    array([[12.18035083, 12.37238906],
           [12.37577081, 12.77668231],
           [12.16196585, 12.33440805]])
    """

    x_out = False
    if isinstance(jday, xr.DataArray):
        x_out = True

    # convert inputs to numpy arrays for function call if necessary
    if not _is_duck_array(jday):
        jday = np.asarray(jday, dtype='float32')
    if not _is_duck_array(lat):
        lat = np.asarray(lat, dtype='float32')

    # check to ensure dimension of lat is not greater than two
    if lat.ndim > 1 or jday.ndim > 1:
        raise ValueError('max_daylight: inputs must have at most one dimension')

    # check if latitude is outside of acceptable ranges
    # warn if more than abs(55)
    # Give stronger warning if more than abs(66)
    if (abs(lat) > 55).all() and (abs(lat) <= 66).all():
        warnings.warn(
            "WARNING: max_daylight has limited validity for abs(lat) > 55 ")
    elif (abs(lat) > 66).all():
        warnings.warn(
            'WARNING: max_daylight: calculation not possible for abs(lat) > 66 for all values of jday, '
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
    dlm = np.transpose(con * ws)

    # handle metadata if xarray output
    if x_out:
        dlm = xr.DataArray(dlm, coords=[jday, lat], dims=["doy", "lat"])
        dlm.attrs['long_name'] = "maximum daylight: FAO_56"
        dlm.attrs['units'] = "hours/day"
        dlm.attrs['url'] = "http://www.fao.org/docrep/X0490E/x0490e07.htm"
        dlm.attrs['info'] = "FAO 56; EQN 34; max_daylight"

    return dlm


def psychrometric_constant(pressure):
    """Compute psychrometric constant [kPa / C] as described in the Food and
    Agriculture Organization (FAO) Irrigation and Drainage Paper 56 entitled:

    Crop evapotranspiration - Guidelines for computing crop water
    requirement. Specifically, see equation 7 of Chapter 3 or equation 3-2 in
    Annex 3.

    From FAO 56:

    The specific heat at constant pressure is the amount of energy required to
    increase the temperature of a unit mass of air by one degree at constant
    pressure. Its value depends on the composition of the air, i.e.,  on its
    humidity. For average atmospheric conditions a value
    cp = 1.013 10-3 MJ kg-1 C-1 can be used. As an average atmospheric
    pressure is used for each location (Equation 7), the psychrometric
    constant is kept constant for each location.

    A table listing the psychometric constant for different altitudes is
    located here: https://www.fao.org/3/X0490E/x0490e0j.htm

    Parameters
    ----------
    pressure : numpy.ndarray, xr.DataArray, list, float
        pressure in kPa/C

    Returns
    -------
    psy_const : numpy.ndarray, xr.DataArray
        the computed psychrometric constant. Same shape as pressure.

    Examples
    --------
    >>> import numpy as np
    >>> from geocat.comp import psychrometric_constant
    >>> pressure = np.array([60, 80, 100])
    >>> psychrometric_constant(pressure)
    array([0.0398844, 0.0531792, 0.066474 ])
    """

    x_out = False
    if isinstance(pressure, xr.DataArray):
        x_out = True
        save_dims = pressure.dims
        save_coords = pressure.coords

    # convert inputs to numpy arrays if necessary
    if not _is_duck_array(pressure):
        pressure = np.asarray(pressure)

    con = 0.66474e-3

    psy_const = con * pressure

    # reformat output for xarray if necessary
    if x_out:
        heatindex = xr.DataArray(psy_const, coords=save_coords, dims=save_dims)
        heatindex.attrs['long_name'] = "psychrometric constant"
        heatindex.attrs['units'] = "kPa/C"
        heatindex.attrs[
            'url'] = "https://www.fao.org/docrep/X0490E/x0490e07.htm"
        heatindex.attrs['info'] = "FAO 56; EQN 8; psychrometric_constant"

    return psy_const


def saturation_vapor_pressure(temperature, tfill=np.NAN):
    """Compute saturation vapor pressure as described in the Food and
    Agriculture Organization (FAO) Irrigation and Drainage Paper 56
    entitled:

    Crop evapotranspiration - Guidelines for computing crop water
    requirement. Specifically, see equation 11 of Chapter 3.

    This is Tetens' Formula: an empirical expression for saturation vapor
    pressure with respect to liquid water that includes the variation of
    latent heat with temperature.

    Note that if temperature = tdew, then this function computes actual vapor
    pressure.

    Parameters
    ----------
    temperature : numpy.ndarray, xr.DataArray, list, float
        Temperature in Fahrenheit

    tfill : float, np.NAN, Optional
        An optional parameter for a fill value in the return value

    Returns
    -------
    svp : numpy.ndarray, xr.DataArray
        the computed actual saturation vapor pressure in kPa.
        Same shape as temperature.

     Examples
    --------
    >>> import numpy as np
    >>> from geocat.comp import saturation_vapor_pressure
    >>> temp = np.array([50, 60, 70])
    >>> saturation_vapor_pressure(temp)
    array([1.22796262, 1.76730647, 2.50402976])
    """

    x_out = False
    if isinstance(temperature, xr.DataArray):
        x_out = True
        save_dims = temperature.dims
        save_coords = temperature.coords

    # convert inputs to numpy arrays for function call if necessary
    if not _is_duck_array(temperature):
        temperature = np.asarray(temperature, dtype='float32')

    temp_c = (temperature - 32) * 5 / 9
    svp = np.where(temp_c > 0, 0.6108 * np.exp(
        (17.27 * temp_c) / (temp_c + 237.3)), tfill)

    # reformat output for xarray if necessary
    if x_out:
        heatindex = xr.DataArray(svp, coords=save_coords, dims=save_dims)
        heatindex.attrs['long_name'] = "saturation vapor pressure"
        heatindex.attrs['units'] = "kPa"
        heatindex.attrs[
            'url'] = "https://www.fao.org/docrep/X0490E/x0490e07.htm"
        heatindex.attrs['info'] = "FAO 56; EQN 11; saturation_vapor_pressure"

    return svp


def actual_saturation_vapor_pressure(tdew, tfill=np.NAN):
    """ Compute 'actual' saturation vapor pressure [kPa] as described in the
    Food and Agriculture Organization (FAO) Irrigation and Drainage Paper 56
    entitled:

    Crop evapotranspiration - Guidelines for computing crop water
    requirement. Specifically, see equation 14 of Chapter 3.

    The dew point temperature is synonymous with the wet bulb temperature.

    Note that this function is the same as saturation_vapor_pressure, but with
    temperature = dew point temperature with different metadata

    Parameters
    ----------
    tdew : numpy.ndarray, xr.DataArray, list, float
        Dew point temperatures in Fahrenheit

    tfill : float, np.NAN, Optional
        An optional parameter for a fill value in the return value

    Returns
    -------
    asvp : numpy.ndarray, xr.DataArray
        the computed actual saturation vapor pressure in kPa.
        Same shape as tdew.

    Examples
    --------
    >>> import numpy as np
    >>> from geocat.comp import actual_saturation_vapor_pressure
    >>> temp = np.array([50, 60, 70])
    >>> actual_saturation_vapor_pressure(temp)
    array([1.22796262, 1.76730647, 2.50402976])
    """

    x_out = False
    if isinstance(tdew, xr.DataArray):
        x_out = True
        save_dims = tdew.dims
        save_coords = tdew.coords

    # convert inputs to numpy arrays if necessary
    if not _is_duck_array(tdew):
        tdew = np.asarray(tdew)

    asvp = saturation_vapor_pressure(tdew, tfill)

    # reformat output for xarray if necessary
    if x_out:
        heatindex = xr.DataArray(asvp, coords=save_coords, dims=save_dims)
        heatindex.attrs[
            'long_name'] = "actual saturation vapor pressure via Tdew"
        heatindex.attrs['units'] = "kPa"
        heatindex.attrs[
            'url'] = "https://www.fao.org/docrep/X0490E/x0490e07.htm"
        heatindex.attrs[
            'info'] = "FAO 56; EQN 14; actual_saturation_vapor_pressure"

    return asvp


def saturation_vapor_pressure_slope(temperature, tfill=np.NAN):
    """Compute the slope [kPa/C] of saturation vapor pressure curve as
    described in the Food and Agriculture Organization (FAO) Irrigation and
    Drainage Paper 56 entitled:

    Crop evapotranspiration - Guidelines for computing crop water
    requirement. Specifically, see equation 13 of Chapter 3.

    Parameters
    ----------
    temperature : numpy.ndarray, xr.DataArray, list, float
        Temperature in Fahrenheit

    tfill : float, np.NAN, Optional
        An optional parameter for a fill value in the return value

    Returns
    -------
    svp_slope : numpy.ndarray, xr.DataArray
        The computed slopes of the saturation vapor pressure curve.
        Will be the same shape as temperature.

    Examples
    --------
    >>> import numpy as np
    >>> from geocat.comp import saturation_vapor_pressure_slope
    >>> temp = np.array([50, 60, 70])
    >>> saturation_vapor_pressure_slope(temp)
    array([0.08224261, 0.11322096, 0.153595  ])
    """

    x_out = False
    if isinstance(temperature, xr.DataArray):
        x_out = True
        save_dims = temperature.dims
        save_coords = temperature.coords

    # convert inputs to numpy arrays for function call if necessary
    if not _is_duck_array(temperature):
        temperature = np.asarray(temperature, dtype='float32')

    # convert to Celsius
    temp_c = (temperature - 32) * 5 / 9

    svp_slope = np.where(
        temp_c > 0, 4096 * (0.6108 * np.exp(
            (17.27 * temp_c) / (temp_c + 237.3)) / (temp_c + 237.3)**2), tfill)

    # reformat output for xarray if necessary
    if x_out:
        heatindex = xr.DataArray(svp_slope, coords=save_coords, dims=save_dims)
        heatindex.attrs['long_name'] = "slope saturation vapor pressure curve"
        heatindex.attrs['units'] = "kPa/C"
        heatindex.attrs[
            'url'] = "https://www.fao.org/docrep/X0490E/x0490e07.htm"
        heatindex.attrs[
            'info'] = "FAO 56; EQN 13; saturation_vapor_pressure_slope"

    return svp_slope
