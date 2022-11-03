import numpy as np
import typing
import warnings
import xarray as xr

from .meteorology import max_daylight, psychrometric_constant, saturation_vapor_pressure, saturation_vapor_pressure_slope, actual_saturation_vapor_pressure


def max_daylight(
    jday: typing.Union[np.ndarray, xr.DataArray, list,
                       float], lat: typing.Union[np.ndarray, xr.DataArray, list,
                                                 float]
) -> typing.Union[np.ndarray, xr.DataArray, float]:
    r""".. deprecated:: 2022.10.0 The ``crop`` module is deprecated.
        ``max_daylight`` has been moved to the ``meteorology`` module for
        future use. Use ``geocat.comp.max_daylight`` or ``geocat.comp.meteorology.max_daylight``
        for the same functionality.

    Computes maximum number of daylight hours as described in the Food and
    Agriculture Organization (FAO) Irrigation and Drainage Paper 56 entitled:

    Crop evapotranspiration - Guidelines for computing crop water
    requirement. Specifically, see equation 34 of Chapter 3.

    Note for abs(lat) > 55 the eqns have limited validity.

    Parameters
    ----------
    jday : ndarray, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        Day of year. Must be 1D

    lat : ndarray, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        Latitude in degrees. Must be 1D

    Returns
    -------
    sunmax : ndarray, :class:`xarray.DataArray`, :class:`float`
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


    See Also
    --------
    Related NCL Functions:
    `daylight_fao56 <https://www.ncl.ucar.edu/Document/Functions/Crop/daylight_fao56.shtml>`__
    """
    warnings.warn(
        "The ``crop`` module is deprecated. ``max_daylight`` has been moved to "
        "the ``meteorology`` module for future use. Use ``geocat.comp.max_daylight`` "
        "or ``geocat.comp.meteorology.max_daylight`` for the same functionality.",
        DeprecationWarning)

    return max_daylight(jday, lat)


def psychrometric_constant(
    pressure: typing.Union[np.ndarray, xr.DataArray, list, float]
) -> typing.Union[np.ndarray, xr.DataArray]:
    r""".. deprecated:: 2022.10.0 The ``crop`` module is deprecated.
        ``psychrometric_constant`` has been moved to the ``meteorology`` module for
        future use. Use ``geocat.comp.psychrometric_constant`` or ``geocat.comp.meteorology.psychrometric_constant``
        for the same functionality.

    Compute psychrometric constant [kPa / C] as described in the Food and
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

    A table listing the psychrometric constant for different altitudes is
    located here: https://www.fao.org/3/X0490E/x0490e0j.htm

    Parameters
    ----------
    pressure : ndarray, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        pressure in kPa/C

    Returns
    -------
    psy_const : ndarray, :class:`xarray.DataArray`
        the computed psychrometric constant. Same shape as pressure.

    Examples
    --------
    >>> import numpy as np
    >>> from geocat.comp import psychrometric_constant
    >>> pressure = np.array([60, 80, 100])
    >>> psychrometric_constant(pressure)
    array([0.0398844, 0.0531792, 0.066474 ])


    See Also
    --------
    Related NCL Functions:
    `psychro_fao56 <https://www.ncl.ucar.edu/Document/Functions/Crop/psychro_fao56.shtml>`__
    """
    warnings.warn(
        "The ``crop`` module is deprecated. ``psychrometric_constant`` has "
        "been moved to the ``meteorology`` module for future use. Use "
        "``geocat.comp.psychrometric_constant`` or "
        "``geocat.comp.meteorology.psychrometric_constant`` for the same "
        "functionality.", DeprecationWarning)

    return psychrometric_constant(pressure)


def saturation_vapor_pressure(
    temperature: typing.Union[np.ndarray, xr.DataArray, list, float],
    tfill: typing.Union[float] = np.nan
) -> typing.Union[np.ndarray, xr.DataArray]:
    r""".. deprecated:: 2022.10.0 The ``crop`` module is deprecated.
        ``saturation_vapor_pressure`` has been moved to the ``meteorology``
        module for future use. Use ``geocat.comp.saturation_vapor_pressure`` or
        ``geocat.comp.meteorology.saturation_vapor_pressure`` for the same
        functionality.

    Compute saturation vapor pressure as described in the Food and
    Agriculture Organization (FAO) Irrigation and Drainage Paper 56
    entitled:

    Crop evapotranspiration - Guidelines for computing crop water
    requirement. Specifically, see equation 11 of Chapter 3.

    This is Tetens' Formula: an empirical expression for saturation vapor
    pressure with respect to liquid water that includes the variation of
    latent heat with temperature.

    Note that if ``temperature`` = ``tdew``, then this function computes actual
    vapor pressure.

    Parameters
    ----------
    temperature : ndarray, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        Temperature in Fahrenheit

    tfill : float, numpy.nan, optional
        An optional parameter for a fill value in the return value

    Returns
    -------
    svp : ndarray, :class:`xarray.DataArray`
        the computed actual saturation vapor pressure in kPa.
        Same shape as temperature.

    Examples
    --------
    >>> import numpy as np
    >>> from geocat.comp import saturation_vapor_pressure
    >>> temp = np.array([50, 60, 70])
    >>> saturation_vapor_pressure(temp)
    array([1.22796262, 1.76730647, 2.50402976])


    See Also
    --------
    Related GeoCAT Functions:
    `actual_saturation_vapor_pressure <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.crop.actual_saturation_vapor_pressure.html#geocat.comp.crop.actual_saturation_vapor_pressure>`__,
    `saturation_vapor_pressure_slope <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.crop.saturation_vapor_pressure_slope.html#geocat.comp.crop.saturation_vapor_pressure_slope>`__

    Related NCL Functions:
    `satvpr_temp_fao56 <https://www.ncl.ucar.edu/Document/Functions/Crop/satvpr_temp_fao56.shtml>`__
    """
    warnings.warn(
        "The ``crop`` module is deprecated. "
        "``saturation_vapor_pressure`` has been moved to the ``meteorology`` "
        "module for future use. Use ``geocat.comp.saturation_vapor_pressure`` or "
        "``geocat.comp.meteorology.saturation_vapor_pressure`` for the same "
        "functionality.", DeprecationWarning)

    return saturation_vapor_pressure(temperature, tfill)


def actual_saturation_vapor_pressure(
    tdew: typing.Union[np.ndarray, xr.DataArray, list, float],
    tfill: typing.Union[float] = np.nan
) -> typing.Union[np.ndarray, xr.DataArray]:
    r""".. deprecated:: 2022.10.0 The ``crop`` module is deprecated.
        ``actual_saturation_vapor_pressure`` has been moved to the ``meteorology``
        module for future use. Use ``geocat.comp.actual_saturation_vapor_pressure``
        or ``geocat.compmeteorology.actual_saturation_vapor_pressure`` for the
        same functionality.

    Compute 'actual' saturation vapor pressure [kPa] as described in the
    Food and Agriculture Organization (FAO) Irrigation and Drainage Paper 56
    entitled:

    Crop evapotranspiration - Guidelines for computing crop water
    requirement. Specifically, see equation 14 of Chapter 3.

    The dew point temperature is synonymous with the wet bulb temperature.

    Note that this function is the same as saturation_vapor_pressure, but with
    temperature = dew point temperature with different metadata

    Parameters
    ----------
    tdew : ndarray, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        Dew point temperatures in Fahrenheit

    tfill : float, numpy.nan, optional
        An optional parameter for a fill value in the return value

    Returns
    -------
    asvp : ndarray, :class:`xarray.DataArray`
        the computed actual saturation vapor pressure in kPa.
        Same shape as tdew.

    Examples
    --------
    >>> import numpy as np
    >>> from geocat.comp import actual_saturation_vapor_pressure
    >>> temp = np.array([50, 60, 70])
    >>> actual_saturation_vapor_pressure(temp)
    array([1.22796262, 1.76730647, 2.50402976])


    See Also
    --------
    Related GeoCAT Functions:
    `saturation_vapor_pressure <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.crop.saturation_vapor_pressure.html#geocat.comp.crop.saturation_vapor_pressure>`__,
    `saturation_vapor_pressure_slope <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.crop.saturation_vapor_pressure_slope.html#geocat.comp.crop.saturation_vapor_pressure_slope>`__

    Related NCL Functions:
    `satvpr_tdew_fao56 <https://www.ncl.ucar.edu/Document/Functions/Crop/satvpr_tdew_fao56.shtml>`__
    """
    warnings.warn(
        "The ``crop`` module is deprecated. "
        "``actual_saturation_vapor_pressure`` has been moved to the ``meteorology`` "
        "module for future use. Use ``geocat.comp.actual_saturation_vapor_pressure`` "
        "or ``geocat.compmeteorology.actual_saturation_vapor_pressure`` for the "
        "same functionality.", DeprecationWarning)

    return actual_saturation_vapor_pressure(tdew, tfill)


def saturation_vapor_pressure_slope(
    temperature: typing.Union[np.ndarray, xr.DataArray, list, float],
    tfill: typing.Union[float] = np.nan
) -> typing.Union[np.ndarray, xr.DataArray]:
    r""".. deprecated:: 2022.10.0 The ``crop`` module is deprecated.
        ``saturation_vapor_pressure_slope`` has been moved to the ``meteorology``
        module for future use. Use ``geocat.comp.saturation_vapor_pressure_slope``
        or ``geocat.comp.meteorology.saturation_vapor_pressure_slope`` for the
        same functionality.

    Compute the slope [kPa/C] of saturation vapor pressure curve as
    described in the Food and Agriculture Organization (FAO) Irrigation and
    Drainage Paper 56 entitled:

    Crop evapotranspiration - Guidelines for computing crop water
    requirement. Specifically, see equation 13 of Chapter 3.

    Parameters
    ----------
    temperature : ndarray, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        Temperature in Fahrenheit

    tfill : float, numpy.nan, optional
        An optional parameter for a fill value in the return value

    Returns
    -------
    svp_slope : ndarray, :class:`xarray.DataArray`
        The computed slopes of the saturation vapor pressure curve.
        Will be the same shape as temperature.

    Examples
    --------
    >>> import numpy as np
    >>> from geocat.comp import saturation_vapor_pressure_slope
    >>> temp = np.array([50, 60, 70])
    >>> saturation_vapor_pressure_slope(temp)
    array([0.08224261, 0.11322096, 0.153595  ])


    See Also
    --------
    Related GeoCAT Functions:
    `actual_saturation_vapor_pressure <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.crop.actual_saturation_vapor_pressure.html#geocat.comp.crop.actual_saturation_vapor_pressure>`__,
    `saturation_vapor_pressure_slope <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.crop.saturation_vapor_pressure_slope.html#geocat.comp.crop.saturation_vapor_pressure_slope>`__

    Related NCL Functions:
    `satvpr_temp_fao56 <https://www.ncl.ucar.edu/Document/Functions/Crop/satvpr_temp_fao56.shtml>`__
    """
    warnings.warn(
        "The ``crop`` module is deprecated. "
        "``saturation_vapor_pressure_slope`` has been moved to the ``meteorology`` "
        "module for future use. Use ``geocat.comp.saturation_vapor_pressure_slope`` "
        "or ``geocat.comp.meteorology.saturation_vapor_pressure_slope`` for the "
        "same functionality.", DeprecationWarning)

    return saturation_vapor_pressure_slope(temperature, tfill)
