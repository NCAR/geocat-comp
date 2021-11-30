import dask.array as da
import numpy as np
import typing
import xarray as xr
import warnings


def _dewtemp(
    tk: typing.Union[np.ndarray, xr.DataArray, list,
                     float], rh: typing.Union[np.ndarray, xr.DataArray, list,
                                              float]
) -> typing.Union[np.ndarray, xr.DataArray, list, float]:
    """This function calculates the dew point temperature given temperature and
    relative humidity using equations from John Dutton's "Ceaseless Wind" (pp
    273-274)

    Parameters
    ----------
    tk : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Temperature in Kelvin

    rh : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Relative humidity. Must be the same dimensions as temperature

    Returns
    -------
    tdk : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Dewpoint temperature in Kelvin. Same size as input variable temperature


    See Also
    --------
    Related GeoCAT Functions:
    `dewtemp <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.dewtemp.html#geocat.comp.meteorology.dewtemp>`_

    Related NCL Functions:
    `dewtemp_trh <https://www.ncl.ucar.edu/Document/Functions/Built-in/dewtemp_trh.shtml>`_
    """

    gc = 461.5  # gas constant for water vapor [j/{kg-k}]
    gcx = gc / (1000 * 4.186)  # [cal/{g-k}]

    lhv = (597.3 - 0.57 * (tk - 273)) / gcx
    tdk = tk * lhv / (lhv - tk * np.log(rh * 0.01))

    return tdk


def _heat_index(temperature: np.ndarray,
                relative_humidity: typing.Union[np.ndarray, xr.DataArray, list,
                                                float],
                alternate_coeffs: bool = False) -> np.ndarray:
    """Compute the 'heat index' as calculated by the National Weather Service.

    Internal function for heat_index

    Parameters
    ----------
    temperature : :class:`numpy.ndarray`
        temperature(s) in Fahrenheit

    relative_humidity : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        relative humidity as a percentage. Must be the same shape as
        temperature

    alternate_coeffs : :class:`bool`, Optional
        flag to use alternate set of coefficients appropriate for
        temperatures from 70F to 115F and humidities between 0% and 80%

    Returns
    -------
    heatindex : :class:`numpy.ndarray`
        Calculated heat index. Same shape as temperature


    See Also
    --------
    Related GeoCAT Functions:
    `heat_index <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.heat_index.html#geocat.comp.meteorology.heat_index>`_,
    `_xheat_index <https://geocat-comp.readthedocs.io/en/latest/internal_api/generated/geocat.comp.meteorology._xheat_index.html#geocat.comp.meteorology._xheat_index>`_

    Related NCL Functions:
    `heat_index_nws <https://www.ncl.ucar.edu/Document/Functions/Contributed/heat_index_nws.shtml>`_
    """
    # Default coefficients for (t>=80F) and (40<gh<100)
    coeffs = [
        -42.379, 2.04901523, 10.14333127, -0.22475541, -0.00683783, -0.05481717,
        0.00122874, 0.00085282, -0.00000199
    ]
    crit = [80, 40, 100]  # [T_low [F], RH_low, RH_high]

    # Optional flag coefficients for (70F<t<115F) and (0<gh<80)
    # within 3F of default coeffs
    if alternate_coeffs:
        coeffs = [
            0.363445176, 0.988622465, 4.777114035, -0.114037667, -0.000850208,
            -0.020716198, 0.000687678, 0.000274954, 0.0
        ]
        crit = [70, 0, 80]  # [T_low [F], RH_low, RH_high]

    # NWS practice
    # average Steadman and t
    heatindex = (0.5 * (temperature + 61.0 + ((temperature - 68.0) * 1.2) +
                        (relative_humidity * 0.094)) + temperature) * 0.5

    # http://ehp.niehs.nih.gov/1206273/
    heatindex = xr.where(temperature < 40, temperature, heatindex)

    # if all t values less than critical, return hi
    # otherwise perform calculation
    eqtype = 0
    if not all(temperature.ravel() < crit[0]):
        eqtype = 1

        heatindex = xr.where(heatindex > crit[0],
                             _nws_eqn(coeffs, temperature, relative_humidity),
                             heatindex)

        # adjustments
        heatindex = xr.where(
            xr.ufuncs.logical_and(
                relative_humidity < 13,
                xr.ufuncs.logical_and(temperature > 80, temperature < 112)),
            heatindex - ((13 - relative_humidity) / 4) * np.sqrt(
                (17 - abs(temperature - 95)) / 17), heatindex)

        heatindex = xr.where(
            xr.ufuncs.logical_and(
                relative_humidity > 85,
                xr.ufuncs.logical_and(temperature > 80, temperature < 87)),
            heatindex + ((relative_humidity - 85.0) / 10.0) *
            ((87.0 - temperature) / 5.0), heatindex)

    return heatindex


def _nws_eqn(coeffs, temp, rel_hum):
    """Helper function to compute the heat index.

    Internal function for heat_index

    Parameters
    ----------
    coeffs : :class:`numpy.ndarray`
        coefficients to calculate heat index

    temp : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        temperaure

    rel_hum : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        relative humidity as a percentage. Must be the same shape as
        temperature

    Returns
    -------
    heatindex : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        Intermediate calculated heat index. Same shape as temperature


    See Also
    --------
    Related GeoCAT Functions:
    `heat_index <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.heat_index.html#geocat.comp.meteorology.heat_index>`_,
    `_heat_index <https://geocat-comp.readthedocs.io/en/latest/internal_api/generated/geocat.comp.meteorology._heat_index.html#geocat.comp.meteorology._heat_index>`_

    Related NCL Functions:
    `heat_index_nws <https://www.ncl.ucar.edu/Document/Functions/Contributed/heat_index_nws.shtml>`_,
    """
    heatindex = coeffs[0] \
                + coeffs[1] * temp \
                + coeffs[2] * rel_hum \
                + coeffs[3] * temp * rel_hum \
                + coeffs[4] * temp ** 2 \
                + coeffs[5] * rel_hum ** 2 \
                + coeffs[6] * temp ** 2 * rel_hum \
                + coeffs[7] * temp * rel_hum ** 2 \
                + coeffs[8] * temp ** 2 * rel_hum ** 2

    return heatindex


def _relhum(
        t: typing.Union[np.ndarray, list, float],
        w: typing.Union[np.ndarray, xr.DataArray, list,
                        float], p: typing.Union[np.ndarray, xr.DataArray, list,
                                                float]) -> np.ndarray:
    """Calculates relative humidity with respect to ice, given temperature,
    mixing ratio, and pressure.

     "Improved Magnus' Form Approx. of Saturation Vapor pressure"
     Oleg A. Alduchov and Robert E. Eskridge
     http://www.osti.gov/scitech/servlets/purl/548871/
     https://doi.org/10.2172/548871

    Parameters
    ----------
    t : :class:`numpy.ndarray`, :obj:`list`, or :obj:`float`
        Temperature in Kelvin

    w : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Mixing ratio in kg/kg. Must have the same dimensions as temperature

    p : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Pressure in Pa. Must have the same dimensions as temperature

    Returns
    -------
    rh :class:`numpy.ndarray)
        Relative humidity. Will have the same dimensions as temperature


    See Also
    --------
    Related GeoCAT Functions:
    `relhum <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum.html#geocat.comp.meteorology.relhum>`_,
    `relhum_ice <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum_ice.html#geocat.comp.meteorology.relhum_ice>`_,
    `relhum_water <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum_water.html#geocat.comp.meteorology.relhum_water>`_,
    `_xrelhum <https://geocat-comp.readthedocs.io/en/latest/internal_api/generated/geocat.comp.meteorology._xrelhum.html#geocat.comp.meteorology._xrelhum>`_

    Related NCL Functions:
    `relhum <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum.shtml>`_,
    `relhum_ttd <https://www.ncl.ucar.edu/Document/Functions/Contributed/relhum_ttd.shtml>`_,
    `relhum_ice <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_ice.shtml>`_,
    `relhum_water <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_water.shtml>`_
    """

    table = np.asarray([
        0.01403, 0.01719, 0.02101, 0.02561, 0.03117, 0.03784, 0.04584, 0.05542,
        0.06685, 0.08049, 0.09672, 0.1160, 0.1388, 0.1658, 0.1977, 0.2353,
        0.2796, 0.3316, 0.3925, 0.4638, 0.5472, 0.6444, 0.7577, 0.8894, 1.042,
        1.220, 1.425, 1.662, 1.936, 2.252, 2.615, 3.032, 3.511, 4.060, 4.688,
        5.406, 6.225, 7.159, 8.223, 9.432, 10.80, 12.36, 14.13, 16.12, 18.38,
        20.92, 23.80, 27.03, 30.67, 34.76, 39.35, 44.49, 50.26, 56.71, 63.93,
        71.98, 80.97, 90.98, 102.1, 114.5, 128.3, 143.6, 160.6, 179.4, 200.2,
        223.3, 248.8, 276.9, 307.9, 342.1, 379.8, 421.3, 466.9, 517.0, 572.0,
        632.3, 698.5, 770.9, 850.2, 937.0, 1032.0, 1146.6, 1272.0, 1408.1,
        1556.7, 1716.9, 1890.3, 2077.6, 2279.6, 2496.7, 2729.8, 2980.0, 3247.8,
        3534.1, 3839.8, 4164.8, 4510.5, 4876.9, 5265.1, 5675.2, 6107.8, 6566.2,
        7054.7, 7575.3, 8129.4, 8719.2, 9346.50, 10013.0, 10722.0, 11474.0,
        12272.0, 13119.0, 14017.0, 14969.0, 15977.0, 17044.0, 18173.0, 19367.0,
        20630.0, 21964.0, 23373.0, 24861.0, 26430.0, 28086.0, 29831.0, 31671.0,
        33608.0, 35649.0, 37796.0, 40055.0, 42430.0, 44927.0, 47551.0, 50307.0,
        53200.0, 56236.0, 59422.0, 62762.0, 66264.0, 69934.0, 73777.0, 77802.0,
        82015.0, 86423.0, 91034.0, 95855.0, 100890.0, 106160.0, 111660.0,
        117400.0, 123400.0, 129650.0, 136170.0, 142980.0, 150070.0, 157460.0,
        165160.0, 173180.0, 181530.0, 190220.0, 199260.0, 208670.0, 218450.0,
        228610.0, 239180.0, 250160.0, 261560.0, 273400.0, 285700.0, 298450.0,
        311690.0, 325420.0, 339650.0, 354410.0, 369710.0, 385560.0, 401980.0,
        418980.0, 436590.0, 454810.0, 473670.0, 493170.0, 513350.0, 534220.0,
        555800.0, 578090.0, 601130.0, 624940.0, 649530.0, 674920.0, 701130.0,
        728190.0, 756110.0, 784920.0, 814630.0, 845280.0, 876880.0, 909450.0,
        943020.0, 977610.0, 1013250.0, 1049940.0, 1087740.0, 1087740.
    ])

    maxtemp = 375.16
    mintemp = 173.16

    # replace values of t above and below max and min values for temperature
    t = np.clip(t, mintemp, maxtemp)

    it = (t - mintemp).astype(int)
    t2 = mintemp + it

    es = (t2 + 1 - t) * table[it] + (t - t2) * table[it + 1]
    es = es * 0.1

    rh = (w * (p - 0.378 * es) / (0.622 * es)) * 100

    # if any value is below 0.0001, set to 0.0001
    rh = np.clip(rh, 0.0001, None)

    return rh


def _relhum_ice(t: typing.Union[np.ndarray, list, float],
                w: typing.Union[np.ndarray, list, float],
                p: typing.Union[np.ndarray, list, float]) -> np.ndarray:
    """Calculates relative humidity with respect to ice, given temperature,
    mixing ratio, and pressure.

     "Improved Magnus' Form Approx. of Saturation Vapor pressure"
     Oleg A. Alduchov and Robert E. Eskridge
     http://www.osti.gov/scitech/servlets/purl/548871/
     https://doi.org/10.2172/548871

    Parameters
    ----------
    t : :class:`numpy.ndarray`, :obj:`list`, :obj:`float`
        Temperature in Kelvin

    w : :class:`numpy.ndarray`, :obj:`list`, :obj:`float`
        Mixing ratio in kg/kg. Must have the same dimensions as temperature

    p : :class:`numpy.ndarray`, :obj:`list`, :obj:`float`
        Pressure in Pa. Must have the same dimensions as temperature

    Returns
    -------
    rh : :class:`numpy.ndarray`
        Relative humidity. Will have the same dimensions as temperature


    See Also
    --------
    Related GeoCAT Functions:
    `relhum <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum.html#geocat.comp.meteorology.relhum>`_,
    `relhum_ice <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum_ice.html#geocat.comp.meteorology.relhum_ice>`_,
    `relhum_water <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum_water.html#geocat.comp.meteorology.relhum_water>`_,
    `_xrelhum <https://geocat-comp.readthedocs.io/en/latest/internal_api/generated/geocat.comp.meteorology._xrelhum.html#geocat.comp.meteorology._xrelhum>`_

    Related NCL Functions:
    `relhum <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum.shtml>`_,
    `relhum_ttd <https://www.ncl.ucar.edu/Document/Functions/Contributed/relhum_ttd.shtml>`_,
    `relhum_ice <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_ice.shtml>`_,
    `relhum_water <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_water.shtml>`_
    """

    # Define data variables

    t0 = 273.15
    ep = 0.622
    onemep = 0.378
    es0 = 6.1128
    a = 22.571
    b = 273.71

    est = es0 * np.exp((a * (t - t0)) / ((t - t0) + b))
    qst = (ep * est) / ((p * 0.01) - onemep * est)

    rh = 100 * (w / qst)

    return rh


def _relhum_water(t: typing.Union[np.ndarray, list, float],
                  w: typing.Union[np.ndarray, list, float],
                  p: typing.Union[np.ndarray, list, float]) -> np.ndarray:
    """Calculates relative humidity with respect to water, given temperature,
    mixing ratio, and pressure.

    Definition of mixing ratio if,

    - es  - is the saturation mixing ratio
    - ep  - is the ratio of the molecular weights of water vapor to dry air
    - p   - is the atmospheric pressure
    - rh  - is the relative humidity (given as a percent)

    rh =  100*  q / ( (ep*es)/(p-es) )

    Parameters
    ----------
    t : :class:`numpy.ndarray`, :obj:`list`, :obj:`float`
        Temperature in Kelvin

    w : :class:`numpy.ndarray`, :obj:`list`, :obj:`float`
        Mixing ratio in kg/kg. Must have the same dimensions as temperature

    p : :class:`numpy.ndarray`, :obj:`list`, :obj:`float`
        Pressure in Pa. Must have the same dimensions as temperature

    Returns
    -------
    rh :class:`numpy.ndarray`
        Relative humidity. Will have the same dimensions as temperature

    See Also
    --------
    Related GeoCAT Functions:
    `relhum <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum.html#geocat.comp.meteorology.relhum>`_,
    `relhum_ice <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum_ice.html#geocat.comp.meteorology.relhum_ice>`_,
    `relhum_water <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum_water.html#geocat.comp.meteorology.relhum_water>`_,
    `_xrelhum <https://geocat-comp.readthedocs.io/en/latest/internal_api/generated/geocat.comp.meteorology._xrelhum.html#geocat.comp.meteorology._xrelhum>`_

    Related NCL Functions:
    `relhum <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum.shtml>`_,
    `relhum_ttd <https://www.ncl.ucar.edu/Document/Functions/Contributed/relhum_ttd.shtml>`_,
    `relhum_ice <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_ice.shtml>`_,
    `relhum_water <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_water.shtml>`_
    """

    # Define data variables

    t0 = 273.15
    ep = 0.622
    onemep = 0.378
    es0 = 6.1128
    a = 17.269
    b = 35.86

    est = es0 * np.exp((a * (t - t0)) / (t - b))
    qst = (ep * est) / ((p * 0.01) - onemep * est)

    rh = 100 * (w / qst)

    return rh


def _xheat_index(temperature: xr.DataArray,
                 relative_humidity: xr.DataArray,
                 alternate_coeffs: bool = False) -> tuple([xr.DataArray, int]):
    """Compute the 'heat index' as calculated by the National Weather Service.

    Internal function for heat_index for dask

    Parameters
    ----------
    temperature : :class:`xarray.DataArray`
        temperature(s) in Fahrenheit

    relative_humidity : :class:`xarray.DataArray`
        relative humidity as a percentage. Must be the same shape as
        temperature

    alternate_coeffs : :class:`bool`, Optional
        flag to use alternate set of coefficients appropriate for
        temperatures from 70F to 115F and humidities between 0% and 80%

    Returns
    -------
    heatindex : :class:`xarray.DataArray`
        Calculated heat index. Same shape as temperature

    eqtype : :class:`int`
        version of equations used, for xarray attrs output

    See Also
    --------
    Related GeoCAT Functions:
    `heat_index <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.heat_index.html#geocat.comp.meteorology.heat_index>`_,
    `_heat_index <https://geocat-comp.readthedocs.io/en/latest/internal_api/generated/geocat.comp.meteorology._heat_index.html#geocat.comp.meteorology._heat_index>`_

    Related NCL Functions:
    `heat_index_nws <https://www.ncl.ucar.edu/Document/Functions/Contributed/heat_index_nws.shtml>`_,
    """
    # Default coefficients for (t>=80F) and (40<gh<100)
    coeffs = [
        -42.379, 2.04901523, 10.14333127, -0.22475541, -0.00683783, -0.05481717,
        0.00122874, 0.00085282, -0.00000199
    ]
    crit = [80, 40, 100]  # [T_low [F], RH_low, RH_high]

    # Optional flag coefficients for (70F<t<115F) and (0<gh<80)
    # within 3F of default coeffs
    if alternate_coeffs:
        coeffs = [
            0.363445176, 0.988622465, 4.777114035, -0.114037667, -0.000850208,
            -0.020716198, 0.000687678, 0.000274954, 0.0
        ]
        crit = [70, 0, 80]  # [T_low [F], RH_low, RH_high]

    # NWS practice
    # average Steadman and t
    heatindex = (0.5 * (temperature + 61.0 + ((temperature - 68.0) * 1.2) +
                        (relative_humidity * 0.094)) + temperature) * 0.5

    # http://ehp.niehs.nih.gov/1206273/
    heatindex = xr.where(temperature < 40, temperature, heatindex)

    # if all t values less than critical, return hi
    # otherwise perform calculation
    eqtype = 0
    if not all(temperature.data.ravel() < crit[0]):
        eqtype = 1

        heatindex = xr.where(heatindex > crit[0],
                             _nws_eqn(coeffs, temperature, relative_humidity),
                             heatindex)

        # adjustments
        heatindex = xr.where(
            xr.ufuncs.logical_and(
                relative_humidity < 13,
                xr.ufuncs.logical_and(temperature > 80, temperature < 112)),
            heatindex - ((13 - relative_humidity) / 4) * np.sqrt(
                (17 - abs(temperature - 95)) / 17), heatindex)

        heatindex = xr.where(
            xr.ufuncs.logical_and(
                relative_humidity > 85,
                xr.ufuncs.logical_and(temperature > 80, temperature < 87)),
            heatindex + ((relative_humidity - 85.0) / 10.0) *
            ((87.0 - temperature) / 5.0), heatindex)

    return heatindex, eqtype


def _xrelhum(t: xr.DataArray, w: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    """Calculates relative humidity with respect to ice, given temperature,
    mixing ratio, and pressure.

     "Improved Magnus' Form Approx. of Saturation Vapor pressure"
     Oleg A. Alduchov and Robert E. Eskridge
     http://www.osti.gov/scitech/servlets/purl/548871/
     https://doi.org/10.2172/548871

    Parameters
    ----------
    t : :class:`xarray.DataArray`
        Temperature in Kelvin

    w : :class:`xarray.DataArray`
        Mixing ratio in kg/kg. Must have the same dimensions as temperature

    p : :class:`xarray.DataArray`
        Pressure in Pa. Must have the same dimensions as temperature

    Returns
    -------
    rh : :class:`xarray.DataArray`
        Relative humidity. Will have the same dimensions as temperature


    See Also
    --------
    Related GeoCAT Functions:
    `relhum <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum.html#geocat.comp.meteorology.relhum>`_,
    `relhum_ice <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum_ice.html#geocat.comp.meteorology.relhum_ice>`_,
    `relhum_water <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum_water.html#geocat.comp.meteorology.relhum_water>`_

    Related NCL Functions:
    `relhum <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum.shtml>`_,
    `relhum_ttd <https://www.ncl.ucar.edu/Document/Functions/Contributed/relhum_ttd.shtml>`_,
    `relhum_ice <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_ice.shtml>`_,
    `relhum_water <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_water.shtml>`_
    """

    table = da.from_array([
        0.01403, 0.01719, 0.02101, 0.02561, 0.03117, 0.03784, 0.04584, 0.05542,
        0.06685, 0.08049, 0.09672, 0.1160, 0.1388, 0.1658, 0.1977, 0.2353,
        0.2796, 0.3316, 0.3925, 0.4638, 0.5472, 0.6444, 0.7577, 0.8894, 1.042,
        1.220, 1.425, 1.662, 1.936, 2.252, 2.615, 3.032, 3.511, 4.060, 4.688,
        5.406, 6.225, 7.159, 8.223, 9.432, 10.80, 12.36, 14.13, 16.12, 18.38,
        20.92, 23.80, 27.03, 30.67, 34.76, 39.35, 44.49, 50.26, 56.71, 63.93,
        71.98, 80.97, 90.98, 102.1, 114.5, 128.3, 143.6, 160.6, 179.4, 200.2,
        223.3, 248.8, 276.9, 307.9, 342.1, 379.8, 421.3, 466.9, 517.0, 572.0,
        632.3, 698.5, 770.9, 850.2, 937.0, 1032.0, 1146.6, 1272.0, 1408.1,
        1556.7, 1716.9, 1890.3, 2077.6, 2279.6, 2496.7, 2729.8, 2980.0, 3247.8,
        3534.1, 3839.8, 4164.8, 4510.5, 4876.9, 5265.1, 5675.2, 6107.8, 6566.2,
        7054.7, 7575.3, 8129.4, 8719.2, 9346.50, 10013.0, 10722.0, 11474.0,
        12272.0, 13119.0, 14017.0, 14969.0, 15977.0, 17044.0, 18173.0, 19367.0,
        20630.0, 21964.0, 23373.0, 24861.0, 26430.0, 28086.0, 29831.0, 31671.0,
        33608.0, 35649.0, 37796.0, 40055.0, 42430.0, 44927.0, 47551.0, 50307.0,
        53200.0, 56236.0, 59422.0, 62762.0, 66264.0, 69934.0, 73777.0, 77802.0,
        82015.0, 86423.0, 91034.0, 95855.0, 100890.0, 106160.0, 111660.0,
        117400.0, 123400.0, 129650.0, 136170.0, 142980.0, 150070.0, 157460.0,
        165160.0, 173180.0, 181530.0, 190220.0, 199260.0, 208670.0, 218450.0,
        228610.0, 239180.0, 250160.0, 261560.0, 273400.0, 285700.0, 298450.0,
        311690.0, 325420.0, 339650.0, 354410.0, 369710.0, 385560.0, 401980.0,
        418980.0, 436590.0, 454810.0, 473670.0, 493170.0, 513350.0, 534220.0,
        555800.0, 578090.0, 601130.0, 624940.0, 649530.0, 674920.0, 701130.0,
        728190.0, 756110.0, 784920.0, 814630.0, 845280.0, 876880.0, 909450.0,
        943020.0, 977610.0, 1013250.0, 1049940.0, 1087740.0, 1087740.
    ])

    maxtemp = 375.16
    mintemp = 173.16

    # replace values of t above and below max and min values for temperature
    t = t.clip(mintemp, maxtemp)

    it = (t - mintemp).astype(int).data
    t2 = mintemp + it

    it_shape = it.shape

    es = (t2 + 1 - t) * table[it.ravel()].reshape(it_shape) + (
        t - t2) * table[it.ravel() + 1].reshape(it_shape)
    es = es * 0.1

    rh = (w * (p - 0.378 * es) / (0.622 * es)) * 100

    rh = rh.clip(0.0001, None)

    return rh


def dewtemp(
    temperature: typing.Union[np.ndarray, xr.DataArray, list, float],
    relative_humidity: typing.Union[np.ndarray, xr.DataArray, list, float]
) -> typing.Union[np.ndarray, float]:
    """This function calculates the dew point temperature given temperature and
    relative humidity using equations from John Dutton's "Ceaseless Wind" (pp
    273-274)

    Parameters
    ----------
    temperature : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Temperature in Kelvin

    relative_humidity : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Relative humidity. Must be the same dimensions as temperature

    Returns
    -------
    dew_pnt_temp : :class:`numpy.ndarray` or :obj:`float`
        Dewpoint temperature in Kelvin. Same size as input variable temperature

    See Also
    --------
    Related GeoCAT Functions:
    `_dewtemp <https://geocat-comp.readthedocs.io/en/latest/internal_api/generated/geocat.comp.meteorology._dewtemp.html#geocat.comp.meteorology._dewtemp>`_

    Related NCL Functions:
    `dewtemp_trh <https://www.ncl.ucar.edu/Document/Functions/Built-in/dewtemp_trh.shtml>`_
    """

    inputs = [temperature, relative_humidity]

    # ensure all inputs same size
    if not (np.shape(x) == np.shape(inputs[0]) for x in inputs):
        raise ValueError("dewtemp: dimensions of inputs are not the same")

    # Get input types
    in_types = [type(item) for item in inputs]

    if xr.DataArray in in_types:

        # check all inputs are xarray.DataArray
        if any(x != xr.DataArray for x in in_types):
            raise TypeError(
                "relhum: if using xarray, all inputs must be xarray")

        # call internal computation function
        # note: no alternative internal function required for dewtemp
        dew_pnt_temp = _dewtemp(temperature, relative_humidity)

        # set xarray attributes
        dew_pnt_temp.attrs['long_name'] = 'dew point temperature'
        dew_pnt_temp.attrs['units'] = 'Kelvin'

    else:
        # ensure in numpy array for function call
        temperature = np.asarray(temperature)
        relative_humidity = np.asarray(relative_humidity)

        # function call for non-dask/xarray
        dew_pnt_temp = _dewtemp(temperature, relative_humidity)

    return dew_pnt_temp


def heat_index(
        temperature: typing.Union[np.ndarray, xr.DataArray, list, float],
        relative_humidity: typing.Union[np.ndarray, xr.DataArray, list, float],
        alternate_coeffs: bool = False
) -> typing.Union[np.ndarray, xr.DataArray]:
    """Compute the 'heat index' as calculated by the National Weather Service.

    The heat index calculation in this funtion is described at:
    https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml

    The 'Heat Index' is a measure of how hot weather "feels" to the body. The combination of temperature an humidity
    produce an "apparent temperature" or the temperature the body "feels". The returned values are for shady
    locations only. Exposure to full sunshine can increase heat index values by up to 15Â°F. Also, strong winds,
    particularly with very hot, dry air, can be extremely hazardous as the wind adds heat to the body

    The computation of the heat index is a refinement of a result obtained by multiple regression analysis carried
    out by Lans P. Rothfusz and described in a 1990 National Weather Service (NWS) Technical Attachment (SR 90-23).
    All values less that 40F/4.4C/277.65K are set to the ambient temperature.

    In practice, the Steadman formula is computed first and the result averaged with the temperature. If this heat
    index value is 80 degrees F or higher, the full regression equation along with any adjustment as described above
    is applied. If the ambient temperature is less the 40F (4.4C/277.65K), the heat index is set to to the ambient
    temperature.

    Parameters
    ----------
    temperature : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        temperature(s) in Fahrenheit

    relative_humidity : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        relative humidity as a percentage. Must be the same shape as
        temperature

    alternate_coeffs : :class:`bool`, Optional
        flag to use alternate set of coefficients appropriate for
        temperatures from 70F to 115F and humidities between 0% and 80%

    Returns
    -------
    heatindex : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        Calculated heat index. Same shape as temperature

    Examples
    --------
    >>> import numpy as np
    >>> import geocat.comp
    >>> t = np.array([104, 100, 92])
    >>> rh = np.array([55, 65, 60])
    >>> hi = heat_index(t,rh)
    >>> hi
    array([137.36135724, 135.8679973 , 104.68441864])


    See Also
    --------
    Related GeoCAT Functions:
    `_heat_index <https://geocat-comp.readthedocs.io/en/latest/internal_api/generated/geocat.comp.meteorology._heat_index.html#geocat.comp.meteorology._heat_index>`_,
    `_xheat_index <https://geocat-comp.readthedocs.io/en/latest/internal_api/generated/geocat.comp.meteorology._xheat_index.html#geocat.comp.meteorology._xheat_index>`_

    Related NCL Functions:
    `heat_index_nws <https://www.ncl.ucar.edu/Document/Functions/Contributed/heat_index_nws.shtml>`_
    """

    inputs = [temperature, relative_humidity]

    # ensure all inputs same size
    if not (np.shape(x) == np.shape(inputs[0]) for x in inputs):
        raise ValueError("heat_index: dimensions of inputs are not the same")

    # Get input types
    in_types = [type(item) for item in inputs]

    # run dask compatible version if input is xarray
    if xr.DataArray in in_types:

        # check all inputs are xarray.DataArray
        if not all(x == xr.DataArray for x in in_types):
            raise TypeError(
                "heat_index: if using xarray, all inputs must be xarray")

        # input validation on relative humidity
        if any(relative_humidity.data.ravel() < 0) or any(
                relative_humidity.data.ravel() > 100):
            raise ValueError('heat_index: invalid values for relative humidity')

        # Check if relative humidity fractional
        if all(relative_humidity.data.ravel() < 1):
            warnings.warn(
                "WARNING: rh must be %, not fractional; All rh are < 1")

        # call internal computation function
        heatindex, eqtype = _xheat_index(temperature, relative_humidity,
                                         alternate_coeffs)

        # set xarray attributes
        heatindex.attrs['long_name'] = "heat index: NWS"
        heatindex.attrs['units'] = "F"
        heatindex.attrs[
            'www'] = "https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml"
        heatindex.attrs['info'] = "appropriate for shady locations with no wind"

        if eqtype == 1:
            heatindex.attrs[
                'tag'] = "NCL: heat_index_nws; (Steadman+t)*0.5 and Rothfusz"
        else:
            heatindex.attrs['tag'] = "NCL: heat_index_nws; (Steadman+t)*0.5"

    else:
        # ensure in numpy array for function call
        temperature = np.atleast_1d(temperature)
        relative_humidity = np.atleast_1d(relative_humidity)

        # input validation on relative humidity
        if any(relative_humidity.ravel() < 0) or any(
                relative_humidity.ravel() > 100):
            raise ValueError('heat_index: invalid values for relative humidity')

        # Check if relative humidity fractional
        if all(relative_humidity.ravel() < 1):
            warnings.warn(
                "WARNING: rh must be %, not fractional; All rh are < 1")

        # function call for non-dask/xarray
        heatindex = _heat_index(temperature, relative_humidity,
                                alternate_coeffs)

    return heatindex


def relhum(
    temperature: typing.Union[np.ndarray, xr.DataArray, list, float],
    mixing_ratio: typing.Union[np.ndarray, xr.DataArray, list, float],
    pressure: typing.Union[np.ndarray, xr.DataArray, list, float]
) -> typing.Union[np.ndarray, xr.DataArray]:
    """This function calculates the relative humidity given temperature, mixing
    ratio, and pressure.

    "Improved Magnus' Form Approx. of Saturation Vapor pressure"
    Oleg A. Alduchov and Robert E. Eskridge
    https://www.osti.gov/scitech/servlets/purl/548871/
    https://doi.org/10.2172/548871

    Parameters
    ----------
    temperature : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Temperature in Kelvin

    mixing_ratio : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Mixing ratio in kg/kg. Must have the same dimensions as temperature

    pressure : :class:`numpy.ndarray`, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Pressure in Pa. Must have the same dimensions as temperature

    Returns
    -------
    relative_humidity : :class:`numpy.ndarray` or :class:`xarray.DataArray`
        Relative humidity. Will have the same dimensions as temperature


    See Also
    --------
    Related GeoCAT Functions:
    `_xrelhum <https://geocat-comp.readthedocs.io/en/latest/internal_api/generated/geocat.comp.meteorology._xrelhum.html#geocat.comp.meteorology._xrelhum>`_,
    `relhum_ice <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum_ice.html#geocat.comp.meteorology.relhum_ice>`_,
    `relhum_water <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum_water.html#geocat.comp.meteorology.relhum_water>`_

    Related NCL Functions:
    `relhum <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum.shtml>`_,
    `relhum_ttd <https://www.ncl.ucar.edu/Document/Functions/Contributed/relhum_ttd.shtml>`_,
    `relhum_ice <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_ice.shtml>`_,
    `relhum_water <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_water.shtml>`_
    """

    inputs = [temperature, mixing_ratio, pressure]

    # ensure all inputs same size
    if not (np.shape(x) == np.shape(inputs[0]) for x in inputs):
        raise ValueError("relhum: dimensions of inputs are not the same")

    # Get input types
    in_types = [type(item) for item in inputs]

    # run dask compatible version if input is xarray
    if xr.DataArray in in_types:

        # check all inputs are xarray.DataArray
        if not all(x == xr.DataArray for x in in_types):
            raise TypeError(
                "relhum: if using xarray, all inputs must be xarray")

        # call internal computation function
        relative_humidity = _xrelhum(temperature, mixing_ratio, pressure)

        # set xarray attributes
        relative_humidity.attrs['long_name'] = "relative humidity"
        relative_humidity.attrs['units'] = 'percentage'
        relative_humidity.attrs['info'] = 'https://doi.org/10.2172/548871'

    else:
        # ensure in numpy array for function call
        temperature = np.asarray(temperature)
        mixing_ratio = np.asarray(mixing_ratio)
        pressure = np.asarray(pressure)

        # function call for non-dask/xarray
        relative_humidity = _relhum(temperature, mixing_ratio, pressure)

    return relative_humidity


def relhum_ice(temperature: typing.Union[np.ndarray, list, float],
               mixing_ratio: typing.Union[np.ndarray, list, float],
               pressure: typing.Union[np.ndarray, list, float]) -> np.ndarray:
    """Calculates relative humidity with respect to ice, given temperature,
    mixing ratio, and pressure.

     "Improved Magnus' Form Approx. of Saturation Vapor pressure"
     Oleg A. Alduchov and Robert E. Eskridge
     http://www.osti.gov/scitech/servlets/purl/548871/
     https://doi.org/10.2172/548871

    Parameters
    ----------
    temperature : :class:`numpy.ndarray`, :obj:`list`, or :obj:`float`
        Temperature in Kelvin

    mixing_ratio : :class:`numpy.ndarray`, :obj:`list`, or :obj:`float`
        Mixing ratio in kg/kg. Must have the same dimensions as temperature

    pressure : :class:`numpy.ndarray`, :obj:`list`, or :obj:`float`
        Pressure in Pa. Must have the same dimensions as temperature

    Returns
    -------
    relative_humidity : :class:`numpy.ndarray`
        Relative humidity. Will have the same dimensions as temperature

    See Also
    --------
    Related GeoCAT Functions:
    `relhum <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum.html#geocat.comp.meteorology.relhum>`_,
    `_xrelhum <https://geocat-comp.readthedocs.io/en/latest/internal_api/generated/geocat.comp.meteorology._xrelhum.html#geocat.comp.meteorology._xrelhum>`_,
    `relhum_water <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum_water.html#geocat.comp.meteorology.relhum_water>`_,
    `_relhum_ice <https://geocat-comp.readthedocs.io/en/latest/internal_api/generated/geocat.comp.meteorology._relhum_ice.html#geocat.comp.meteorology._relhum_ice>_

    Related NCL Functions:
    `relhum <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum.shtml>`_,
    `relhum_ttd <https://www.ncl.ucar.edu/Document/Functions/Contributed/relhum_ttd.shtml>`_,
    `relhum_ice <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_ice.shtml>`_,
    `relhum_water <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_water.shtml>`_
    """

    # If xarray input, pull data and store metadata
    x_out = False
    if isinstance(temperature, xr.DataArray):
        x_out = True
        save_dims = temperature.dims
        save_coords = temperature.coords
        save_attrs = temperature.attrs

    # ensure in numpy array for function call
    temperature = np.asarray(temperature)
    mixing_ratio = np.asarray(mixing_ratio)
    pressure = np.asarray(pressure)

    # ensure all inputs same size
    if np.shape(temperature) != np.shape(mixing_ratio) or np.shape(
            temperature) != np.shape(pressure):
        raise ValueError(f"relhum_ice: dimensions of inputs are not the same")

    relative_humidity = _relhum_ice(temperature, mixing_ratio, pressure)

    # output as xarray if input as xarray
    if x_out:
        relative_humidity = xr.DataArray(data=relative_humidity,
                                         coords=save_coords,
                                         dims=save_dims,
                                         attrs=save_attrs)

    return relative_humidity


def relhum_water(temperature: typing.Union[np.ndarray, list, float],
                 mixing_ratio: typing.Union[np.ndarray, list, float],
                 pressure: typing.Union[np.ndarray, list, float]) -> np.ndarray:
    """Calculates relative humidity with respect to water, given temperature,
    mixing ratio, and pressure.

    Definition of mixing ratio if,
    es  - is the saturation mixing ratio
    ep  - is the ratio of the molecular weights of water vapor to dry air
    p   - is the atmospheric pressure
    rh  - is the relative humidity (given as a percent)

    rh =  100*  q / ( (ep*es)/(p-es) )

    Parameters
    ----------
    temperature : :class:`numpy.ndarray`, :obj:`list`, or :obj:`float`
        Temperature in Kelvin

    mixing_ratio : :class:`numpy.ndarray`, :obj:`list`, or :obj:`float`
        Mixing ratio in kg/kg. Must have the same dimensions as temperature

    pressure : :class:`numpy.ndarray`, :obj:`list`, or :obj:`float`
        Pressure in Pa. Must have the same dimensions as temperature

    Returns
    -------
    relative_humidity : :class:`numpy.ndarray`
        Relative humidity. Will have the same dimensions as temperature

    See Also
    --------
    Related GeoCAT Functions:
    `relhum <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum.html#geocat.comp.meteorology.relhum>_`,
    `_xrelhum <https://geocat-comp.readthedocs.io/en/latest/internal_api/generated/geocat.comp.meteorology._xrelhum.html#geocat.comp.meteorology._xrelhum>`_,
    `relhum_ice <https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.meteorology.relhum_ice.html#geocat.comp.meteorology.relhum_ice>`_,
    `relhum_water <https://geocat-comp.readthedocs.io/en/latest/internal_api/generated/geocat.comp.meteorology._relhum_water.html#geocat.comp.meteorology._relhum_water>`_

    Related NCL Functions:
    `relhum <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum.shtml>`_,
    `relhum_ttd <https://www.ncl.ucar.edu/Document/Functions/Contributed/relhum_ttd.shtml>`_,
    `relhum_ice <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_ice.shtml>`_,
    `relhum_water <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_water.shtml>`_
    """

    # If xarray input, pull data and store metadata
    x_out = False
    if isinstance(temperature, xr.DataArray):
        x_out = True
        save_dims = temperature.dims
        save_coords = temperature.coords
        save_attrs = temperature.attrs

    # ensure in numpy array for function call
    temperature = np.asarray(temperature)
    mixing_ratio = np.asarray(mixing_ratio)
    pressure = np.asarray(pressure)

    # ensure all inputs same size
    if np.shape(temperature) != np.shape(mixing_ratio) or np.shape(
            temperature) != np.shape(pressure):
        raise ValueError(f"relhum_water: dimensions of inputs are not the same")

    relative_humidity = _relhum_water(temperature, mixing_ratio, pressure)

    # output as xarray if input as xarray
    if x_out:
        relative_humidity = xr.DataArray(data=relative_humidity,
                                         coords=save_coords,
                                         dims=save_dims,
                                         attrs=save_attrs)

    return relative_humidity
