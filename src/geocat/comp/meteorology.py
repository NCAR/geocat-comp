import dask.array as da
import numpy as np
import typing
import warnings
import xarray as xr

from .gc_util import _generate_wrapper_docstring


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
    tk : ndarray, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Temperature in Kelvin

    rh : ndarray, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Relative humidity. Must be the same dimensions as ``temperature``

    Returns
    -------
    tdk : ndarray, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Dewpoint temperature in Kelvin. Same size as input variable temperature


    See Also
    --------
    Related GeoCAT Functions:
    :func:`dewtemp`

    Related NCL Functions:
    `dewtemp_trh <https://www.ncl.ucar.edu/Document/Functions/Built-in/dewtemp_trh.shtml>`__
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
    temperature : ndarray
        temperature(s) in Fahrenheit

    relative_humidity : ndarray, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        relative humidity as a percentage. Must be the same shape as
        ``temperature``

    alternate_coeffs : bool, optional
        flag to use alternate set of coefficients appropriate for
        temperatures from 70F to 115F and humidities between 0% and 80%

    Returns
    -------
    heatindex : ndarray
        Calculated heat index. Same shape as ``temperature``


    See Also
    --------
    Related GeoCAT Functions:
    :func:`heat_index`,
    :func:`_xheat_index`

    Related NCL Functions:
    `heat_index_nws <https://www.ncl.ucar.edu/Document/Functions/Contributed/heat_index_nws.shtml>`__
    """
    # Default coefficients for (t>=80F) and (40<gh<100)
    coeffs = [
        -42.379, 2.04901523, 10.14333127, -0.22475541, -0.00683783, -0.05481717,
        0.00122874, 0.00085282, -0.00000199
    ]
    crit = [80, 40, 100]  # [T_low [F], RH_low, RH_high]

    # optional flag coefficients for (70F<t<115F) and (0<gh<80)
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
            np.logical_and(relative_humidity < 13,
                           np.logical_and(temperature > 80, temperature < 112)),
            heatindex - ((13 - relative_humidity) / 4) * np.sqrt(
                (17 - abs(temperature - 95)) / 17), heatindex)

        heatindex = xr.where(
            np.logical_and(relative_humidity > 85,
                           np.logical_and(temperature > 80, temperature < 87)),
            heatindex + ((relative_humidity - 85.0) / 10.0) *
            ((87.0 - temperature) / 5.0), heatindex)

    return heatindex


def _nws_eqn(coeffs, temp, rel_hum):
    """Helper function to compute the heat index.

    Internal function for heat_index

    Parameters
    ----------
    coeffs : ndarray
        coefficients to calculate heat index

    temp : ndarray, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        temperature

    rel_hum : ndarray, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        relative humidity as a percentage. Must be the same shape as
        ``temperature``

    Returns
    -------
    heatindex : ndarray, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        Intermediate calculated heat index. Same shape as ``temperature``


    See Also
    --------
    Related GeoCAT Functions:
    :func:`heat_index`,
    :func:`_heat_index`

    Related NCL Functions:
    `heat_index_nws <https://www.ncl.ucar.edu/Document/Functions/Contributed/heat_index_nws.shtml>`__,
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
     https://journals.ametsoc.org/view/journals/apme/35/4/1520-0450_1996_035_0601_imfaos_2_0_co_2.xml

    Parameters
    ----------
    t : ndarray, :obj:`list`, or :obj:`float`
        Temperature in Kelvin

    w : ndarray, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Mixing ratio in kg/kg. Must have the same dimensions as ``temperature``

    p : ndarray, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Pressure in Pa. Must have the same dimensions as ``temperature``

    Returns
    -------
    rh : ndarray
        Relative humidity. Will have the same dimensions as ``temperature``


    See Also
    --------
    Related GeoCAT Functions:
    :func:`relhum`
    :func:`relhum_ice`
    :func:`relhum_water`
    :func:`_xrelhum`

    Related NCL Functions:
    `relhum <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum.shtml>`__,
    `relhum_ttd <https://www.ncl.ucar.edu/Document/Functions/Contributed/relhum_ttd.shtml>`__,
    `relhum_ice <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_ice.shtml>`__,
    `relhum_water <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_water.shtml>`__
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
    https://journals.ametsoc.org/view/journals/apme/35/4/1520-0450_1996_035_0601_imfaos_2_0_co_2.xml

    Parameters
    ----------
    t : ndarray, :obj:`list`, :obj:`float`
        Temperature in Kelvin

    w : ndarray, :obj:`list`, :obj:`float`
        Mixing ratio in kg/kg. Must have the same dimensions as ``temperature``

    p : ndarray, :obj:`list`, :obj:`float`
        Pressure in Pa. Must have the same dimensions as ``temperature``

    Returns
    -------
    rh : ndarray
        Relative humidity. Will have the same dimensions as ``temperature``


    See Also
    --------
    Related GeoCAT Functions:
    :func:`relhum`
    :func:`relhum_ice`
    :func:`relhum_water`
    :func:`_xrelhum`

    Related NCL Functions:
    `relhum <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum.shtml>`__,
    `relhum_ttd <https://www.ncl.ucar.edu/Document/Functions/Contributed/relhum_ttd.shtml>`__,
    `relhum_ice <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_ice.shtml>`__,
    `relhum_water <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_water.shtml>`__
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

    Definition of mixing ratio if:

    - ``es``  - is the saturation mixing ratio
    - ``ep``  - is the ratio of the molecular weights of water vapor to dry air
    - ``p``   - is the atmospheric pressure
    - ``rh``  - is the relative humidity (given as a percent)

    .. math::
        rh =  100*  q / ( (ep*es)/(p-es) )

    Parameters
    ----------
    t : ndarray, :obj:`list`, :obj:`float`
        Temperature in Kelvin

    w : ndarray, :obj:`list`, :obj:`float`
        Mixing ratio in kg/kg. Must have the same dimensions as ``temperature``

    p : ndarray, :obj:`list`, :obj:`float`
        Pressure in Pa. Must have the same dimensions as ``temperature``

    Returns
    -------
    rh : ndarray
        Relative humidity. Will have the same dimensions as ``temperature``

    See Also
    --------
    Related GeoCAT Functions:
    :func:`relhum`
    :func:`relhum_ice`
    :func:`relhum_water`
    :func:`_xrelhum`

    Related NCL Functions:
    `relhum <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum.shtml>`__,
    `relhum_ttd <https://www.ncl.ucar.edu/Document/Functions/Contributed/relhum_ttd.shtml>`__,
    `relhum_ice <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_ice.shtml>`__,
    `relhum_water <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_water.shtml>`__
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
        ``temperature``

    alternate_coeffs : bool, optional
        flag to use alternate set of coefficients appropriate for
        temperatures from 70F to 115F and humidities between 0% and 80%

    Returns
    -------
    heatindex : :class:`xarray.DataArray`
        Calculated heat index. Same shape as ``temperature``

    eqtype : :class:`int`
        version of equations used, for xarray attrs output

    See Also
    --------
    Related GeoCAT Functions:
    :func:`heat_index`
    :func:`_heat_index`

    Related NCL Functions:
    `heat_index_nws <https://www.ncl.ucar.edu/Document/Functions/Contributed/heat_index_nws.shtml>`__,
    """
    # Default coefficients for (t>=80F) and (40<gh<100)
    coeffs = [
        -42.379, 2.04901523, 10.14333127, -0.22475541, -0.00683783, -0.05481717,
        0.00122874, 0.00085282, -0.00000199
    ]
    crit = [80, 40, 100]  # [T_low [F], RH_low, RH_high]

    # optional flag coefficients for (70F<t<115F) and (0<gh<80)
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
            np.logical_and(relative_humidity < 13,
                           np.logical_and(temperature > 80, temperature < 112)),
            heatindex - ((13 - relative_humidity) / 4) * np.sqrt(
                (17 - abs(temperature - 95)) / 17), heatindex)

        heatindex = xr.where(
            np.logical_and(relative_humidity > 85,
                           np.logical_and(temperature > 80, temperature < 87)),
            heatindex + ((relative_humidity - 85.0) / 10.0) *
            ((87.0 - temperature) / 5.0), heatindex)

    return heatindex, eqtype


def _xrelhum(t: xr.DataArray, w: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    """Calculates relative humidity with respect to ice, given temperature,
    mixing ratio, and pressure.

     "Improved Magnus' Form Approx. of Saturation Vapor pressure"
     Oleg A. Alduchov and Robert E. Eskridge
     https://journals.ametsoc.org/view/journals/apme/35/4/1520-0450_1996_035_0601_imfaos_2_0_co_2.xml

    Parameters
    ----------
    t : :class:`xarray.DataArray`
        Temperature in Kelvin

    w : :class:`xarray.DataArray`
        Mixing ratio in kg/kg. Must have the same dimensions as ``temperature``

    p : :class:`xarray.DataArray`
        Pressure in Pa. Must have the same dimensions as ``temperature``

    Returns
    -------
    rh : :class:`xarray.DataArray`
        Relative humidity. Will have the same dimensions as ``temperature``


    See Also
    --------
    Related GeoCAT Functions:
    :func:`relhum`
    :func:`relhum_ice`
    :func:`relhum_water`

    Related NCL Functions:
    `relhum <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum.shtml>`__,
    `relhum_ttd <https://www.ncl.ucar.edu/Document/Functions/Contributed/relhum_ttd.shtml>`__,
    `relhum_ice <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_ice.shtml>`__,
    `relhum_water <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_water.shtml>`__
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
    temperature : ndarray, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Temperature in Kelvin

    relative_humidity : ndarray, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Relative humidity. Must be the same dimensions as ``temperature``

    Returns
    -------
    dew_pnt_temp : ndarray or :obj:`float`
        Dewpoint temperature in Kelvin. Same size as input variable temperature

    See Also
    --------
    Related GeoCAT Functions:
    :func:`_dewtemp`

    Related NCL Functions:
    `dewtemp_trh <https://www.ncl.ucar.edu/Document/Functions/Built-in/dewtemp_trh.shtml>`__
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

    The 'Heat Index' is a measure of how hot weather "feels" to the body. The combination of temperature and humidity
    produce an "apparent temperature" or the temperature the body "feels". The returned values are for shady
    locations only. Exposure to full sunshine can increase heat index values by up to 15°F. Also, strong winds,
    particularly with very hot, dry air, can be extremely hazardous as the wind adds heat to the body

    The computation of the heat index is a refinement of a result obtained by multiple regression analysis carried
    out by Lans P. Rothfusz and described in a 1990 National Weather Service (NWS) Technical Attachment (SR 90-23).
    All values less that 40F/4.4C/277.65K are set to the ambient temperature.

    In practice, the Steadman formula is computed first and the result averaged with the temperature. If this heat
    index value is 80 degrees F or higher, the full regression equation along with any adjustment as described above
    is applied. If the ambient temperature is less the 40F (4.4C/277.65K), the heat index is set to the ambient
    temperature.

    Parameters
    ----------
    temperature : ndarray, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        temperature(s) in Fahrenheit

    relative_humidity : ndarray, :class:`xarray.DataArray`, :class:`list`, :class:`float`
        relative humidity as a percentage. Must be the same shape as
        ``temperature``

    alternate_coeffs : bool, optional
        flag to use alternate set of coefficients appropriate for
        temperatures from 70F to 115F and humidities between 0% and 80%

    Returns
    -------
    heatindex : ndarray, :class:`xarray.DataArray`
        Calculated heat index. Same shape as ``temperature``

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
    :func:`_heat_index`
    :func:`_xheat_index`

    Related NCL Functions:
    `heat_index_nws <https://www.ncl.ucar.edu/Document/Functions/Contributed/heat_index_nws.shtml>`__
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
    https://journals.ametsoc.org/view/journals/apme/35/4/1520-0450_1996_035_0601_imfaos_2_0_co_2.xml

    Parameters
    ----------
    temperature : ndarray, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Temperature in Kelvin

    mixing_ratio : ndarray, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Mixing ratio in kg/kg. Must have the same dimensions as ``temperature``

    pressure : ndarray, :class:`xarray.DataArray`, :obj:`list`, or :obj:`float`
        Pressure in Pa. Must have the same dimensions as ``temperature``

    Returns
    -------
    relative_humidity : ndarray or :class:`xarray.DataArray`
        Relative humidity. Will have the same dimensions as ``temperature``


    See Also
    --------
    Related GeoCAT Functions:
    :func:`_xrelhum`
    :func:`relhum_ice`
    :func:`relhum_water`

    Related NCL Functions:
    `relhum <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum.shtml>`__,
    `relhum_ttd <https://www.ncl.ucar.edu/Document/Functions/Contributed/relhum_ttd.shtml>`__,
    `relhum_ice <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_ice.shtml>`__,
    `relhum_water <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_water.shtml>`__
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
        relative_humidity.attrs[
            'info'] = 'https://journals.ametsoc.org/view/journals/apme/35/4/1520-0450_1996_035_0601_imfaos_2_0_co_2.xml'

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
    https://journals.ametsoc.org/view/journals/apme/35/4/1520-0450_1996_035_0601_imfaos_2_0_co_2.xml

    Parameters
    ----------
    temperature : ndarray, :obj:`list`, or :obj:`float`
        Temperature in Kelvin

    mixing_ratio : ndarray, :obj:`list`, or :obj:`float`
        Mixing ratio in kg/kg. Must have the same dimensions as ``temperature``

    pressure : ndarray, :obj:`list`, or :obj:`float`
        Pressure in Pa. Must have the same dimensions as ``temperature``

    Returns
    -------
    relative_humidity : ndarray
        Relative humidity. Will have the same dimensions as ``temperature``

    See Also
    --------
    Related GeoCAT Functions:
    :func:`relhum`
    :func:`_xrelhum`
    :func:`relhum_water`
    :func:`_relhum_ice`

    Related NCL Functions:
    `relhum <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum.shtml>`__,
    `relhum_ttd <https://www.ncl.ucar.edu/Document/Functions/Contributed/relhum_ttd.shtml>`__,
    `relhum_ice <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_ice.shtml>`__,
    `relhum_water <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_water.shtml>`__
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

    Definition of mixing ratio if:

    - `es`  - is the saturation mixing ratio
    - `ep`  - is the ratio of the molecular weights of water vapor to dry air
    - `p`   - is the atmospheric pressure
    - `rh`  - is the relative humidity (given as a percent)

    .. math::
        rh =  100  q / ( (ep*es)/(p-es) )

    Parameters
    ----------
    temperature : ndarray, :obj:`list`, or :obj:`float`
        Temperature in Kelvin

    mixing_ratio : ndarray, :obj:`list`, or :obj:`float`
        Mixing ratio in kg/kg. Must have the same dimensions as ``temperature``

    pressure : ndarray, :obj:`list`, or :obj:`float`
        Pressure in Pa. Must have the same dimensions as ``temperature``

    Returns
    -------
    relative_humidity : ndarray
        Relative humidity. Will have the same dimensions as ``temperature``

    See Also
    --------
    Related GeoCAT Functions:
    :func:`relhum`
    :func:`_xrelhum`
    :func:`relhum_ice`
    :func:`relhum_water`

    Related NCL Functions:
    `relhum <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum.shtml>`__,
    `relhum_ttd <https://www.ncl.ucar.edu/Document/Functions/Contributed/relhum_ttd.shtml>`__,
    `relhum_ice <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_ice.shtml>`__,
    `relhum_water <https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum_water.shtml>`__
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


def max_daylight(
    jday: typing.Union[np.ndarray, xr.DataArray, list,
                       float], lat: typing.Union[np.ndarray, xr.DataArray, list,
                                                 float]
) -> typing.Union[np.ndarray, xr.DataArray, float]:
    """Computes maximum number of daylight hours as described in the Food and
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

    x_out = False
    if isinstance(jday, xr.DataArray):
        x_out = True

    # convert inputs to numpy arrays for function call if necessary
    if not xr.core.utils.is_duck_array(jday):
        jday = np.asarray(jday, dtype='float32')
    if not xr.core.utils.is_duck_array(lat):
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


def psychrometric_constant(
    pressure: typing.Union[np.ndarray, xr.DataArray, list, float]
) -> typing.Union[np.ndarray, xr.DataArray]:
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

    # Constant
    con = 0.66474e-3

    in_type = type(pressure)

    # Psychrometric constant calculation
    # if input not xarray, make sure in numpy for calculation
    if in_type is not xr.DataArray:
        psy_const = con * np.asarray(pressure)

    # else if input is xarray, add relevant metadata for xarray output
    else:
        psy_const = con * pressure
        psy_const.attrs['long_name'] = "psychrometric constant"
        psy_const.attrs['units'] = "kPa/C"
        psy_const.attrs[
            'url'] = "https://www.fao.org/docrep/X0490E/x0490e07.htm"
        psy_const.attrs['info'] = "FAO 56; EQN 8; psychrometric_constant"

    return psy_const


def saturation_vapor_pressure(
    temperature: typing.Union[np.ndarray, xr.DataArray, list, float],
    tfill: typing.Union[float] = np.nan
) -> typing.Union[np.ndarray, xr.DataArray]:
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
    :func:`actual_saturation_vapor_pressure`
    :func:`saturation_vapor_pressure_slope`

    Related NCL Functions:
    `satvpr_temp_fao56 <https://www.ncl.ucar.edu/Document/Functions/Crop/satvpr_temp_fao56.shtml>`__
    """

    in_type = type(temperature)

    if in_type is xr.DataArray:

        # convert temperature to Celsius
        temp_c = (temperature - 32) * 5 / 9

        # calculate svp
        svp = xr.where(temp_c > 0, 0.6108 * np.exp(
            (17.27 * temp_c) / (temp_c + 237.3)), tfill)

        # add relevant metadata
        svp.attrs['long_name'] = "saturation vapor pressure"
        svp.attrs['units'] = "kPa"
        svp.attrs['url'] = "https://www.fao.org/docrep/X0490E/x0490e07.htm"
        svp.attrs['info'] = "FAO 56; EQN 11; saturation_vapor_pressure"

    else:
        temperature = np.asarray(temperature)

        temp_c = (temperature - 32) * 5 / 9
        svp = np.where(temp_c > 0, 0.6108 * np.exp(
            (17.27 * temp_c) / (temp_c + 237.3)), tfill)

    return svp


def actual_saturation_vapor_pressure(
    tdew: typing.Union[np.ndarray, xr.DataArray, list, float],
    tfill: typing.Union[float] = np.nan
) -> typing.Union[np.ndarray, xr.DataArray]:
    """Compute 'actual' saturation vapor pressure [kPa] as described in the
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
    :func:`saturation_vapor_pressure`
    :func:`saturation_vapor_pressure_slope`

    Related NCL Functions:
    `satvpr_tdew_fao56 <https://www.ncl.ucar.edu/Document/Functions/Crop/satvpr_tdew_fao56.shtml>`__
    """

    in_type = type(tdew)

    asvp = saturation_vapor_pressure(tdew, tfill)

    # reformat metadata for xarray
    if in_type is xr.DataArray:
        asvp.attrs['long_name'] = "actual saturation vapor pressure via Tdew"
        asvp.attrs['units'] = "kPa"
        asvp.attrs['url'] = "https://www.fao.org/docrep/X0490E/x0490e07.htm"
        asvp.attrs['info'] = "FAO 56; EQN 14; actual_saturation_vapor_pressure"

    return asvp


def saturation_vapor_pressure_slope(
    temperature: typing.Union[np.ndarray, xr.DataArray, list, float],
    tfill: typing.Union[float] = np.nan
) -> typing.Union[np.ndarray, xr.DataArray]:
    """Compute the slope [kPa/C] of saturation vapor pressure curve as
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
    :func:`actual_saturation_vapor_pressure`
    :func:`saturation_vapor_pressure_slope`

    Related NCL Functions:
    `satvpr_temp_fao56 <https://www.ncl.ucar.edu/Document/Functions/Crop/satvpr_temp_fao56.shtml>`__
    """
    in_type = type(temperature)

    if in_type is xr.DataArray:

        # convert to Celsius
        temp_c = (temperature - 32) * 5 / 9

        # calculate svp_slope
        svp_slope = xr.where(
            temp_c > 0, 4096 * (0.6108 * np.exp(
                (17.27 * temp_c) / (temp_c + 237.3)) / (temp_c + 237.3)**2),
            tfill)

        # add relevant metadata
        svp_slope.attrs['long_name'] = "slope saturation vapor pressure curve"
        svp_slope.attrs['units'] = "kPa/C"
        svp_slope.attrs[
            'url'] = "https://www.fao.org/docrep/X0490E/x0490e07.htm"
        svp_slope.attrs[
            'info'] = "FAO 56; EQN 13; saturation_vapor_pressure_slope"

    else:
        temperature = np.asarray(temperature)

        # convert to Celsius
        temp_c = (temperature - 32) * 5 / 9

        # calculate svp_slope
        svp_slope = np.where(
            temp_c > 0, 4096 * (0.6108 * np.exp(
                (17.27 * temp_c) / (temp_c + 237.3)) / (temp_c + 237.3)**2),
            tfill)

    return svp_slope


def _delta_pressure1D(pressure_lev, surface_pressure):
    """Helper function for `delta_pressure`. Calculates the pressure layer
    thickness (delta pressure) of a one-dimensional pressure level array.

    Returns an array of length matching `pressure_lev`.

    Parameters
    ----------
    pressure_lev : :class:`numpy.ndarray`
        The pressure level array. May be in ascending or descending order.
        Must have the same units as `surface_pressure`.

    surface_pressure : :class:`float`
        The scalar surface pressure. Must have the same units as
        `pressure_lev`.

    Returns
    -------
    delta_pressure : :class:`numpy.ndarray`
        The pressure layer thickness array. Shares dimensions and units of
        `pressure_lev`.
    """
    pressure_top = min(pressure_lev)

    # Safety checks
    if pressure_top <= 0:
        warnings.warn("'pressure_lev` values must all be positive.")
    if pressure_top > surface_pressure:
        warnings.warn(
            "`surface_pressure` must be greater than minimum `pressure_lev` value."
        )

    # Sort so pressure increases (array goes from top of atmosphere to bottom)
    is_pressuredecreasing = pressure_lev[1] < pressure_lev[0]
    if is_pressuredecreasing:
        pressure_lev = np.flip(pressure_lev)

    # Calculate delta pressure
    delta_pressure = np.empty_like(pressure_lev)

    delta_pressure[0] = (pressure_lev[0] +
                         pressure_lev[1]) / 2 - pressure_top  # top level
    delta_pressure[1:-1] = [
        (a - b) / 2 for a, b in zip(pressure_lev[2:], pressure_lev[:-1])
    ]
    delta_pressure[-1] = surface_pressure - (
        pressure_lev[-1] + pressure_lev[-2]) / 2  # bottom level

    # Return delta_pressure to original order
    if is_pressuredecreasing:
        delta_pressure = np.flip(delta_pressure)

    return delta_pressure


def delta_pressure(pressure_lev, surface_pressure):
    """Calculates the pressure layer thickness (delta pressure) of a constant
    pressure level coordinate system.

    Returns an array of shape matching (``surface_pressure``, ``pressure_lev``).

    Parameters
    ----------
    pressure_lev : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        The pressure level array. May be in ascending or descending order.
        Must have the same units as ``surface_pressure``.

    surface_pressure : :class:`int`, :class:`float`, :class:`numpy.ndarray`, :class:`xarray.DataArray`
        The scalar or N-dimensional surface pressure array. Must have the same
        units as ``pressure_lev``.

    Returns
    -------
    delta_pressure : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        The pressure layer thickness array. Shares units with ``pressure_lev``.
        If ``surface_pressure`` is scalar, shares dimensions with
        ``pressure_level``. If ``surface_pressure`` is an array than the returned
        array will have an additional dimension [e.g. (lat, lon, time) becomes
        (lat, lon, time, lev)]. Will always be the same type as ``surface_pressure``.

    See Also
    --------
    Related NCL Functions:
    `dpres_plev <https://www.ncl.ucar.edu/Document/Functions/Built-in/dpres_plevel.shtml>`__
    """
    # Get original array types
    type_surface_pressure = type(
        surface_pressure
    )  # save type for delta_pressure to same type as surface_pressure at end
    type_pressure_level = type(pressure_lev)

    # Preserve attributes for Xarray
    if type_surface_pressure == xr.DataArray:
        da_coords = dict(surface_pressure.coords)
        da_attrs = dict(surface_pressure.attrs)
        da_dims = surface_pressure.dims
    if type_pressure_level == xr.DataArray:
        da_attrs = dict(
            pressure_lev.attrs)  # Overwrite attributes to match pressure_lev

    # Calculate delta pressure
    if np.isscalar(surface_pressure):  # scalar case
        delta_pressure = _delta_pressure1D(pressure_lev, surface_pressure)
    else:  # multi-dimensional cases
        shape = surface_pressure.shape
        delta_pressure_shape = shape + (len(pressure_lev),
                                       )  # preserve shape for reshaping

        surface_pressure_flattened = np.ravel(
            surface_pressure)  # flatten to avoid nested for loops
        delta_pressure = [
            _delta_pressure1D(pressure_lev, e)
            for e in surface_pressure_flattened
        ]

        delta_pressure = np.array(delta_pressure).reshape(delta_pressure_shape)

    # If passed in an Xarray array, return an Xarray array
    # Change this to return a dataset that has both surface pressure and delta pressure?
    if type_surface_pressure == xr.DataArray:
        da_coords['lev'] = pressure_lev.values if (
            type_pressure_level == xr.DataArray) else pressure_lev
        da_dims = da_dims + ("lev",)
        da_attrs.update({"long name": "pressure layer thickness"})
        delta_pressure = xr.DataArray(delta_pressure,
                                      coords=da_coords,
                                      dims=da_dims,
                                      attrs=da_attrs,
                                      name="delta pressure")

    return delta_pressure


# NCL NAME WRAPPER FUNCTIONS BELOW
def dpres_plev(pressure_lev, surface_pressure):
    return delta_pressure(pressure_lev, surface_pressure)


_generate_wrapper_docstring(dpres_plev, delta_pressure)
