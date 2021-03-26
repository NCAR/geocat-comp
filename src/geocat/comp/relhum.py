import numpy as np
import xarray as xr


def relhum(temperature, mixing_ratio, pressure):
    """This function calculates the relative humidity given temperature, mixing
    ratio, and pressure.

    "Improved Magnus' Form Approx. of Saturation Vapor pressure"
    Oleg A. Alduchov and Robert E. Eskridge
    http://www.osti.gov/scitech/servlets/purl/548871/
    https://doi.org/10.2172/548871

        Args:

            temperature (:class:`numpy.ndarray`, :class:`xr.DataArray`, :obj:`list`, or :obj:`float`):
                Temperature in Kelvin

            mixing_ratio (:class:`numpy.ndarray`, :class:`xr.DataArray`, :obj:`list`, or :obj:`float`):
                Mixing ratio in kg/kg. Must have the same dimensions as temperature

            pressure (:class:`numpy.ndarray`, :class:`xr.DataArray`, :obj:`list`, or :obj:`float`):
                Pressure in Pa. Must have the same dimensions as temperature

        Returns:

            relative_humidity (:class:`numpy.ndarray` or :class:`xr.DataArray`):
                Relative humidity. Will have the same dimensions as temperature
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
        raise ValueError(f"relhum: dimensions of inputs are not the same")

    relative_humidity = _relhum(temperature, mixing_ratio, pressure)

    # output as xarray if input as xarray
    if x_out:
        relative_humidity = xr.DataArray(data=relative_humidity,
                                         coords=save_coords,
                                         dims=save_dims,
                                         attrs=save_attrs)

    return relative_humidity


def relhum_water(temperature, mixing_ratio, pressure):
    """Calculates relative humidity with respect to water, given temperature,
    mixing ratio, and pressure.

     Definition of mixing ratio if,
     es  - is the saturation mixing ratio
     ep  - is the ratio of the molecular weights of water vapor to dry air
     p   - is the atmospheric pressure
     rh  - is the relative humidity (given as a percent)

     rh =  100*  q / ( (ep*es)/(p-es) )

    Args:

         temperature (:class:`numpy.ndarray`, :obj:`list`, or :obj:`float`):
             Temperature in Kelvin

         mixing_ratio (:class:`numpy.ndarray`, :obj:`list`, or :obj:`float`):
             Mixing ratio in kg/kg. Must have the same dimensions as temperature

         pressure (:class:`numpy.ndarray`, :obj:`list`, or :obj:`float`):
             Pressure in Pa. Must have the same dimensions as temperature

     Returns:

         relative_humidity (:class:`numpy.ndarray`):
             Relative humidity. Will have the same dimensions as temperature
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


def relhum_ice(temperature, mixing_ratio, pressure):
    """Calculates relative humidity with respect to ice, given temperature,
    mixing ratio, and pressure.

     "Improved Magnus' Form Approx. of Saturation Vapor pressure"
     Oleg A. Alduchov and Robert E. Eskridge
     http://www.osti.gov/scitech/servlets/purl/548871/
     https://doi.org/10.2172/548871

    Args:

         temperature (:class:`numpy.ndarray`, :obj:`list`, or :obj:`float`):
             Temperature in Kelvin

         mixing_ratio (:class:`numpy.ndarray`, :obj:`list`, or :obj:`float`):
             Mixing ratio in kg/kg. Must have the same dimensions as temperature

         pressure (:class:`numpy.ndarray`, :obj:`list`, or :obj:`float`):
             Pressure in Pa. Must have the same dimensions as temperature

     Returns:

         relative_humidity (:class:`numpy.ndarray`):
             Relative humidity. Will have the same dimensions as temperature
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


def _relhum(t, w, p):
    """Calculates relative humidity with respect to ice, given temperature,
    mixing ratio, and pressure.

     "Improved Magnus' Form Approx. of Saturation Vapor pressure"
     Oleg A. Alduchov and Robert E. Eskridge
     http://www.osti.gov/scitech/servlets/purl/548871/
     https://doi.org/10.2172/548871

    Args:

         t (:class:`numpy.ndarray`, :obj:`list`, or :obj:`float`):
             Temperature in Kelvin

         w (:class:`numpy.ndarray`, :class:`xr.DataArray`, :obj:`list`, or :obj:`float`):
             Mixing ratio in kg/kg. Must have the same dimensions as temperature

         p (:class:`numpy.ndarray`, :class:`xr.DataArray`, :obj:`list`, or :obj:`float`):
             Pressure in Pa. Must have the same dimensions as temperature

     Returns:

         rh (:class:`numpy.ndarray):
             Relative humidity. Will have the same dimensions as temperature
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


def _relhum_ice(t, w, p):
    """Calculates relative humidity with respect to ice, given temperature,
    mixing ratio, and pressure.

     "Improved Magnus' Form Approx. of Saturation Vapor pressure"
     Oleg A. Alduchov and Robert E. Eskridge
     http://www.osti.gov/scitech/servlets/purl/548871/
     https://doi.org/10.2172/548871

    Args:

         t (:class:`numpy.ndarray`, :obj:`list`, or :obj:`float`):
             Temperature in Kelvin

         w (:class:`numpy.ndarray`, :obj:`list`, or :obj:`float`):
             Mixing ratio in kg/kg. Must have the same dimensions as temperature

         p (:class:`numpy.ndarray`, :obj:`list`, or :obj:`float`):
             Pressure in Pa. Must have the same dimensions as temperature

     Returns:

         rh (:class:`numpy.ndarray`):
             Relative humidity. Will have the same dimensions as temperature
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


def _relhum_water(t, w, p):
    """Calculates relative humidity with respect to water, given temperature,
    mixing ratio, and pressure.

     Definition of mixing ratio if,
     es  - is the saturation mixing ratio
     ep  - is the ratio of the molecular weights of water vapor to dry air
     p   - is the atmospheric pressure
     rh  - is the relative humidity (given as a percent)

     rh =  100*  q / ( (ep*es)/(p-es) )

    Args:

         t (:class:`numpy.ndarray`, :obj:`list`, or :obj:`float`):
             Temperature in Kelvin

         w (:class:`numpy.ndarray`, :obj:`list`, or :obj:`float`):
             Mixing ratio in kg/kg. Must have the same dimensions as temperature

         p (:class:`numpy.ndarray`, :obj:`list`, or :obj:`float`):
             Pressure in Pa. Must have the same dimensions as temperature

     Returns:

         rh (:class:`numpy.ndarray`):
             Relative humidity. Will have the same dimensions as temperature
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
