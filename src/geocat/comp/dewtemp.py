import numpy as np
import xarray as xr


def dewtemp(temperature, relative_humidity):
    """This function calculates the dew point temperature given temperature and
    relative humidity using equations from John Dutton's "Ceaseless Wind" (pp
    273-274)

    Args:

        temperature (:class:`numpy.ndarray`, :class:`xr.DataArray`, :obj:`list`, or :obj:`float`):
            Temperature in K

        relative_humidity (:class:`numpy.ndarray`, :class:`xr.DataArray`, :obj:`list`, or :obj:`float`):
            Relative humidity. Must be the same dimensions as temperature


    Returns:

        dew_pnt_temp (:class:`numpy.ndarray` or :obj:`float`):
            Dewpoint temperature in Kelvin. Same size as input variable temperature
    """

    inputs = [temperature, relative_humidity]

    # ensure all inputs same size
    if not (np.shape(x) == np.shape(inputs[0]) for x in inputs):
        raise ValueError("dewtemp: dimensions of inputs are not the same")

    # Get input types
    in_types = [type(item) for item in inputs]

    if xr.DataArray in in_types:

        # check all inputs are xr.DataArray
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


def _dewtemp(tk, rh):
    """This function calculates the dew point temperature given temperature and
    relative humidity using equations from John Dutton's "Ceaseless Wind" (pp
    273-274)

    Args:

        tk (:class:`numpy.ndarray`, :class:`xr.DataArray`, :obj:`list`, or :obj:`float`):
            Temperature in K

        rh (:class:`numpy.ndarray`, :class:`xr.DataArray`, :obj:`list`, or :obj:`float`):
            Relative humidity. Must be the same dimensions as temperature


    Returns:

        tdk (:class:`numpy.ndarray`, :class:`xr.DataArray`, :obj:`list`, or :obj:`float`):
            Dewpoint temperature in Kelvin. Same size as input variable temperature
    """

    gc = 461.5  # gas constant for water vapor [j/{kg-k}]
    gcx = gc / (1000 * 4.186)  # [cal/{g-k}]

    lhv = (597.3 - 0.57 * (tk - 273)) / gcx
    tdk = tk * lhv / (lhv - tk * np.log(rh * 0.01))

    return tdk
