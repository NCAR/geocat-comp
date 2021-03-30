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

    # If xarray input, pull data and store metadata
    x_out = False
    if isinstance(temperature, xr.DataArray):
        x_out = True
        save_dims = temperature.dims
        save_coords = temperature.coords
        save_attrs = temperature.attrs

    # ensure in numpy array for function call
    temperature = np.asarray(temperature)
    relative_humidity = np.asarray(relative_humidity)

    # make sure the input arrays are of the same size
    if np.shape(temperature) != np.shape(relative_humidity):
        raise ValueError(
            f"dewtemp_trh: dimensions of temperature, {np.shape(temperature)}, and relative_humidity, "
            f"{np.shape(relative_humidity)}, do not match")

    # Call mapblocks to run function
    dew_pnt_temp = _dewtemp(temperature, relative_humidity)

    # output as xarray if input as xarray
    if x_out:
        dew_pnt_temp = xr.DataArray(data=dew_pnt_temp,
                                    coords=save_coords,
                                    dims=save_dims,
                                    attrs=save_attrs)

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
