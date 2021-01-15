import numpy as np
import xarray as xr
import dask.array as da
from dask.array.core import map_blocks


def dewtemp_trh(temperature, relative_humidity):
    """ This function calculates the dew point temperature given temperature and relative humidity
            using equations from John Dutton's "Ceaseless Wind" (pp 273-274)

            Parameters
            ----------
            temperature : numpy.ndarray, xr.DataArray, list, or float
                Temperature in K
            relative_humidity : numpy.ndarray, xr.DataArray, list, or float
                Relative humidity. Must be the same dimensions as temperature


            Returns
            -------
            tdk : numpy.ndarray, xr.DataArray, list, or float
                Relative humidity. Same size as input variable tk
    """

    # make sure the input arrays are of the same size
    if np.shape(temperature) != np.shape(relative_humidity):
        raise ValueError(
            f"dewtemp_trh: dimensions of temperature, {np.shape(temperature)}, and relative_humidity, "
            f"{np.shape(relative_humidity)}, do not match")

    # see if single value input and skip dask if appropriate
    if np.size(temperature) == 1:
        return _dewtemp(temperature, relative_humidity)

    # ''' Start of boilerplate
    if not isinstance(temperature, xr.DataArray):
        temperature = xr.DataArray(temperature)
        temperature = da.from_array(temperature, chunks="auto")

    if not isinstance(relative_humidity, xr.DataArray):
        relative_humidity = xr.DataArray(relative_humidity)
        relative_humidity = da.from_array(relative_humidity, chunks="auto")

    tdk = map_blocks(_dewtemp, temperature, relative_humidity)
    tdk = tdk.compute()

    # tdk = _dewtemp(temperature, relative_humidity)
    return tdk


def _dewtemp(tk, rh):
    """ This function calculates the dew point temperature given temperature and relative humidity
        using equations from John Dutton's "Ceaseless Wind" (pp 273-274)

        Parameters
        ----------
        tk : numpy.ndarray, xr.DataArray, list, or float
            Temperature in K
        rh : numpy.ndarray, xr.DataArray, list, or float
            Relative humidity


        Returns
        -------
        tdk : numpy.ndarray, xr.DataArray, list, or float
            Relative humidity

    """

    gc = 461.5  # gas constant for water vapor [j/{kg-k}]
    gcx = gc / (1000 * 4.186)  # [cal/{g-k}]

    lhv = (597.3 - 0.57 * (tk - 273)) / gcx
    tdk = tk * lhv / (lhv - tk * np.log(rh * 0.01))

    return tdk
