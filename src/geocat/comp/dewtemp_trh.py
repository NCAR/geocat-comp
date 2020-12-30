import numpy as np
import math


def dewtemp_trh(temperature, relative_humidity):
    """ This function calculates the dew point temperature given temperature and relative humidity
            using equations from John Dutton's "Ceaseless Wind" (pp 273-274)

            Parameters
            ----------
            temperature : numpy.ndarray, list, or float
                Temperature in K
            relative_humidity : numpy.ndarray, list, or float
                Relative humidity. Must be the same dimensions as temperature


            Returns
            -------
            tdk : numpy.ndarray
                Relative humidity. Same size as input variable tk
    """

    # make sure the input arrays are of the same size
    if np.shape(temperature) != np.shape(relative_humidity):
        raise ValueError(
            f"dewtemp_trh: dimensions of temperature, {np.shape(temperature)}, and relative_humidity, "
            f"{np.shape(relative_humidity)}, do not match")
    else:
        # store original shape
        shape = np.shape(temperature)

    # convert inputs to np arrays
    temperature = np.asarray(temperature)
    relative_humidity = np.asarray(relative_humidity)

    # make an empty space for output array
    tdk = np.zeros(np.size(temperature))

    # fill in output array
    for i in range(np.size(temperature)):
        tdk[i] = _dewtemp(
            np.ravel(temperature)[i],
            np.ravel(relative_humidity)[i])

    # reshape output array to match the input dimensions
    tdk = np.reshape(tdk, shape)

    return tdk


def _dewtemp(tk, rh):
    """ This function calculates the dew point temperature given temperature and relative humidity
        using equations from John Dutton's "Ceaseless Wind" (pp 273-274)

        Parameters
        ----------
        tk : float
            Temperature in K
        rh : float
            Relative humidity


        Returns
        -------
        tdk : float
            Relative humidity

    """

    gc = 461.5  # gas constant for water vapor [j/{kg-k}]
    gcx = gc / (1000 * 4.186)  # [cal/{g-k}]

    lhv = (597.3 - 0.57 * (tk - 273)) / gcx
    tdk = tk * lhv / (lhv - tk * math.log(rh * 0.01))

    return tdk
