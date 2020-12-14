import numpy as np
import math


def dewtemp_trh(tk, rh):
    """ This function calculates the dew point temperature given temperature and relative humidity.

        Parameters
        ----------
        tk : numpy.ndarray
            Temperature in K
        rh : numpy.ndarray
            Relative humidity. Must be the same size as tk


        Returns
        -------
        tdk : numpy.ndarray
            Relative humidity. Same size as input variable tk

    """

    gc = 461.5              # gas constant for water vapor [j/{kg-k}]
    gcx = gc/(1000*4.186)   # [cal/{g-k}]

    lhv = (597.3-0.57 * (tk-273))/gcx
    tdk = tk*lhv / (lhv - tk * math.log(rh*0.01))

    return tdk
