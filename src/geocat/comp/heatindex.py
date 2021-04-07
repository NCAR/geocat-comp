import warnings

import numpy as np


def heat_index(t, rh, opt):
    """Compute the 'heat index' as calculated by the National Weather Service.

    Parameters
    ----------
    t : numpy.ndarray, xr.DataArray, float
        temperature(s) in Fahrenheit

    rh : numpy.ndarray, xr.DataArray, float
        relative humidity as a percentage. Must be the same shape as t

    opt : Boolean

    Returns
    -------
    hi : numpy.ndarray, xr.DataArray, list, float
        Calculated heat index. Same shape as t
    """

    # Default coefficients for (t>=80F) and (40<gh<100)
    c = [
        -42.379, 2.04901523, 10.14333127, -0.22475541, -0.00683783, -0.05481717,
        0.00122874, 0.00085282, -0.00000199
    ]
    crit = [80, 40, 100]  # [T_low [F], RH_low, RH_high]

    # Optional flag coefficients for (70F<t<115F) and (0<gh<80)
    # within 3F of default coefs
    if opt:
        c = [
            0.363445176, 0.988622465, 4.777114035, -0.114037667, -0.000850208,
            -0.020716198, 0.000687678, 0.000274954, 0.0
        ]
        crit = [70, 0, 80]  # [T_low [F], RH_low, RH_high]

    # NWS practice
    # average Steadman and t
    hi = np.asarray(
        (0.5 * (t + 61.0 + ((t - 68.0) * 1.2) + (rh * 0.094)) + t) * 0.5)

    # http://ehp.niehs.nih.gov/1206273/
    hi = [(ti if ti < 40.0 else hii) for hii, ti in zip(hi, t)]

    # if all t values less than critical, return hi
    # otherwise perform calculation
    if all(ti < crit[0] for ti in t):
        eqnType = 0
        return hi
    else:
        eqnType = 1
        for i in range(len(hi)):
            if hi[i] > crit[0]:
                hi[i] = c[0] + c[1] * t[i] + c[2] * rh[i] + c[3] * t[i] * rh[i]
                +c[4] * t[i]**2 + c[5] * rh[i]**2 + c[6] * t[i]**2 * rh
                +c[7] * t[i] * rh[i]**2 + c[8] * t[i]**2 * rh[i]**2

            # adjustments
            if rh[i] < 13 and (80 < t[i] < 112):
                hi[i] = hi[i] - ((13 - rh[i]) / 4) * np.sqrt(
                    (17 - abs(t[i] - 95)) / 17)

            if rh[i] > 85 and (80 < t[i] < 87):
                hi[i] = hi[i] + ((rh[i] - 85) / 10) * ((87 - t[i]) / 5)

        return hi
