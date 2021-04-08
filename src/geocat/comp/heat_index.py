import warnings

import numpy as np
import xarray as xr

from .comp_util import _is_duck_array


def heat_index(t, rh, alt_coef):
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
    t : numpy.ndarray, xr.DataArray, float
        temperature(s) in Fahrenheit

    rh : numpy.ndarray, xr.DataArray, float
        relative humidity as a percentage. Must be the same shape as t

    alt_coef : Boolean
        option to use alternate set of coefficients appropriate for
        temperatures from 70F to 115F and humidities between 0% and 80%

    Returns
    -------
    hi : numpy.ndarray, xr.DataArray, list, float
        Calculated heat index. Same shape as t
    """

    x_out = False
    if isinstance(t, xr.DataArray):
        x_out = True

    # convert inputs to numpy arrays if necessary
    if not _is_duck_array(t):
        jday = np.asarray(t, dtype='float32')
    if not _is_duck_array(rh):

        lat = np.asarray(rh, dtype='float32')

    # check to ensure dimensions of inputs not greater than 1
    if t.ndim > 1 or rh.ndim > 1:
        raise ValueError('heat_index: inputs must have at most one dimension')

    # Check if relative humidity fractional
    if all(rh < 1):
        warnings.warn("WARNING: rh must be %, not fractional; All rh are < 1")

    # Default coefficients for (t>=80F) and (40<gh<100)
    c = [
        -42.379, 2.04901523, 10.14333127, -0.22475541, -0.00683783, -0.05481717,
        0.00122874, 0.00085282, -0.00000199
    ]
    crit = [80, 40, 100]  # [T_low [F], RH_low, RH_high]

    # Optional flag coefficients for (70F<t<115F) and (0<gh<80)
    # within 3F of default coefs
    if alt_coef:
        c = [
            0.363445176, 0.988622465, 4.777114035, -0.114037667, -0.000850208,
            -0.020716198, 0.000687678, 0.000274954, 0.0
        ]
        crit = [70, 0, 80]  # [T_low [F], RH_low, RH_high]

    # NWS practice
    # average Steadman and t
    hi = (0.5 * (t + 61.0 + ((t - 68.0) * 1.2) + (rh * 0.094)) + t) * 0.5

    # http://ehp.niehs.nih.gov/1206273/
    hi = [(ti if ti < 40.0 else hii) for hii, ti in zip(hi, t)]

    # if all t values less than critical, return hi
    # otherwise perform calculation
    if all(ti < crit[0] for ti in t):
        return hi
    else:
        for i in range(len(hi)):
            if hi[i] >= crit[0]:
                hi[i] = c[0] \
                        + c[1] * t[i] \
                        + c[2] * rh[i] \
                        + c[3] * t[i] * rh[i] \
                        + c[4] * t[i] ** 2 \
                        + c[5] * rh[i] ** 2 \
                        + c[6] * t[i] ** 2 * rh[i] \
                        + c[7] * t[i] * rh[i] ** 2 \
                        + c[8] * t[i] ** 2 * rh[i] ** 2

            # adjustments
            if rh[i] < 13 and (80 < t[i] < 112):
                hi[i] = hi[i] - ((13 - rh[i]) / 4) * np.sqrt(
                    (17 - abs(t[i] - 95)) / 17)

            if rh[i] > 85 and (80 < t[i] < 87):
                hi[i] = hi[i] + ((rh[i] - 85.0) / 10.0) * ((87.0 - t[i]) / 5.0)

        return hi


# t = [104, 100, 92, 92, 86, 80, 80, 60, 30]
# rh = [55, 65, 60, 90, 90, 40, 75, 90, 50]
#
# print(heat_index(t, rh, False))
