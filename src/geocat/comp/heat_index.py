import warnings

import numpy as np
import xarray as xr

from .comp_util import _is_duck_array


def _nws_eqn(coeffs, temp, rel_hum):
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


def heat_index(temperature, relative_humidity, alternate_coeffs=False):
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
    temperature : numpy.ndarray, xr.DataArray, float
        temperature(s) in Fahrenheit

    relative_humidity : numpy.ndarray, xr.DataArray, float
        relative humidity as a percentage. Must be the same shape as
        temperature

    alternate_coeffs : Boolean, Optional
        flag to use alternate set of coefficients appropriate for
        temperatures from 70F to 115F and humidities between 0% and 80%

    Returns
    -------
    heatindex : numpy.ndarray, xr.DataArray
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
    """

    x_out = False
    if isinstance(temperature, xr.DataArray):
        x_out = True
        save_dims = temperature.dims
        save_coords = temperature.coords

    # convert inputs to numpy arrays if necessary
    if not _is_duck_array(temperature):
        temperature = np.atleast_1d(temperature)
    if not _is_duck_array(relative_humidity):
        relative_humidity = np.atleast_1d(relative_humidity)

    # Input validation on relative humidity
    if any(relative_humidity.ravel() < 0) or any(
            relative_humidity.ravel() > 100):
        raise ValueError('heat_index: invalid values for relative humidity')

    # Check if relative humidity fractional
    if all(relative_humidity.ravel() < 1):
        warnings.warn("WARNING: rh must be %, not fractional; All rh are < 1")

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
    heatindex = np.where(temperature < 40, temperature, heatindex)

    # if all t values less than critical, return hi
    # otherwise perform calculation
    eqtype = 0
    if not all(temperature.ravel() < crit[0]):
        eqtype = 1

        heatindex = np.where(heatindex > crit[0],
                             _nws_eqn(coeffs, temperature, relative_humidity),
                             heatindex)

        # adjustments
        heatindex = np.where(
            np.logical_and(relative_humidity < 13,
                           np.logical_and(temperature > 80, temperature < 112)),
            heatindex - ((13 - relative_humidity) / 4) * np.sqrt(
                (17 - abs(temperature - 95)) / 17), heatindex)

        heatindex = np.where(
            np.logical_and(relative_humidity > 85,
                           np.logical_and(temperature > 80, temperature < 87)),
            heatindex + ((relative_humidity - 85.0) / 10.0) *
            ((87.0 - temperature) / 5.0), heatindex)

    # reformat output for xarray if necessary
    if x_out:
        heatindex = xr.DataArray(heatindex, coords=save_coords, dims=save_dims)
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

    return heatindex
