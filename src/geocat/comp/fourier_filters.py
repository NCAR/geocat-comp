import math as m

import numpy as np
import xarray as xr


def fourier_filter(signal,
                   frequency,
                   cutoff_frequency_low=0,
                   cutoff_frequency_high=0,
                   time_axis=0,
                   high_pass=False,
                   low_pass=False,
                   band_pass=False,
                   band_block=False):
    """Filter a dataset by frequency. This function allowes for low-pass, high-
    pass, band-pass, or band-block filtering of the data's freqency
    representation.

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
    Example 1: The tidal cycle needs to be removed from a 10/hr oceanic dataset,
    (https://tidesandcurrents.noaa.gov/waterlevels.html?id=9415020&units=standard&bdate=20210101&edate=20210131&timezone=GMT&datum=MLLW&interval=6&action=data)
    numpy_dataset(lat,lon,time,[sea_surface_height,sea_surface_temperature])
    dataset shape is assumed to be (7440,)
    the frequency resolution will be 0.00137 (cycles/day)
    to remove the semidiurnal tidal cycle, ~1.9323 cycles per day
    to remove the semidiurnal tidal cycle, use a band-block filter centered on
    1.9323 cycles per day, with a width of
    lower bound should be 1.9323-0.00137
    >>> result = fourier_filter(dataset,1,cutoff_frequency_low=1.9,,)

    >>> xarray_dataset = xarray.DataArray(dataset)
    dataset can, but should not, be chunked in the time dimension,
    all other dimensions may be chunked at will, we will chunk by lat and lon
    >>> xarray_dataset.chunk((1,1,730,2))
    """
    resolution = frequency / len(signal)
    signal = np.swapaxes(signal, time_axis, 0)
    res_fft = np.fft.fft(signal, axis=0)
    cfl_index = m.floor(cutoff_frequency_low / resolution)
    cfln_index = 1 - cfl_index
    cfh_index = m.ceil(cutoff_frequency_high / resolution)
    cfhn_index = 1 - cfh_index
    if low_pass:
        res_fft[cfl_index:cfln_index] = np.zeros(
            res_fft[cfl_index:cfln_index].shape)
    if high_pass:
        res_fft[:cfh_index] = np.zeros(res_fft[:cfh_index].shape)
        res_fft[cfhn_index:] = np.zeros(res_fft[cfhn_index:].shape)
    if band_pass:
        res_fft[:cfl_index] = np.zeros(res_fft[:cfl_index].shape)
        res_fft[cfh_index:cfhn_index] = np.zeros(
            res_fft[cfh_index:cfhn_index].shape)
        res_fft[cfln_index:] = np.zeros(res_fft[cfln_index:].shape)
    if band_block:
        res_fft[cfl_index:cfh_index] = np.zeros(
            res_fft[cfl_index:cfh_index].shape)
        res_fft[cfhn_index:cfln_index] = np.zeros(
            res_fft[cfhn_index:cfln_index].shape)
    result = np.real(np.fft.ifft(res_fft, axis=0))
    result = np.swapaxes(result, time_axis, 0)
    if type(signal) == xr.DataArray:
        xr_result = signal.copy()
        xr_result.data = result
        result = xr_result
    return result


def fourier_low_pass(signal, frequency, cutoff_frequency, time_axis=0):
    return fourier_filter(signal,
                          frequency,
                          cutoff_frequency_low=cutoff_frequency,
                          time_axis=time_axis,
                          low_pass=True)


def fourier_high_pass(signal, frequency, cutoff_frequency, time_axis=0):
    return fourier_filter(signal,
                          frequency,
                          cutoff_frequency_high=cutoff_frequency,
                          time_axis=time_axis,
                          high_pass=True)


def fourier_band_pass(signal,
                      frequency,
                      cutoff_frequency_low,
                      cutoff_frequency_high,
                      time_axis=0):
    return fourier_filter(signal,
                          frequency,
                          cutoff_frequency_low=cutoff_frequency_low,
                          cutoff_frequency_high=cutoff_frequency_high,
                          time_axis=time_axis,
                          band_pass=True)


def fourier_band_block(signal,
                       frequency,
                       cutoff_frequency_low,
                       cutoff_frequency_high,
                       time_axis=0):
    return fourier_filter(signal,
                          frequency,
                          cutoff_frequency_low=cutoff_frequency_low,
                          cutoff_frequency_high=cutoff_frequency_high,
                          time_axis=time_axis,
                          band_block=True)
