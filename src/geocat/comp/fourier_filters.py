import math as m

import numpy as np


def fourier_filter(signal,
                   frequency,
                   cutoff_frequency_low=0,
                   cutoff_frequency_high=0,
                   time_axis=0,
                   high_pass=False,
                   low_pass=False,
                   band_pass=False,
                   band_block=False):
    """TODO list:

    [ ] axis reordering is time axis is not 0 [ ] test against
    dask.client.submit() [ ] test against xarray packaged data
    """
    resolution = frequency / len(signal)
    signal = np.swapaxes(signal, time_axis, 0)
    res_fft = np.fft.fft(signal, axis=0)
    cfl_index = m.floor(cutoff_frequency_low / resolution)
    cfln_index = m.ceil((frequency - cutoff_frequency_low) / resolution)
    cfh_index = m.ceil(cutoff_frequency_high / resolution)
    cfhn_index = m.floor((frequency - cutoff_frequency_high) / resolution)
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
    result = np.real(np.fft.ifft(res_fft, axis=time_axis))
    result = np.swapaxes(result, time_axis, 0)
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
