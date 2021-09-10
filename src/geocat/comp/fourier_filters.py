import math as m
import numpy as np
import typing
import xarray as xr


def fourier_filter(
        signal: typing.Union[np.ndarray, xr.DataArray],
        frequency: float,
        cutoff_frequency_low: float = 0,
        cutoff_frequency_high: float = 0,
        time_axis: int = 0,
        low_pass: bool = False,
        high_pass: bool = False,
        band_pass: bool = False,
        band_block: bool = False) -> typing.Union[np.ndarray, xr.DataArray]:
    """Filter a dataset by frequency. This function allowes for low_pass, high_
    pass, band_pass, or band_block filtering of the data's freqency
    representation.

    Parameters
    ----------
    signal : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        n-dimensional dataset

    frequency : :class:`float`
        sample frequency of dataset

    cutoff_frequency_low : :class:`float`, Optional
        low frequency for cutting fourier transform, used by low_pass, band_pass, band_block. Defaults to 0.

    cutoff_frequency_high : :class:`float`, Optional
        high frequency for cutting fourier transform, used by low_pass, band_pass, band_block. Defaults to 0.

    time_axis : :class:`int`, Optional
        the time axis of the data set. Defaults to 0.

    low_pass : :class:`bool`, Optional
        runs a low_pass filter on the data if set to True. Defaults to False.

    high_pass : :class:`bool`, Optional
        runs a high_pass filter on the data if set to True. Defaults to False.

    band_pass : :class:`bool`, Optional
        runs a band_pass filter on the data if set to True. Defaults to False.

    band_block : :class:`bool`, Optional
        runs a band_block filter on the data if set to True. Defaults to False.

    Returns
    -------
    return_signal : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        signal with specified filters applied

    Examples
    --------
    Example 1: The tidal cycle needs to be removed from a 10/hr `oceanic dataset <https://tidesandcurrents.noaa.gov/waterlevels.html?id=9415020&units=standard&bdate=20210101&edate=20210131&timezone=GMT&datum=MLLW&interval=6&action=data>`_


    >>> from geocat.comp import fourier_filter
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> import numpy as np
    >>> import pandas as pd
    >>> import xarray as xr

    >>> dataset = xr.DataArray(pd.read_csv('CO-OPS_9415020_wl.csv'))
    >>> xr_data = dataset.loc[:,'Verified (ft)']

    >>> data_freq = 10 #points per hour
    >>> tide_freq1 = 1/(1*12.4206) #tides per hour
    >>> tide_freq2 = 1/(2*12.4206) #tides per hour
    >>> res = data_freq/(len(xr_data))
    >>> cflow1 = tide_freq1 - res*5
    >>> cfhigh1 = tide_freq1 + res*5
    >>> cflow2 = tide_freq2 - res*5
    >>> cfhigh2 = tide_freq2 + res*5

    >>> fig, ax = plt.subplots(1,1,dpi=100,figsize=(8,4),constrained_layout=True)
    >>> no_tide = xr_data
    >>> ax.plot(no_tide[2000:3000])
    >>> no_tide = fourier_filter(no_tide, data_freq, cutoff_frequency_low=cflow1,cutoff_frequency_high=cfhigh1,band_block=True)
    >>> ax.plot(no_tide[2000:3000])
    >>> no_tide = fourier_filter(no_tide, data_freq, cutoff_frequency_low=cflow2,cutoff_frequency_high=cfhigh2,band_block=True)
    >>> ax.plot(no_tide[2000:3000])
    >>> fig.show()

    >>> fig, axs = plt.subplots(2,1, dpi=100,figsize=(8,4),constrained_layout=True)
    >>> axs[0].set_title('real')
    >>> axs[0].plot(np.real(np.fft.fft(xr_data)[1:100]))
    >>> axs[0].plot(np.real(np.fft.fft(no_tide)[1:100]))
    >>> axs[1].set_title('imag')
    >>> axs[1].plot(np.imag(np.fft.fft(xr_data)[1:100]))
    >>> axs[1].plot(np.imag(np.fft.fft(no_tide)[1:100]))
    >>> fig.show()

    >>> fig, axs = plt.subplots(2,1, dpi=100,figsize=(8,4),constrained_layout=True)
    >>> start = 0
    >>> end = -1
    >>> axs[0].set_title('real')
    >>> axs[0].plot(np.real(xr_data)[start:end])
    >>> axs[0].plot(np.real(no_tide)[start:end])
    >>> axs[1].set_title('imag')
    >>> axs[1].plot(np.imag(xr_data)[start:end])
    >>> axs[1].plot(np.imag(no_tide)[start:end])
    >>> fig.show()
    """
    resolution = frequency / len(signal)
    signal = np.swapaxes(signal, time_axis, 0)
    res_fft = np.fft.fft(signal, axis=0)
    cfl_index = m.floor(cutoff_frequency_low / resolution)
    cfln_index = 1 - cfl_index
    cfh_index = m.ceil(cutoff_frequency_high / resolution)
    cfhn_index = 1 - cfh_index
    if low_pass:
        if cfl_index > 1:
            res_fft[cfl_index:cfln_index] = np.zeros(
                res_fft[cfl_index:cfln_index].shape)
        else:
            res_fft[cfl_index:] = np.zeros(res_fft[cfl_index:].shape)
    if high_pass:
        res_fft[:cfh_index] = np.zeros(res_fft[:cfh_index].shape)
        if cfh_index > 1:
            res_fft[cfhn_index:] = np.zeros(res_fft[cfhn_index:].shape)
    if band_pass:
        res_fft[:cfl_index] = np.zeros(res_fft[:cfl_index].shape)
        if cfh_index > 1:
            res_fft[cfh_index:cfhn_index] = np.zeros(
                res_fft[cfh_index:cfhn_index].shape)
        else:
            res_fft[cfh_index:] = np.zeros(res_fft[cfh_index:].shape)
        if cfl_index > 1:
            res_fft[cfln_index:] = np.zeros(res_fft[cfln_index:].shape)
    if band_block:
        res_fft[cfl_index:cfh_index] = np.zeros(
            res_fft[cfl_index:cfh_index].shape)
        if cfl_index > 1 and cfh_index > 1:
            res_fft[cfhn_index:cfln_index] = np.zeros(
                res_fft[cfhn_index:cfln_index].shape)
        elif cfh_index > 1:
            res_fft[cfhn_index:] = np.zeros(res_fft[cfhn_index:].shape)
    result = np.fft.ifft(res_fft, axis=0)
    result = np.real(result)
    result = np.swapaxes(result, time_axis, 0)
    if type(signal) == xr.DataArray:
        xr_result = signal.copy()
        xr_result.data = result
        result = xr_result
    return result


def fourier_low_pass(
        signal: typing.Union[np.ndarray, xr.DataArray],
        frequency: float,
        cutoff_frequency_low: float,
        time_axis: int = 0) -> typing.Union[np.ndarray, xr.DataArray]:
    """Filter a dataset by frequency. This function allowes for low_pass
    filtering of the data's freqency representation.

    Parameters
    ----------
    signal : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        n-dimensional dataset

    frequency : :class:`float`
        sample frequency of dataset

    cutoff_frequency_low : :class:`float`
        low frequency for cutting fourier transform

    time_axis : :class:`int`, Optional
        the time axis of the data set. Defaults to 0.

    Returns
    -------
    return_signal : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        signal with specified filters applied
    """
    return fourier_filter(signal,
                          frequency,
                          cutoff_frequency_low=cutoff_frequency_low,
                          time_axis=time_axis,
                          low_pass=True)


def fourier_high_pass(
        signal: typing.Union[np.ndarray, xr.DataArray],
        frequency: float,
        cutoff_frequency_high: float,
        time_axis: int = 0) -> typing.Union[np.ndarray, xr.DataArray]:
    """Filter a dataset by frequency. This function allowes for high_pass
    filtering of the data's freqency representation.

    Parameters
    ----------
    signal : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        n-dimensional dataset

    frequency : :class:`float`
        sample frequency of dataset

    cutoff_frequency_high : :class:`float`
        high frequency for cutting fourier transform

    time_axis : :class:`int`, Optional
        the time axis of the data set. Defaults to 0.

    Returns
    -------
    return_signal : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        signal with specified filters applied
    """
    return fourier_filter(signal,
                          frequency,
                          cutoff_frequency_high=cutoff_frequency_high,
                          time_axis=time_axis,
                          high_pass=True)


def fourier_band_pass(
        signal: typing.Union[np.ndarray, xr.DataArray],
        frequency: float,
        cutoff_frequency_low: float,
        cutoff_frequency_high: float,
        time_axis: int = 0) -> typing.Union[np.ndarray, xr.DataArray]:
    """Filter a dataset by frequency. This function allowes for band_pass
    filtering of the data's freqency representation.

    Parameters
    ----------
    signal : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        n-dimensional dataset

    frequency : :class:`float`
        sample frequency of dataset

    cutoff_frequency_low : :class:`float`
        low frequency for cutting fourier transform

    cutoff_frequency_high : :class:`float`
        high frequency for cutting fourier transform

    time_axis : :class:`int`, Optional
        the time axis of the data set. Defaults to 0.

    Returns
    -------
    return_signal : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        signal with specified filters applied
    """
    return fourier_filter(signal,
                          frequency,
                          cutoff_frequency_low=cutoff_frequency_low,
                          cutoff_frequency_high=cutoff_frequency_high,
                          time_axis=time_axis,
                          band_pass=True)


def fourier_band_block(
        signal: typing.Union[np.ndarray, xr.DataArray],
        frequency: float,
        cutoff_frequency_low: float,
        cutoff_frequency_high: float,
        time_axis: int = 0) -> typing.Union[np.ndarray, xr.DataArray]:
    """Filter a dataset by frequency. This function allowes for band_block
    filtering of the data's freqency representation.

    Parameters
    ----------
    signal : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        n-dimensional dataset

    frequency : :class:`float`
        sample frequency of dataset

    cutoff_frequency_low : :class:`float`
        low frequency for cutting fourier transform

    cutoff_frequency_high : :class:`float`
        high frequency for cutting fourier transform

    time_axis : :class:`int`, Optional
        the time axis of the data set. Defaults to 0.

    Returns
    -------
    return_signal : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        signal with specified filters applied
    """
    return fourier_filter(signal,
                          frequency,
                          cutoff_frequency_low=cutoff_frequency_low,
                          cutoff_frequency_high=cutoff_frequency_high,
                          time_axis=time_axis,
                          band_block=True)
