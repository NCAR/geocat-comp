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
    ## Fourier Pass
    resolution = frequency / len(signal)
    res_fft = np.fft.fft(signal, axis=time_axis)
    for index in range(len(res_fft)):
        ffreq = index * resolution
        if low_pass and (ffreq >= cutoff_frequency_low and
                         ffreq <= frequency - cutoff_frequency_low):
            res_fft[index] = 0
        if high_pass and (ffreq <= cutoff_frequency_high or
                          ffreq >= frequency - cutoff_frequency_high):
            res_fft[index] = 0
        if band_pass and (ffreq <= cutoff_frequency_low or
                          (ffreq >= cutoff_frequency_high and
                           ffreq <= frequency - cutoff_frequency_high) or
                          ffreq >= frequency - cutoff_frequency_low):
            res_fft[index] = 0
        if band_block and ((ffreq >= cutoff_frequency_low and
                            ffreq <= cutoff_frequency_high) or
                           (ffreq <= frequency - cutoff_frequency_low and
                            ffreq >= frequency - cutoff_frequency_high)):
            res_fft[index] = 0
    result = np.real(np.fft.ifft(res_fft, axis=time_axis))  # why times 2?
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
