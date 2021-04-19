import math as m

import matplotlib.pyplot as plt
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
        if low_pass and (index * resolution >= cutoff_frequency_low):
            res_fft[index] = 0
        if high_pass and (index * resolution <= cutoff_frequency_high):
            res_fft[index] = 0
        if band_pass and (index * resolution <= cutoff_frequency_low or
                          index * resolution >= cutoff_frequency_high):
            res_fft[index] = 0
        if band_block and (index * resolution >= cutoff_frequency_low and
                           index * resolution <= cutoff_frequency_high):
            res_fft[index] = 0
    result = np.fft.ifft(res_fft, axis=time_axis)
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


freq = 10000

t = np.arange(30000) / freq  # ten seconds of 1kHz

t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
          np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
          np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
          np.sin(100 * t * m.tau) / 10)

s_data_lp = fourier_low_pass(t_data, freq, 15)
s_data_hp = fourier_high_pass(t_data, freq, 15)
s_data_bp = fourier_band_pass(t_data, freq, 3, 30)
s_data_bb = fourier_band_block(t_data, freq, 3, 30)
res = freq / len(t)
x_axis = np.array([x * res for x in range(len(t))])

#plt.semilogx(x_axis[:400], np.fft.fft(s_data_lp)[:400])
plt.plot(t_data)
# plt.plot(flp(t_data, freq, 30))
plt.plot(s_data_bb)
plt.show()
