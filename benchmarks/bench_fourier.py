from math import tau
import numpy as np
import xarray as xr

from geocat.comp import (
    fourier_band_block,
)


class Bench_fourier:
    def setup(self):
        self.freq = 200
        t = np.arange(200) / self.freq
        t = t[:, None] + t
        t = t[:, :, None] + t
        self.t_data = (
            np.sin(t * tau) / 0.1
            + np.sin(2 * t * tau) / 0.2
            + np.sin(5 * t * tau) / 0.5
            + np.sin(10 * t * tau)
            + np.sin(20 * t * tau) / 2
            + np.sin(50 * t * tau) / 5
            + np.sin(100 * t * tau) / 10
        )

    def time_band_block(self):
        fourier_band_block(self.t_data, self.freq, 3, 30, time_axis=0)

    def peakmem_band_block(self):
        fourier_band_block(self.t_data, self.freq, 3, 30, time_axis=0)
