import numpy as np
import xarray as xr
import os
import sys

from geocat.comp import eofunc_eofs, eofunc_pcs, pearson_r

# Get repo directory to access test data
dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Bench_eofunc_eofs:
    def setup(self):
        self.sample_data_eof = np.arange(64, dtype='int64').reshape((4, 4, 4))

    def time_eofunc(self):
        eofunc_eofs(self.sample_data_eof, neofs=1, time_dim=2)

    def peakmem_eofunc(self):
        eofunc_eofs(self.sample_data_eof, neofs=1, time_dim=2)


class Bench_eofunc_ps:
    def setup(self):
        try:
            self.nc_ds = xr.open_dataset(dir + "/test/eofunc_dataset.nc")
        except Exception:
            self.nc_ds = xr.open_dataset(dir + "/../test/eofunc_dataset.nc")

        self.sst = self.nc_ds.sst

    def time_eofunc_ps(self):
        eofunc_pcs(self.sst, npcs=5, meta=True)

    def peakmem_eofunc_ps(self):
        eofunc_pcs(self.sst, npcs=5, meta=True)


class Bench_pearson_r:
    def setup(self):
        times = xr.date_range(
            start='2022-08-01', end='2022-08-05', freq='D', use_cftime=True
        )
        lats = np.linspace(start=-45, stop=45, num=3, dtype='float32')
        lons = np.linspace(start=-180, stop=180, num=4, dtype='float32')

        x, y, z = np.meshgrid(lons, lats, times)
        np.random.seed(0)

        self.a = np.random.random_sample((len(lats), len(lons), len(times)))
        self.b = np.power(self.a, 2)
        self.weights = np.cos(np.deg2rad(y))

    def time_pearson_r(self):
        pearson_r(self.a, self.b, weights=self.weights)
