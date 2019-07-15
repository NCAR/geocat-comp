import numpy as np
import xarray as xr
import geocat.comp

import sys
import time
import unittest as ut

n = 127

xi = np.linspace(0, n, num=n//2 + 1, dtype=np.float64)
yi = np.linspace(0, n, num=n//2 + 1, dtype=np.float64)
xo = np.linspace(xi.min(), xi.max(), num=xi.shape[0] * 2 - 1)
yo = np.linspace(yi.min(), yi.max(), num=yi.shape[0] * 2 - 1)
fi_np = np.random.rand(96, 3, len(yi), len(xi)).astype(np.float64)

class Test(ut.TestCase):
    def test_linint2(self):
        chunks = {'time': fi_np.shape[0], 'level': fi_np.shape[1], 'lat': fi_np.shape[2], 'lon': fi_np.shape[3]}
        fi = xr.DataArray(fi_np, dims=['time', 'level', 'lat', 'lon'], coords={'lat': yi, 'lon': xi}).chunk(chunks)
        fo = geocat.comp.linint2(fi, xo, yo, 0)
        np.testing.assert_array_equal(fi.values, fo[..., ::2, ::2].values)
