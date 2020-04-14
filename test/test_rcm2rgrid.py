import numpy as np
import xarray as xr
import geocat.comp

import sys
import time
import unittest as ut


# create and fill the input 2D grid (lat2D, lon2D)
lat2d = np.asarray([1, 2, 5, 1, 2, 5, 1, 2, 5]).reshape((3, 3))
lon2d = np.asarray([1, 1, 1, 2, 2, 2, 5, 5, 5]).reshape((3, 3))

# nominal input
fi = np.asarray(
    [1.870327, 1.872924, 2.946794, 1.98253, 1.353965, 0.8730035, 0.1410671, 1.877125, 1.931963, -0.1676207, 1.917912, 1.735453, -1.82497, 1.01385, 1.053591,
     1.754721, 1.177423, 0.381366, 2.015617, 0.4975608, 2.169137, 0.3293635, 0.6676366, 2.691788, 2.510986, 1.027274, 1.351906]).reshape((3, 3, 3))

# nan input
fi_nan = fi.copy()
fi_nan[0, 1, 1] = np.nan
fi_nan[2, 1, 1] = np.nan
fi_nan[1, 0, 1] = np.nan
fi_nan[1, 2, 1] = np.nan
fi_nan[1, 1, 0] = np.nan
fi_nan[1, 1, 2] = np.nan
fi_nan[1, 1, 1] = np.nan

# msg input
fi_msg = fi_nan.copy()
fi_msg[np.isnan(fi_msg)] = -99

#  create and fill the output grid indices (lat, lon)
lat = np.asarray([1, 2, 5])
lon = np.asarray([1, 2, 5])

# DEPRECIATE BELOW
"""
# Size of the grids
n = 6
m = 6

# create and fill the input 2D grid (lat2D, lon2D)
in_size_M = int(m / 2) + 1
in_size_N = int(n / 2) + 1

lat2D = np.zeros((in_size_M, in_size_N))
lon2D = np.zeros((in_size_M, in_size_N))
for i in range(in_size_M):
    lat2D[i, :] = float(i)
for j in range(in_size_N):
    lon2D[:, j] = float(j)

# create and fill input data array (fi)
fi = np.random.randn(3, in_size_M, in_size_N)

# create and fill the output 1D grid (lat1D, lon1D)
out_size_M = m + 1
out_size_N = n + 1

lat1D = np.zeros((out_size_M))
lon1D = np.zeros((out_size_N))
for i in range(out_size_M):
    lat1D[i] = float(i) * 0.5
for j in range(out_size_N):
    lon1D[j] = float(j) * 0.5


class Test_rcm2rgrid_float64(ut.TestCase):
    def test_rcm2rgrid_float64(self):
        fo = geocat.comp.rcm2rgrid(lat2D, lon2D, fi, lat1D, lon1D)
        np.testing.assert_array_equal(fi, fo[..., ::2, ::2].values)

    def test_rcm2rgrid_msg_float64(self):
        fo = geocat.comp.rcm2rgrid(lat2D, lon2D, fi, lat1D, lon1D, msg=fi[0, 0, 0])
        np.testing.assert_array_equal(fi, fo[..., ::2, ::2].values)

    def test_rcm2rgrid_nan_float64(self):
        fi_np_copy = fi.copy()
        fi_np_copy[:, 0, 0] = np.nan
        fo = geocat.comp.rcm2rgrid(lat2D, lon2D, fi_np_copy, lat1D, lon1D)
        np.testing.assert_array_equal(fi_np_copy[:, 1:, 1:], fo[:, 2::2, 2::2].values)


class Test_rcm2rgrid_float32(ut.TestCase):
    def test_rcm2rgrid_float32(self):
        fi_asfloat32 = fi.astype(np.float32)
        fo = geocat.comp.rcm2rgrid(lat2D.astype(np.float32), lon2D.astype(np.float32), fi_asfloat32,
                                   lat1D.astype(np.float32), lon1D.astype(np.float32))
        np.testing.assert_array_equal(fi_asfloat32, fo[..., ::2, ::2].values)

    def test_rcm2rgrid_msg_float32(self):
        fi_np_copy = fi.astype(np.float32)
        fo = geocat.comp.rcm2rgrid(lat2D.astype(np.float32), lon2D.astype(np.float32), fi_np_copy,
                                   lat1D.astype(np.float32), lon1D.astype(np.float32), msg=fi_np_copy[0, 0, 0])
        np.testing.assert_array_equal(fi_np_copy, fo[..., ::2, ::2].values)

    def test_rcm2rgrid_nan_float32(self):
        fi_np_copy = fi.astype(np.float32)
        fi_np_copy[:, 0, 0] = np.nan
        fo = geocat.comp.rcm2rgrid(lat2D, lon2D, fi_np_copy, lat1D, lon1D)
        np.testing.assert_array_equal(fi_np_copy[:, 1:, 1:], fo[:, 2::2, 2::2].values)
"""
