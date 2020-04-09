import numpy as np
import xarray as xr
import geocat.comp

import sys
import time
import unittest as ut


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
fi = np.random.randn(1, in_size_M, in_size_N)

# create and fill the output 1D grid (lat1D, lon1D)
out_size_M = m + 1
out_size_N = n + 1

lat1D = np.zeros((out_size_M))
lon1D = np.zeros((out_size_N))
for i in range(out_size_M):
    lat1D[i] = float(i) * 0.5
for j in range(out_size_N):
    lon1D[j] = float(j) * 0.5

fo = geocat.comp.rcm2points(lat2D, lon2D, fi, lat1D, lon1D, 2)

fi_diag = np.asarray([np.diag(fi[0, :, :])])
fi_diag_asfloat32 = fi_diag.astype(np.float32)


class Test_rcm2points_float64(ut.TestCase):
    def test_rcm2points_float64(self):
        fo = geocat.comp.rcm2points(lat2D, lon2D, fi, lat1D, lon1D, 2)
        np.testing.assert_array_equal(fi_diag, fo[..., ::2, ::2].values)

    def test_rcm2points_msg_float64(self):
        fo = geocat.comp.rcm2points(lat2D, lon2D, fi, lat1D, lon1D, 0, msg=fi[0, 0, 0])
        np.testing.assert_array_equal(fi_diag, fo[..., ::2, ::2].values)

    def test_rcm2points_nan_float64(self):
        fi_np_copy = fi.copy()
        fi_np_copy[:, 0, 0] = np.nan
        fi_np_diag = np.asarray([np.diag(fi_np_copy[0, :, :])])
        fo = geocat.comp.rcm2points(lat2D, lon2D, fi_np_copy, lat1D, lon1D)
        np.testing.assert_array_equal(fi_diag[:, 1:], fo[..., 2::2].values)


class Test_rcm2points_float32(ut.TestCase):
    def test_rcm2points_float32(self):
        fi_asfloat32 = fi.astype(np.float32)
        fo = geocat.comp.rcm2points(lat2D.astype(np.float32), lon2D.astype(np.float32), fi_asfloat32, lat1D.astype(np.float32), lon1D.astype(np.float32))
        np.testing.assert_array_equal(fi_diag_asfloat32, fo[..., ::2, ::2].values)

    def test_rcm2points_msg_float32(self):
        fi_np_copy = fi.astype(np.float32)
        fo = geocat.comp.rcm2points(lat2D.astype(np.float32), lon2D.astype(np.float32), fi_np_copy, lat1D.astype(np.float32), lon1D.astype(np.float32), 0,
            msg=fi_np_copy[0, 0, 0])
        np.testing.assert_array_equal(fi_diag_asfloat32, fo[..., ::2, ::2].values)

    def test_rcm2points_nan_float32(self):
        fi_np_copy = fi.astype(np.float32)
        fi_np_copy[:, 0, 0] = np.nan
        fo = geocat.comp.rcm2points(lat2D, lon2D, fi_np_copy, lat1D, lon1D)
        np.testing.assert_array_equal(fi_diag_asfloat32[:, 1:], fo[..., 2::2].values)
