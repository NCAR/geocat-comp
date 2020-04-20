import numpy as np
import xarray as xr
import geocat.comp

import sys
import time
import unittest as ut


# nominal input
fi = np.asarray(
    [1.870327, 1.872924, 2.946794, 1.98253, 1.353965, 0.8730035, 0.1410671, 1.877125, 1.931963, -0.1676207, 1.917912, 1.735453, -1.82497, 1.01385, 1.053591,
     1.754721, 1.177423, 0.381366, 2.015617, 0.4975608, 2.169137, 0.3293635, 0.6676366, 2.691788, 2.510986, 1.027274, 1.351906]).reshape((3, 3, 3))

# nan input
fi_nan = fi.copy()
fi_nan[0, 1, 1] = np.nan
fi_nan[1, 1, 1] = np.nan
fi_nan[2, 1, 1] = np.nan

# msg input
fi_msg = fi_nan.copy()
fi_msg[np.isnan(fi_msg)] = -99

#  grids
lat = np.asarray([1, 2, 5])
lon = np.asarray([1, 2, 5])
lat2d = np.asarray([1, 2, 5, 1, 2, 5, 1, 2, 5]).reshape((3, 3))
lon2d = np.asarray([1, 1, 1, 2, 2, 2, 5, 5, 5]).reshape((3, 3))

msg64 = fi_msg[1, 1, 1].astype(np.float64)
msg32 = fi_msg[1, 1, 1].astype(np.float32)


# these two interpolations do not provide the same result, so both should be tested
def tests(fi, msg=None):
    fo = geocat.comp.rgrid2rcm(lat, lon, fi, lat2d, lon2d, msg=msg)
    return [fo]


def assertions(expected_results, results):
    np.testing.assert_array_almost_equal(expected_results[0], results[0])


fo_expected = np.asarray(
    [1.870327, 1.98253, 0.1410671, 1.872924, 1.353965, 1.877125, 2.946794, 0.8730035, 1.931963, -0.1676207, -1.82497, 1.754721, 1.917912, 1.01385, 1.177423,
     1.735453, 1.053591, 0.381366, 2.015617, 0.3293635, 2.510986, 0.4975608, 0.6676366, 1.027274, 2.169137, 2.691788, 1.351906]).reshape((3, 3, 3))
expected_results = [fo_expected]

fo_expected_nan = np.array(
    [1.870327, 1.98253, 0.1410671, 1.872924, 1.486321, 1.877125, 2.946794, 0.8730035, 1.931963, -0.1676207, -1.82497, 1.754721, 1.917912, 0.9684872, 1.177423,
     1.735453, 1.053591, 0.381366, 2.015617, 0.3293635, 2.510986, 0.4975608, 1.7583, 1.027274, 2.169137, 2.691788, 1.351906]).reshape((3, 3, 3))
expected_results_nan = [fo_expected_nan]

fo_expected_msg = np.array(
    [1.870327, 1.98253, 0.1410671, 1.872924, 1.486321, 1.877125, 2.946794, 0.8730035, 1.931963, -0.1676207, -1.82497, 1.754721, 1.917912, 0.9684872, 1.177423,
     1.735453, 1.053591, 0.381366, 2.015617, 0.3293635, 2.510986, 0.4975608, 1.7583, 1.027274, 2.169137, 2.691788, 1.351906]).reshape((3, 3, 3))
expected_results_msg = [fo_expected_msg]


# run tests

class Test_rgrid2rcm_float64(ut.TestCase):
    """
    Test_rgrid2rcm_float64
    This unit test covers the nominal, nan, and msg cases of 64 bit float input for rgrid2rcm
    """

    def test_rgrid2rcm_float64(self):
        assertions(expected_results, tests(fi.astype(np.float64)))

    def test_rgrid2rcm_float64_nan(self):
        assertions(expected_results_nan, tests(fi_nan.astype(np.float64)))

    def test_rgrid2rcm_float64_msg(self):
        assertions(expected_results_msg, tests(fi_msg.astype(np.float64), msg=msg64))


class Test_rgrid2rcm_float32(ut.TestCase):
    """
    Test_rgrid2rcm_float32
    This unit test covers the nominal, nan, and msg cases of 32 bit float input for rgrid2rcm
    """

    def test_rgrid2rcm_float32(self):
        assertions(expected_results, tests(fi.astype(np.float32)))

    def test_rgrid2rcm_float32_nan(self):
        assertions(expected_results_nan, tests(fi_nan.astype(np.float32)))

    def test_rgrid2rcm_float32_msg(self):
        assertions(expected_results_msg, tests(fi_msg.astype(np.float32), msg=msg32))


# Depriciate the below
"""
# Size of the grids
n = 6
m = 6

# create and fill the input 1D grid (lat1D, lon1D)
in_size_M = int(m / 2) + 1
in_size_N = int(n / 2) + 1

lat1D = np.zeros((in_size_M))
lon1D = np.zeros((in_size_N))
for i in range(in_size_M):
    lat1D[i] = float(i)
for j in range(in_size_N):
    lon1D[j] = float(j)

# create and fill input data array (fi)
fi = np.random.randn(3, in_size_M, in_size_N)

# create and fill the output 2D grid (lat2D, lon2D)
out_size_M = m + 1
out_size_N = n + 1

lat2D = np.zeros((out_size_M, out_size_N))
lon2D = np.zeros((out_size_M, out_size_N))
for i in range(out_size_M):
    lat2D[i, :] = float(i) * 0.5
for j in range(out_size_N):
    lon2D[:, j] = float(j) * 0.5


class Test_rgrid2rcm_float64(ut.TestCase):
    def test_rgrid2rcm_float64(self):
        fo = geocat.comp.rgrid2rcm(lat1D, lon1D, fi, lat2D, lon2D)
        np.testing.assert_array_equal(fi, fo[..., ::2, ::2].values)

    def test_rgrid2rcm_msg_float64(self):
        fo = geocat.comp.rgrid2rcm(lat1D, lon1D, fi, lat2D, lon2D, msg=fi[0, 0, 0])
        np.testing.assert_array_equal(fi, fo[..., ::2, ::2].values)

    def test_rgrid2rcm_nan_float64(self):
        fi_np_copy = fi.copy()
        fi_np_copy[:, 0, 0] = np.nan
        fo = geocat.comp.rgrid2rcm(lat1D, lon1D, fi_np_copy, lat2D, lon2D)
        np.testing.assert_array_equal(fi_np_copy[:, 1:, 1:], fo[:, 2::2, 2::2].values)


class Test_rgrid2rcm_float32(ut.TestCase):
    def test_rgrid2rcm_float32(self):
        fi_asfloat32 = fi.astype(np.float32)
        fo = geocat.comp.rgrid2rcm(lat1D.astype(np.float32), lon1D.astype(np.float32), fi_asfloat32,
                                   lat2D.astype(np.float32), lon2D.astype(np.float32))
        np.testing.assert_array_equal(fi_asfloat32, fo[..., ::2, ::2].values)

    def test_rgrid2rcm_msg_float32(self):
        fi_np_copy = fi.astype(np.float32)
        fo = geocat.comp.rgrid2rcm(lat1D.astype(np.float32), lon1D.astype(np.float32), fi_np_copy,
                                   lat2D.astype(np.float32), lon2D.astype(np.float32), msg=fi_np_copy[0, 0, 0])
        np.testing.assert_array_equal(fi_np_copy, fo[..., ::2, ::2].values)

    def test_rgrid2rcm_nan_float32(self):
        fi_np_copy = fi.astype(np.float32)
        fi_np_copy[:, 0, 0] = np.nan
        fo = geocat.comp.rgrid2rcm(lat1D, lon1D, fi_np_copy, lat2D, lon2D)
        np.testing.assert_array_equal(fi_np_copy[:, 1:, 1:], fo[:, 2::2, 2::2].values)
"""
