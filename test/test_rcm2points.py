import numpy as np
import xarray as xr
import geocat.comp

import sys
import time
import unittest as ut


# create and fill the input 2D grid (lat2D, lon2D)

lat2d = np.asarray([1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3]).reshape((3, 3))
lon2d = np.asarray([1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3]).reshape((3, 3))

# nominal input
fi = np.asarray(
    [1.870327, 1.872924, 2.946794, 1.98253, 1.353965, 0.8730035, 0.1410671, 1.877125, 1.931963, -0.1676207, 1.917912, 1.735453, -1.82497, 1.01385, 1.053591,
     1.754721, 1.177423, 0.381366, 2.015617, 0.4975608, 2.169137, 0.3293635, 0.6676366, 2.691788, 2.510986, 1.027274, 1.351906]).reshape((3, 3, 3))

# nan input
fi_nan = fi
fi_nan[0, 1, 1] = np.nan
fi_nan[2, 1, 1] = np.nan
fi_nan[1, 0, 1] = np.nan
fi_nan[1, 2, 1] = np.nan
fi_nan[1, 1, 0] = np.nan
fi_nan[1, 1, 2] = np.nan
fi_nan[1, 1, 1] = np.nan

# msg input
fi_msg = fi_nan
fi_msg[np.isnan(fi_msg)] = -99

# output grid for this function
lat = np.asarray([1, 2, 3])
lon = np.asarray([1, 2, 3])


# these two interpolations provide the same result, consider refactoring to just one of the two cases.
def tests(fi):
    fo0 = geocat.comp.rcm2points(lat2d, lon2d, fi, lat, lon, 0)  # inverse distance weighting
    fo2 = geocat.comp.rcm2points(lat2d, lon2d, fi, lat, lon, 2)  # bilinear interpolation
    return [fo0, fo2]


def assertions(expected_results, results):
    np.testing.assert_array_equal(expected_results[0], results[0])
    np.testing.assert_array_equal(expected_results[1], results[1])


# expected output
fo0_expected = np.asarray([1.948043, 1.786732, 0.6411167, 0.3673298, -0.9549527, 1.492616, 1.7391, 0.6348206, 2.115912]).reshape((3, 3))
fo2_expected = np.asarray([1.948043, 1.786732, 0.6411167, 0.3673298, -0.9549527, 1.492616, 1.7391, 0.6348206, 2.115912]).reshape((3, 3))
expected_results = [fo0_expected, fo2_expected]

fo0_expected_nan = np.asarray([1.95103, 1.878375, 0.6331819, 0.02682439, 1.067587, 1.613317, 1.744488, 0.6278715, 2.132033]).reshape((3, 3))
fo2_expected_nan = np.asarray([1.95103, 1.878375, 0.6331819, 0.02682439, 1.067587, 1.613317, 1.744488, 0.6278715, 2.132033]).reshape((3, 3))
expected_results_nan = [fo0_expected_nan, fo2_expected_nan]

fo0_expected_msg = np.asarray([1.95103, 1.878375, 0.6331819, 0.02682439, 1.067587, 1.613317, 1.744488, 0.6278715, 2.132033]).reshape((3, 3))
fo2_expected_msg = np.asarray([1.95103, 1.878375, 0.6331819, 0.02682439, 1.067587, 1.613317, 1.744488, 0.6278715, 2.132033]).reshape((3, 3))
expected_results_msg = [fo0_expected_msg, fo2_expected_msg]


# run tests

class Test_rcm2points_float64(ut.TestCase):
    """
    Test_rcm2points_float64
    This unit test covers the nominal, nan, and msg cases of 64 bit float input for rcm2points
    """

    def test_rcm2points_float64_nan(self):
        assertions(expected_results_nan, tests(fi_nan.astype(np.float64)))

    def test_rcm2points_float64(self):
        assertions(expected_results, tests(fi.np.astype(np.float64)))

    def test_rcm2points_float64_msg(self):
        assertions(expected_results_msg, tests(fi_msg.astype(np.float64)))


class Test_rcm2points_float32(ut.TestCase):
    """
    Test_rcm2points_float32
    This unit test covers the nominal, nan, and msg cases of 32 bit float input for rcm2points
    """

    def test_rcm2points_float32_nan(self):
        assertions(expected_results_nan, tests(fi_nan.astype(np.float32)))

    def test_rcm2points_float32(self):
        assertions(expected_results, tests(fi.np.astype(np.float32)))

    def test_rcm2points_float32_msg(self):
        assertions(expected_results_msg, tests(fi_msg.astype(np.float32)))


# EVERYTHING BELOW THIS LINE IS GOING TO BE GONE


'''
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
        np.testing.assert_array_equal(fi_diag_asfloat32[:, 1:], fo[..., 2::2].values)'''
