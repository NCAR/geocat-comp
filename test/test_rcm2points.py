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
lat = np.asarray([1, 2, 3])
lon = np.asarray([1, 2, 3])
lat2d = np.asarray([1, 2, 5, 1, 2, 5, 1, 2, 5]).reshape((3, 3))
lon2d = np.asarray([1, 1, 1, 2, 2, 2, 5, 5, 5]).reshape((3, 3))

msg64 = fi_msg[1, 1, 1].astype(np.float64)
msg32 = fi_msg[1, 1, 1].astype(np.float32)


# these two interpolations do not provide the same result, so both should be tested
def tests(fi, msg=None):
    if msg is None:
        fo0 = geocat.comp.rcm2points(lat2d, lon2d, fi, lat, lon, opt=0)  # inverse distance weighting
        fo2 = geocat.comp.rcm2points(lat2d, lon2d, fi, lat, lon, opt=2)  # bilinear interpolation
    else:
        fo0 = geocat.comp.rcm2points(lat2d, lon2d, fi, lat, lon, opt=0, msg=msg)  # inverse distance weighting
        fo2 = geocat.comp.rcm2points(lat2d, lon2d, fi, lat, lon, opt=2, msg=msg)  # bilinear interpolation
    return [fo0, fo2]


def assertions(expected_results, results):
    np.testing.assert_array_almost_equal(expected_results[0], results[0])
    np.testing.assert_array_almost_equal(expected_results[1], results[1])


# expected output
fo0_expected = np.asarray([1.870327, 1.353965, 1.588746, -0.1676207, 1.01385, 0.7974159, 2.015617, 0.6676366, 1.249507]).reshape((3, 3))
fo2_expected = np.asarray([1.870327, 1.353965, 1.588746, -0.1676207, 1.01385, 0.7974159, 2.015617, 0.6676366, 1.249507]).reshape((3, 3))
expected_results = [fo0_expected, fo2_expected]

fo0_expected_nan = np.asarray([1.870327, 1.486811, 1.679019, -0.1676207, 0.9685476, 0.7141976, 2.015617, 1.757489, 1.473235]).reshape((3, 3))
fo2_expected_nan = np.asarray([1.870327, 1.851139, 1.679019, -0.1676207, 0.2114156, 0.7141976, 2.015617, 0.9372569, 1.473235]).reshape((3, 3))
expected_results_nan = [fo0_expected_nan, fo2_expected_nan]

fo0_expected_msg = np.asarray([1.870327, 1.486811, 1.679019, -0.1676207, 0.9685476, 0.7141976, 2.015617, 1.757489, 1.473235]).reshape((3, 3))
fo2_expected_msg = np.asarray([1.870327, 1.851139, 1.679019, -0.1676207, 0.2114156, 0.7141976, 2.015617, 0.9372569, 1.473235]).reshape((3, 3))
expected_results_msg = [fo0_expected_msg, fo2_expected_msg]


# run tests

class Test_rcm2points_float64(ut.TestCase):
    """
    Test_rcm2points_float64
    This unit test covers the nominal, nan, and msg cases of 64 bit float input for rcm2points
    """

    def test_rcm2points_float64(self):
        assertions(expected_results, tests(fi.astype(np.float64)))

    def test_rcm2points_float64_nan(self):
        assertions(expected_results_nan, tests(fi_nan.astype(np.float64)))

    def test_rcm2points_float64_msg(self):
        assertions(expected_results_msg, tests(fi_msg.astype(np.float64), msg=msg64))


class Test_rcm2points_float32(ut.TestCase):
    """
    Test_rcm2points_float32
    This unit test covers the nominal, nan, and msg cases of 32 bit float input for rcm2points
    """

    def test_rcm2points_float32(self):
        assertions(expected_results, tests(fi.astype(np.float32)))

    def test_rcm2points_float32_nan(self):
        assertions(expected_results_nan, tests(fi_nan.astype(np.float32)))

    def test_rcm2points_float32_msg(self):
        assertions(expected_results_msg, tests(fi_msg.astype(np.float32), msg=msg32))
