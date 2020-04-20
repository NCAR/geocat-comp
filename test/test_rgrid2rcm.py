import numpy as np
import xarray as xr
import geocat.comp

import sys
import time
import pytest as pt
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


@pt.mark.skip(reason="Not a test.")
def tests(fi, msg=None):
    fo = geocat.comp.rgrid2rcm(lat, lon, fi, lat2d, lon2d, msg=msg)
    return [fo]


@pt.mark.skip(reason="Not a test.")
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
