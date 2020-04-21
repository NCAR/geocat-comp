import numpy as np
import numpy.testing as nt
import xarray as xr
import geocat.comp as gc

import sys
import time
import unittest as ut


# nominal input
fi_nom = np.asarray(
    [1.870327, 1.872924, 2.946794, 1.98253, 1.353965, 0.8730035, 0.1410671, 1.877125, 1.931963, -0.1676207, 1.917912, 1.735453, -1.82497, 1.01385, 1.053591,
     1.754721, 1.177423, 0.381366, 2.015617, 0.4975608, 2.169137, 0.3293635, 0.6676366, 2.691788, 2.510986, 1.027274, 1.351906]).reshape((3, 3, 3))

# nan input
fi_nan = fi_nom.copy()
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

fo_nom_expected = np.asarray(
    [1.870327, 1.98253, 0.1410671, 1.872924, 1.353965, 1.877125, 2.946794, 0.8730035, 1.931963, -0.1676207, -1.82497, 1.754721, 1.917912, 1.01385, 1.177423,
     1.735453, 1.053591, 0.381366, 2.015617, 0.3293635, 2.510986, 0.4975608, 0.6676366, 1.027274, 2.169137, 2.691788, 1.351906]).reshape((3, 3, 3))

fo_nan_expected = np.array(
    [1.870327, 1.98253, 0.1410671, 1.872924, 1.875024, 1.877125, 2.946794, 0.8730035, 1.931963, -0.1676207, -1.82497, 1.754721, 1.917912, 1.547667, 1.177423,
     1.735453, 1.053591, 0.381366, 2.015617, 0.3293635, 2.510986, 0.4975608, 0.7624173, 1.027274, 2.169137, 2.691788, 1.351906]).reshape((3, 3, 3))

fo_msg_expected = np.array(
    [1.870327, 1.98253, 0.1410671, 1.872924, 1.875024, 1.877125, 2.946794, 0.8730035, 1.931963, -0.1676207, -1.82497, 1.754721, 1.917912, 1.547667, 1.177423,
     1.735453, 1.053591, 0.381366, 2.015617, 0.3293635, 2.510986, 0.4975608, 0.7624173, 1.027274, 2.169137, 2.691788, 1.351906]).reshape((3, 3, 3))


# run tests
class Test_rcm2rgrid(ut.TestCase):
    """
    Test_rcm2rgrid_float64
    This unit test covers the nominal, nan, and msg cases of 64 bit float input for rcm2rgrid
    """

    def test_rcm2rgrid_float64_nom(self):
        nt.assert_array_almost_equal(fo_nom_expected, gc.rcm2rgrid(lat2d, lon2d, fi_nom.astype(np.float64), lat, lon))

    def test_rcm2rgrid_float64_nan(self):
        nt.assert_array_almost_equal(fo_nan_expected, gc.rcm2rgrid(lat2d, lon2d, fi_nan.astype(np.float64), lat, lon))

    def test_rcm2rgrid_float64_msg(self):
        nt.assert_array_almost_equal(fo_msg_expected, gc.rcm2rgrid(lat2d, lon2d, fi_msg.astype(np.float64), lat, lon, msg=msg64))

    def test_rcm2rgrid_float32_nom(self):
        nt.assert_array_almost_equal(fo_nom_expected, gc.rcm2rgrid(lat2d, lon2d, fi_nom.astype(np.float32), lat, lon))

    def test_rcm2rgrid_float32_nan(self):
        nt.assert_array_almost_equal(fo_nan_expected, gc.rcm2rgrid(lat2d, lon2d, fi_nan.astype(np.float32), lat, lon))

    def test_rcm2rgrid_float32_msg(self):
        nt.assert_array_almost_equal(fo_msg_expected, gc.rcm2rgrid(lat2d, lon2d, fi_msg.astype(np.float32), lat, lon, msg=msg32))
