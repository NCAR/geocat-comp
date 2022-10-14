import sys
import unittest

import numpy as np
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp import gradient
elif "-v" in str(sys.argv):
    from src.geocat.comp import gradient
else:
    from geocat.comp import gradient


class Test_Gradient(unittest.TestCase):
    test_data = None
    test_results_lon = None
    test_results_lat = None

    results = None
    results_lon = None
    results_lat = None

    @classmethod
    def setUpClass(cls):
        cls.test_data = xr.load_dataset(
            'test/gradient_test_data.nc').to_array().squeeze()
        cls.test_results_lon = xr.load_dataset(
            'test/gradient_test_results_longitude.nc').to_array().squeeze()
        cls.test_results_lat = xr.load_dataset(
            'test/gradient_test_results_latitude.nc').to_array().squeeze()
        cls.results = gradient(cls.test_data)
        cls.results_axis0 = cls.results[0]
        cls.results_axis1 = cls.results[1]

    def test_gradient_axis0_xr(self):
        np.testing.assert_almost_equal(
            self.results_axis0.data,
            self.test_results_lon.data,
            decimal=3,
        )

    def test_gradient_axis1_xr(self):
        np.testing.assert_almost_equal(
            self.results_axis1.data,
            self.test_results_lat.data,
            decimal=3,
        )
