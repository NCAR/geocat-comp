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
    test_data_xr = None
    test_data_dask = None
    test_results_lon = None
    test_results_lat = None

    results = None
    results_lon = None
    results_lat = None

    @classmethod
    def setUpClass(cls):
        cls.test_data_xr = xr.load_dataset(
            'test/gradient_test_data.nc').to_array().squeeze()
        cls.test_data_dask = cls.test_data_xr.chunk(10)
        cls.test_results_lon = xr.load_dataset(
            'test/gradient_test_results_longitude.nc').to_array().squeeze()
        cls.test_results_lat = xr.load_dataset(
            'test/gradient_test_results_latitude.nc').to_array().squeeze()

    def test_gradient_axis0_xr(self):
        self.results = gradient(self.test_data_xr)
        self.results_axis0 = self.results[0]
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis0.data,
            self.test_results_lon.data,
            decimal=3,
        )

    def test_gradient_axis1_xr(self):
        self.results = gradient(self.test_data_xr)
        self.results_axis0 = self.results[0]
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1.data,
            self.test_results_lat.data,
            decimal=3,
        )

    def test_gradient_axis0_dask(self):
        self.results = gradient(self.test_data_dask)
        self.results_axis0 = self.results[0]
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis0.data,
            self.test_results_lon.data,
            decimal=3,
        )

    def test_gradient_axis1_dask(self):
        self.results = gradient(self.test_data_dask)
        self.results_axis0 = self.results[0]
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1.data,
            self.test_results_lat.data,
            decimal=3,
        )
