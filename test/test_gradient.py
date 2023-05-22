import unittest

import numpy as np
import xarray as xr

from src.geocat.comp import gradient


class Test_Gradient(unittest.TestCase):
    test_data_xr = None
    test_data_np = None
    test_data_dask = None
    test_results_lon = None
    test_results_lat = None
    test_coords_1d_lon = None
    test_coords_1d_lat = None
    test_coords_2d_lon_np = None
    test_coords_2d_lat_np = None
    test_coords_1d_lat_np = None
    test_coords_1d_lon_np = None

    results = None
    results_lon = None
    results_lat = None

    @classmethod
    def setUpClass(cls):
        cls.test_data_xr = xr.load_dataset(
            'test/gradient_test_data.nc').to_array().squeeze()
        cls.test_data_xr_nocoords = xr.DataArray(cls.test_data_xr, coords={})
        cls.test_data_np = cls.test_data_xr.values
        cls.test_data_dask = cls.test_data_xr.chunk(10)
        cls.test_results_lon = xr.load_dataset(
            'test/gradient_test_results_longitude.nc').to_array().squeeze()
        cls.test_results_lat = xr.load_dataset(
            'test/gradient_test_results_latitude.nc').to_array().squeeze()
        cls.test_coords_1d_lon = cls.test_data_xr.coords['lon']
        cls.test_coords_1d_lat = cls.test_data_xr.coords['lat']
        cls.test_coords_2d_lon_np, cls.test_coords_2d_lat_np = np.meshgrid(
            cls.test_coords_1d_lon, cls.test_coords_1d_lat)
        cls.test_data_xr_2d_coords = xr.DataArray(
            cls.test_data_xr,
            dims=['x', 'y'],
            coords=dict(
                lon=(['x', 'y'], cls.test_coords_2d_lon_np),
                lat=(['x', 'y'], cls.test_coords_2d_lat_np),
            ),
        )
        cls.test_coords_1d_lon_np = cls.test_coords_1d_lon.values
        cls.test_coords_1d_lat_np = cls.test_coords_1d_lat.values

    def test_gradient_axis0_xr(self):
        self.results = gradient(self.test_data_xr)
        self.results_axis0 = self.results[0]
        np.testing.assert_almost_equal(
            self.results_axis0.values,
            self.test_results_lon.values,
            decimal=3,
        )

    def test_gradient_axis1_xr(self):
        self.results = gradient(self.test_data_xr)
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1.values,
            self.test_results_lat.values,
            decimal=3,
        )

    def test_gradient_axis0_dask(self):
        self.results = gradient(self.test_data_dask)
        self.results_axis0 = self.results[0]
        np.testing.assert_almost_equal(
            self.results_axis0.values,
            self.test_results_lon.values,
            decimal=3,
        )

    def test_gradient_axis1_dask(self):
        self.results = gradient(self.test_data_dask)
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1.values,
            self.test_results_lat.values,
            decimal=3,
        )

    def test_gradient_axis0_xr_1d_nocoords(self):
        self.results = gradient(self.test_data_xr_nocoords,
                                lon=self.test_coords_1d_lon,
                                lat=self.test_coords_1d_lat)
        self.results_axis0 = self.results[0]
        np.testing.assert_almost_equal(
            self.results_axis0.values,
            self.test_results_lon.values,
            decimal=3,
        )

    def test_gradient_axis1_xr_1d_nocoords(self):
        self.results = gradient(self.test_data_xr_nocoords,
                                lon=self.test_coords_1d_lon,
                                lat=self.test_coords_1d_lat)
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1.values,
            self.test_results_lat.values,
            decimal=3,
        )

    def test_gradient_axis0_xr_2d_nocoords(self):
        self.results = gradient(self.test_data_xr_nocoords,
                                self.test_coords_2d_lon_np,
                                self.test_coords_2d_lat_np)
        self.results_axis0 = self.results[0]
        np.testing.assert_almost_equal(
            self.results_axis0.values,
            self.test_results_lon.values,
            decimal=3,
        )

    def test_gradient_axis1_xr_2d_nocoords(self):
        self.results = gradient(self.test_data_xr_nocoords,
                                self.test_coords_2d_lon_np,
                                self.test_coords_2d_lat_np)
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1.values,
            self.test_results_lat.values,
            decimal=3,
        )

    def test_gradient_axis0_xr_2d_coords(self):
        self.results = gradient(self.test_data_xr_2d_coords)
        self.results_axis0 = self.results[0]
        np.testing.assert_almost_equal(
            self.results_axis0.values,
            self.test_results_lon.values,
            decimal=3,
        )

    def test_gradient_axis1_xr_2d_coords(self):
        self.results = gradient(self.test_data_xr_2d_coords)
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1.values,
            self.test_results_lat.values,
            decimal=3,
        )

    def test_gradient_axis0_np_1d_nocoords(self):
        self.results = gradient(self.test_data_np,
                                lon=self.test_coords_1d_lon_np,
                                lat=self.test_coords_1d_lat_np)
        self.results_axis0 = self.results[0]
        np.testing.assert_almost_equal(
            self.results_axis0,
            self.test_results_lon.values,
            decimal=3,
        )

    def test_gradient_axis1_np_1d_nocoords(self):
        self.results = gradient(self.test_data_np,
                                lon=self.test_coords_1d_lon_np,
                                lat=self.test_coords_1d_lat_np)
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1,
            self.test_results_lat.values,
            decimal=3,
        )

    def test_gradient_axis0_np_2d_nocoords(self):
        self.results = gradient(self.test_data_np, self.test_coords_2d_lon_np,
                                self.test_coords_2d_lat_np)
        self.results_axis0 = self.results[0]
        np.testing.assert_almost_equal(
            self.results_axis0,
            self.test_results_lon.values,
            decimal=3,
        )

    def test_gradient_axis1_np_2d_nocoords(self):
        self.results = gradient(self.test_data_np, self.test_coords_2d_lon_np,
                                self.test_coords_2d_lat_np)
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1,
            self.test_results_lat.values,
            decimal=3,
        )
