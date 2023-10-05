import sys
import pytest

import numpy as np
import xarray as xr

from geocat.comp import gradient


class Test_Gradient:
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

    @pytest.fixture(scope="class")
    def test_data_xr(self):
        return xr.load_dataset(
            'test/gradient_test_data.nc').to_array().squeeze()

    @pytest.fixture(scope="class")
    def test_results_lon(self):
        return xr.load_dataset(
            'test/gradient_test_results_longitude.nc').to_array().squeeze()

    @pytest.fixture(scope="class")
    def test_results_lat(self):
        return xr.load_dataset(
            'test/gradient_test_results_latitude.nc').to_array().squeeze()

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setUpClass(cls, test_data_xr) -> None:
        cls.__name__ = "setUpClass"  # Python 3.9
        cls.test_data_xr_nocoords = xr.DataArray(test_data_xr, coords={})
        cls.test_data_np = test_data_xr.values
        cls.test_data_dask = test_data_xr.chunk(10)
        cls.test_coords_1d_lon = test_data_xr.coords['lon']
        cls.test_coords_1d_lat = test_data_xr.coords['lat']
        cls.test_coords_2d_lon_np, cls.test_coords_2d_lat_np = np.meshgrid(
            cls.test_coords_1d_lon, cls.test_coords_1d_lat)
        cls.test_data_xr_2d_coords = xr.DataArray(
            test_data_xr,
            dims=['x', 'y'],
            coords=dict(
                lon=(['x', 'y'], cls.test_coords_2d_lon_np),
                lat=(['x', 'y'], cls.test_coords_2d_lat_np),
            ),
        )
        cls.test_coords_1d_lon_np = cls.test_coords_1d_lon.values
        cls.test_coords_1d_lat_np = cls.test_coords_1d_lat.values

    def test_gradient_axis0_xr(self, test_data_xr, test_results_lon) -> None:
        self.results = gradient(test_data_xr)
        self.results_axis0 = self.results[0]
        np.testing.assert_almost_equal(
            self.results_axis0.values,
            test_results_lon.values,
            decimal=3,
        )

    def test_gradient_axis1_xr(self, test_data_xr, test_results_lat) -> None:
        self.results = gradient(test_data_xr)
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1.values,
            test_results_lat.values,
            decimal=3,
        )

    def test_gradient_axis0_dask(self, test_results_lon) -> None:
        self.results = gradient(self.test_data_dask)
        self.results_axis0 = self.results[0]
        np.testing.assert_almost_equal(
            self.results_axis0.values,
            test_results_lon.values,
            decimal=3,
        )

    def test_gradient_axis1_dask(self, test_results_lat) -> None:
        self.results = gradient(self.test_data_dask)
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1.values,
            test_results_lat.values,
            decimal=3,
        )

    def test_gradient_axis0_xr_1d_nocoords(self, test_results_lon) -> None:
        self.results = gradient(self.test_data_xr_nocoords,
                                lon=self.test_coords_1d_lon,
                                lat=self.test_coords_1d_lat)
        self.results_axis0 = self.results[0]
        np.testing.assert_almost_equal(
            self.results_axis0.values,
            test_results_lon.values,
            decimal=3,
        )

    def test_gradient_axis1_xr_1d_nocoords(self, test_results_lat) -> None:
        self.results = gradient(self.test_data_xr_nocoords,
                                lon=self.test_coords_1d_lon,
                                lat=self.test_coords_1d_lat)
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1.values,
            test_results_lat.values,
            decimal=3,
        )

    def test_gradient_axis0_xr_2d_nocoords(self, test_results_lon) -> None:
        self.results = gradient(self.test_data_xr_nocoords,
                                self.test_coords_2d_lon_np,
                                self.test_coords_2d_lat_np)
        self.results_axis0 = self.results[0]
        np.testing.assert_almost_equal(
            self.results_axis0.values,
            test_results_lon.values,
            decimal=3,
        )

    def test_gradient_axis1_xr_2d_nocoords(self, test_results_lat) -> None:
        self.results = gradient(self.test_data_xr_nocoords,
                                self.test_coords_2d_lon_np,
                                self.test_coords_2d_lat_np)
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1.values,
            test_results_lat.values,
            decimal=3,
        )

    def test_gradient_axis0_xr_2d_coords(self, test_results_lon) -> None:
        self.results = gradient(self.test_data_xr_2d_coords)
        self.results_axis0 = self.results[0]
        np.testing.assert_almost_equal(
            self.results_axis0.values,
            test_results_lon.values,
            decimal=3,
        )

    def test_gradient_axis1_xr_2d_coords(self, test_results_lat) -> None:
        self.results = gradient(self.test_data_xr_2d_coords)
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1.values,
            test_results_lat.values,
            decimal=3,
        )

    def test_gradient_axis0_np_1d_nocoords(self, test_results_lon) -> None:
        self.results = gradient(self.test_data_np,
                                lon=self.test_coords_1d_lon_np,
                                lat=self.test_coords_1d_lat_np)
        self.results_axis0 = self.results[0]
        np.testing.assert_almost_equal(
            self.results_axis0,
            test_results_lon.values,
            decimal=3,
        )

    def test_gradient_axis1_np_1d_nocoords(self, test_results_lat) -> None:
        self.results = gradient(self.test_data_np,
                                lon=self.test_coords_1d_lon_np,
                                lat=self.test_coords_1d_lat_np)
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1,
            test_results_lat.values,
            decimal=3,
        )

    def test_gradient_axis0_np_2d_nocoords(self, test_results_lon) -> None:
        self.results = gradient(self.test_data_np, self.test_coords_2d_lon_np,
                                self.test_coords_2d_lat_np)
        self.results_axis0 = self.results[0]
        np.testing.assert_almost_equal(
            self.results_axis0,
            test_results_lon.values,
            decimal=3,
        )

    def test_gradient_axis1_np_2d_nocoords(self, test_results_lat) -> None:
        self.results = gradient(self.test_data_np, self.test_coords_2d_lon_np,
                                self.test_coords_2d_lat_np)
        self.results_axis1 = self.results[1]
        np.testing.assert_almost_equal(
            self.results_axis1,
            test_results_lat.values,
            decimal=3,
        )
