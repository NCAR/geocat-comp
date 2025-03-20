import pytest

import numpy as np
import xarray as xr

from geocat.comp import gradient


class Test_Gradient:
    @pytest.fixture(scope="class")
    def test_data_xr(self):
        return xr.load_dataset('test/gradient_test_data.nc').to_array().squeeze()

    @pytest.fixture(scope="class")
    def expected_results(self):
        return [
            xr.load_dataset('test/gradient_test_results_longitude.nc')
            .to_array()
            .squeeze(),
            xr.load_dataset('test/gradient_test_results_latitude.nc')
            .to_array()
            .squeeze(),
        ]

    @pytest.fixture(scope="class")
    def lat_lon_meshgrid(self, test_data_xr):
        return np.meshgrid(test_data_xr.coords["lon"], test_data_xr.coords["lat"])

    def test_gradient_xr(self, test_data_xr, expected_results) -> None:
        actual_result = gradient(test_data_xr)
        np.testing.assert_almost_equal(
            np.array(actual_result), np.array(expected_results), decimal=3
        )

    def test_gradient_xr_1d_nocoords(self, test_data_xr, expected_results) -> None:
        actual_result = gradient(
            xr.DataArray(test_data_xr, coords={}),
            lon=test_data_xr.coords["lon"],
            lat=test_data_xr.coords["lat"],
        )
        np.testing.assert_almost_equal(
            np.array(actual_result), np.array(expected_results), decimal=3
        )

    def test_gradient_xr_2d_nocoords(
        self, test_data_xr, expected_results, lat_lon_meshgrid
    ) -> None:
        (lon_2d, lat_2d) = lat_lon_meshgrid
        actual_result = gradient(
            xr.DataArray(test_data_xr, coords={}),
            lon=lon_2d,
            lat=lat_2d,
        )
        np.testing.assert_almost_equal(
            np.array(actual_result), np.array(expected_results), decimal=3
        )

    def test_gradient_xr_2d_coords(
        self, test_data_xr, expected_results, lat_lon_meshgrid
    ) -> None:
        test_data_xr_2d_coords = xr.DataArray(
            test_data_xr,
            dims=["x", "y"],
            coords=dict(
                lon=(["x", "y"], lat_lon_meshgrid[0]),
                lat=(["x", "y"], lat_lon_meshgrid[1]),
            ),
        )
        actual_result = gradient(test_data_xr_2d_coords)
        np.testing.assert_almost_equal(
            np.array(actual_result), np.array(expected_results), decimal=3
        )

    def test_gradient_np_1d_nocoords(self, test_data_xr, expected_results) -> None:
        actual_result = gradient(
            test_data_xr.values,
            lon=test_data_xr.coords["lon"].values,
            lat=test_data_xr.coords["lat"].values,
        )
        np.testing.assert_almost_equal(
            actual_result, np.array(expected_results), decimal=3
        )

    def test_gradient_np_2d_nocoords(
        self, test_data_xr, expected_results, lat_lon_meshgrid
    ) -> None:
        (lon_2d, lat_2d) = lat_lon_meshgrid
        actual_result = gradient(test_data_xr.values, lon_2d, lat_2d)

        np.testing.assert_almost_equal(
            actual_result, np.array(expected_results), decimal=3
        )
