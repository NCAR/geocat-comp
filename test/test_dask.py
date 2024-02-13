import pytest

import dask.array
import xarray as xr
import numpy as np

from geocat.comp.meteorology import (
    dewtemp, heat_index, relhum, relhum_ice, relhum_water,
    actual_saturation_vapor_pressure, max_daylight, psychrometric_constant,
    saturation_vapor_pressure, saturation_vapor_pressure_slope, delta_pressure)


@pytest.fixture(scope="module")
def client() -> None:
    # dask client reference for all subsequent tests
    client = dd.Client()
    yield client
    client.close()


class Test_dask_compat:

    def test_dewtemp_dask(self):
        t_def = [
            29.3, 28.1, 23.5, 20.9, 18.4, 15.9, 13.1, 10.1, 6.7, 3.1, -0.5,
            -4.5, -9.0, -14.8, -21.5, -29.7, -40.0, -52.4
        ]

        rh_def = [
            75.0, 60.0, 61.1, 76.7, 90.5, 89.8, 78.3, 76.5, 46.0, 55.0, 63.8,
            53.2, 42.9, 41.7, 51.0, 70.6, 50.0, 50.0
        ]

        dt_2 = [
            24.38342, 19.55563, 15.53281, 16.64218, 16.81433, 14.22482,
            9.401337, 6.149719, -4.1604, -5.096619, -6.528168, -12.61957,
            -19.38332, -25.00714, -28.9841, -33.34853, -46.51273, -58.18289
        ]
        tk = xr.DataArray(np.asarray(t_def) + 273.15).chunk(6)
        rh = xr.DataArray(rh_def).chunk(6)

        out = dewtemp(tk, rh)
        assert isinstance((out - 273.15).data, dask.array.Array)
        assert np.allclose(out - 273.15, dt_2, atol=0.1)

    def test_heat_index_dask(self):

        ncl_gt_1 = [
            137.36142, 135.86795, 104.684456, 131.25621, 105.39449, 79.78999,
            83.57511, 59.965, 30.
        ]

        t1 = np.array([104, 100, 92, 92, 86, 80, 80, 60, 30])
        rh1 = np.array([55, 65, 60, 90, 90, 40, 75, 90, 50])

        t = xr.DataArray(t1).chunk(3)
        rh = xr.DataArray(rh1).chunk(3)

        out = heat_index(t, rh)
        assert isinstance(out.data, dask.array.Array)
        assert np.allclose(out, ncl_gt_1, atol=0.005)

    def test_relhum_dask(self):
        p_def = [
            100800, 100000, 95000, 90000, 85000, 80000, 75000, 70000, 65000,
            60000, 55000, 50000, 45000, 40000, 35000, 30000, 25000, 20000,
            17500, 15000, 12500, 10000, 8000, 7000, 6000, 5000, 4000, 3000,
            2500, 2000
        ]

        t_def = [
            302.45, 301.25, 296.65, 294.05, 291.55, 289.05, 286.25, 283.25,
            279.85, 276.25, 272.65, 268.65, 264.15, 258.35, 251.65, 243.45,
            233.15, 220.75, 213.95, 206.65, 199.05, 194.65, 197.15, 201.55,
            206.45, 211.85, 216.85, 221.45, 222.45, 225.65
        ]

        q_def = [
            0.02038, 0.01903, 0.01614, 0.01371, 0.01156, 0.0098, 0.00833,
            0.00675, 0.00606, 0.00507, 0.00388, 0.00329, 0.00239, 0.0017, 0.001,
            0.0006, 0.0002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]

        rh_gt_2 = [
            79.8228, 79.3578, 84.1962, 79.4898, 73.989, 69.2401, 66.1896,
            61.1084, 64.21, 63.8305, 58.0412, 60.8194, 57.927, 62.3734, 62.9706,
            73.8184, 62.71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]

        p = xr.DataArray(p_def).chunk(10)
        t = xr.DataArray(t_def).chunk(10)
        q = xr.DataArray(q_def).chunk(10)

        out = relhum(t, q, p)
        assert isinstance(out.data, dask.array.Array)
        assert np.allclose(relhum(t, q, p), rh_gt_2, atol=0.1)
