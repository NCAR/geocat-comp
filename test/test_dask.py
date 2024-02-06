import pytest

import dask.array
import xarray as xr

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
