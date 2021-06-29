import sys
import unittest

import dask.array
import dask.distributed as dd
import numpy as np
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp import dewtemp
else:
    from geocat.comp import dewtemp


class Test_dewtemp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # set up ground truths
        cls.t_def = [
            29.3, 28.1, 23.5, 20.9, 18.4, 15.9, 13.1, 10.1, 6.7, 3.1, -0.5,
            -4.5, -9.0, -14.8, -21.5, -29.7, -40.0, -52.4
        ]

        cls.rh_def = [
            75.0, 60.0, 61.1, 76.7, 90.5, 89.8, 78.3, 76.5, 46.0, 55.0, 63.8,
            53.2, 42.9, 41.7, 51.0, 70.6, 50.0, 50.0
        ]

        cls.dt_1 = 6.3

        cls.dt_2 = [
            24.38342, 19.55563, 15.53281, 16.64218, 16.81433, 14.22482,
            9.401337, 6.149719, -4.1604, -5.096619, -6.528168, -12.61957,
            -19.38332, -25.00714, -28.9841, -33.34853, -46.51273, -58.18289
        ]

        # make dask client to reference in subsequent tests
        cls.client = dd.Client()

    def test_float_input(self):
        tk = 18. + 273.15
        rh = 46.5

        assert np.allclose(dewtemp(tk, rh) - 273.15, self.dt_1, 0.1)

    def test_list_input(self):
        tk = (np.asarray(self.t_def) + 273.15).tolist()

        assert np.allclose(dewtemp(tk, self.rh_def) - 273.15, self.dt_2, 0.1)

    def test_numpy_input(self):
        tk = np.asarray(self.t_def) + 273.15
        rh = np.asarray(self.rh_def)

        assert np.allclose(dewtemp(tk, rh) - 273.15, self.dt_2, 0.1)

    def test_xarray_input(self):
        tk = xr.DataArray(np.asarray(self.t_def) + 273.15)
        rh = xr.DataArray(self.rh_def)

        assert np.allclose(dewtemp(tk, rh) - 273.15, self.dt_2, 0.1)

    def test_dims_error(self):
        self.assertRaises(ValueError, dewtemp, self.t_def[:10], self.rh_def[:8])

    def test_xarray_type_error(self):
        self.assertRaises(TypeError, dewtemp, self.t_def,
                          xr.DataArray(self.rh_def))

    def test_dask_compute(self):
        tk = xr.DataArray(np.asarray(self.t_def) + 273.15).chunk(6)
        rh = xr.DataArray(self.rh_def).chunk(6)

        assert np.allclose(dewtemp(tk, rh) - 273.15, self.dt_2, atol=0.1)

    def test_dask_lazy(self):
        tk = xr.DataArray(np.asarray(self.t_def) + 273.15).chunk(6)
        rh = xr.DataArray(self.rh_def).chunk(6)

        assert isinstance((dewtemp(tk, rh) - 273.15).data, dask.array.Array)
