import sys
import unittest

import dask
import dask.distributed as dd
import numpy as np
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp import relhum, relhum_ice, relhum_water
else:
    from geocat.comp import relhum, relhum_ice, relhum_water


class Test_relhum(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # set up ground truths
        cls.p_def = [
            100800, 100000, 95000, 90000, 85000, 80000, 75000, 70000, 65000,
            60000, 55000, 50000, 45000, 40000, 35000, 30000, 25000, 20000,
            17500, 15000, 12500, 10000, 8000, 7000, 6000, 5000, 4000, 3000,
            2500, 2000
        ]

        cls.t_def = [
            302.45, 301.25, 296.65, 294.05, 291.55, 289.05, 286.25, 283.25,
            279.85, 276.25, 272.65, 268.65, 264.15, 258.35, 251.65, 243.45,
            233.15, 220.75, 213.95, 206.65, 199.05, 194.65, 197.15, 201.55,
            206.45, 211.85, 216.85, 221.45, 222.45, 225.65
        ]

        cls.q_def = [
            0.02038, 0.01903, 0.01614, 0.01371, 0.01156, 0.0098, 0.00833,
            0.00675, 0.00606, 0.00507, 0.00388, 0.00329, 0.00239, 0.0017, 0.001,
            0.0006, 0.0002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]

        cls.rh_gt_1 = 46.4

        cls.rh_gt_2 = [
            79.8228, 79.3578, 84.1962, 79.4898, 73.989, 69.2401, 66.1896,
            61.1084, 64.21, 63.8305, 58.0412, 60.8194, 57.927, 62.3734, 62.9706,
            73.8184, 62.71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]

        # make dask client to reference in subsequent tests
        cls.client = dd.Client()

    def test_float_input(self):
        p = 1000. * 100
        t = 18. + 273.15
        q = 6. / 1000.

        assert np.allclose(relhum(t, q, p), self.rh_gt_1, atol=0.1)

    def test_list_input(self):

        assert np.allclose(relhum(self.t_def, self.q_def, self.p_def),
                           self.rh_gt_2,
                           atol=0.1)

    def test_numpy_input(self):
        p = np.asarray(self.p_def)
        t = np.asarray(self.t_def)
        q = np.asarray(self.q_def)

        assert np.allclose(relhum(t, q, p), self.rh_gt_2, atol=0.1)

    def test_dims_error(self):
        self.assertRaises(ValueError, relhum, self.t_def[:10], self.q_def[:10],
                          self.p_def[:9])

    def test_xarray_type_error(self):
        self.assertRaises(TypeError, relhum, self.t_def,
                          xr.DataArray(self.q_def), self.p_def)

    def test_dask_compute(self):
        p = xr.DataArray(self.p_def).chunk(10)
        t = xr.DataArray(self.t_def).chunk(10)
        q = xr.DataArray(self.q_def).chunk(10)

        assert np.allclose(relhum(t, q, p), self.rh_gt_2, atol=0.1)

    def test_dask_lazy(self):
        p = xr.DataArray(self.p_def).chunk(10)
        t = xr.DataArray(self.t_def).chunk(10)
        q = xr.DataArray(self.q_def).chunk(10)

        assert isinstance(relhum(t, q, p).data, dask.array.Array)


class Test_relhum_water(unittest.TestCase):

    rh_gt_1 = 46.3574

    def test_float_input(self):
        p = 1000. * 100
        t = 18. + 273.15
        q = 6. / 1000.

        assert np.allclose(relhum_water(t, q, p), self.rh_gt_1, atol=0.1)


class Test_relhum_ice(unittest.TestCase):

    rh_gt_1 = 147.8802

    def test_float_input(self):
        tc = -5.
        tk = tc + 273.15
        w = 3.7 / 1000.
        p = 1000. * 100.

        assert np.allclose(relhum_ice(tk, w, p), self.rh_gt_1, atol=0.1)
