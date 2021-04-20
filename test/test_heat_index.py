import sys
import unittest

import dask.array as da
import dask.distributed as dd
import numpy as np
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp import heat_index
else:
    from geocat.comp import heat_index


class Test_heat_index(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # set up ground truths
        cls.ncl_gt_1 = [
            137.36142, 135.86795, 104.684456, 131.25621, 105.39449, 79.78999,
            83.57511, 59.965, 30.
        ]
        cls.ncl_gt_2 = [
            68.585, 76.13114, 75.12854, 99.43573, 104.93261, 93.73293,
            104.328705, 123.23398, 150.34001, 106.87023
        ]

        cls.t1 = np.array([104, 100, 92, 92, 86, 80, 80, 60, 30])
        cls.rh1 = np.array([55, 65, 60, 90, 90, 40, 75, 90, 50])

        cls.t2 = np.array([70, 75, 80, 85, 90, 95, 100, 105, 110, 115])
        cls.rh2 = np.array([10, 75, 15, 80, 65, 25, 30, 40, 50, 5])

        # make client to reference in subsequent tests
        cls.client = dd.Client()

    def test_numpy_input(self):
        assert np.allclose(heat_index(self.t1, self.rh1, False),
                           self.ncl_gt_1,
                           atol=0.005)

    def test_multi_dimensional_input(self):
        assert np.allclose(heat_index(self.t2.reshape(2, 5),
                                      self.rh2.reshape(2, 5), True),
                           np.asarray(self.ncl_gt_2).reshape(2, 5),
                           atol=0.005)

    def test_alt_coef(self):
        assert np.allclose(heat_index(self.t2, self.rh2, True),
                           self.ncl_gt_2,
                           atol=0.005)

    def test_float_input(self):
        assert np.allclose(heat_index(80, 75), 83.5751, atol=0.005)

    def test_list_input(self):
        assert np.allclose(heat_index(self.t1.tolist(), self.rh1.tolist()),
                           self.ncl_gt_1,
                           atol=0.005)

    def test_xarray_input(self):
        t = xr.DataArray(self.t1)
        rh = xr.DataArray(self.rh1)

        assert np.allclose(heat_index(t, rh), self.ncl_gt_1, atol=0.005)

    def test_alternate_xarray_tag(self):
        t = xr.DataArray([15, 20])
        rh = xr.DataArray([15, 20])

        out = heat_index(t, rh)
        assert out.tag == "NCL: heat_index_nws; (Steadman+t)*0.5"

    def test_rh_warning(self):
        self.assertWarns(UserWarning, heat_index, [50, 80, 90], [0.1, 0.2, 0.5])

    def test_rh_valid(self):
        self.assertRaises(ValueError, heat_index, [50, 80, 90], [-1, 101, 50])

    def test_dask_unchunked_input(self):
        t = da.from_array(self.t1)
        rh = da.from_array(self.rh1)

        out = self.client.submit(heat_index, t, rh).result()

        assert np.allclose(out, self.ncl_gt_1, atol=0.005)

    def test_dask_chunked_input(self):
        t = da.from_array(self.t1, chunks='auto')
        rh = da.from_array(self.rh1, chunks='auto')

        out = self.client.submit(heat_index, t, rh).result()

        assert np.allclose(out, self.ncl_gt_1, atol=0.005)
