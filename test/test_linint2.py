import sys
import time
import unittest as ut
import logging
logging.basicConfig(level=logging.DEBUG)
mylogger = logging.getLogger()

import numpy as np
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
# if "--cov" in str(sys.argv):
#     from src.geocat.comp import ChunkError, CoordinateError, linint1
# else:
#     from geocat.comp import ChunkError, CoordinateError, linint1

from src.geocat.comp import ChunkError, CoordinateError, linint1a, linint1

n = 127

xi = np.linspace(0, n, num=n // 2 + 1, dtype=np.float64)
xo = np.linspace(xi.min(), xi.max(), num=xi.shape[0] * 2 - 1)
fi_np = np.random.rand(96, 3, len(xi)).astype(np.float64)

chunks = {
    'time': fi_np.shape[0],
    'level': fi_np.shape[1],
    'lon': fi_np.shape[2]
}

#dummy data
class Test_linint1(ut.TestCase):

    def test_example(self):
        xi = np.array([2, 4, 6, 8])
        fi = np.array([1, 2, 3, 4])
        xo = np.array([2, 3, 4, 5, 6, 7, 8])
        fo = linint1a(fi, xo, xi)
        np.testing.assert_array_equal([1, 1.5, 2, 2.5, 3, 3.5, 4], fo)
    
    def test_cyclic(self):
        xi = np.array([0, 50, 100, 350])
        fi = np.array([1, 2, 3, 4])
        xo = np.array([0, 40, 355])
        fo = linint1a(fi, xo, xi, 1)
        np.testing.assert_array_equal([1.0, 1.8, 3.94], fo)

    def test_nonmcinc(self):
        xi = np.array([1, 2, 4, 6, 3, 5])
        fi = np.array([3, 4, 5, 7, 6, 5])
        xo = np.array([1.5, 2, 2.5])
        fo = linint1a(fi, xo, xi)
        np.testing.assert_array_equal([3.5, 4.0, 5.0], fo)

    def test_linint1(self):
        fi = xr.DataArray(fi_np,
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo, xi)

        # d = ~np.equal(fi.values, fo[..., ::2].values)
        # where = np.where(d)
        # for i in range(3):
        #     for j in range(len(where[0])):
        #         logging.warning("where[%i][%i] = %i", i, j, where[i][j])
        # for i in range(12):
        #     logging.warning("fi[%i][%i][%i] = %f", where[0][0], where[1][0], i, fi[where[0][0]][where[1][0]][i].values)
        # for i in range(25):
        #     logging.warning("fo[%i][%i][%i] = %f", where[0][0], where[1][0], i, fo[where[0][0]][where[1][0]][i].values)
        
        np.testing.assert_almost_equal(fi.values, fo[..., ::2].values, decimal=16)


# TODO: All of these tests should be revisited since they indeed are not
#  checking any actually interpolated values.
class Test_linint1_float64(ut.TestCase):

    def test_linint1(self):
        fi = xr.DataArray(fi_np,
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo)
        np.testing.assert_almost_equal(fi.values, fo[..., ::2].values, decimal=16)

    def test_linint1_msg(self):
        fi = xr.DataArray(fi_np,
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo, msg_py=fi_np[0, 0, 0])
        np.testing.assert_almost_equal(fi.values, fo[..., ::2].values, decimal=16)

    def test_linint2_nan(self):
        fi_np_copy = fi_np.copy()
        fi_np_copy[:, :, 0] = np.nan
        fi = xr.DataArray(fi_np_copy,
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo)
        np.testing.assert_almost_equal(fi.values, fo[..., ::2].values, decimal=16)

    def test_linint1_masked(self):
        fi_mask = np.zeros(fi_np.shape, dtype=bool)
        fi_mask[:, :, 0] = True
        fi_ma = np.ma.MaskedArray(fi_np, mask=fi_mask)
        fi = xr.DataArray(fi_ma,
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo)
        np.testing.assert_almost_equal(fi, fo[..., ::2].values, decimal=16)


class Test_linint1_float32(ut.TestCase):

    def test_linint1_float32(self):
        fi = xr.DataArray(fi_np.astype(np.float32),
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo)
        np.testing.assert_almost_equal(fi.values, fo[..., ::2].values, decimal=16)

    def test_linint1_msg_float32(self):
        fi_np_copy = fi_np.astype(np.float32)
        fi = xr.DataArray(fi_np,
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo, msg_py=fi_np_copy[0, 0, 0])
        np.testing.assert_almost_equal(fi.values, fo[..., ::2].values, decimal=16)

    def test_linint1_nan_float32(self):
        fi_np_copy = fi_np.astype(np.float32)
        fi_np_copy[:, :, 0] = np.nan
        fi = xr.DataArray(fi_np_copy,
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo)
        np.testing.assert_almost_equal(fi.values, fo[..., ::2].values, decimal=16)

    def test_linint1_masked_float32(self):
        fi_mask = np.zeros(fi_np.shape, dtype=bool)
        fi_mask[:, :, 0] = True
        fi_ma = np.ma.MaskedArray((fi_np * 100).astype(np.float32),
                                  mask=fi_mask)
        fi = xr.DataArray(fi_ma,
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo)
        np.testing.assert_almost_equal(fi, fo[..., ::2].values, decimal=16)


class Test_linint1_int32(ut.TestCase):

    def test_linint1_int32(self):
        fi = xr.DataArray((fi_np * 100).astype(np.int32),
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo)
        np.testing.assert_array_equal(fi.values, fo[..., ::2].values, decimal=16)

    def test_linint1_msg_int32(self):
        fi_np_copy = fi_np.astype(np.int32)
        fi = xr.DataArray(fi_np,
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo, msg_py=fi_np_copy[0, 0, 0])
        np.testing.assert_array_equal(fi.values, fo[..., ::2].values, decimal=16)

    def test_linint1_masked_int32(self):
        fi_mask = np.zeros(fi_np.shape, dtype=bool)
        fi_mask[:, :, 0] = True
        fi_ma = np.ma.MaskedArray((fi_np * 100).astype(np.int32), mask=fi_mask)
        fi = xr.DataArray(fi_ma,
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo)
        np.testing.assert_array_equal(fi, fo[..., ::2].values, decimal=16)


class Test_linint1_int64(ut.TestCase):

    def test_linint1_int64(self):
        fi = xr.DataArray((fi_np * 100).astype(np.int64),
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo)
        np.testing.assert_array_equal(fi.values, fo[..., ::2].values, decimal=16)
        print("Good after assertion")

    def test_linint1_msg_int64(self):
        fi_np_copy = (fi_np * 100).astype(np.int64)
        fi = xr.DataArray(fi_np,
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo, msg_py=fi_np_copy[0, 0, 0])
        np.testing.assert_array_equal(fi.values, fo[..., ::2].values, decimal=16)

    def test_linint1_masked_int64(self):
        fi_mask = np.zeros(fi_np.shape, dtype=bool)
        fi_mask[:, :, 0] = True
        fi_ma = np.ma.MaskedArray((fi_np * 100).astype(np.int64), mask=fi_mask)
        fi = xr.DataArray(fi_ma,
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo)
        np.testing.assert_array_equal(fi, fo[..., ::2].values, decimal=16)

class Test_linint1_dask(ut.TestCase):

    def test_linint1_chunked_leftmost(self):
        # use 1 for time and level chunk sizes
        chunks = {
            'time': 1,
            'level': 1,
            'lon': fi_np.shape[3]
        }
        fi = xr.DataArray(fi_np,
                           dims=['time', 'level', 'lon'],
                           coords={
                              'lon': xi
                          }).chunk(chunks)
        fo = linint1a(fi, xo)
        np.testing.assert_array_equal(fi.values, fo[..., ::2].values)

    def test_linint1_chunked_interp(self):
        # use 1 for interpolated dimension chunk sizes -- this should throw a ChunkError
        chunks = {'time': 1, 'level': 1, 'lon': 1}
        fi = xr.DataArray(fi_np,
                          dims=['time', 'level', 'lon'],
                          coords={
                              'lon': xi
                          }).chunk(chunks)
        with self.assertRaises(ChunkError):
            fo = linint1a(fi, xo)


class Test_linint1_numpy(ut.TestCase):

    def test_linint1_fi_np(self):
        fo = linint1a(fi_np, xo, xi=xi)
        np.testing.assert_array_equal(fi_np, fo[..., ::2])

    def test_linint1_fi_np_no_xi(self):
        with self.assertRaises(CoordinateError):
            fo = linint1a(fi_np, xo, 0)

# # class Test_linint2_non_monotonic(ut.TestCase):
# #
# #     def test_linint2_non_monotonic_xr(self):
# #         fi = xr.DataArray(fi_np[:, :, ::-1, :],
# #                           dims=['time', 'level', 'lat', 'lon'],
# #                           coords={
# #                               'lat': yi_reverse,
# #                               'lon': xi
# #                           }).chunk(chunks)
# #         with self.assertWarns(_ncomp.NcompWarning):
# #             linint2(fi, xo, yo).compute()
# #
# #     def test_linint2_non_monotonic_np(self):
# #         with self.assertWarns(geocat.ncomp._ncomp.NcompWarning):
# #             linint2(fi_np[:, :, ::-1, :],
# #                                  xo,
# #                                  yo,
# #                                  xi=xi,
# #                                  yi=yi_reverse)


# class Test_linint1_non_contiguous(ut.TestCase):

#     def test_linint1_non_contiguous_xr(self):
#         fi = xr.DataArray(fi_np[:, :, ::-1, :],
#                           dims=['time', 'level', 'lat', 'lon'],
#                           coords={
#                               'lat': yi_reverse,
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint2(fi[:, :, ::-1, :], xo, yo)
#         np.testing.assert_array_equal(fi[:, :, ::-1, :].values,
#                                       fo[..., ::2, ::2].values)

#     def test_linint2_non_contiguous_np(self):
#         fi = xr.DataArray(fi_np[:, :, ::-1, :],
#                           dims=['time', 'level', 'lat', 'lon'],
#                           coords={
#                               'lat': yi_reverse,
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint2(fi[:, :, ::-1, :].values,
#                      xo,
#                      yo,
#                      xi=xi,
#                      yi=yi_reverse[::-1])
#         np.testing.assert_array_equal(fi[:, :, ::-1, :], fo[..., ::2, ::2])
