import numpy as np
import xarray as xr
import geocat.comp

from abc import ABCMeta
import unittest as ut


class BaseTestClass(metaclass=ABCMeta):

    # Dimensions and shape indices
    _no = 7
    _ni = (int)(_no // 2 + 1)
    _shape0 = 3
    _shape1 = 2

    _xi = np.linspace(0, _no, num=_ni, dtype=np.float64)
    _yi = np.linspace(0, _no, num=_ni, dtype=np.float64)

    _yi_reverse = _yi[::-1].copy()
    _xo = np.linspace(_xi.min(), _xi.max(), num=_no)
    _yo = np.linspace(_yi.min(), _yi.max(), num=_no)
    # _fi_np = np.random.rand(_shape0, _shape1, _ni, _ni).astype(np.float64)
    _fi_np = np.array([0.3237105, 0.324392, 0.4933876, 0.2887298, 0.3530066, 0.1539609, 0.5662112, 0.8207198, 0.3254946,
                       0.3400797, 0.8951765, 0.3365994, 0.285023, 0.503381, 0.7735307, 0.02115688, 0.0370131, 0.2905971,
                       0.08607747, 0.7475868, 0.3594459, 0.208128, 0.07976629, 0.147932, 0.7081553, 0.3954463, 0.76442,
                       0.6459406, 0.4730613, 0.8595491, 0.4975119, 0.002354512, 0.9233359, 0.7525227, 0.3873016, 0.4400381,
                       0.801115 ,0.02650658, 0.1531967, 0.08475874, 0.9324431, 0.307627, 0.4906232, 0.7363205, 0.1770434,
                       0.7332226, 0.3352886 ,0.8860217, 0.6281158, 0.4734903, 0.2919583, 0.1400909, 0.4408124, 0.4808828,
                       0.715275, 0.07891378, 0.7466062, 0.5349883, 0.6757824, 0.1525627, 0.4337856, 0.5456881, 0.9060123,
                       0.3816891, 0.1366455, 0.7225873, 0.5033991, 0.5035226, 0.8972847, 0.4618556, 0.5903139, 0.8825227,
                       0.487358, 0.7638103, 0.003553235, 0.5545163, 0.9651239, 0.8871626, 0.2278673, 0.7623955, 0.1769017,
                       0.7437277, 0.2431449, 0.7062375, 0.3831751, 0.4336452, 0.5618274, 0.3706031, 0.4842921, 0.1042453,
                       0.1383722, 0.06250896, 0.1314681, 0.8344675, 0.1938357, 0.1400811]).reshape((_shape0, _shape1, _ni, _ni))

    _chunks = {'time': _fi_np.shape[0], 'level': _fi_np.shape[1], 'lat': _fi_np.shape[2], 'lon': _fi_np.shape[3]}

    _ncl_truth = [0.3237105, 0.2887675, 0.1539609, 0.4888571, 0.8951765, 0.5066158, np.nan, 0.0370131, 0.223796,
                  0.208128, 0.3619402, 0.76442, 0.4775567, np.nan, 0.9233359, 0.62587, 0.02650658, 0.2444884,
                  0.4906232, 0.6120635, np.nan, 0.6281158, 0.5058253, 0.4808828, 0.6017321, 0.6757824, 0.5290115,
                  np.nan, 0.1366455, 0.5545933, 0.4618556, 0.4548832, 0.003553235, 0.3870832, np.nan,
                  0.1769017, 0.4343624, 0.4336452, 0.3095225, 0.1383722, 0.1336995, np.nan]


class Test_linint2points_numpy(ut.TestCase, BaseTestClass):
    def test_linint2points_fi_np(self):
        fo = geocat.comp.linint2_points(self._fi_np, self._xo, self._yo, 0, xi=self._xi, yi=self._yi)

        self.assertEqual((self._shape0, self._shape1), fo.shape[:-1])

        self.assertEqual(np.float64, fo.dtype)

        newshape = (self._shape0 * self._shape1 * self._no, )

        fo_vals = fo.values.reshape(newshape).tolist()

        print(self._fi_np.reshape((self._shape0 * self._shape1 * self._ni  * self._ni, )))

        print(fo_vals)

        print(self._xo)

        # Use numpy.testing.assert_almost_equal() instead of ut.TestCase.assertAlmostEqual() because the former can
        # handle NaNs but the latter cannot.
        # Compare the function-generated fo array to NCL ground-truth up to 5 decimal points
        np.testing.assert_almost_equal(self._ncl_truth, fo_vals, decimal=5)

    def test_linint2points_fi_np_no_xi_yi(self):
        with self.assertRaises(geocat.comp.CoordinateError):
            fo = geocat.comp.linint2_points(self._fi_np, self._xo, self._yo, 0)

    def test_linint2points_fi_np_no_yi(self):
        with self.assertRaises(geocat.comp.CoordinateError):
            fo = geocat.comp.linint2_points(self._fi_np, self._xo, self._yo, 0, xi=self._xi)

    def test_linint2points_fi_np_no_xi(self):
        with self.assertRaises(geocat.comp.CoordinateError):
            fo = geocat.comp.linint2_points(self._fi_np, self._xo, self._yo, 0, yi=self._yi)


class Test_linint2points_non_monotonic(ut.TestCase, BaseTestClass):
    # def test_linint2points_non_monotonic_xr(self):
    #     fi = xr.DataArray(self._fi_np[:, :, ::-1, :], dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi_reverse, 'lon': self._xi}).chunk(_chunks)
    #     with self.assertWarns(geocat.comp._ncomp.NcompWarning):
    #         geocat.comp.linint2_points(fi, self._xo, self._yo, 0).compute()

    def test_linint2points_non_monotonic_np(self):
        with self.assertWarns(geocat.comp._ncomp.NcompWarning):
            geocat.comp.linint2_points(self._fi_np[:, :, ::-1, :], self._xo, self._yo, 0, xi=self._xi, yi=self._yi_reverse)


# class Test_linint2points_float64(ut.TestCase, BaseTestClass):
#     def test_linint2points(self):
#         fi = xr.DataArray(self._fi_np, dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0)
#         print(fi.values.shape)
#         print(fo[..., ::2, ::2].values.shape)
#         np.testing.assert_array_equal(fi.values, fo[..., ::2, ::2].values)
#
#     def test_linint2points_msg(self):
#         fi = xr.DataArray(self._fi_np, dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0, msg=self._fi_np[0, 0, 0, 0])
#         np.testing.assert_array_equal(fi.values, fo[..., ::2, ::2].values)
#
#     def test_linint2points_nan(self):
#         fi_np_copy = self._fi_np.copy()
#         fi_np_copy[:,:,0,0] = np.nan
#         fi = xr.DataArray(fi_np_copy, dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0)
#         np.testing.assert_array_equal(fi.values, fo[..., ::2, ::2].values)
#
#     def test_linint2points_masked(self):
#         fi_mask = np.zeros(self._fi_np.shape, dtype=np.bool)
#         fi_mask[:,:,0,0] = True
#         fi_ma = np.ma.MaskedArray(self._fi_np, mask=fi_mask)
#         fi = xr.DataArray(fi_ma, dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0)
#         np.testing.assert_array_equal(fi, fo[..., ::2, ::2].values)
#
#
# class Test_linint2points_float32(ut.TestCase, BaseTestClass):
#     def test_linint2points_float32(self):
#         fi = xr.DataArray(self._fi_np.astype(np.float32), dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0)
#         np.testing.assert_array_equal(fi.values, fo[..., ::2, ::2].values)
#
#     def test_linint2points_msg_float32(self):
#         fi_np_copy = self._fi_np.astype(np.float32)
#         fi = xr.DataArray(self._fi_np, dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0, msg=fi_np_copy[0, 0, 0, 0])
#         np.testing.assert_array_equal(fi.values, fo[..., ::2, ::2].values)
#
#     def test_linint2points_nan_float32(self):
#         fi_np_copy = self._fi_np.astype(np.float32)
#         fi_np_copy[:,:,0,0] = np.nan
#         fi = xr.DataArray(fi_np_copy, dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0)
#         np.testing.assert_array_equal(fi.values, fo[..., ::2, ::2].values)
#
#     def test_linint2points_masked_float32(self):
#         fi_mask = np.zeros(self._fi_np.shape, dtype=np.bool)
#         fi_mask[:,:,0,0] = True
#         fi_ma = np.ma.MaskedArray((self._fi_np * 100).astype(np.float32), mask=fi_mask)
#         fi = xr.DataArray(fi_ma, dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0)
#         np.testing.assert_array_equal(fi, fo[..., ::2, ::2].values)
#
#
# class Test_linint2points_int32(ut.TestCase, BaseTestClass):
#     def test_linint2points_int32(self):
#         fi = xr.DataArray((self._fi_np * 100).astype(np.int32), dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0)
#         np.testing.assert_array_equal(fi.values, fo[..., ::2, ::2].values)
#
#     def test_linint2points_msg_int32(self):
#         fi_np_copy = self._fi_np.astype(np.int32)
#         fi = xr.DataArray(self._fi_np, dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0, msg=fi_np_copy[0, 0, 0, 0])
#         np.testing.assert_array_equal(fi.values, fo[..., ::2, ::2].values)
#
#     def test_linint2points_masked_int32(self):
#         fi_mask = np.zeros(self._fi_np.shape, dtype=np.bool)
#         fi_mask[:,:,0,0] = True
#         fi_ma = np.ma.MaskedArray((self._fi_np * 100).astype(np.int32), mask=fi_mask)
#         fi = xr.DataArray(fi_ma, dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0)
#         np.testing.assert_array_equal(fi, fo[..., ::2, ::2].values)
#
#
# class Test_linint2points_int64(ut.TestCase, BaseTestClass):
#     def test_linint2points_int64(self):
#         fi = xr.DataArray((self._fi_np * 100).astype(np.int64), dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0)
#         np.testing.assert_array_equal(fi.values, fo[..., ::2, ::2].values)
#
#     def test_linint2points_msg_int64(self):
#         fi_np_copy = (self._fi_np * 100).astype(np.int64)
#         fi = xr.DataArray(self._fi_np, dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0, msg=fi_np_copy[0, 0, 0, 0])
#         np.testing.assert_array_equal(fi.values, fo[..., ::2, ::2].values)
#
#     def test_linint2points_masked_int64(self):
#         fi_mask = np.zeros(self._fi_np.shape, dtype=np.bool)
#         fi_mask[:,:,0,0] = True
#         fi_ma = np.ma.MaskedArray((self._fi_np * 100).astype(np.int64), mask=fi_mask)
#         fi = xr.DataArray(fi_ma, dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0)
#         np.testing.assert_array_equal(fi, fo[..., ::2, ::2].values)
#
#
# class Test_linint2points_dask(ut.TestCase, BaseTestClass):
#     def test_linint2points_chunked_leftmost(self):
#         # use 1 for time and level chunk sizes
#         chunks = {'time': 1, 'level': 1, 'lat': self._fi_np.shape[2], 'lon': self._fi_np.shape[3]}
#         fi = xr.DataArray(self._fi_np, dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(chunks)
#         fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0)
#         np.testing.assert_array_equal(fi.values, fo[..., ::2, ::2].values)
#
#     def test_linint2points_chunked_interp(self):
#         # use 1 for interpolated dimension chunk sizes -- this should throw a ChunkError
#         chunks = {'time': 1, 'level': 1, 'lat': 1, 'lon': 1}
#         fi = xr.DataArray(self._fi_np, dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi, 'lon': self._xi}).chunk(chunks)
#         with self.assertRaises(geocat.comp.ChunkError):
#             fo = geocat.comp.linint2_points(fi, self._xo, self._yo, 0)
#
#
# class Test_linint2points_non_contiguous(ut.TestCase, BaseTestClass):
#     def test_linint2points_non_contiguous_xr(self):
#         fi = xr.DataArray(self._fi_np[:, :, ::-1, :], dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi_reverse, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi[:,:,::-1,:], self._xo, self._yo, 0)
#         np.testing.assert_array_equal(fi[:,:,::-1,:].values, fo[..., ::2, ::2].values)
#
#     def test_linint2points_non_contiguous_np(self):
#         fi = xr.DataArray(self._fi_np[:, :, ::-1, :], dims=['time', 'level', 'lat', 'lon'], coords={'lat': self._yi_reverse, 'lon': self._xi}).chunk(_chunks)
#         fo = geocat.comp.linint2_points(fi[:,:,::-1,:].values, self._xo, self._yo, 0, xi=self._xi, yi=self._yi_reverse[::-1])
#         np.testing.assert_array_equal(fi[:,:,::-1,:].values, fo[..., ::2, ::2].values)
