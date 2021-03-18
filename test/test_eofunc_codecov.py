from abc import ABCMeta
from unittest import TestCase

import numpy as np
import numpy.testing as nt
# from dask.array.tests.test_xarray import xr
import xarray as xr

from src.geocat.comp import eofunc, eofunc_eofs, eofunc_pcs, eofunc_ts


class BaseEOFTestClass(metaclass=ABCMeta):
    _sample_data_eof = []

    # _sample_data[ 0 ]
    _sample_data_eof.append([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11],
                              [12, 13, 14, 15]],
                             [[16, 17, 18, 19], [20, 21, 22, 23],
                              [24, 25, 26, 27], [28, 29, 30, 31]],
                             [[32, 33, 34, 35], [36, 37, 38, 39],
                              [40, 41, 42, 43], [44, 45, 46, 47]],
                             [[48, 49, 50, 51], [52, 53, 54, 55],
                              [56, 57, 58, 59], [60, 61, 62, 63]]])

    # _sample_data[ 1 ]
    _sample_data_eof.append(np.arange(64, dtype='double').reshape((4, 4, 4)))

    # _sample_data[ 2 ]
    tmp_data = np.asarray([
        0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99,
        18, -99, 20, 21, 22, -99, 24, 25, 26, 27, 28, -99, 30, -99, 32, 33, 34,
        35, 36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50, 51,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    ],
                          dtype='double').reshape((4, 4, 4))
    _sample_data_eof.append(tmp_data)

    # _sample_data[ 3 ]
    tmp_data = np.asarray([
        0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99,
        18, -99, 20, 21, 22, -99, 24, 25, 26, 27, 28, -99, 30, -99, 32, 33, 34,
        35, 36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50, 51,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    ],
                          dtype='double').reshape((4, 4, 4))
    tmp_data[tmp_data == -99] = np.nan
    _sample_data_eof.append(tmp_data)

    # _sample_data[ 4 ]
    _sample_data_eof.append(np.arange(64, dtype='int64').reshape((4, 4, 4)))

    try:
        _nc_ds = xr.open_dataset("eofunc_dataset.nc")
    except:
        _nc_ds = xr.open_dataset("test/eofunc_dataset.nc")

    _num_attrs = 4

    expected_output = np.full((1, 4, 4), 0.25)
    expected_eigen_val_time_dim_2 = 26.66666
    expected_eigen_val_time_dim_1 = 426.66666
    expected_eigen_val_time_dim_0 = 6826.66667


class Test_eof(TestCase, BaseEOFTestClass):

    def test_eof_00(self):
        data = self._sample_data_eof[0]

        results = eofunc_eofs(data, neofs=1, time_dim=2)
        eof = results.data
        attrs = results.attrs

        nt.assert_equal(self.expected_output.shape, results.shape)

        nt.assert_array_almost_equal(self.expected_output, eof, 5)

        nt.assert_equal(self._num_attrs, len(attrs))

        nt.assert_almost_equal(self.expected_eigen_val_time_dim_2,
                               attrs['eigenvalues'].values[0], 5)

    def test_eof_deprecated(self):
        data = self._sample_data_eof[0]

        results = eofunc(data, neval=1)
        eof = results.data
        attrs = results.attrs

        nt.assert_equal(self.expected_output.shape, results.shape)

        nt.assert_array_almost_equal(self.expected_output, eof, 5)

        nt.assert_equal(self._num_attrs, len(attrs))

        nt.assert_almost_equal(self.expected_eigen_val_time_dim_2,
                               attrs['eigenvalues'].values[0], 5)

    def test_eof_01(self):
        data = self._sample_data_eof[1]

        results = eofunc_eofs(data, neofs=1, time_dim=2)
        eof = results.data
        attrs = results.attrs

        nt.assert_equal(self.expected_output.shape, results.shape)

        nt.assert_array_almost_equal(self.expected_output, eof, 5)

        nt.assert_equal(self._num_attrs, len(attrs))

        nt.assert_almost_equal(self.expected_eigen_val_time_dim_2,
                               attrs['eigenvalues'].values[0], 5)

    def test_eof_02(self):
        data = self._sample_data_eof[1]

        results = eofunc_eofs(data, neofs=1, time_dim=2)
        eof = results.data
        attrs = results.attrs

        nt.assert_equal(self.expected_output.shape, results.shape)

        nt.assert_array_almost_equal(self.expected_output, eof, 5)

        nt.assert_equal(self._num_attrs, len(attrs))

        nt.assert_almost_equal(self.expected_eigen_val_time_dim_2,
                               attrs['eigenvalues'].values[0], 5)

    def test_eof_14(self):
        data = self._sample_data_eof[4]

        results = eofunc_eofs(data, neofs=1, time_dim=2)
        eof = results.data
        attrs = results.attrs

        nt.assert_equal(self.expected_output.shape, results.shape)

        nt.assert_array_almost_equal(self.expected_output, eof, 5)

        nt.assert_equal(self._num_attrs, len(attrs))

        nt.assert_almost_equal(self.expected_eigen_val_time_dim_2,
                               attrs['eigenvalues'].values[0], 5)

    def test_eof_15(self):

        data = np.asarray(self._sample_data_eof[0])
        data = np.transpose(data, axes=(2, 1, 0))

        dims = [f"dim_{i}" for i in range(data.ndim)]
        dims[0] = 'time'

        data = xr.DataArray(data,
                            dims=dims,
                            attrs={
                                "prop1": "prop1",
                                "prop2": 2
                            })

        results = eofunc_eofs(data, neofs=1)
        eof = results.data
        attrs = results.attrs

        nt.assert_equal(self.expected_output.shape, results.shape)

        nt.assert_array_almost_equal(self.expected_output, eof, 5)

        nt.assert_equal(self._num_attrs, len(attrs))

        nt.assert_almost_equal(self.expected_eigen_val_time_dim_2,
                               attrs['eigenvalues'].values[0], 5)

        nt.assert_equal(False, ("prop1" in attrs))
        nt.assert_equal(False, ("prop2" in attrs))

    # TODO: Maybe revisited to add time_dim support for Xarray in addition to numpy inputs
    # def test_eof_15_time_dim(self):
    #
    #     data = np.asarray(self._sample_data_eof[0])
    #
    #     dims = [f"dim_{i}" for i in range(data.ndim)]
    #     dims[2] = 'time'
    #
    #     data = xr.DataArray(
    #         data,
    #         dims=dims,
    #         attrs={"prop1": "prop1",
    #                "prop2": 2,
    #                }
    #     )
    #
    #     results = eofunc_eofs(data, num_eofs=1, time_dim=2)
    #     eof = results.data
    #     attrs = results.attrs
    #
    #     nt.assert_equal(self.expected_output.shape, results.shape)
    #
    #     nt.assert_array_almost_equal(self.expected_output, eof, 5)
    #
    #     nt.assert_equal(self._num_attrs + 2, len(attrs))
    #
    #     # self.assertAlmostEqual(5.33333, attrs['eval_transpose'][0], 4)
    #     # self.assertAlmostEqual(100.0, attrs['pcvar'][0], 1)
    #     self.assertAlmostEqual(26.66666, attrs['eigenvalues'].values[0], 4)
    #     # self.assertEqual("covariance", attrs['matrix'])
    #     # self.assertEqual("transpose", attrs['method'])
    #     self.assertFalse("prop1" in attrs)
    #     self.assertFalse("prop2" in attrs)

    def test_eof_16(self):
        data = np.asarray(self._sample_data_eof[0])
        data = np.transpose(data, axes=(2, 1, 0))

        dims = [f"dim_{i}" for i in range(data.ndim)]
        dims[0] = 'time'

        data = xr.DataArray(data,
                            dims=dims,
                            attrs={
                                "prop1": "prop1",
                                "prop2": 2,
                            })

        results = eofunc_eofs(data, 1, meta=True)
        eof = results.data
        attrs = results.attrs

        nt.assert_equal(self.expected_output.shape, results.shape)

        nt.assert_array_almost_equal(self.expected_output, eof, 5)

        nt.assert_equal(self._num_attrs + 2, len(attrs))

        nt.assert_almost_equal(self.expected_eigen_val_time_dim_2,
                               attrs['eigenvalues'].values[0], 5)

        nt.assert_equal(True, ("prop1" in attrs))
        nt.assert_equal(True, ("prop2" in attrs))
        nt.assert_equal("prop1", attrs["prop1"])
        nt.assert_equal(2, attrs["prop2"])

    def test_eof_n_01(self):
        data = self._sample_data_eof[1]

        results = eofunc_eofs(data, neofs=1, time_dim=1)
        eof = results.data
        attrs = results.attrs

        nt.assert_equal(self.expected_output.shape, results.shape)

        nt.assert_array_almost_equal(self.expected_output, eof, 5)

        nt.assert_equal(self._num_attrs, len(attrs))

        nt.assert_almost_equal(self.expected_eigen_val_time_dim_1,
                               attrs['eigenvalues'].values[0], 5)

    def test_eof_n_03(self):
        data = self._sample_data_eof[1]

        results = eofunc_eofs(data, 1, time_dim=0)
        eof = results.data
        attrs = results.attrs

        nt.assert_equal(self.expected_output.shape, results.shape)

        nt.assert_array_almost_equal(self.expected_output, eof, 5)

        nt.assert_equal(self._num_attrs, len(attrs))

        nt.assert_almost_equal(self.expected_eigen_val_time_dim_0,
                               attrs['eigenvalues'].values[0], 5)

    def test_eof_n_03_1(self):
        data = self._sample_data_eof[1]

        results = eofunc_eofs(data, 1, time_dim=0)
        eof = results.data
        attrs = results.attrs

        nt.assert_equal(self.expected_output.shape, results.shape)

        nt.assert_array_almost_equal(self.expected_output, eof, 5)

        nt.assert_equal(self._num_attrs, len(attrs))

        nt.assert_almost_equal(self.expected_eigen_val_time_dim_0,
                               attrs['eigenvalues'].values[0], 5)


class Test_eof_ts(TestCase, BaseEOFTestClass):

    def test_01(self):
        sst = self._nc_ds.sst
        evec = self._nc_ds.evec
        expected_tsout = self._nc_ds.tsout

        actual_tsout = eofunc_pcs(sst, npcs=5)

        nt.assert_equal(actual_tsout.shape, expected_tsout.shape)

        nt.assert_array_almost_equal(actual_tsout, expected_tsout.data, 3)

    def test_01_deprecated(self):
        sst = self._nc_ds.sst
        evec = self._nc_ds.evec
        expected_tsout = self._nc_ds.tsout

        actual_tsout = eofunc_ts(sst, evec, time_dim=0)

        nt.assert_equal(actual_tsout.shape, expected_tsout.shape)

        nt.assert_array_almost_equal(actual_tsout, expected_tsout.data, 3)

    def test_02(self):
        sst = self._nc_ds.sst
        evec = self._nc_ds.evec
        expected_tsout = self._nc_ds.tsout

        actual_tsout = eofunc_pcs(sst, npcs=5, meta=True)

        nt.assert_equal(actual_tsout.shape, expected_tsout.shape)

        nt.assert_array_almost_equal(actual_tsout, expected_tsout.data, 3)

        nt.assert_equal(actual_tsout.coords["time"].data,
                        sst.coords["time"].data)
