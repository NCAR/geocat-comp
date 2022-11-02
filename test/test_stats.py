from unittest import TestCase
import sys
from abc import ABCMeta
import numpy as np
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp.stats import eofunc, eofunc_eofs, eofunc_pcs, eofunc_ts, pearson_r
else:
    from geocat.comp.stats import eofunc, eofunc_eofs, eofunc_pcs, eofunc_ts, pearson_r


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

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(self.expected_output, eof, 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(self.expected_eigen_val_time_dim_2,
                                       attrs['eigenvalues'].values[0], 5)

    def test_eof_deprecated(self):
        data = self._sample_data_eof[0]

        results = eofunc(data, neval=1)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(self.expected_output, eof, 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(self.expected_eigen_val_time_dim_2,
                                       attrs['eigenvalues'].values[0], 5)

    def test_eof_01(self):
        data = self._sample_data_eof[1]

        results = eofunc_eofs(data, neofs=1, time_dim=2)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(self.expected_output, eof, 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(self.expected_eigen_val_time_dim_2,
                                       attrs['eigenvalues'].values[0], 5)

    def test_eof_02(self):
        data = self._sample_data_eof[1]

        results = eofunc_eofs(data, neofs=1, time_dim=2)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(self.expected_output, eof, 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(self.expected_eigen_val_time_dim_2,
                                       attrs['eigenvalues'].values[0], 5)

    def test_eof_14(self):
        data = self._sample_data_eof[4]

        results = eofunc_eofs(data, neofs=1, time_dim=2)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(self.expected_output, eof, 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(self.expected_eigen_val_time_dim_2,
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

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(self.expected_output, eof, 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(self.expected_eigen_val_time_dim_2,
                                       attrs['eigenvalues'].values[0], 5)

        np.testing.assert_equal(False, ("prop1" in attrs))
        np.testing.assert_equal(False, ("prop2" in attrs))

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
    #     np.testing.assert_equal(self.expected_output.shape, results.shape)
    #
    #     np.testing.assert_array_almost_equal(self.expected_output, eof, 5)
    #
    #     np.testing.assert_equal(self._num_attrs + 2, len(attrs))
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

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(self.expected_output, eof, 5)

        np.testing.assert_equal(self._num_attrs + 2, len(attrs))

        np.testing.assert_almost_equal(self.expected_eigen_val_time_dim_2,
                                       attrs['eigenvalues'].values[0], 5)

        np.testing.assert_equal(True, ("prop1" in attrs))
        np.testing.assert_equal(True, ("prop2" in attrs))
        np.testing.assert_equal("prop1", attrs["prop1"])
        np.testing.assert_equal(2, attrs["prop2"])

    def test_eof_n_01(self):
        data = self._sample_data_eof[1]

        results = eofunc_eofs(data, neofs=1, time_dim=1)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(self.expected_output, eof, 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(self.expected_eigen_val_time_dim_1,
                                       attrs['eigenvalues'].values[0], 5)

    def test_eof_n_03(self):
        data = self._sample_data_eof[1]

        results = eofunc_eofs(data, 1, time_dim=0)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(self.expected_output, eof, 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(self.expected_eigen_val_time_dim_0,
                                       attrs['eigenvalues'].values[0], 5)

    def test_eof_n_03_1(self):
        data = self._sample_data_eof[1]

        results = eofunc_eofs(data, 1, time_dim=0)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(self.expected_output, eof, 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(self.expected_eigen_val_time_dim_0,
                                       attrs['eigenvalues'].values[0], 5)


class Test_eof_ts(TestCase, BaseEOFTestClass):

    def test_01(self):
        sst = self._nc_ds.sst
        evec = self._nc_ds.evec
        expected_tsout = self._nc_ds.tsout

        actual_tsout = eofunc_pcs(sst, npcs=5)

        np.testing.assert_equal(actual_tsout.shape, expected_tsout.shape)

        np.testing.assert_array_almost_equal(actual_tsout, expected_tsout.data,
                                             3)

    def test_01_deprecated(self):
        sst = self._nc_ds.sst
        evec = self._nc_ds.evec
        expected_tsout = self._nc_ds.tsout

        actual_tsout = eofunc_ts(sst, evec, time_dim=0)

        np.testing.assert_equal(actual_tsout.shape, expected_tsout.shape)

        np.testing.assert_array_almost_equal(actual_tsout, expected_tsout.data,
                                             3)

    def test_02(self):
        sst = self._nc_ds.sst
        evec = self._nc_ds.evec
        expected_tsout = self._nc_ds.tsout

        actual_tsout = eofunc_pcs(sst, npcs=5, meta=True)

        np.testing.assert_equal(actual_tsout.shape, expected_tsout.shape)

        np.testing.assert_array_almost_equal(actual_tsout, expected_tsout.data,
                                             3)

        np.testing.assert_equal(actual_tsout.coords["time"].data,
                                sst.coords["time"].data)


class Test_pearson_r(TestCase):

    @classmethod
    def setUpClass(cls):
        # Coordinates
        times = xr.cftime_range(start='2022-08-01', end='2022-08-05', freq='D')
        lats = np.linspace(start=-45, stop=45, num=3, dtype='float32')
        lons = np.linspace(start=-180, stop=180, num=4, dtype='float32')

        # Create data variables
        x, y, z = np.meshgrid(lons, lats, times)
        np.random.seed(0)
        cls.a = np.random.random_sample((len(lats), len(lons), len(times)))
        cls.b = np.power(cls.a, 2)
        cls.weights = np.cos(np.deg2rad(y))
        cls.ds = xr.Dataset(data_vars={
            'a': (('lat', 'lon', 'time'), cls.a),
            'b': (('lat', 'lon', 'time'), cls.b),
            'weights': (('lat', 'lon', 'time'), cls.weights)
        },
                            coords={
                                'lat': lats,
                                'lon': lons,
                                'time': times
                            },
                            attrs={'description': 'Test data'})

        cls.unweighted_r = 0.963472086
        cls.unweighted_r_skipnan = 0.96383798
        cls.weighted_r = 0.963209755
        cls.weighted_r_lat = [
            [0.995454445, 0.998450821, 0.99863877, 0.978765291, 0.982350092],
            [0.99999275, 0.995778831, 0.998994355, 0.991634937, 0.999868279],
            [0.991344899, 0.998632079, 0.99801552, 0.968517489, 0.985215828],
            [0.997034735, 0.99834464, 0.987382522, 0.99646236, 0.989222738]
        ]

    # Testing numpy inputs
    def test_np_inputs(self):
        a = self.a
        b = self.b
        result = pearson_r(a, b)
        assert np.allclose(self.unweighted_r, result)

    def test_np_inputs_weighted(self):
        a = self.a
        b = self.b
        w = self.weights
        result = pearson_r(a, b, weights=w)
        assert np.allclose(self.weighted_r, result)

    def test_np_inputs_warn(self):
        a = self.a
        b = self.b
        self.assertWarns(Warning, pearson_r, a, b, dim='lat', axis=0)

    def test_np_inputs_across_lats(self):
        a = self.a
        b = self.b
        w = self.weights
        result = pearson_r(a, b, weights=w, axis=0)
        assert np.allclose(self.weighted_r_lat, result)

    def test_np_inputs_skipna(self):
        # deep copy to prevent adding nans to the test data for other tests
        a = self.a.copy()
        a[0] = np.nan
        b = self.b
        result = pearson_r(a, b, skipna=True)
        assert np.allclose(self.unweighted_r_skipnan, result)

    # Testing xarray inputs
    def test_xr_inputs(self):
        a = self.ds.a
        b = self.ds.b
        result = pearson_r(a, b)
        assert np.allclose(self.unweighted_r, result)

    def test_xr_inputs_weighted(self):
        a = self.ds.a
        b = self.ds.b
        w = self.ds.weights
        result = pearson_r(a, b, weights=w)
        assert np.allclose(self.weighted_r, result)

    def test_xr_inputs_warn(self):
        a = self.ds.a
        b = self.ds.b
        self.assertWarns(Warning, pearson_r, a, b, dim='lat', axis=0)

    def test_xr_inputs_across_lats(self):
        a = self.ds.a
        b = self.ds.b
        w = self.ds.weights[:, 0, 0]
        result = pearson_r(a, b, weights=w, dim='lat')
        assert np.allclose(self.weighted_r_lat, result)

    def test_xr_inputs_skipna(self):
        # deep copy to prevent adding nans to the test data for other tests
        a = self.ds.a.copy(deep=True)
        a[0] = np.nan
        b = self.ds.b
        result = pearson_r(a, b, skipna=True)
        assert np.allclose(self.unweighted_r_skipnan, result)

    def test_keep_attrs(self):
        a = self.ds.a
        b = self.ds.b
        a.attrs.update({'Description': 'Test Data'})
        b.attrs.update({'2nd Description': 'Dummy Data'})
        result = pearson_r(a, b, keep_attrs=True)
        assert result.attrs == a.attrs
