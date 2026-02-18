from abc import ABCMeta
from cf_xarray.datasets import popds as pop
import numpy as np
import xarray as xr
import pytest

from .util import make_toy_temp_dataset
from geocat.comp.stats import (
    eofunc,
    eofunc_eofs,
    eofunc_pcs,
    eofunc_ts,
    pearson_r,
    nmse,
)

if hasattr(xr, "AlignmentError"):
    AlignmentError = xr.AlignmentError
else:
    AlignmentError = ValueError


def cupid_nmse(obs, mod):
    """Isla Simpson's NMSE calculation from CUPID toolbox
    https://github.com/NCAR/CUPiD/blob/b6a32b5dd7b88369689dbc3746c3df21af8ce40a/nblibrary/atm/nmse_utils.py
    """
    # get the weights and weight by zero if the model or obs is nan
    w = np.cos(np.deg2rad(obs.lat))
    w = w.expand_dims({"lon": obs.lon}, axis=1)
    w = w.where(~(np.isnan(obs) | np.isnan(mod)), 0)
    obs = obs.where(w != 0, 0)
    mod = mod.where(w != 0, 0)

    # edit: make sure weights dataarray
    if isinstance(w, xr.Dataset):
        w = w.to_dataarray()
    if not isinstance(w, xr.DataArray):
        w = xr.DataArray(w)

    # numerator
    num = (mod - obs) ** 2.0
    numw = num.weighted(w)
    numwm = numw.mean(["lon", "lat"])

    # denominator
    obsw = obs.weighted(w)
    obswm = obsw.mean(["lon", "lat"])
    obsprime = obs - obswm
    obsprime2 = obsprime**2.0
    obsprime2w = obsprime2.weighted(w)
    obsprime2wm = obsprime2w.mean(["lon", "lat"])

    nmse = numwm / obsprime2wm

    # edit: match attrs for testing comparison
    # clear out existing metadata on return object
    nmse = nmse.drop_attrs()
    nmse.attrs['description'] = (
        "Normalized Mean Squared Error (NMSE) between modeled and observed fields"
    )

    return nmse


class Test_nmse:
    def test_nmse(self):
        nlat = 10
        nlon = 10
        nt = 2

        m = make_toy_temp_dataset(nlat=nlat, nlon=nlon, nt=nt, nans=True)
        o = make_toy_temp_dataset(nlat=nlat, nlon=nlon, nt=nt, nans=True)

        # test on full datasets
        xr.testing.assert_allclose(cupid_nmse(o, m), nmse(o, m))

        # test on dataarrays
        xr.testing.assert_allclose(cupid_nmse(o.t, m.t), nmse(o.t, m.t))

        # test dataset var is same as dataarray calc, np to avoid metadata + dataset coord differences
        np.testing.assert_allclose(
            nmse(o, m).t.sel({"variable": "t"}).values, nmse(o.t, m.t).values
        )

    def test_nmse_validation(self):
        nlat = 10
        nlon = 10
        nt = 2

        m = make_toy_temp_dataset(nlat=nlat, nlon=nlon, nt=nt, nans=True, cf=False)
        o = make_toy_temp_dataset(nlat=nlat, nlon=nlon, nt=nt, nans=True, cf=False)

        # try non-xarray
        with pytest.raises(TypeError):
            nmse(o.t.values, m)

        # try mixed DataArray and Dataset
        with pytest.raises(TypeError):
            nmse(o.t, m.drop_vars('t2'))

        # try with mismatched lat and lon coordinate names
        with pytest.raises(KeyError):
            nmse(o.rename({'lat': 'latitude', 'lon': 'longitude'}), m)

        # try mismatched dataset vars
        with pytest.raises(ValueError):
            # raises clear error from xarray
            nmse(o.drop_vars('t'), m)

        # try mismatched dims
        with pytest.raises((AlignmentError, ValueError)):
            # raises clear error from xarray
            nmse(o.drop_isel({'lat': 0}), m.drop_isel({'lat': 1}))
        with pytest.raises(AlignmentError):
            # raises clear error from xarray
            nmse(o.drop_isel({'lon': 0}), m)

    def test_nmse_cf(self):
        nlat = 10
        nlon = 10
        nt = 2

        m = make_toy_temp_dataset(nlat=nlat, nlon=nlon, nt=nt, nans=True, cf=False)
        o = make_toy_temp_dataset(nlat=nlat, nlon=nlon, nt=nt, nans=True, cf=False)

        m_cf = m.copy()
        o_cf = o.copy()

        m_cf.lat.attrs["standard_name"] = "latitude"
        m_cf.lon.attrs["standard_name"] = "longitude"
        m_cf.time.attrs["standard_name"] = "time"
        o_cf.lat.attrs["standard_name"] = "latitude"
        o_cf.lon.attrs["standard_name"] = "longitude"
        o_cf.time.attrs["standard_name"] = "time"

        # test mixed cf
        xr.testing.assert_allclose(cupid_nmse(o_cf, m), nmse(o, m_cf))

        # test pre-cf coordinated
        xr.testing.assert_allclose(
            cupid_nmse(o_cf.cf.guess_coord_axis(), m),
            nmse(o.cf.guess_coord_axis(), m_cf),
        )

    def test_nmse_grids(self):
        nlat = 10
        nlon = 15
        nt = 2

        m = make_toy_temp_dataset(nlat=nlat, nlon=nlon, nt=nt, nans=True)
        o = make_toy_temp_dataset(nlat=nlat, nlon=nlon, nt=nt, nans=True)

        # make meshgrids
        x, y = np.meshgrid(m.lat, m.lon)
        m_t_mesh = xr.DataArray(
            data=m.t.values,
            dims=["time", "x", "y"],
            coords=dict(
                lat=(["y", "x"], x),
                lon=(["y", "x"], y),
                time=m.time,
            ),
        )
        o_t_mesh = xr.DataArray(
            data=o.t.values,
            dims=["time", "x", "y"],
            coords=dict(
                lat=(["y", "x"], x),
                lon=(["y", "x"], y),
                time=o.time,
            ),
        )

        # test evenly spaced mesh grid against 1d lat/lon
        xr.testing.assert_allclose(nmse(o.t, m.t), nmse(o_t_mesh, m_t_mesh))

        with pytest.raises(KeyError):
            nmse(pop, pop)

        # gauss lats (from https://www.ncl.ucar.edu/Document/Functions/Built-in/gaus_lobat_wgt.shtml)
        # fmt: off
        glat = [-90., -78.45661, -53.25302, -18.83693, 18.83693, 53.25302, 78.45661, 90.]
        # fmt: on

        mg = make_toy_temp_dataset(lat=glat)
        og = make_toy_temp_dataset(lat=glat)

        with pytest.raises(ValueError):
            nmse(mg, og)


class BaseEOFTestClass(metaclass=ABCMeta):
    _sample_data_eof = []

    # _sample_data[ 0 ]
    _sample_data_eof.append(
        [
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            [[16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]],
            [[32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47]],
            [[48, 49, 50, 51], [52, 53, 54, 55], [56, 57, 58, 59], [60, 61, 62, 63]],
        ]
    )

    # _sample_data[ 1 ]
    _sample_data_eof.append(np.arange(64, dtype='double').reshape((4, 4, 4)))

    # _sample_data[ 2 ]
    # fmt: off
    tmp_data = np.asarray(
        [0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99,
         18, -99, 20, 21, 22, -99, 24, 25, 26, 27, 28, -99, 30, -99, 32, 33, 34, 35,
         36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50, 51, 52, 53,
         54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
        dtype='double',
    ).reshape((4, 4, 4))
    # fmt: on
    _sample_data_eof.append(tmp_data)

    # _sample_data[ 3 ]
    # fmt: off
    tmp_data = np.asarray(
        [0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99,
         18, -99, 20, 21, 22, -99, 24, 25, 26, 27, 28, -99, 30, -99, 32, 33, 34, 35,
         36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50, 51, 52, 53,
         54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
        dtype='double',
    ).reshape((4, 4, 4))
    # fmt: onß
    tmp_data[tmp_data == -99] = np.nan
    _sample_data_eof.append(tmp_data)

    # _sample_data[ 4 ]
    _sample_data_eof.append(np.arange(64, dtype='int64').reshape((4, 4, 4)))

    _num_attrs = 4

    expected_output = np.full((1, 4, 4), 0.25)
    expected_eigen_val_time_dim_2 = 26.66666
    expected_eigen_val_time_dim_1 = 426.66666
    expected_eigen_val_time_dim_0 = 6826.66667


class Test_eof(BaseEOFTestClass):
    def test_eof_00(self) -> None:
        data = self._sample_data_eof[0]

        results = eofunc_eofs(data, neofs=1, time_dim=2)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(
            np.linalg.norm(self.expected_output), np.linalg.norm(eof), 5
        )

        np.testing.assert_array_almost_equal(self.expected_output, abs(eof), 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(
            self.expected_eigen_val_time_dim_2, attrs['eigenvalues'].values[0], 5
        )

    def test_eof_deprecated(self) -> None:
        data = self._sample_data_eof[0]

        results = eofunc(data, neval=1)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(
            np.linalg.norm(self.expected_output), np.linalg.norm(eof), 5
        )

        np.testing.assert_array_almost_equal(self.expected_output, abs(eof), 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(
            self.expected_eigen_val_time_dim_2, attrs['eigenvalues'].values[0], 5
        )

    def test_eof_01(self) -> None:
        data = self._sample_data_eof[1]

        results = eofunc_eofs(data, neofs=1, time_dim=2)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(
            np.linalg.norm(self.expected_output), np.linalg.norm(eof), 5
        )

        np.testing.assert_array_almost_equal(self.expected_output, abs(eof), 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(
            self.expected_eigen_val_time_dim_2, attrs['eigenvalues'].values[0], 5
        )

    def test_eof_02(self) -> None:
        data = self._sample_data_eof[1]

        results = eofunc_eofs(data, neofs=1, time_dim=2)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(
            np.linalg.norm(self.expected_output), np.linalg.norm(eof), 5
        )

        np.testing.assert_array_almost_equal(self.expected_output, abs(eof), 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(
            self.expected_eigen_val_time_dim_2, attrs['eigenvalues'].values[0], 5
        )

    def test_eof_14(self) -> None:
        data = self._sample_data_eof[4]

        results = eofunc_eofs(data, neofs=1, time_dim=2)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(
            np.linalg.norm(self.expected_output), np.linalg.norm(eof), 5
        )

        np.testing.assert_array_almost_equal(self.expected_output, abs(eof), 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(
            self.expected_eigen_val_time_dim_2, attrs['eigenvalues'].values[0], 5
        )

    def test_eof_15(self) -> None:
        data = np.asarray(self._sample_data_eof[0])
        data = np.transpose(data, axes=(2, 1, 0))

        dims = [f"dim_{i}" for i in range(data.ndim)]
        dims[0] = 'time'

        data = xr.DataArray(data, dims=dims, attrs={"prop1": "prop1", "prop2": 2})

        results = eofunc_eofs(data, neofs=1)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(
            np.linalg.norm(self.expected_output), np.linalg.norm(eof), 5
        )

        np.testing.assert_array_almost_equal(self.expected_output, abs(eof), 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(
            self.expected_eigen_val_time_dim_2, attrs['eigenvalues'].values[0], 5
        )

        np.testing.assert_equal(False, ("prop1" in attrs))
        np.testing.assert_equal(False, ("prop2" in attrs))

    # TODO: Maybe revisited to add time_dim support for Xarray in addition to numpy inputs
    # def test_eof_15_time_dim(self) -> None:
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
    #     np.testing.assert_array_almost_equal(np.linalg.norm(self.expected_output), np.linalg.norm(eof), 5)
    #
    #     np.testing.assert_array_almost_equal(self.expected_output, abs(eof), 5)
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

    def test_eof_16(self) -> None:
        data = np.asarray(self._sample_data_eof[0])
        data = np.transpose(data, axes=(2, 1, 0))

        dims = [f"dim_{i}" for i in range(data.ndim)]
        dims[0] = 'time'

        data = xr.DataArray(
            data,
            dims=dims,
            attrs={
                "prop1": "prop1",
                "prop2": 2,
            },
        )

        results = eofunc_eofs(data, 1, meta=True)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(
            np.linalg.norm(self.expected_output), np.linalg.norm(eof), 5
        )

        np.testing.assert_array_almost_equal(self.expected_output, abs(eof), 5)

        np.testing.assert_equal(self._num_attrs + 2, len(attrs))

        np.testing.assert_almost_equal(
            self.expected_eigen_val_time_dim_2, attrs['eigenvalues'].values[0], 5
        )

        np.testing.assert_equal(True, ("prop1" in attrs))
        np.testing.assert_equal(True, ("prop2" in attrs))
        np.testing.assert_equal("prop1", attrs["prop1"])
        np.testing.assert_equal(2, attrs["prop2"])

    def test_eof_n_01(self) -> None:
        data = self._sample_data_eof[1]

        results = eofunc_eofs(data, neofs=1, time_dim=1)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(
            np.linalg.norm(self.expected_output), np.linalg.norm(eof), 5
        )

        np.testing.assert_array_almost_equal(self.expected_output, abs(eof), 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(
            self.expected_eigen_val_time_dim_1, attrs['eigenvalues'].values[0], 5
        )

    def test_eof_n_03(self) -> None:
        data = self._sample_data_eof[1]

        results = eofunc_eofs(data, 1, time_dim=0)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(
            np.linalg.norm(self.expected_output), np.linalg.norm(eof), 5
        )

        np.testing.assert_array_almost_equal(self.expected_output, abs(eof), 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(
            self.expected_eigen_val_time_dim_0, attrs['eigenvalues'].values[0], 5
        )

    def test_eof_n_03_1(self) -> None:
        data = self._sample_data_eof[1]

        results = eofunc_eofs(data, 1, time_dim=0)
        eof = results.data
        attrs = results.attrs

        np.testing.assert_equal(self.expected_output.shape, results.shape)

        np.testing.assert_array_almost_equal(
            np.linalg.norm(self.expected_output), np.linalg.norm(eof), 5
        )

        np.testing.assert_array_almost_equal(self.expected_output, abs(eof), 5)

        np.testing.assert_equal(self._num_attrs, len(attrs))

        np.testing.assert_almost_equal(
            self.expected_eigen_val_time_dim_0, attrs['eigenvalues'].values[0], 5
        )


class Test_eof_ts(BaseEOFTestClass):
    @pytest.fixture(scope="class")
    def _nc_ds(self):
        try:
            return xr.open_dataset("eofunc_dataset.nc")
        except Exception:
            return xr.open_dataset("test/eofunc_dataset.nc")

    def test_01(self, _nc_ds) -> None:
        sst = _nc_ds.sst
        expected_tsout = _nc_ds.tsout

        actual_tsout = eofunc_pcs(sst, npcs=5)

        np.testing.assert_equal(actual_tsout.shape, expected_tsout.shape)

        np.testing.assert_array_almost_equal(actual_tsout, expected_tsout.data, 3)

    def test_01_deprecated(self, _nc_ds) -> None:
        sst = _nc_ds.sst
        evec = _nc_ds.evec
        expected_tsout = _nc_ds.tsout

        actual_tsout = eofunc_ts(sst, evec, time_dim=0)

        np.testing.assert_equal(actual_tsout.shape, expected_tsout.shape)

        np.testing.assert_array_almost_equal(actual_tsout, expected_tsout.data, 3)

    def test_02(self, _nc_ds) -> None:
        sst = _nc_ds.sst
        expected_tsout = _nc_ds.tsout

        actual_tsout = eofunc_pcs(sst, npcs=5, meta=True)

        np.testing.assert_equal(actual_tsout.shape, expected_tsout.shape)

        np.testing.assert_array_almost_equal(actual_tsout, expected_tsout.data, 3)

        np.testing.assert_equal(
            actual_tsout.coords["time"].data, sst.coords["time"].data
        )


class Test_pearson_r:
    # Coordinates
    times = xr.date_range(
        start='2022-08-01', end='2022-08-05', freq='D', use_cftime=True
    )
    lats = np.linspace(start=-45, stop=45, num=3, dtype='float32')
    lons = np.linspace(start=-180, stop=180, num=4, dtype='float32')

    # Create data variables
    x, y, z = np.meshgrid(lons, lats, times)
    np.random.seed(0)
    a = np.random.random_sample((len(lats), len(lons), len(times)))
    b = np.power(a, 2)
    weights = np.cos(np.deg2rad(y))
    ds = xr.Dataset(
        data_vars={
            'a': (('lat', 'lon', 'time'), a),
            'b': (('lat', 'lon', 'time'), b),
            'weights': (('lat', 'lon', 'time'), weights),
        },
        coords={'lat': lats, 'lon': lons, 'time': times},
        attrs={'description': 'Test data'},
    )

    unweighted_r = 0.963472086
    unweighted_r_skipnan = 0.96383798
    weighted_r = 0.963209755
    weighted_r_lat = [
        [0.995454445, 0.998450821, 0.99863877, 0.978765291, 0.982350092],
        [0.99999275, 0.995778831, 0.998994355, 0.991634937, 0.999868279],
        [0.991344899, 0.998632079, 0.99801552, 0.968517489, 0.985215828],
        [0.997034735, 0.99834464, 0.987382522, 0.99646236, 0.989222738],
    ]

    # Testing numpy inputs
    def test_np_inputs(self) -> None:
        a = self.a
        b = self.b
        result = pearson_r(a, b)
        assert np.allclose(self.unweighted_r, result)

    def test_np_inputs_weighted(self) -> None:
        a = self.a
        b = self.b
        w = self.weights
        result = pearson_r(a, b, weights=w)
        assert np.allclose(self.weighted_r, result)

    def test_np_inputs_warn(self) -> None:
        a = self.a
        b = self.b
        with pytest.warns(UserWarning):
            pearson_r(a, b, dim='lat', axis=0)

    def test_np_inputs_across_lats(self) -> None:
        a = self.a
        b = self.b
        w = self.weights
        result = pearson_r(a, b, weights=w, axis=0)
        assert np.allclose(self.weighted_r_lat, result)

    def test_np_inputs_skipna(self) -> None:
        # deep copy to prevent adding nans to the test data for other tests
        a = self.a.copy()
        a[0] = np.nan
        b = self.b
        result = pearson_r(a, b, skipna=True)
        assert np.allclose(self.unweighted_r_skipnan, result)

    # Testing xarray inputs
    def test_xr_inputs(self) -> None:
        a = self.ds.a
        b = self.ds.b
        result = pearson_r(a, b)
        assert np.allclose(self.unweighted_r, result)

    def test_xr_inputs_weighted(self) -> None:
        a = self.ds.a
        b = self.ds.b
        w = self.ds.weights
        result = pearson_r(a, b, weights=w)
        assert np.allclose(self.weighted_r, result)

    def test_xr_inputs_warn(self) -> None:
        a = self.ds.a
        b = self.ds.b
        with pytest.warns(UserWarning):
            pearson_r(a, b, dim='lat', axis=0)

    def test_xr_inputs_across_lats(self) -> None:
        a = self.ds.a
        b = self.ds.b
        w = self.ds.weights[:, 0, 0]
        result = pearson_r(a, b, weights=w, dim='lat')
        assert np.allclose(self.weighted_r_lat, result)

    def test_xr_inputs_skipna(self) -> None:
        # deep copy to prevent adding nans to the test data for other tests
        a = self.ds.a.copy(deep=True)
        a[0] = np.nan
        b = self.ds.b
        result = pearson_r(a, b, skipna=True)
        assert np.allclose(self.unweighted_r_skipnan, result)

    def test_keep_attrs(self) -> None:
        a = self.ds.a
        b = self.ds.b
        a.attrs.update({'Description': 'Test Data'})
        b.attrs.update({'2nd Description': 'Dummy Data'})
        result = pearson_r(a, b, keep_attrs=True)
        assert result.attrs == a.attrs
