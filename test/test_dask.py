import pytest

import dask.array
from dask.distributed import Client
import xarray as xr
import numpy as np
import geocat.datafiles as gdf
from math import tau
from packaging.version import Version
from scipy import __version__ as scipy_version


from .util import (
    _get_toy_climatology_data,
    get_fake_climatology_dataset,
    make_toy_temp_dataset,
)

# import everything for dask compatibility and performance tests
from geocat.comp.meteorology import (
    delta_pressure,
    dewtemp,
    heat_index,
    relhum,
    relhum_ice,
    relhum_water,
    actual_saturation_vapor_pressure,
    saturation_vapor_pressure,
    saturation_vapor_pressure_slope,
    max_daylight,
    psychrometric_constant,
)

from geocat.comp.gradient import (
    gradient,
)

from geocat.comp.interpolation import (
    interp_hybrid_to_pressure,
    interp_sigma_to_hybrid,
    interp_multidim,
    pressure_at_hybrid_levels,
    delta_pressure_hybrid,
)

from geocat.comp.climatologies import (
    climate_anomaly,
    month_to_season,
    calendar_average,
    climatology_average,
)

from geocat.comp.fourier_filters import (
    fourier_band_block,
    fourier_high_pass,
    fourier_low_pass,
    fourier_band_pass,
)

from geocat.comp.spherical import decomposition, recomposition, scale_voronoi

from geocat.comp.stats import eofunc_eofs, eofunc_pcs, pearson_r, nmse


@pytest.fixture(scope="module")
def client():
    # dask client reference for all subsequent tests
    client = Client()
    yield client
    client.close()


class TestDaskCompat_meteorology:
    @pytest.mark.xfail(
        reason="delta_pressure not compatible with dask, downcasts to np"
    )
    def test_delta_pressure_dask(self):
        pressure_lev = xr.DataArray(np.array([1, 5, 100, 1000])).chunk()
        surface_pressure_1D = xr.DataArray((np.array([1018, 1019]))).chunk()

        out = delta_pressure(pressure_lev, surface_pressure_1D)

        assert isinstance(out.data, dask.array.Array)

    def test_dewtemp_dask(self):
        # fmt: off
        t_def = [29.3, 28.1, 23.5, 20.9, 18.4, 15.9, 13.1, 10.1, 6.7, 3.1,
                 -0.5, -4.5, -9.0, -14.8, -21.5, -29.7, -40.0, -52.4]

        rh_def = [75.0, 60.0, 61.1, 76.7, 90.5, 89.8, 78.3, 76.5, 46.0,
                  55.0, 63.8, 53.2, 42.9, 41.7, 51.0, 70.6, 50.0, 50.0]
        # fmt: on

        tk = xr.DataArray(np.asarray(t_def) + 273.15).chunk(6)
        rh = xr.DataArray(rh_def).chunk(6)

        out = dewtemp(tk, rh)
        assert isinstance((out - 273.15).data, dask.array.Array)

    def test_heat_index_dask(self):
        # fmt: off
        t1 = np.array([75, 80, 85, 90, 95, 100, 105, 110, 115])
        rh1 = np.array([75, 15, 80, 65, 25, 30, 40, 50, 5])
        # fmt: on

        t = xr.DataArray(t1).chunk(3)
        rh = xr.DataArray(rh1).chunk(3)

        out = heat_index(t, rh, alternate_coeffs=True)
        assert isinstance(out.data, dask.array.Array)

    def test_relhum_dask(self):
        # fmt: off
        p_def = [100800, 100000, 95000, 90000, 85000, 80000, 75000, 70000,
                 65000, 60000, 55000, 50000, 45000, 40000, 35000, 30000,
                 25000, 20000, 17500, 15000, 12500, 10000, 8000, 7000, 6000,
                 5000, 4000, 3000, 2500, 2000]

        t_def = [ 302.45, 301.25, 296.65, 294.05, 291.55, 289.05, 286.25,
                  283.25, 279.85, 276.25, 272.65, 268.65, 264.15, 258.35,
                  251.65, 243.45, 233.15, 220.75, 213.95, 206.65, 199.05,
                  194.65, 197.15, 201.55, 206.45, 211.85, 216.85, 221.45,
                  222.45, 225.65]

        q_def = [ 0.02038, 0.01903, 0.01614, 0.01371, 0.01156, 0.0098, 0.00833,
                  0.00675, 0.00606, 0.00507, 0.00388, 0.00329, 0.00239, 0.0017,
                  0.001, 0.0006, 0.0002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # fmt: on

        p = xr.DataArray(p_def).chunk(10)
        t = xr.DataArray(t_def).chunk(10)
        q = xr.DataArray(q_def).chunk(10)

        out = relhum(t, q, p)
        assert isinstance(out.data, dask.array.Array)

    def test_relhum_ice_dask(self):
        tc = -5.0
        tk = tc + 273.15
        w = 3.7 / 1000.0
        p = 1000.0 * 100.0

        tk = xr.DataArray(np.full((9, 9), tk)).chunk((1, 9))
        w = xr.DataArray(np.full((9, 9), w)).chunk((1, 9))
        p = xr.DataArray(np.full((9, 9), p)).chunk((1, 9))

        out = relhum_ice(tk, w, p)

        assert isinstance(out.data, dask.array.Array)

    def test_relhum_water_dask(self):
        p = 1000.0 * 100
        t = 18.0 + 273.15
        q = 6.0 / 1000.0

        p = xr.DataArray(np.full((9, 9), p)).chunk((1, 9))
        t = xr.DataArray(np.full((9, 9), t)).chunk((1, 9))
        q = xr.DataArray(np.full((9, 9), q)).chunk((1, 9))

        out = relhum_water(t, q, p)

        assert isinstance(out.data, dask.array.Array)

    def test_actual_saturation_vapor_pressure_dask(self):
        tempf = xr.DataArray(np.arange(1, 101, 1)).chunk(10)

        out = actual_saturation_vapor_pressure(tempf, tfill=1.0000000e20)

        assert isinstance(out.data, dask.array.Array)

    def test_saturation_vapor_pressure_dask(self):
        tempf = xr.DataArray(np.arange(1, 101, 1)).chunk(10)

        out = saturation_vapor_pressure(tempf, tfill=1.0000000e20)

        assert isinstance(out.data, dask.array.Array)

    def test_saturation_vapor_pressure_slope_dask(self):
        tempf = xr.DataArray(np.arange(1, 101, 1)).chunk(10)

        out = saturation_vapor_pressure_slope(tempf)

        assert isinstance(out.data, dask.array.Array)

    @pytest.mark.xfail(reason="max_daylight not compatible with dask")
    def test_max_daylight_dask(self):
        jday_gt = xr.DataArray(np.linspace(1, 365, num=365)).chunk("auto")
        lat_gt = xr.DataArray(np.linspace(-66, 66, num=133)).chunk("auto")

        out = max_daylight(jday_gt, lat_gt)

        assert isinstance(out.data, dask.array.Array)

    def test_psychrometric_constant_dask(self):
        pressure_gt = xr.DataArray(np.arange(1, 101, 1)).chunk(10)

        try:
            # Generated by running ncl_tests/test_psychro_fao56.ncl
            ncl_gt = xr.open_dataarray("psychro_fao56_output.nc").values
        except FileNotFoundError:
            ncl_gt = xr.open_dataarray("test/psychro_fao56_output.nc").values

        out = psychrometric_constant(pressure_gt)

        assert isinstance(out.data, dask.array.Array)
        assert np.allclose(out, ncl_gt, atol=0.005)


class TestDaskCompat_gradient:
    @pytest.mark.xfail(reason="gradient not compatible with dask")
    def test_gradient_dask(self):
        test_data = (
            xr.load_dataset('test/gradient_test_data.nc').to_array().squeeze().chunk(10)
        )

        expected_results = [
            xr.load_dataset('test/gradient_test_results_longitude.nc')
            .to_array()
            .squeeze(),
            xr.load_dataset('test/gradient_test_results_latitude.nc')
            .to_array()
            .squeeze(),
        ]

        out = gradient(test_data)
        assert isinstance(out[0].data, dask.array.Array)
        np.testing.assert_almost_equal(
            np.array(out), np.array(expected_results), decimal=3
        )


class TestDaskCompat_interpolation:
    @pytest.fixture(scope="class")
    def ds_atmos(self):
        # Open the netCDF data file "atmos.nc" and read in common variables
        try:
            return xr.open_dataset(gdf.get("netcdf_files/atmos.nc"), decode_times=False)
        except FileNotFoundError:
            return xr.open_dataset("test/atmos.nc", decode_times=False)

    def test_interp_hybrid_to_pressure_dask(self, ds_atmos, vinth2p_output):
        pres3d = np.asarray([1000, 950, 800, 700, 600, 500, 400, 300, 200]) * 100

        data = ds_atmos.U[0, :, :, :].chunk({'lev': 5})
        ps = ds_atmos.PS[0, :, :]

        out = interp_hybrid_to_pressure(
            data,
            ps,
            ds_atmos.hyam,
            ds_atmos.hybm,
            p0=1000 * 100,
            new_levels=pres3d,
            method='log',
        )

        assert isinstance(out.data, dask.array.Array)
        assert np.allclose(
            out.mean(dim='lon'), vinth2p_output.uzon, atol=5e-6, equal_nan=True
        )

    def test_interp_hybrid_to_pressure_dask_lat_chunk(self, ds_atmos, vinth2p_output):
        pres3d = np.asarray([1000, 950, 800, 700, 600, 500, 400, 300, 200]) * 100

        data = ds_atmos.U[0, :, :, :].chunk({'lat': 32, 'lon': 32})
        ps = ds_atmos.PS[0, :, :].chunk({'lat': 32, 'lon': 32})

        out = interp_hybrid_to_pressure(
            data,
            ps,
            ds_atmos.hyam,
            ds_atmos.hybm,
            p0=1000 * 100,
            new_levels=pres3d,
            method='log',
        )

        assert isinstance(out.data, dask.array.Array)
        assert np.allclose(
            out.mean(dim='lon'), vinth2p_output.uzon, atol=5e-6, equal_nan=True
        )

    def test_interp_hybrid_to_pressure_dask_chunk_length_1(self, ds_atmos):
        pres3d = np.asarray([1000, 950, 800, 700, 600, 500, 400, 300, 200]) * 100

        data = ds_atmos.U[0, :, :, :].chunk({'lev': 1})
        ps = ds_atmos.PS[0, :, :]

        pytest.warns(
            UserWarning,
            interp_hybrid_to_pressure,
            data,
            ps,
            ds_atmos.hyam,
            ds_atmos.hybm,
            p0=1000 * 100,
            new_levels=pres3d,
            method='log',
        )

    def test_interp_sigma_to_hybrid_dask(self):
        try:
            ds_ps = xr.open_dataset(
                gdf.get("netcdf_files/ps.89335.1.nc"), decode_times=False
            )
        except FileNotFoundError:
            ds_ps = xr.open_dataset("test/ps.89335.1.nc", decode_times=False)

        try:
            ds_u = xr.open_dataset(
                gdf.get("netcdf_files/u.89335.1_subset_time361.nc"), decode_times=False
            )
        except FileNotFoundError:
            ds_u = xr.open_dataset(
                "test/u.89335.1_subset_time361.nc", decode_times=False
            )

        hyam = xr.DataArray([0.0108093, 0.0130731, 0.03255911, 0.0639471])
        hybm = xr.DataArray([0.0108093, 0.0173664, 0.06069280, 0.1158237])

        ps = ds_ps.ps[361, 0:3, 0:2] * 100
        ps = ps.chunk()
        u = ds_u.u[:, 0:3, 0:2].chunk()
        sigma = ds_ps.sigma

        out = interp_sigma_to_hybrid(
            u, sigma, ps, hyam, hybm, 1000 * 100, method='linear'
        )

        assert isinstance(out.data, dask.array.Array)

    def test_interp_multidim(self):
        nlat = 10
        nlon = 10
        out_lat = np.linspace(-90, 90, nlat - 2)
        out_lon = np.linspace(-180, 180, nlon - 2)
        ds = make_toy_temp_dataset(nlat=nlat, nlon=nlon)

        out = interp_multidim(ds.t.chunk(), lat_out=out_lat, lon_out=out_lon)
        assert isinstance(out.data, dask.array.Array)

    def test_pressure_at_hybrid_levels_dask(self, ds_atmos, p_out):
        ps = ds_atmos.PS.chunk()

        out = pressure_at_hybrid_levels(ps, p_out.hyam, p_out.hybm)
        assert isinstance(out.data, dask.array.Array)

    def test_delta_pressure_hybrid_dask(self, ds_atmos):
        ps = ds_atmos.PS[0, :, :].drop('time').chunk()
        p0 = 1000.0 * 100
        out = delta_pressure_hybrid(ps, ds_atmos.hyam, ds_atmos.hybm, p0)

        assert isinstance(out.data, dask.array.Array)


class TestDaskCompat_climatology:
    def test_climate_anomaly_dask(self):
        ds = _get_toy_climatology_data('2020-01-01', '2021-12-31', 'D', 4, 1)
        daily = ds.data.chunk({'lat': 1})

        out = climate_anomaly(daily, 'season')

        assert isinstance(out.data, dask.array.Array)

    def test_month_to_season_dask(self):
        ds = get_fake_climatology_dataset(
            start_month="2000-01", nmonths=12, nlats=1, nlons=1
        )
        da = ds.my_var.chunk({'time': 4})
        out = month_to_season(da, 'JFM')
        assert isinstance(out.data, dask.array.Array)

    def test_calendar_average_dask(self):
        ds = _get_toy_climatology_data('2020-01-01', '2021-12-01', 'MS', 1, 1)
        monthly = ds.data.chunk({'time': 6})
        out = calendar_average(monthly, 'season')
        assert isinstance(out.data, dask.array.Array)

    def test_climatology_average_dask(self):
        ds = _get_toy_climatology_data('2020-01-01', '2021-12-01', 'MS', 1, 1)
        monthly = ds.data.chunk({'time': 6})
        out = climatology_average(monthly, 'season')
        assert isinstance(out.data, dask.array.Array)


class TestDaskCompat_fourier:
    freq = 1000
    t = np.arange(1000) / freq
    t_data_1 = (
        np.sin(t * tau) / 0.1
        + np.sin(2 * t * tau) / 0.2
        + np.sin(5 * t * tau) / 0.5
        + np.sin(10 * t * tau)
        + np.sin(20 * t * tau) / 2
        + np.sin(50 * t * tau) / 5
        + np.sin(100 * t * tau) / 10
    )

    t_data_2 = (
        np.sin(t * tau) / 0.1
        + np.sin(2 * t * tau) / 0.2
        + np.sin(10 * t * tau)
        + np.sin(20 * t * tau) / 2
        + np.sin(100 * t * tau) / 10
    )

    da = xr.DataArray(
        data=[t_data_1, t_data_2],
        dims=['signal', "frequency"],
        coords={"frequency": t},
    )

    @pytest.mark.xfail(reason="fourier_high_pass not compatible with dask")
    def test_fourier_high_pass_dask(self):
        signal = self.da.chunk({'signal': 1})
        out = fourier_high_pass(signal, self.freq, 15)

        assert isinstance(out.data, dask.array.Array)

    @pytest.mark.xfail(reason="fourier_low_pass not compatible with dask")
    def test_fourier_low_pass_dask(self):
        signal = self.da.chunk({'signal': 1})
        out = fourier_low_pass(signal, self.freq, 15)

        assert isinstance(out.data, dask.array.Array)

    @pytest.mark.xfail(reason="fourier_band_pass not compatible with dask")
    def test_fourier_band_pass_dask(self):
        signal = self.da.chunk({'signal': 1})
        out = fourier_band_pass(signal, self.freq, 3, 15)

        assert isinstance(out.data, dask.array.Array)

    @pytest.mark.xfail(reason="fourier_band_block not compatible with dask")
    def test_fourier_band_block_dask(self):
        signal = self.da.chunk({'signal': 1})
        out = fourier_band_block(signal, self.freq, 3, 30)

        assert isinstance(out.data, dask.array.Array)


class TestDaskCompat_spherical:
    def test_decomposition_dask(self, spherical_data):
        out = decomposition(
            spherical_data['test_data_xr'].chunk(),
            spherical_data['test_scale_xr'].chunk(),
            spherical_data['theta_xr'].chunk(),
            spherical_data['phi_xr'].chunk(),
        )

        assert isinstance(out.data, dask.array.Array)

    @pytest.mark.xfail(
        Version(scipy_version) < Version('1.15.0'),
        reason="old scipy version not compatible with dask",
    )
    def test_recomputation_dask(self, spherical_data):
        out = recomposition(
            spherical_data['test_results_xr'].chunk(),
            spherical_data['theta_xr'].chunk(),
            spherical_data['phi_xr'].chunk(),
        )

        assert isinstance(out.data, dask.array.Array)

    @pytest.mark.xfail(reason="scale_voroni not compatible with dask")
    def test_scale_voroni_dask(self, spherical_data):
        out = scale_voronoi(
            spherical_data['theta_xr'].chunk(),
            spherical_data['phi_xr'].chunk(),
        )
        # out.data is a memory view?
        assert isinstance(out.data, dask.array.Array)


class TestDaskCompat_stats:
    @pytest.mark.xfail(reason="eofunc_eofs not compatible with dask")
    def test_eofunc_eofs_dask(self):
        data = xr.DataArray(
            np.arange(64, dtype='double').reshape((4, 4, 4)), dims=["time", "x1", "x2"]
        ).chunk()
        out = eofunc_eofs(data, neofs=1, time_dim=2)

        assert isinstance(out.data, dask.array.Array)

    @pytest.mark.xfail(reason="eofunc_eofs not compatible with dask")
    def test_eofunc_pcs_dask(self):
        data = xr.DataArray(
            np.arange(64, dtype='double').reshape((4, 4, 4)), dims=["time", "x1", "x2"]
        ).chunk()
        out = eofunc_pcs(data, npcs=5)

        assert isinstance(out.data, dask.array.Array)

    def test_pearson_r_dask(self):
        times = xr.date_range(
            start='2022-08-01', end='2022-08-03', freq='D', use_cftime=True
        )
        lats = np.linspace(start=-45, stop=45, num=3, dtype='float32')
        lons = np.linspace(start=-180, stop=180, num=2, dtype='float32')
        x, y, z = np.meshgrid(lons, lats, times)
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
        axr = ds.a.chunk()
        bxr = ds.b.chunk()
        wxr = ds.weights.chunk()
        out = pearson_r(axr, bxr, weights=wxr)

        assert isinstance(out.data, dask.array.Array)

    def test_nmse_dask(self):
        nlat = 10
        nlon = 10
        nt = 2

        m = make_toy_temp_dataset(nlat=nlat, nlon=nlon, nt=nt, nans=True).t.chunk()
        o = make_toy_temp_dataset(nlat=nlat, nlon=nlon, nt=nt, nans=True).t.chunk()

        out = nmse(o, m)
        assert isinstance(out.data, dask.array.Array)
