import pytest

import dask.array
from dask.distributed import Client
import xarray as xr
import numpy as np
import geocat.datafiles as gdf

# import everything for dask compatibility and performance tests
from geocat.comp import (
    dewtemp,
    heat_index,
    relhum,
    actual_saturation_vapor_pressure,
    saturation_vapor_pressure,
    saturation_vapor_pressure_slope,
    max_daylight,
    psychrometric_constant,
    gradient,
    interp_hybrid_to_pressure,
    interp_sigma_to_hybrid,
)


@pytest.fixture(scope="module")
def client() -> None:
    # dask client reference for all subsequent tests
    client = Client()
    yield client
    client.close()


class TestDaskCompat:
    def test_dewtemp_dask(self):
        t_def = [
            29.3,
            28.1,
            23.5,
            20.9,
            18.4,
            15.9,
            13.1,
            10.1,
            6.7,
            3.1,
            -0.5,
            -4.5,
            -9.0,
            -14.8,
            -21.5,
            -29.7,
            -40.0,
            -52.4,
        ]

        rh_def = [
            75.0,
            60.0,
            61.1,
            76.7,
            90.5,
            89.8,
            78.3,
            76.5,
            46.0,
            55.0,
            63.8,
            53.2,
            42.9,
            41.7,
            51.0,
            70.6,
            50.0,
            50.0,
        ]

        dt_2 = [
            24.38342,
            19.55563,
            15.53281,
            16.64218,
            16.81433,
            14.22482,
            9.401337,
            6.149719,
            -4.1604,
            -5.096619,
            -6.528168,
            -12.61957,
            -19.38332,
            -25.00714,
            -28.9841,
            -33.34853,
            -46.51273,
            -58.18289,
        ]
        tk = xr.DataArray(np.asarray(t_def) + 273.15).chunk(6)
        rh = xr.DataArray(rh_def).chunk(6)

        out = dewtemp(tk, rh)
        assert isinstance((out - 273.15).data, dask.array.Array)
        assert np.allclose(out - 273.15, dt_2, atol=0.1)

    def test_heat_index_dask(self):
        ncl_gt_1 = [
            137.36142,
            135.86795,
            104.684456,
            131.25621,
            105.39449,
            79.78999,
            83.57511,
            59.965,
            30.0,
        ]

        t1 = np.array([104, 100, 92, 92, 86, 80, 80, 60, 30])
        rh1 = np.array([55, 65, 60, 90, 90, 40, 75, 90, 50])

        t = xr.DataArray(t1).chunk(3)
        rh = xr.DataArray(rh1).chunk(3)

        out = heat_index(t, rh)
        assert isinstance(out.data, dask.array.Array)
        assert np.allclose(out, ncl_gt_1, atol=0.005)

    def test_relhum_dask(self):
        p_def = [
            100800,
            100000,
            95000,
            90000,
            85000,
            80000,
            75000,
            70000,
            65000,
            60000,
            55000,
            50000,
            45000,
            40000,
            35000,
            30000,
            25000,
            20000,
            17500,
            15000,
            12500,
            10000,
            8000,
            7000,
            6000,
            5000,
            4000,
            3000,
            2500,
            2000,
        ]

        t_def = [
            302.45,
            301.25,
            296.65,
            294.05,
            291.55,
            289.05,
            286.25,
            283.25,
            279.85,
            276.25,
            272.65,
            268.65,
            264.15,
            258.35,
            251.65,
            243.45,
            233.15,
            220.75,
            213.95,
            206.65,
            199.05,
            194.65,
            197.15,
            201.55,
            206.45,
            211.85,
            216.85,
            221.45,
            222.45,
            225.65,
        ]

        q_def = [
            0.02038,
            0.01903,
            0.01614,
            0.01371,
            0.01156,
            0.0098,
            0.00833,
            0.00675,
            0.00606,
            0.00507,
            0.00388,
            0.00329,
            0.00239,
            0.0017,
            0.001,
            0.0006,
            0.0002,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        rh_gt_2 = [
            79.8228,
            79.3578,
            84.1962,
            79.4898,
            73.989,
            69.2401,
            66.1896,
            61.1084,
            64.21,
            63.8305,
            58.0412,
            60.8194,
            57.927,
            62.3734,
            62.9706,
            73.8184,
            62.71,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        p = xr.DataArray(p_def).chunk(10)
        t = xr.DataArray(t_def).chunk(10)
        q = xr.DataArray(q_def).chunk(10)

        out = relhum(t, q, p)
        assert isinstance(out.data, dask.array.Array)
        assert np.allclose(relhum(t, q, p), rh_gt_2, atol=0.1)

    def test_actual_saturation_vapor_pressure_dask(self):
        tempf = xr.DataArray(np.arange(1, 101, 1)).chunk(10)

        try:
            ncl_gt = xr.open_dataarray("satvpr_tdew_fao56_output.nc").values
        except FileNotFoundError:
            ncl_gt = xr.open_dataarray("test/satvpr_tdew_fao56_output.nc").values

        out = actual_saturation_vapor_pressure(tempf, tfill=1.0000000e20)

        assert isinstance(out.data, dask.array.Array)
        assert np.allclose(out, ncl_gt, atol=0.005)

    def test_saturation_vapor_pressure_dask(self):
        tempf = xr.DataArray(np.arange(1, 101, 1)).chunk(10)

        try:
            # Generated by running ncl_tests/test_satvpr_temp_fao56.ncl
            ncl_gt = xr.open_dataarray("satvpr_temp_fao56_output.nc").values
        except FileNotFoundError:
            ncl_gt = xr.open_dataarray("test/satvpr_temp_fao56_output.nc").values

        out = saturation_vapor_pressure(tempf, tfill=1.0000000e20)

        assert isinstance(out.data, dask.array.Array)
        assert np.allclose(out, ncl_gt, atol=0.005)

    def test_saturation_vapor_pressure_slope_dask(self):
        tempf = xr.DataArray(np.arange(1, 101, 1)).chunk(10)

        try:
            ncl_gt = xr.open_dataarray("satvpr_slope_fao56_output.nc").values
        except FileNotFoundError:
            ncl_gt = xr.open_dataarray("test/satvpr_slope_fao56_output.nc").values

        out = saturation_vapor_pressure_slope(tempf)

        assert isinstance(out.data, dask.array.Array)
        assert np.allclose(out, ncl_gt, atol=0.005, equal_nan=True)

    @pytest.mark.xfail(reason="max_daylight not compatible with dask")
    def test_max_daylight_dask(self):
        # set up ground truths
        jday_gt = xr.DataArray(np.linspace(1, 365, num=365)).chunk("auto")
        lat_gt = xr.DataArray(np.linspace(-66, 66, num=133)).chunk("auto")

        try:
            # Generated by running ncl_tests/test_max_daylight.ncl
            ncl_gt = xr.open_dataarray("max_daylight_test.nc").values
        except FileNotFoundError:
            ncl_gt = xr.open_dataarray("test/max_daylight_test.nc").values

        out = max_daylight(jday_gt, lat_gt)

        assert isinstance(out.data, dask.array.Array)
        assert np.allclose(out, ncl_gt, atol=0.005)

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

    def test_interp_hybrid_to_pressure_dask(self):
        pres3d = np.asarray([1000, 950, 800, 700, 600, 500, 400, 300, 200]) * 100

        # Open the netCDF data file "atmos.nc" and read in common variables
        try:
            ds_atmos = xr.open_dataset(
                gdf.get("netcdf_files/atmos.nc"), decode_times=False
            )
        except FileNotFoundError:
            ds_atmos = xr.open_dataset("test/atmos.nc", decode_times=False)

        try:
            # Generated by running ncl_tests/vinth2p_test_conwomap_5.ncl on
            # atmos.nc
            ds_out = xr.open_dataset("vinth2p_output.nc")
        except FileNotFoundError:
            ds_out = xr.open_dataset("test/vinth2p_output.nc")

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
        assert np.allclose(out.mean(dim='lon'), ds_out.uzon, atol=5e-6, equal_nan=True)

    def test_interp_hybrid_to_pressure_dask_lat_chunk(self):
        pres3d = np.asarray([1000, 950, 800, 700, 600, 500, 400, 300, 200]) * 100

        # Open the netCDF data file "atmos.nc" and read in common variables
        try:
            ds_atmos = xr.open_dataset(
                gdf.get("netcdf_files/atmos.nc"), decode_times=False
            )
        except FileNotFoundError:
            ds_atmos = xr.open_dataset("test/atmos.nc", decode_times=False)

        try:
            # Generated by running ncl_tests/vinth2p_test_conwomap_5.ncl on
            # atmos.nc
            ds_out = xr.open_dataset("vinth2p_output.nc")
        except FileNotFoundError:
            ds_out = xr.open_dataset("test/vinth2p_output.nc")

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
        assert np.allclose(out.mean(dim='lon'), ds_out.uzon, atol=5e-6, equal_nan=True)

    def test_interp_hybrid_to_pressure_dask_chunk_length_1(self):
        pres3d = np.asarray([1000, 950, 800, 700, 600, 500, 400, 300, 200]) * 100

        # Open the netCDF data file "atmos.nc" and read in common variables
        try:
            ds_atmos = xr.open_dataset(
                gdf.get("netcdf_files/atmos.nc"), decode_times=False
            )
        except FileNotFoundError:
            ds_atmos = xr.open_dataset("test/atmos.nc", decode_times=False)

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

        try:
            ds_out = xr.open_dataset(
                "sigma2hybrid_output.nc"
            )  # Generated by running ncl_tests/test_sigma2hybrid.ncl
        except FileNotFoundError:
            ds_out = xr.open_dataset("test/sigma2hybrid_output.nc")

        hyam = xr.DataArray([0.0108093, 0.0130731, 0.03255911, 0.0639471])
        hybm = xr.DataArray([0.0108093, 0.0173664, 0.06069280, 0.1158237])

        ps = ds_ps.ps[361, 0:3, 0:2] * 100
        ps = ps.chunk()
        u = ds_u.u[:, 0:3, 0:2].chunk()
        sigma = ds_ps.sigma
        xh_expected = ds_out.xh.transpose('ncl3', 'ncl1', 'ncl2')

        out = interp_sigma_to_hybrid(
            u, sigma, ps, hyam, hybm, 1000 * 100, method='linear'
        )

        assert isinstance(out.data, dask.array.Array)
        assert np.allclose(xh_expected, out, atol=5e-6, equal_nan=True)
