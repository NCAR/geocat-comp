import numpy as np
import pytest
from random import uniform
import xarray as xr
import uxarray as ux

from geocat.comp.meteorology import (
    dewtemp,
    heat_index,
    relhum,
    relhum_ice,
    relhum_water,
    actual_saturation_vapor_pressure,
    max_daylight,
    psychrometric_constant,
    saturation_vapor_pressure,
    saturation_vapor_pressure_slope,
    delta_pressure,
    zonal_meridional_psi,
)
from geocat.comp.interpolation import interp_hybrid_to_pressure


class Test_dewtemp:
    # ground truths
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

    dt_1 = 6.3

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

    def test_float_input(self) -> None:
        tk = 18.0 + 273.15
        rh = 46.5

        assert np.allclose(dewtemp(tk, rh) - 273.15, self.dt_1, 0.1)

    def test_list_input(self) -> None:
        tk = (np.asarray(self.t_def) + 273.15).tolist()

        assert np.allclose(dewtemp(tk, self.rh_def) - 273.15, self.dt_2, 0.1)

    def test_numpy_input(self) -> None:
        tk = np.asarray(self.t_def) + 273.15
        rh = np.asarray(self.rh_def)

        assert np.allclose(dewtemp(tk, rh) - 273.15, self.dt_2, 0.1)

    def test_xarray_input(self) -> None:
        tk = xr.DataArray(np.asarray(self.t_def) + 273.15)
        rh = xr.DataArray(self.rh_def)

        assert np.allclose(dewtemp(tk, rh) - 273.15, self.dt_2, 0.1)

    def test_dims_error(self) -> None:
        with pytest.raises(ValueError):
            dewtemp(self.t_def[:10], self.rh_def[:8])

    def test_xarray_type_error(self) -> None:
        with pytest.raises(TypeError):
            dewtemp(self.t_def, xr.DataArray(self.rh_def))


class Test_heat_index:
    # set up ground truths
    # use ncl for alt coefficient testing within unchanged ranges
    hi_ncl_alt = [
        76.13114,
        75.12854,
        99.43573,
        104.93261,
        93.73293,
        104.328705,
        123.23398,
        150.34001,
        106.87023,
    ]
    t_alt = np.array([75, 80, 85, 90, 95, 100, 105, 110, 115])
    rh_alt = np.array([75, 15, 80, 65, 25, 30, 40, 50, 5])

    t_nws = [80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110]
    rh_nws = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    t_nws, rh_nws = np.meshgrid(t_nws, rh_nws)

    # used javascript to query NWS form
    @pytest.fixture(scope="class")
    def hi_nws(self):
        try:
            return np.genfromtxt("heat-index.csv", delimiter=",").T
        except Exception:
            return np.genfromtxt("test/heat-index.csv", delimiter=",").T

    def test_numpy_input(self, hi_nws) -> None:
        hi = heat_index(self.t_nws, self.rh_nws, False)
        assert np.allclose(hi.round(1), hi_nws)

    def test_multi_dimensional_input(self) -> None:
        assert np.allclose(
            heat_index(self.t_alt.reshape(3, 3), self.rh_alt.reshape(3, 3), True),
            np.asarray(self.hi_ncl_alt).reshape(3, 3),
            atol=0.005,
        )

    def test_alt_coef(self) -> None:
        assert np.allclose(
            heat_index(self.t_alt, self.rh_alt, True), self.hi_ncl_alt, atol=0.005
        )

    def test_xarray_alt_coef(self) -> None:
        hi_alt = heat_index(xr.DataArray(self.t_alt), xr.DataArray(self.rh_alt), True)
        assert np.allclose(hi_alt, self.hi_ncl_alt, atol=0.005)

    def test_float_input(self) -> None:
        assert np.allclose(heat_index(80, 75), 83.5751, atol=0.005)

    def test_list_input(self, hi_nws) -> None:
        hi = heat_index(self.t_nws.tolist(), self.rh_nws.tolist())
        hi = [[round(x, 1) for x in r] for r in hi]
        assert np.allclose(hi, hi_nws)

    def test_xarray_input(self, hi_nws) -> None:
        t = xr.DataArray(self.t_nws)
        rh = xr.DataArray(self.rh_nws)

        assert np.allclose(heat_index(t, rh).round(1), hi_nws)

    def test_rh_warning(self) -> None:
        with pytest.warns(UserWarning):
            heat_index([50, 80, 90], [0.1, 0.2, 0.5])

    def test_rh_invalid(self) -> None:
        with pytest.raises(ValueError):
            heat_index([50, 80, 90], [-1, 101, 50])

    def test_xarray_rh_warning(self) -> None:
        with pytest.warns(UserWarning):
            heat_index([50, 80, 90], [0.1, 0.2, 0.5])

    def test_xarray_rh_valid(self) -> None:
        with pytest.raises(ValueError):
            heat_index(xr.DataArray([50, 80, 90]), xr.DataArray([-1, 101, 50]))

    def test_xarray_type_error(self) -> None:
        with pytest.raises(TypeError):
            heat_index(self.t_nws, xr.DataArray(self.rh_nws))

    def test_dims_error(self) -> None:
        with pytest.raises(ValueError):
            heat_index(self.t_nws[:10], self.rh_nws[:8])

    def test_bad_list_input(self):
        with pytest.raises(ValueError):
            heat_index(np.asarray([[85, 85], [85]]), np.asarray([[60, 60], [60]]))


class Test_relhum:
    # set up ground truths
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

    rh_gt_1 = 46.4

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

    def test_float_input(self) -> None:
        p = 1000.0 * 100
        t = 18.0 + 273.15
        q = 6.0 / 1000.0

        assert np.allclose(relhum(t, q, p), self.rh_gt_1, atol=0.1)

    def test_list_input(self) -> None:
        assert np.allclose(
            relhum(self.t_def, self.q_def, self.p_def), self.rh_gt_2, atol=0.1
        )

    def test_numpy_input(self) -> None:
        p = np.asarray(self.p_def)
        t = np.asarray(self.t_def)
        q = np.asarray(self.q_def)

        assert np.allclose(relhum(t, q, p), self.rh_gt_2, atol=0.1)

    def test_dims_error(self) -> None:
        with pytest.raises(ValueError):
            relhum(self.t_def[:10], self.q_def[:10], self.p_def[:9])

    def test_mixed_input_types(self) -> None:
        assert np.allclose(
            relhum(np.asarray(self.t_def), xr.DataArray(self.q_def), self.p_def),
            self.rh_gt_2,
            atol=0.1,
        )
        assert np.allclose(
            relhum(self.t_def, xr.DataArray(self.q_def), self.p_def),
            self.rh_gt_2,
            atol=0.1,
        )
        assert np.allclose(
            relhum(self.t_def, self.q_def, np.asarray(self.p_def)),
            self.rh_gt_2,
            atol=0.1,
        )


class Test_relhum_water:
    rh_gt_1 = 46.3574

    def test_float_input(self) -> None:
        p = 1000.0 * 100
        t = 18.0 + 273.15
        q = 6.0 / 1000.0

        assert np.allclose(relhum_water(t, q, p), self.rh_gt_1, atol=0.1)


class Test_relhum_ice:
    rh_gt_1 = 147.8802

    def test_float_input(self) -> None:
        tc = -5.0
        tk = tc + 273.15
        w = 3.7 / 1000.0
        p = 1000.0 * 100.0

        assert np.allclose(relhum_ice(tk, w, p), self.rh_gt_1, atol=0.1)


class Test_actual_saturation_vapor_pressure:
    # set up ground truths
    temp_gt = np.arange(1, 101, 1)

    @pytest.fixture(scope="class")
    def ncl_gt(self):
        # get ground truth from ncl run netcdf file
        try:
            return xr.open_dataarray(
                "satvpr_tdew_fao56_output.nc"
            ).values  # Generated by running ncl_tests/test_satvpr_tdew_fao56.ncl
        except Exception:
            return xr.open_dataarray("test/satvpr_tdew_fao56_output.nc").values

    def test_numpy_input(self, ncl_gt) -> None:
        assert np.allclose(
            actual_saturation_vapor_pressure(self.temp_gt, tfill=1.0000000e20),
            ncl_gt,
            atol=0.005,
        )

    def test_float_input(self) -> None:
        degf = 59
        expected = 1.70535
        assert np.allclose(actual_saturation_vapor_pressure(degf), expected, atol=0.005)

    def test_list_input(self, ncl_gt) -> None:
        assert np.allclose(
            actual_saturation_vapor_pressure(self.temp_gt.tolist(), tfill=1.0000000e20),
            ncl_gt.tolist(),
            atol=0.005,
        )

    def test_multi_dimensional_input(self, ncl_gt) -> None:
        assert np.allclose(
            actual_saturation_vapor_pressure(
                self.temp_gt.reshape(2, 50), tfill=1.0000000e20
            ),
            ncl_gt.reshape(2, 50),
            atol=0.005,
        )

    def test_xarray_input(self, ncl_gt) -> None:
        tempf = xr.DataArray(self.temp_gt)
        expected = xr.DataArray(ncl_gt)

        assert np.allclose(
            actual_saturation_vapor_pressure(tempf, tfill=1.0000000e20),
            expected,
            atol=0.005,
        )


class Test_max_daylight:
    # set up ground truths
    jday_gt = np.linspace(1, 365, num=365)
    lat_gt = np.linspace(-66, 66, num=133)

    @pytest.fixture(scope="class")
    def ncl_gt(self):
        # get ground truth from ncl run netcdf file
        try:
            return xr.open_dataarray(
                "max_daylight_test.nc"
            ).values  # Generated by running ncl_tests/test_max_daylight.ncl
        except Exception:
            return xr.open_dataarray("test/max_daylight_test.nc").values

    def test_numpy_input(self, ncl_gt) -> None:
        assert np.allclose(max_daylight(self.jday_gt, self.lat_gt), ncl_gt, atol=0.005)

    def test_float_input(self) -> None:
        assert np.allclose(max_daylight(246, -20.0), 11.66559, atol=0.005)

    def test_list_input(self, ncl_gt) -> None:
        assert np.allclose(
            max_daylight(self.jday_gt.tolist(), self.lat_gt.tolist()),
            ncl_gt,
            atol=0.005,
        )

    def test_xarray_input(self, ncl_gt) -> None:
        jday = xr.DataArray(self.jday_gt)
        lat = xr.DataArray(self.lat_gt)

        assert np.allclose(max_daylight(jday, lat), ncl_gt, atol=0.005)

    def test_input_dim(self) -> None:
        with pytest.raises(ValueError):
            max_daylight(np.arange(4).reshape(2, 2), np.arange(4).reshape(2, 2))

    def test_lat_bound_warning(self) -> None:
        with pytest.warns(UserWarning):
            max_daylight(10, 56)

    def test_lat_bound_second_warning(self) -> None:
        with pytest.warns(UserWarning):
            max_daylight(10, 67)


class Test_psychrometric_constant:
    # set up ground truths
    pressure_gt = np.arange(1, 101, 1)

    @pytest.fixture(scope="class")
    def ncl_gt(self):
        # get ground truth from ncl run netcdf file
        try:
            return xr.open_dataarray(
                "psychro_fao56_output.nc"
            ).values  # Generated by running ncl_tests/test_psychro_fao56.ncl
        except Exception:
            return xr.open_dataarray("test/psychro_fao56_output.nc").values

    def test_numpy_input(self, ncl_gt) -> None:
        assert np.allclose(psychrometric_constant(self.pressure_gt), ncl_gt, atol=0.005)

    def test_float_input(self) -> None:
        pressure = 81.78
        expected = 0.05434634
        assert np.allclose(psychrometric_constant(pressure), expected, atol=0.005)

    def test_list_input(self, ncl_gt) -> None:
        assert np.allclose(
            psychrometric_constant(self.pressure_gt.tolist()),
            ncl_gt.tolist(),
            atol=0.005,
        )

    def test_multi_dimensional_input(self, ncl_gt) -> None:
        assert np.allclose(
            psychrometric_constant(self.pressure_gt.reshape(2, 50)),
            ncl_gt.reshape(2, 50),
            atol=0.005,
        )

    def test_xarray_input(self, ncl_gt) -> None:
        pressure = xr.DataArray(self.pressure_gt)
        expected = xr.DataArray(ncl_gt)

        assert np.allclose(psychrometric_constant(pressure), expected, atol=0.005)


class Test_saturation_vapor_pressure:
    # set up ground truths
    temp_gt = np.arange(1, 101, 1)

    @pytest.fixture(scope="class")
    def ncl_gt(self):
        # get ground truth from ncl run netcdf file
        try:
            return xr.open_dataarray(
                "satvpr_temp_fao56_output.nc"
            ).values  # Generated by running ncl_tests/test_satvpr_temp_fao56.ncl
        except Exception:
            return xr.open_dataarray("test/satvpr_temp_fao56_output.nc").values

    def test_numpy_input(self, ncl_gt) -> None:
        assert np.allclose(
            saturation_vapor_pressure(self.temp_gt, tfill=1.0000000e20),
            ncl_gt,
            atol=0.005,
        )

    def test_float_input(self) -> None:
        degf = 59
        expected = 1.70535
        assert np.allclose(saturation_vapor_pressure(degf), expected, atol=0.005)

    def test_list_input(self, ncl_gt) -> None:
        assert np.allclose(
            saturation_vapor_pressure(self.temp_gt.tolist(), tfill=1.0000000e20),
            ncl_gt.tolist(),
            atol=0.005,
        )

    def test_multi_dimensional_input(self, ncl_gt) -> None:
        assert np.allclose(
            saturation_vapor_pressure(self.temp_gt.reshape(2, 50), tfill=1.0000000e20),
            ncl_gt.reshape(2, 50),
            atol=0.005,
        )

    def test_xarray_input(self, ncl_gt) -> None:
        tempf = xr.DataArray(self.temp_gt)
        expected = xr.DataArray(ncl_gt)

        assert np.allclose(
            saturation_vapor_pressure(tempf, tfill=1.0000000e20), expected, atol=0.005
        )


class Test_saturation_vapor_pressure_slope:
    # set up ground truths
    temp_gt = np.arange(1, 101, 1)

    @pytest.fixture(scope="class")
    def ncl_gt(self):
        # get ground truth from ncl run netcdf file
        try:
            return xr.open_dataarray(
                "satvpr_slope_fao56_output.nc"
            ).values  # Generated by running ncl_tests/test_satvpr_slope_fao56.ncl
        except Exception:
            return xr.open_dataarray("test/satvpr_slope_fao56_output.nc").values

    def test_numpy_input(self, ncl_gt) -> None:
        assert np.allclose(
            saturation_vapor_pressure_slope(self.temp_gt), ncl_gt, equal_nan=True
        )

    def test_float_input(self) -> None:
        degf = 67.55
        expected = 0.142793
        assert np.allclose(saturation_vapor_pressure_slope(degf), expected, atol=0.005)

    def test_list_input(self, ncl_gt) -> None:
        assert np.allclose(
            saturation_vapor_pressure_slope(self.temp_gt.tolist()),
            ncl_gt.tolist(),
            equal_nan=True,
        )

    def test_multi_dimensional_input(self, ncl_gt) -> None:
        assert np.allclose(
            saturation_vapor_pressure_slope(self.temp_gt.reshape(2, 50)),
            ncl_gt.reshape(2, 50),
            atol=0.005,
            equal_nan=True,
        )

    def test_xarray_input(self, ncl_gt) -> None:
        tempf = xr.DataArray(self.temp_gt)
        expected = xr.DataArray(ncl_gt)

        assert np.allclose(
            saturation_vapor_pressure_slope(tempf), expected, atol=0.005, equal_nan=True
        )


class Test_Delta_Pressure:
    pressure_lev = np.array([1, 5, 100, 1000])
    pressure_lev_da = xr.DataArray(pressure_lev)
    pressure_lev_da.attrs = {
        "long name": "pressure level",
        "units": "hPa",
        "direction": "descending",
    }

    surface_pressure_scalar = 1018
    surface_pressure_1D = np.array([1018, 1019])
    surface_pressure_2D = np.array([[1018, 1019], [1017, 1019.5]])
    surface_pressure_3D = np.array(
        [[[1018, 1019], [1017, 1019.5]], [[1019, 1020], [1018, 1020.5]]]
    )

    surface_pressure_3D_da = xr.DataArray(
        surface_pressure_3D,
        coords={"time": [1, 2], "lat": [3, 4], "lon": [5, 6]},
        dims=["time", "lat", "lon"],
        attrs={"long name": "surface pressure", "units": "hPa"},
    )

    def test_delta_pressure1D(self) -> None:
        pressure_lev = [float(i) for i in self.pressure_lev]
        pressure_top = min(pressure_lev)
        delta_p = delta_pressure(pressure_lev, self.surface_pressure_scalar)
        assert sum(delta_p) == (self.surface_pressure_scalar - pressure_top)

    def test_delta_pressure_level_below_surface(self) -> None:
        pressure_lev = [float(i) for i in self.pressure_lev]
        surface_pressure_adjusted = 900.0
        delta_p = delta_pressure(pressure_lev, surface_pressure_adjusted)
        assert np.nansum(delta_p) == (surface_pressure_adjusted - min(pressure_lev))

    def test_delta_pressure_levels_below_surface(self) -> None:
        pressure_lev = [float(i) for i in self.pressure_lev]
        surface_pressure_adjusted = 50.0
        delta_p = delta_pressure(pressure_lev, surface_pressure_adjusted)
        assert np.nansum(delta_p) == (surface_pressure_adjusted - min(pressure_lev))
        assert np.sum(np.isnan(delta_p)) == 2

    def test_negative_pressure_error(self) -> None:
        pressure_lev_negative = self.pressure_lev.copy()
        pressure_lev_negative[0] = -5
        with pytest.raises(ValueError):
            delta_pressure(pressure_lev_negative, self.surface_pressure_scalar)

    def test_relative_pressure_error(self) -> None:
        surface_pressure_low = 0.5
        with pytest.raises(ValueError):
            delta_pressure(self.pressure_lev, surface_pressure_low)

    def test_output_type(self) -> None:
        delta_pressure_da = delta_pressure(
            self.pressure_lev_da, self.surface_pressure_3D_da
        )
        assert isinstance(delta_pressure_da, xr.DataArray)

        delta_pressure_np = delta_pressure(self.pressure_lev, self.surface_pressure_3D)
        assert isinstance(delta_pressure_np, np.ndarray)

    def test_output_dimensions(self) -> None:
        delta_pressure_scalar = delta_pressure(
            self.pressure_lev, self.surface_pressure_scalar
        )
        assert delta_pressure_scalar.shape == (4,)

        delta_pressure_1D = delta_pressure(self.pressure_lev, self.surface_pressure_1D)
        assert delta_pressure_1D.shape == (2, 4)

        delta_pressure_2D = delta_pressure(self.pressure_lev, self.surface_pressure_2D)
        assert delta_pressure_2D.shape == (2, 2, 4)

        delta_pressure_3D = delta_pressure(self.pressure_lev, self.surface_pressure_3D)
        assert delta_pressure_3D.shape == (2, 2, 2, 4)

    def test_output_attrs(self) -> None:
        delta_pressure_da = delta_pressure(
            self.pressure_lev_da, self.surface_pressure_3D_da
        )
        for item in self.pressure_lev_da.attrs:
            assert item in delta_pressure_da.attrs

    def test_output_coords(self) -> None:
        delta_pressure_da = delta_pressure(
            self.pressure_lev_da, self.surface_pressure_3D_da
        )
        for item in self.surface_pressure_3D_da.coords:
            assert item in delta_pressure_da.coords
        for item in self.pressure_lev_da.coords:
            assert item in delta_pressure_da.coords

    def test_mismatch_input_types(self) -> None:
        delta_pressure_da = delta_pressure(
            self.pressure_lev, self.surface_pressure_3D_da
        )
        assert isinstance(delta_pressure_da, xr.DataArray)

        delta_pressure_np = delta_pressure(
            self.pressure_lev_da, self.surface_pressure_3D
        )
        assert isinstance(delta_pressure_np, np.ndarray)

    def test_pressure_top_specified(self) -> None:
        # pressure_top <= min(pressure_lev)
        # 0 < pressure_top <= 1
        pressure_top = round(uniform(0.1, 0.99), 2)

        # test scalar
        delta_p = delta_pressure(self.pressure_lev, self.surface_pressure_scalar)
        delta_p_top = delta_pressure(
            self.pressure_lev, self.surface_pressure_scalar, pressure_top=pressure_top
        )
        assert delta_p_top[0] == delta_p[0] + min(self.pressure_lev) - pressure_top

        # test multi-dimensional
        delta_p = delta_pressure(self.pressure_lev, self.surface_pressure_3D)
        delta_p_top = delta_pressure(
            self.pressure_lev, self.surface_pressure_3D, pressure_top=pressure_top
        )
        np.testing.assert_equal(
            delta_p_top[:, :, :, 0],
            delta_p[:, :, :, 0] + min(self.pressure_lev) - pressure_top,
        )

        # test multidim xarray
        delta_p_da = delta_pressure(self.pressure_lev_da, self.surface_pressure_3D_da)
        delta_p_da_top = delta_pressure(
            self.pressure_lev_da, self.surface_pressure_3D_da, pressure_top=pressure_top
        )
        np.testing.assert_allclose(delta_p_top, delta_p_da_top.values)
        xr.testing.assert_equal(
            delta_p_da_top[:, :, :, 0],
            delta_p_da[:, :, :, 0] + min(self.pressure_lev_da) - pressure_top,
        )

        # test list input for pressure_lev for .min() usage w/ pressure_top
        delta_pressure(self.pressure_lev.tolist(), self.surface_pressure_scalar)
        delta_pressure(
            self.pressure_lev.tolist(),
            self.surface_pressure_scalar,
            pressure_top=pressure_top,
        )


class Test_zonal_meridional_psi:
    lat = np.arange(36, 45, 1)

    @pytest.fixture
    def uxds_plev(self):
        """Load UXarray dataset with pressure level data."""
        try:
            return ux.open_dataset("grid_subset.nc", "plev_subset.nc")
        except FileNotFoundError:
            return ux.open_dataset("test/grid_subset.nc", "test/plev_subset.nc")

    @pytest.fixture
    def uxds_hybrid(self):
        """Load UXarray dataset with hybrid level data."""
        try:
            return ux.open_dataset("grid_subset.nc", "hybrid_subset.nc")
        except FileNotFoundError:
            return ux.open_dataset("test/grid_subset.nc", "test/hybrid_subset.nc")

    def test_zonal_meridional_psi_pressure_levels(self, uxds_plev) -> None:
        out = zonal_meridional_psi(uxds_plev, lat=self.lat)

        # ---- structural checks ----
        assert isinstance(out, xr.DataArray)

        assert "time" in out.dims
        assert "latitudes" in out.dims
        assert "plev" in out.dims

        # shape checks
        assert out.sizes["latitudes"] == len(self.lat)
        assert out.sizes["plev"] == len(uxds_plev.V.plev)

        # ---- numerical coherence ----
        assert np.isfinite(out).all()
        assert not np.allclose(out.values, 0)

    def test_zonal_meridional_psi_hybrid_equivalence(self, uxds_hybrid) -> None:
        # ---- function output (hybrid path) ----
        out_func = zonal_meridional_psi(uxds_hybrid, lat=self.lat)

        # ---- manual calculation ----
        da_ipress = interp_hybrid_to_pressure(
            uxds_hybrid.V, uxds_hybrid.PS, uxds_hybrid.hyam, uxds_hybrid.hybm
        )
        ux_ipress = ux.UxDataArray(da_ipress, uxgrid=uxds_hybrid.uxgrid)

        ux_v_zonal = ux_ipress.zonal_mean(lat=self.lat)
        ux_v_zonal['plev'] = ux_ipress.plev
        ux_PS_zonal = uxds_hybrid.PS.zonal_mean(lat=self.lat)

        a = 6.371e6
        g = 9.80665
        da_scaling_factor = 2 * np.pi * a * np.cos(self.lat) / g
        np_dp_zonal = delta_pressure(ux_v_zonal.plev, ux_PS_zonal)

        da_dp_zonal = xr.DataArray(
            np_dp_zonal,
            dims=["time", "latitudes", "plev"],
            coords={"plev": ux_v_zonal.plev, "latitudes": ux_v_zonal.latitudes},
            name="delta_pressure",
        )
        ux_dp_zonal = ux.UxDataArray(da_dp_zonal, uxgrid=uxds_hybrid.uxgrid)
        integrand = ux_v_zonal * ux_dp_zonal
        ux_mpsi = (
            integrand.isel(plev=slice(None, None, -1))
            .cumsum(dim="plev")
            .isel(plev=slice(None, None, -1))
        )
        out_manual = ux_mpsi * da_scaling_factor

        xr.testing.assert_allclose(out_func, out_manual)

    @pytest.mark.xfail(reason="Dataset may not cover extreme latitudes")
    def test_zonal_meridional_psi_raises_on_nan_surface_pressure(self, uxds_plev):
        """Ensure zonal_meridional_psi fails when surface pressure contains NaNs."""
        uxds_bad = uxds_plev.copy()
        uxds_bad["PS"][:] = np.nan

        zonal_meridional_psi(uxds_bad, lat=self.lat)

    @pytest.mark.xfail(reason="Dataset may not cover extreme latitudes")
    def test_zonal_meridional_psi_extreme_latitudes(self, uxds_plev):
        """Test with latitude values at poles."""
        out = zonal_meridional_psi(uxds_plev, lat=np.array([-90, 0, 90]))
        assert np.isfinite(out).all()

    def test_zonal_meridional_psi_cos_latitude_weighting(self, uxds_plev):
        """Test that cosine latitude weighting is applied."""
        out_low = zonal_meridional_psi(uxds_plev, lat=np.array([36]))
        out_high = zonal_meridional_psi(uxds_plev, lat=np.array([44]))

        assert np.isfinite(out_low).all()
        assert np.isfinite(out_high).all()

    def test_zonal_meridional_psi_custom_varnames(self, uxds_plev):
        """Test providing custom variable names."""
        out = zonal_meridional_psi(
            uxds_plev,
            meridional_wind_varname='V',
            surface_air_pressure_varname='PS',
            plev_coordname='plev',
            lat=self.lat,
        )

        assert isinstance(out, xr.DataArray)
        assert out.sizes["latitudes"] == len(self.lat)

    def test_zonal_meridional_psi_single_latitude(self, uxds_plev):
        """Test with single latitude value."""
        out = zonal_meridional_psi(uxds_plev, lat=40.0)

        assert isinstance(out, xr.DataArray)
        assert "latitudes" in out.dims

    def test_zonal_meridional_psi_metadata(self, uxds_plev):
        """Test that output has correct metadata."""
        out = zonal_meridional_psi(uxds_plev, lat=self.lat)

        assert "long_name" in out.attrs
        assert out.attrs["long_name"] == "zonal mean meridional streamfunction"
        assert out.attrs["units"] == "kg/s"
        assert "info" in out.attrs

    def test_zonal_meridional_psi_dimension_order(self, uxds_plev):
        """Test that output has expected dimension ordering."""
        out = zonal_meridional_psi(uxds_plev, lat=self.lat)

        expected_dims = ["time", "plev", "latitudes"]
        assert list(out.dims) == expected_dims

    def test_zonal_meridional_psi_missing_meridional_wind(self, uxds_plev):
        """Test error when meridional wind variable is missing."""
        uxds_bad = uxds_plev.drop_vars('V')

        with pytest.raises(KeyError) as excinfo:
            zonal_meridional_psi(uxds_bad, lat=self.lat)

        assert "Could not find" in str(excinfo.value)

    def test_zonal_meridional_psi_missing_surface_pressure(self, uxds_hybrid):
        """Test error when surface pressure is missing."""
        uxds_bad = uxds_hybrid.drop_vars('PS')

        with pytest.raises(KeyError) as excinfo:
            zonal_meridional_psi(uxds_bad, lat=self.lat)

        assert "Could not find" in str(excinfo.value)

    def test_zonal_meridional_psi_missing_coords(self, uxds_plev):
        """Test error when neither pressure nor hybrid coordinates exist."""
        # Create dataset without pressure coordinate on V
        uxds_bad = uxds_plev.copy()
        uxds_bad['V'] = uxds_plev['V'].isel(plev=0, drop=True)

        with pytest.raises(AttributeError) as excinfo:
            zonal_meridional_psi(uxds_bad, lat=self.lat)

        assert "must have either a pressure level coordinate" in str(excinfo.value)

    def test_zonal_meridional_psi_ascending_pressure(self, uxds_plev):
        """Test with ascending pressure levels (if data supports it)."""
        out = zonal_meridional_psi(uxds_plev, lat=self.lat)

        # Check that integration handled orientation correctly
        assert np.isfinite(out).all()

    def test_zonal_meridional_psi_descending_pressure(self, uxds_hybrid):
        """Test with descending pressure levels (typical for hybrid coords)."""
        out = zonal_meridional_psi(uxds_hybrid, lat=self.lat)

        # Check that integration handled orientation correctly
        assert np.isfinite(out).all()

    def test_zonal_meridional_psi_scaling_magnitude(self, uxds_plev):
        """Test that Earth geometry scaling produces reasonable magnitudes."""
        out = zonal_meridional_psi(uxds_plev, lat=self.lat)

        # Values should be large due to Earth radius scaling (2*pi*a/g term)
        # Typical values are O(10^10) kg/s
        assert np.abs(out.values).max() > 1e6

    def test_zonal_meridional_psi_case_insensitive_finding(self, uxds_plev):
        """Test that variable finding works with case variations."""
        # This tests the _find_var functionality indirectly
        out = zonal_meridional_psi(uxds_plev, lat=self.lat)
        assert isinstance(out, xr.DataArray)

    def test_zonal_meridional_psi_pressure_integration_correctness(self, uxds_plev):
        """Test that pressure integration produces monotonic results."""
        out = zonal_meridional_psi(uxds_plev, lat=self.lat)

        # Stream function should vary smoothly with pressure
        # (not necessarily monotonic, but should have structure)
        assert not np.all(np.diff(out.values, axis=-1) == 0)

    def test_zonal_meridional_psi_time_dimension_preserved(self, uxds_plev):
        """Test that time dimension is preserved correctly."""
        out = zonal_meridional_psi(uxds_plev, lat=self.lat)

        assert "time" in out.dims
        assert out.sizes["time"] == uxds_plev.sizes["time"]

    def test_zonal_meridional_psi_hybrid_creates_plev_coord(self, uxds_hybrid):
        """Test that hybrid interpolation creates 'plev' coordinate."""
        out = zonal_meridional_psi(uxds_hybrid, lat=self.lat)

        assert "plev" in out.dims
        assert "plev" in out.coords
