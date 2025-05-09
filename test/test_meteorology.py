import pytest

import numpy as np
import xarray as xr

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
)


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
    ncl_gt_2 = [
        68.585,
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

    t1 = np.array([104, 100, 92, 92, 86, 80, 80, 60, 30])
    rh1 = np.array([55, 65, 60, 90, 90, 40, 75, 90, 50])

    t2 = np.array([70, 75, 80, 85, 90, 95, 100, 105, 110, 115])
    rh2 = np.array([10, 75, 15, 80, 65, 25, 30, 40, 50, 5])

    def test_numpy_input(self) -> None:
        assert np.allclose(
            heat_index(self.t1, self.rh1, False), self.ncl_gt_1, atol=0.005
        )

    def test_multi_dimensional_input(self) -> None:
        assert np.allclose(
            heat_index(self.t2.reshape(2, 5), self.rh2.reshape(2, 5), True),
            np.asarray(self.ncl_gt_2).reshape(2, 5),
            atol=0.005,
        )

    def test_alt_coef(self) -> None:
        assert np.allclose(
            heat_index(self.t2, self.rh2, True), self.ncl_gt_2, atol=0.005
        )

    def test_xarray_alt_coef(self) -> None:
        assert np.allclose(
            heat_index(xr.DataArray(self.t2), xr.DataArray(self.rh2), True),
            self.ncl_gt_2,
            atol=0.005,
        )

    def test_float_input(self) -> None:
        assert np.allclose(heat_index(80, 75), 83.5751, atol=0.005)

    def test_list_input(self) -> None:
        assert np.allclose(
            heat_index(self.t1.tolist(), self.rh1.tolist()), self.ncl_gt_1, atol=0.005
        )

    def test_xarray_input(self) -> None:
        t = xr.DataArray(self.t1)
        rh = xr.DataArray(self.rh1)

        assert np.allclose(heat_index(t, rh), self.ncl_gt_1, atol=0.005)

    def test_alternate_xarray_tag(self) -> None:
        t = xr.DataArray([15, 20])
        rh = xr.DataArray([15, 20])

        out = heat_index(t, rh)
        assert out.tag == "NCL: heat_index_nws; (Steadman+t)*0.5"

    def test_rh_warning(self) -> None:
        with pytest.warns(UserWarning):
            heat_index([50, 80, 90], [0.1, 0.2, 0.5])

    def test_rh_valid(self) -> None:
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
            heat_index(self.t1, xr.DataArray(self.rh1))

    def test_dims_error(self) -> None:
        with pytest.raises(ValueError):
            heat_index(self.t1[:10], self.rh1[:8])


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

    def test_xarray_type_error(self) -> None:
        with pytest.raises(TypeError):
            relhum(self.t_def, xr.DataArray(self.q_def), self.p_def)


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
