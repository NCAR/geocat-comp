import sys
import unittest

import dask.array
import dask.distributed as dd
import geocat.datafiles as gdf
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
import pandas as pd
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp.meteorology import (
        dewtemp, heat_index, relhum, relhum_ice, relhum_water, showalter_index,
        actual_saturation_vapor_pressure, max_daylight, psychrometric_constant,
        saturation_vapor_pressure, saturation_vapor_pressure_slope)
else:
    from geocat.comp.meteorology import (
        dewtemp, heat_index, relhum, relhum_ice, relhum_water, showalter_index,
        actual_saturation_vapor_pressure, max_daylight, psychrometric_constant,
        saturation_vapor_pressure, saturation_vapor_pressure_slope)


class Test_dewtemp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # set up ground truths
        cls.t_def = [
            29.3, 28.1, 23.5, 20.9, 18.4, 15.9, 13.1, 10.1, 6.7, 3.1, -0.5,
            -4.5, -9.0, -14.8, -21.5, -29.7, -40.0, -52.4
        ]

        cls.rh_def = [
            75.0, 60.0, 61.1, 76.7, 90.5, 89.8, 78.3, 76.5, 46.0, 55.0, 63.8,
            53.2, 42.9, 41.7, 51.0, 70.6, 50.0, 50.0
        ]

        cls.dt_1 = 6.3

        cls.dt_2 = [
            24.38342, 19.55563, 15.53281, 16.64218, 16.81433, 14.22482,
            9.401337, 6.149719, -4.1604, -5.096619, -6.528168, -12.61957,
            -19.38332, -25.00714, -28.9841, -33.34853, -46.51273, -58.18289
        ]

        # make dask client to reference in subsequent tests
        cls.client = dd.Client()

    def test_float_input(self):
        tk = 18. + 273.15
        rh = 46.5

        assert np.allclose(dewtemp(tk, rh) - 273.15, self.dt_1, 0.1)

    def test_list_input(self):
        tk = (np.asarray(self.t_def) + 273.15).tolist()

        assert np.allclose(dewtemp(tk, self.rh_def) - 273.15, self.dt_2, 0.1)

    def test_numpy_input(self):
        tk = np.asarray(self.t_def) + 273.15
        rh = np.asarray(self.rh_def)

        assert np.allclose(dewtemp(tk, rh) - 273.15, self.dt_2, 0.1)

    def test_xarray_input(self):
        tk = xr.DataArray(np.asarray(self.t_def) + 273.15)
        rh = xr.DataArray(self.rh_def)

        assert np.allclose(dewtemp(tk, rh) - 273.15, self.dt_2, 0.1)

    def test_dims_error(self):
        self.assertRaises(ValueError, dewtemp, self.t_def[:10], self.rh_def[:8])

    def test_xarray_type_error(self):
        self.assertRaises(TypeError, dewtemp, self.t_def,
                          xr.DataArray(self.rh_def))

    def test_dask_compute(self):
        tk = xr.DataArray(np.asarray(self.t_def) + 273.15).chunk(6)
        rh = xr.DataArray(self.rh_def).chunk(6)

        assert np.allclose(dewtemp(tk, rh) - 273.15, self.dt_2, atol=0.1)

    def test_dask_lazy(self):
        tk = xr.DataArray(np.asarray(self.t_def) + 273.15).chunk(6)
        rh = xr.DataArray(self.rh_def).chunk(6)

        assert isinstance((dewtemp(tk, rh) - 273.15).data, dask.array.Array)


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

    def test_xarray_alt_coef(self):
        assert np.allclose(heat_index(xr.DataArray(self.t2),
                                      xr.DataArray(self.rh2), True),
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

    def test_xarray_rh_warning(self):
        self.assertWarns(UserWarning, heat_index, [50, 80, 90], [0.1, 0.2, 0.5])

    def test_xarray_rh_valid(self):
        self.assertRaises(ValueError, heat_index, xr.DataArray([50, 80, 90]),
                          xr.DataArray([-1, 101, 50]))

    def test_xarray_type_error(self):
        self.assertRaises(TypeError, heat_index, self.t1,
                          xr.DataArray(self.rh1))

    def test_dims_error(self):
        self.assertRaises(ValueError, heat_index, self.t1[:10], self.rh1[:8])

    def test_dask_compute(self):
        t = xr.DataArray(self.t1).chunk(3)
        rh = xr.DataArray(self.rh1).chunk(3)

        assert np.allclose(heat_index(t, rh), self.ncl_gt_1, atol=0.005)

    def test_dask_lazy(self):
        t = xr.DataArray(self.t1).chunk(3)
        rh = xr.DataArray(self.rh1).chunk(3)

        assert isinstance((heat_index(t, rh)).data, dask.array.Array)


class Test_relhum(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # set up ground truths
        cls.p_def = [
            100800, 100000, 95000, 90000, 85000, 80000, 75000, 70000, 65000,
            60000, 55000, 50000, 45000, 40000, 35000, 30000, 25000, 20000,
            17500, 15000, 12500, 10000, 8000, 7000, 6000, 5000, 4000, 3000,
            2500, 2000
        ]

        cls.t_def = [
            302.45, 301.25, 296.65, 294.05, 291.55, 289.05, 286.25, 283.25,
            279.85, 276.25, 272.65, 268.65, 264.15, 258.35, 251.65, 243.45,
            233.15, 220.75, 213.95, 206.65, 199.05, 194.65, 197.15, 201.55,
            206.45, 211.85, 216.85, 221.45, 222.45, 225.65
        ]

        cls.q_def = [
            0.02038, 0.01903, 0.01614, 0.01371, 0.01156, 0.0098, 0.00833,
            0.00675, 0.00606, 0.00507, 0.00388, 0.00329, 0.00239, 0.0017, 0.001,
            0.0006, 0.0002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]

        cls.rh_gt_1 = 46.4

        cls.rh_gt_2 = [
            79.8228, 79.3578, 84.1962, 79.4898, 73.989, 69.2401, 66.1896,
            61.1084, 64.21, 63.8305, 58.0412, 60.8194, 57.927, 62.3734, 62.9706,
            73.8184, 62.71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]

        # make dask client to reference in subsequent tests
        cls.client = dd.Client()

    def test_float_input(self):
        p = 1000. * 100
        t = 18. + 273.15
        q = 6. / 1000.

        assert np.allclose(relhum(t, q, p), self.rh_gt_1, atol=0.1)

    def test_list_input(self):

        assert np.allclose(relhum(self.t_def, self.q_def, self.p_def),
                           self.rh_gt_2,
                           atol=0.1)

    def test_numpy_input(self):
        p = np.asarray(self.p_def)
        t = np.asarray(self.t_def)
        q = np.asarray(self.q_def)

        assert np.allclose(relhum(t, q, p), self.rh_gt_2, atol=0.1)

    def test_dims_error(self):
        self.assertRaises(ValueError, relhum, self.t_def[:10], self.q_def[:10],
                          self.p_def[:9])

    def test_xarray_type_error(self):
        self.assertRaises(TypeError, relhum, self.t_def,
                          xr.DataArray(self.q_def), self.p_def)

    def test_dask_compute(self):
        p = xr.DataArray(self.p_def).chunk(10)
        t = xr.DataArray(self.t_def).chunk(10)
        q = xr.DataArray(self.q_def).chunk(10)

        assert np.allclose(relhum(t, q, p), self.rh_gt_2, atol=0.1)

    def test_dask_lazy(self):
        p = xr.DataArray(self.p_def).chunk(10)
        t = xr.DataArray(self.t_def).chunk(10)
        q = xr.DataArray(self.q_def).chunk(10)

        assert isinstance(relhum(t, q, p).data, dask.array.Array)


class Test_relhum_water(unittest.TestCase):

    rh_gt_1 = 46.3574

    def test_float_input(self):
        p = 1000. * 100
        t = 18. + 273.15
        q = 6. / 1000.

        assert np.allclose(relhum_water(t, q, p), self.rh_gt_1, atol=0.1)


class Test_relhum_ice(unittest.TestCase):

    rh_gt_1 = 147.8802

    def test_float_input(self):
        tc = -5.
        tk = tc + 273.15
        w = 3.7 / 1000.
        p = 1000. * 100.

        assert np.allclose(relhum_ice(tk, w, p), self.rh_gt_1, atol=0.1)


class Test_showalter_index(unittest.TestCase):
    ds = pd.read_csv(gdf.get('ascii_files/sounding.testdata'),
                     delimiter='\\s+',
                     header=None)
    # get ground truth from ncl run netcdf file
    try:
        out = xr.open_dataset(
            "skewt_params_output.nc"
        )  # Generated by running ncl_tests/skewt_params_test.ncl
    except:
        out = xr.open_dataset("test/skewt_params_output.nc")

    # Extract the data from ds
    p = ds[1].values * units.hPa  # Pressure [mb/hPa]
    tc = (ds[5].values + 2) * units.degC  # Temperature [C]
    tdc = ds[9].values * units.degC  # Dew pt temp  [C]
    pro = mpcalc.parcel_profile(p, tc[0], tdc[0]).to('degC')

    # Extract Showalter Index from NCL out file and convert to int
    Shox = np.round(out['Shox'])  # Use np.round to avoid rounding issues
    NCL_shox = int(Shox[0])  # Convert to int

    def test_shox_vals(self):
        self.assertWarns(DeprecationWarning)
        parameters = [((self.p, self.tc, self.tdc),
                       units.Quantity(self.NCL_shox, 'delta_degree_Celsius')),
                      ((self.p.magnitude, self.tc.magnitude,
                        self.tdc.magnitude), self.NCL_shox)]
        for params in parameters:
            # Showalter index
            shox = showalter_index(*params[0])

            # Place calculated values in iterable list
            vals = np.round(shox).astype(int)

            # Compare calculated values with expected
            np.testing.assert_equal(vals, params[1])


class Test_actual_saturation_vapor_pressure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # get ground truth from ncl run netcdf file
        try:
            ncl_xr_gt = xr.open_dataarray(
                "satvpr_tdew_fao56_output.nc"
            )  # Generated by running ncl_tests/test_satvpr_tdew_fao56.ncl
        except:
            ncl_xr_gt = xr.open_dataarray("test/satvpr_tdew_fao56_output.nc")

        # set up ground truths
        cls.ncl_gt = np.asarray(ncl_xr_gt)

        cls.temp_gt = np.arange(1, 101, 1)

        # make client to reference in subsequent tests
        cls.client = dd.Client()

    def test_numpy_input(self):
        assert np.allclose(actual_saturation_vapor_pressure(
            self.temp_gt, tfill=1.0000000e+20),
                           self.ncl_gt,
                           atol=0.005)

    def test_float_input(self):
        degf = 59
        expected = 1.70535
        assert np.allclose(actual_saturation_vapor_pressure(degf),
                           expected,
                           atol=0.005)

    def test_list_input(self):
        assert np.allclose(actual_saturation_vapor_pressure(
            self.temp_gt.tolist(), tfill=1.0000000e+20),
                           self.ncl_gt.tolist(),
                           atol=0.005)

    def test_multi_dimensional_input(self):
        assert np.allclose(actual_saturation_vapor_pressure(
            self.temp_gt.reshape(2, 50), tfill=1.0000000e+20),
                           self.ncl_gt.reshape(2, 50),
                           atol=0.005)

    def test_xarray_input(self):
        tempf = xr.DataArray(self.temp_gt)
        expected = xr.DataArray(self.ncl_gt)

        assert np.allclose(actual_saturation_vapor_pressure(
            tempf, tfill=1.0000000e+20),
                           expected,
                           atol=0.005)

    def test_dask_compute(self):
        tempf = xr.DataArray(self.temp_gt).chunk(10)

        assert np.allclose(actual_saturation_vapor_pressure(
            tempf, tfill=1.0000000e+20),
                           self.ncl_gt,
                           atol=0.005)

    def test_dask_lazy(self):
        tempf = xr.DataArray(self.temp_gt).chunk(10)

        assert isinstance(
            (actual_saturation_vapor_pressure(tempf, tfill=1.0000000e+20)).data,
            dask.array.Array)


class Test_max_daylight(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # get ground truth from ncl run netcdf file
        try:
            ncl_xr_gt = xr.open_dataarray(
                "max_daylight_test.nc"
            )  # Generated by running ncl_tests/test_max_daylight.ncl
        except:
            ncl_xr_gt = xr.open_dataarray("test/max_daylight_test.nc")

        # set up ground truths
        cls.ncl_gt = np.asarray(ncl_xr_gt)

        cls.jday_gt = np.linspace(1, 365, num=365)
        cls.lat_gt = np.linspace(-66, 66, num=133)

        # make client to reference in subsequent tests
        cls.client = dd.Client()

    def test_numpy_input(self):
        assert np.allclose(max_daylight(self.jday_gt, self.lat_gt),
                           self.ncl_gt,
                           atol=0.005)

    def test_float_input(self):
        assert np.allclose(max_daylight(246, -20.0), 11.66559, atol=0.005)

    def test_list_input(self):
        assert np.allclose(max_daylight(self.jday_gt.tolist(),
                                        self.lat_gt.tolist()),
                           self.ncl_gt,
                           atol=0.005)

    def test_xarray_input(self):
        jday = xr.DataArray(self.jday_gt)
        lat = xr.DataArray(self.lat_gt)

        assert np.allclose(max_daylight(jday, lat), self.ncl_gt, atol=0.005)

    def test_dask_unchunked_input(self):
        jday = dask.array.from_array(self.jday_gt)
        lat = dask.array.from_array(self.lat_gt)

        out = self.client.submit(max_daylight, jday, lat).result()

        assert np.allclose(out, self.ncl_gt, atol=0.005)

    def test_dask_chunked_input(self):
        jday = dask.array.from_array(self.jday_gt, chunks='auto')
        lat = dask.array.from_array(self.lat_gt, chunks='auto')

        out = self.client.submit(max_daylight, jday, lat).result()

        assert np.allclose(out, self.ncl_gt, atol=0.005)

    def test_input_dim(self):
        self.assertRaises(ValueError, max_daylight,
                          np.arange(4).reshape(2, 2),
                          np.arange(4).reshape(2, 2))

    def test_lat_bound_warning(self):
        self.assertWarns(UserWarning, max_daylight, 10, 56)

    def test_lat_bound_second_warning(self):
        self.assertWarns(UserWarning, max_daylight, 10, 67)


class Test_psychrometric_constant(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # get ground truth from ncl run netcdf file
        try:
            ncl_xr_gt = xr.open_dataarray(
                "psychro_fao56_output.nc"
            )  # Generated by running ncl_tests/test_psychro_fao56.ncl
        except:
            ncl_xr_gt = xr.open_dataarray("test/psychro_fao56_output.nc")

        # set up ground truths
        cls.ncl_gt = np.asarray(ncl_xr_gt)

        cls.pressure_gt = np.arange(1, 101, 1)

        # make client to reference in subsequent tests
        cls.client = dd.Client()

    def test_numpy_input(self):
        assert np.allclose(psychrometric_constant(self.pressure_gt),
                           self.ncl_gt,
                           atol=0.005)

    def test_float_input(self):
        pressure = 81.78
        expected = 0.05434634
        assert np.allclose(psychrometric_constant(pressure),
                           expected,
                           atol=0.005)

    def test_list_input(self):
        assert np.allclose(psychrometric_constant(self.pressure_gt.tolist()),
                           self.ncl_gt.tolist(),
                           atol=0.005)

    def test_multi_dimensional_input(self):
        assert np.allclose(psychrometric_constant(
            self.pressure_gt.reshape(2, 50)),
                           self.ncl_gt.reshape(2, 50),
                           atol=0.005)

    def test_xarray_input(self):
        pressure = xr.DataArray(self.pressure_gt)
        expected = xr.DataArray(self.ncl_gt)

        assert np.allclose(psychrometric_constant(pressure),
                           expected,
                           atol=0.005)

    def test_dask_compute(self):
        pressure = xr.DataArray(self.pressure_gt).chunk(10)

        assert np.allclose(psychrometric_constant(pressure),
                           self.ncl_gt,
                           atol=0.005)

    def test_dask_lazy(self):
        pressure = xr.DataArray(self.pressure_gt).chunk(10)

        assert isinstance((psychrometric_constant(pressure)).data,
                          dask.array.Array)


class Test_saturation_vapor_pressure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # get ground truth from ncl run netcdf file
        try:
            ncl_xr_gt = xr.open_dataarray(
                "satvpr_temp_fao56_output.nc"
            )  # Generated by running ncl_tests/test_satvpr_temp_fao56.ncl
        except:
            ncl_xr_gt = xr.open_dataarray("test/satvpr_temp_fao56_output.nc")

        # set up ground truths
        cls.ncl_gt = np.asarray(ncl_xr_gt)

        cls.temp_gt = np.arange(1, 101, 1)

        # make client to reference in subsequent tests
        cls.client = dd.Client()

    def test_numpy_input(self):
        assert np.allclose(saturation_vapor_pressure(self.temp_gt,
                                                     tfill=1.0000000e+20),
                           self.ncl_gt,
                           atol=0.005)

    def test_float_input(self):
        degf = 59
        expected = 1.70535
        assert np.allclose(saturation_vapor_pressure(degf),
                           expected,
                           atol=0.005)

    def test_list_input(self):
        assert np.allclose(saturation_vapor_pressure(self.temp_gt.tolist(),
                                                     tfill=1.0000000e+20),
                           self.ncl_gt.tolist(),
                           atol=0.005)

    def test_multi_dimensional_input(self):
        assert np.allclose(saturation_vapor_pressure(self.temp_gt.reshape(
            2, 50),
                                                     tfill=1.0000000e+20),
                           self.ncl_gt.reshape(2, 50),
                           atol=0.005)

    def test_xarray_input(self):
        tempf = xr.DataArray(self.temp_gt)
        expected = xr.DataArray(self.ncl_gt)

        assert np.allclose(saturation_vapor_pressure(tempf,
                                                     tfill=1.0000000e+20),
                           expected,
                           atol=0.005)

    def test_dask_compute(self):
        tempf = xr.DataArray(self.temp_gt).chunk(10)

        assert np.allclose(saturation_vapor_pressure(tempf,
                                                     tfill=1.0000000e+20),
                           self.ncl_gt,
                           atol=0.005)

    def test_dask_lazy(self):
        tempf = xr.DataArray(self.temp_gt).chunk(10)

        assert isinstance((saturation_vapor_pressure(tempf,
                                                     tfill=1.0000000e+20)).data,
                          dask.array.Array)


class Test_saturation_vapor_pressure_slope(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # get ground truth from ncl run netcdf file
        try:
            ncl_xr_gt = xr.open_dataarray(
                "satvpr_slope_fao56_output.nc"
            )  # Generated by running ncl_tests/test_satvpr_slope_fao56.ncl
        except:
            ncl_xr_gt = xr.open_dataarray("test/satvpr_slope_fao56_output.nc")

        # set up ground truths
        cls.ncl_gt = np.asarray(ncl_xr_gt)

        cls.temp_gt = np.arange(1, 101, 1)

        # make client to reference in subsequent tests
        cls.client = dd.Client()

    def test_numpy_input(self):
        assert np.allclose(saturation_vapor_pressure_slope(self.temp_gt),
                           self.ncl_gt,
                           equal_nan=True)

    def test_float_input(self):
        degf = 67.55
        expected = 0.142793
        assert np.allclose(saturation_vapor_pressure_slope(degf),
                           expected,
                           atol=0.005)

    def test_list_input(self):
        assert np.allclose(saturation_vapor_pressure_slope(
            self.temp_gt.tolist()),
                           self.ncl_gt.tolist(),
                           equal_nan=True)

    def test_multi_dimensional_input(self):
        assert np.allclose(saturation_vapor_pressure_slope(
            self.temp_gt.reshape(2, 50)),
                           self.ncl_gt.reshape(2, 50),
                           atol=0.005,
                           equal_nan=True)

    def test_xarray_input(self):
        tempf = xr.DataArray(self.temp_gt)
        expected = xr.DataArray(self.ncl_gt)

        assert np.allclose(saturation_vapor_pressure_slope(tempf),
                           expected,
                           atol=0.005,
                           equal_nan=True)

    def test_dask_compute(self):
        tempf = xr.DataArray(self.temp_gt).chunk(10)

        assert np.allclose(saturation_vapor_pressure_slope(tempf),
                           self.ncl_gt,
                           atol=0.005,
                           equal_nan=True)

    def test_dask_lazy(self):
        tempf = xr.DataArray(self.temp_gt).chunk(10)

        assert isinstance((saturation_vapor_pressure_slope(tempf)).data,
                          dask.array.Array)
