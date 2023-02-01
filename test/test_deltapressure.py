from unittest import TestCase
import xarray as xr
import numpy as np
from geocat.comp.delta_pressure import _calc_deltapressure_1D, calc_deltapressure


class TestDeltaPressure(TestCase):
    pressure_lev = [1, 2, 50, 100, 200, 500, 700, 1000]
    pressure_lev_da = xr.DataArray(pressure_lev)
    pressure_lev_da.attrs = {
        "long name": "pressure level",
        "units": "hPa",
        "direction": "descending"
    }

    surface_pressure_scalar = 1018

    surface_pressure_1D = [1018, 1019]
    coords = {'lon': [5, 6]}
    dims = ["lon"]
    attrs = {"long name": "surface pressure", "units": "hPa"}
    surface_pressure_1D_da = xr.DataArray(surface_pressure_1D,
                                          coords=coords,
                                          dims=dims,
                                          attrs=attrs)

    surface_pressure_2D = [[1018, 1019], [1017, 1019.5]]
    coords = {'lat': [3, 4], 'lon': [5, 6]}
    dims = ["lat", "lon"]
    surface_pressure_2D_da = xr.DataArray(surface_pressure_2D,
                                          coords=coords,
                                          dims=dims,
                                          attrs=attrs)

    surface_pressure_3D = [[[1018, 1019], [1017, 1019.5]],
                           [[1019, 1020], [1018, 1020.5]]]
    coords = {'time': [1, 2], 'lat': [3, 4], 'lon': [5, 6]}
    dims = ["time", "lat", "lon"]
    surface_pressure_3D_da = xr.DataArray(surface_pressure_3D,
                                          coords=coords,
                                          dims=dims,
                                          attrs=attrs)

    def test_deltapressure_1D(self):
        pressure_top = min(self.pressure_lev)
        delta_pressure = _calc_deltapressure_1D(self.pressure_lev,
                                                self.surface_pressure_scalar)
        self.assertEqual(sum(delta_pressure),
                         self.surface_pressure_scalar - pressure_top)

    def test_missing_surface_pressure_warning(self):
        surface_pressure_missing = NaN
        with self.assertWarns(Warning):
            delta_pressure = _calc_deltapressure_1D(self.pressure_lev,
                                                    surface_pressure_missing)

    def test_negative_pressure_warning(self):
        pressure_lev_negative = self.pressure_lev.copy()
        pressure_lev_negative[0] = -5
        with self.assertWarns(Warning):
            delta_pressure = _calc_deltapressure_1D(
                pressure_lev_negative, self.surface_pressure_scalar)

    def test_relative_pressure_warning(self):
        surface_pressure_low = 0.5
        with self.assertWarns(Warning):
            delta_pressure = _calc_deltapressure_1D(self.pressure_lev,
                                                    surface_pressure_low)

    def test_convert_numpy_error(self):
        surface_pressure_long = 40 * 2 * 10**30
        self.assertRaises(
            AttributeError,
            calc_deltapressure(self.pressure_lev, surface_pressure_long))

        pressure_lev_long = 40 * 2 * 10**30
        self.assertRaises(
            AttributeError,
            calc_deltapressure(pressure_lev_long, self.surface_pressure_1D))

    def test_4_dimensions(self):
        surface_pressure_4D = [[[[1018, 1019], [1017, 1019.5]],
                                [[1019, 1020], [1018, 1020.5]]],
                               [[[1018, 1019], [1017, 1019.5]],
                                [[1019, 1020], [1018, 1020.5]]]]
        with self.assertWarns(Warning):
            delta_pressure = calc_deltapressure(self.pressure_lev,
                                                surface_pressure_4D)

    def test_output_type(self):
        delta_pressure_da = calc_deltapressure(self.pressure_lev_da,
                                               self.surface_pressure_3D_da)
        assert type(delta_pressure_da) == xr.DataArray

        delta_pressure_np = calc_deltapressure(self.pressure_lev,
                                               self.surface_pressure_3D)
        assert type(delta_pressure_np) == np.Array

    def test_output_dimensions(self):
        delta_pressure_scalar = calc_deltapressure(self.pressure_lev,
                                                   self.surface_pressure_scalar)
        assert delta_pressure_scalar.shape == (8)

        delta_pressure_1D = calc_deltapressure(self.pressure_lev,
                                               self.surface_pressure_1D)
        assert delta_pressure_1D.shape == (2, 8)

        delta_pressure_2D = calc_deltapressure(self.pressure_lev,
                                               self.surface_pressure_2D)
        assert delta_pressure_2D.shape == (2, 2, 8)

        delta_pressure_3D = calc_deltapressure(self.pressure_lev,
                                               self.surface_pressure_3D)
        assert delta_pressure_3D.shape == (2, 2, 2, 8)

    def test_output_attrs(self):
        delta_pressure_da = calc_deltapressure(self.pressure_lev_da,
                                               self.surface_pressure_3D_da)
        assert delta_pressure_da.attrs.pop(
            "long name") == self.pressure_lev_da.attrs.pop("long name")

    def test_output_coords(self):
        delta_pressure_da = calc_deltapressure(self.pressure_lev_da,
                                               self.surface_pressure_3D_da)
        for item in self.surface_pressure_3D_da.coords:
            self.assertIn(item, delta_pressure_da.coords)
        for item in self.pressure_lev_da.coords:
            self.assertIn(item, delta_pressure_da.coords)
