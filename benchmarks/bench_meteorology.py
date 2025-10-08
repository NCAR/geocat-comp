import numpy as np
import xarray as xr

from geocat.comp.meteorology import (
    dewtemp,
    heat_index,
    relhum,
    actual_saturation_vapor_pressure,
    max_daylight,
    psychrometric_constant,
    saturation_vapor_pressure,
    saturation_vapor_pressure_slope,
    delta_pressure,
)


class Bench_dewtemp:
    def setup(self):
        # fmt: off
        self. rh_def = [75.0, 60.0, 61.1, 76.7, 90.5, 89.8, 78.3, 76.5, 46.0, 55.0, 63.8, 53.2, 42.9, 41.7, 51.0, 70.6, 50.0, 50.0]
        # fmt: on

    def time_dewtemp(self):
        xr.DataArray(self.rh_def)


class Bench_heat_index:
    def setup(self):
        self.t = np.array([70, 75, 80, 85, 90, 95, 100, 105, 110, 115])
        self.rh = np.array([10, 75, 15, 80, 65, 25, 30, 40, 50, 5])

    def time_heat_index(self):
        heat_index(self.t, self.rh, True)


class Bench_relhum:
    def setup(self):
        # fmt: off
        self.p_def = [
            100800, 100000, 95000, 90000, 85000, 80000, 75000, 70000, 65000, 60000, 55000, 50000, 45000, 40000, 35000, 30000,
            25000, 20000, 17500, 15000, 12500, 10000, 8000, 7000, 6000, 5000, 4000, 3000, 2500, 2000
        ]

        self.t_def = [
            302.45, 301.25, 296.65, 294.05, 291.55, 289.05, 286.25, 283.25, 279.85, 276.25, 272.65, 268.65, 264.15, 258.35, 251.65,
            243.45, 233.15, 220.75, 213.95, 206.65, 199.05, 194.65, 197.15, 201.55, 206.45, 211.85, 216.85, 221.45, 222.45, 225.65
        ]

        self.q_def = [
            0.02038, 0.01903, 0.01614, 0.01371, 0.01156, 0.0098, 0.00833, 0.00675, 0.00606, 0.00507, 0.00388, 0.00329, 0.00239,
            0.0017, 0.001, 0.0006, 0.0002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        # fmt: on

    def time_relhum(self):
        relhum(self.t_def, self.q_def, self.p_def)


class Bench_actual_saturation_vapor_pressure:
    def setup(self):
        self.temp = np.arange(1, 101, 1)

    def time_actual_saturation_vapor_pressure(self):
        actual_saturation_vapor_pressure(self.temp, tfill=1.0000000e20)


class Bench_max_daylight:
    def setup(self):
        self.jday = np.linspace(1, 365, num=365)
        self.lat = np.linspace(-66, 66, num=133)

    def time_max_daylight(self):
        max_daylight(self.jday, self.lat)


class Bench_psychrometric_constant:
    def setup(self):
        self.pressure = np.arange(1, 101, 1)

    def time_psychrometric_constant(self):
        psychrometric_constant(self.pressure)


class Bench_saturation_vapor_pressure:
    def setup(self):
        self.temp = np.arange(1, 101, 1)

    def time_saturation_vapor_pressure(self):
        saturation_vapor_pressure(self.temp, tfill=1.0000000e20)


class Bench_saturation_vapor_pressure_slope:
    def setup(self):
        self.temp = np.arange(1, 101, 1)

    def time_saturation_vapor_pressure_slope(self):
        saturation_vapor_pressure_slope(self.temp)


class Bench_delta_pressure:
    def setup(self):
        self.pressure_lev = np.array([1, 5, 100, 1000], dtype=np.float64)
        self.surface_pressure_scalar = 1018.0

    def time_delta_pressure(self):
        delta_pressure(self.pressure_lev, self.surface_pressure_scalar)
