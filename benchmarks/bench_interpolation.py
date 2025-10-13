import numpy as np
import xarray as xr
import geocat.datafiles as gdf

from geocat.comp import (
    interp_hybrid_to_pressure,
    interp_sigma_to_hybrid,
    interp_multidim,
)


class Bench_interp_hybrid_to_pressure_extrap_temp:
    def setup(self):
        # Load test data
        try:
            self.ds_ccsm = xr.open_dataset(
                gdf.get("netcdf_files/ccsm35.h0.0021-01.demo.nc"), decode_times=False
            )
        except Exception:
            self.ds_ccsm = xr.open_dataset(
                "test/ccsm35.h0.0021-01.demo.nc", decode_times=False
            )

        # Set up input parameters similar to test class
        self.temp_in = self.ds_ccsm.T[:, :, :3, :2]
        self.press_in = self.ds_ccsm.PS[:, :3, :2]
        self._hyam = self.ds_ccsm.hyam
        self._hybm = self.ds_ccsm.hybm
        self._p0 = 1000 * 100  # reference pressure in Pa
        self.new_levels = np.asarray([500, 925, 950, 1000])
        self.new_levels *= 100  # new levels in Pa
        self.phis = self.ds_ccsm.PHIS[:, :3, :2]

    def time_interp_hybrid_to_pressure_extrap_temp(self):
        interp_hybrid_to_pressure(
            self.temp_in,
            self.press_in,
            self._hyam,
            self._hybm,
            p0=self._p0,
            new_levels=self.new_levels,
            method="linear",
            extrapolate=True,
            variable='temperature',
            t_bot=self.temp_in.isel(lev=-1, drop=True),
            phi_sfc=self.phis,
        )

    def peakmem_interp_hybrid_to_pressure_extrap_temp(self):
        interp_hybrid_to_pressure(
            self.temp_in,
            self.press_in,
            self._hyam,
            self._hybm,
            p0=self._p0,
            new_levels=self.new_levels,
            method="linear",
            extrapolate=True,
            variable='temperature',
            t_bot=self.temp_in.isel(lev=-1, drop=True),
            phi_sfc=self.phis,
        )


class Bench_interp_hybrid_to_pressure:
    def setup(self):
        # Load test data
        try:
            self.ds_atmos = xr.open_dataset(
                gdf.get("netcdf_files/atmos.nc"), decode_times=False
            )
        except Exception:
            self.ds_atmos = xr.open_dataset("test/atmos.nc", decode_times=False)

        # Set up input parameters similar to test class
        self._hyam = self.ds_atmos.hyam
        self._hybm = self.ds_atmos.hybm
        self._p0 = 1000.0 * 100  # Pa

        self.data = self.ds_atmos.U[0, :, :, :]
        self.ps = self.ds_atmos.PS[0, :, :]
        self.pres3d = np.asarray([1000, 950, 800, 700, 600, 500, 400, 300, 200]) * 100

    def time_interp_hybrid_to_pressure(self):
        interp_hybrid_to_pressure(
            self.data,
            self.ps,
            self._hyam,
            self._hybm,
            p0=self._p0,
            new_levels=self.pres3d,
            method="log",
        )

    def peakmem_interp_hybrid_to_pressure(self):
        interp_hybrid_to_pressure(
            self.data,
            self.ps,
            self._hyam,
            self._hybm,
            p0=self._p0,
            new_levels=self.pres3d,
            method="log",
        )


class Bench_interp_sigma_to_hybrid:
    def setup(self):
        # Open the netCDF data file "u.89335.1.nc" and read in input data
        try:
            ds_u = xr.open_dataset(
                gdf.get("netcdf_files/u.89335.1_subset_time361.nc"), decode_times=False
            )
        except Exception:
            ds_u = xr.open_dataset(
                "test/u.89335.1_subset_time361.nc", decode_times=False
            )

        try:
            ds_ps = xr.open_dataset(
                gdf.get("netcdf_files/ps.89335.1.nc"), decode_times=False
            )
        except Exception:
            ds_ps = xr.open_dataset("test/ps.89335.1.nc", decode_times=False)

        self.hyam = xr.DataArray([0.0108093, 0.0130731, 0.03255911, 0.0639471])
        self.hybm = xr.DataArray([0.0108093, 0.0173664, 0.06069280, 0.1158237])

        self.u = ds_u.u[:, 0:3, 0:2]
        self.ps = ds_ps.ps[361, 0:3, 0:2] * 100  # Pa
        self._p0 = 1000.0 * 100  # Pa
        self.sigma = ds_ps.sigma

    def time_interp_sigma_to_hybrid_1d(self):
        interp_sigma_to_hybrid(
            self.u[:, 0, 0],
            self.sigma,
            self.ps[0, 0],
            self.hyam,
            self.hybm,
            p0=self._p0,
            method="linear",
        )

    def time_interp_sigma_to_hybrid_3d(self):
        interp_sigma_to_hybrid(
            self.u,
            self.sigma,
            self.ps,
            self.hyam,
            self.hybm,
            p0=self._p0,
            method="linear",
        )


class Bench_interp_multidim:
    def setup(self):
        self.test_input = xr.load_dataset(
            gdf.get("netcdf_files/spherical_noise_input.nc")
        )['spherical_noise']
        self.test_output = xr.load_dataset(
            gdf.get("netcdf_files/spherical_noise_output.nc")
        )['spherical_noise']

    def time_interp_multidim_chunk(self):
        interp_multidim(
            self.test_input.chunk(2),
            self.test_output.coords['lat'],
            self.test_output.coords['lon'],
        )

    def peakmem_interp_multidim_chunk(self):
        interp_multidim(
            self.test_input.chunk(2),
            self.test_output.coords['lat'],
            self.test_output.coords['lon'],
        )
