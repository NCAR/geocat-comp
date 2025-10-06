from geocat.comp import interp_hybrid_to_pressure
import numpy as np
import xarray as xr
import geocat.datafiles as gdf


class bench_interp:
    """Initial benchmarks for interpolation module"""

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
        """Benchmark basic interpolation"""
        interp_hybrid_to_pressure(
            self.data,
            self.ps,
            self._hyam,
            self._hybm,
            p0=self._p0,
            new_levels=self.pres3d,
            method="log",
        )
