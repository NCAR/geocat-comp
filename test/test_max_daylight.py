import unittest
import numpy as np
import xarray as xr
import dask.array as da
import dask.distributed as dd
from dask.array.core import map_blocks

from geocat.comp.crop import (max_daylight)


class Test_max_daylight(unittest.TestCase):
    # get ground truth from ncl run netcdf file
    ncl_gt = np.asarray(xr.open_dataarray('./ncl_tests/max_daylight_test.nc'))
    jday_gt = np.linspace(1, 365, num=365)
    lat_gt = np.linspace(-66, 66, num=133)

    def test_numpy_input(self):
        assert np.allclose(max_daylight(self.jday_gt, self.lat_gt), self.ncl_gt, atol=0.1)

