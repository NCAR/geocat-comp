import numpy as np
import xarray as xr
import geocat.comp

import sys
import time
import unittest as ut

plev = np.array( [1000.,950.,900.,850.,800.,750.,700.,650.,600.,
		    550.,500.,450.,400.,350.,300.,250.,200.,
		    175.,150.,125.,100., 80., 70., 60., 50.,
		    40., 30., 25., 20., 10.] ) # units hPa
# convert hPa to Pa
plev = plev * 100.0;
psfc = 101800.0 # Units of Pa

#Expected Output
expected_dp = np.array( [4300., 5000., 5000., 5000., 5000., 5000., 5000., 5000.,
			 5000., 5000., 5000., 5000., 5000., 5000., 5000., 5000.,
			 3750., 2500., 2500., 2500., 2250., 1500., 1000., 1000.,
			 1000., 1000., 750., 500., 750., 500.] )

class Test_rcm2points_float64(ut.TestCase):
    def test_rcm2points_float64(self):
        result_dp = geocat.comp.dpres_plevel(plev, psfc)
        np.testing.assert_array_equal(expected_dp, result_dp.values)

class Test_dpres_plevel_float32(ut.TestCase):
    def test_dpres_plevel_float32(self):
        plev_asfloat32 = plev.astype(np.float32)
        result_dp = geocat.comp.dpres_plevel(plev_asfloat32, psfc)
        np.testing.assert_array_equal(expected_dp, result_dp.values)
