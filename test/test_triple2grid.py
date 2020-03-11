import numpy as np
import xarray as xr
import geocat.comp

import sys
import time
import unittest as ut

def prepare_masked(fo_masked, fo):
    for i in range(fo.shape[0]):
        for j in range(fo.shape[1]):
            fo_masked[i, j] = np.diag(fo[i,j,::2,::2])
    
# Size of the grids
n = 12
m = 12

in_size_M = int(m / 2) + 1
x = np.zeros((in_size_M))
y = np.zeros((in_size_M))
for i in range(in_size_M):
    x[i] = float(i)
    y[i] = float(i)    

xgrid_npoints_M = m + 1
ygrid_npoints_N = n + 1
xgrid = np.zeros((xgrid_npoints_M))
ygrid = np.zeros((ygrid_npoints_N))
for i in range(xgrid_npoints_M):
    xgrid[i] = float(i) * 0.5
for i in range(ygrid_npoints_N):
    ygrid[i] = float(i) * 0.5
    
#create and fill input data array (fi)
fi = np.random.randn(3, in_size_M, in_size_M)
fo_masked = np.zeros((3, in_size_M, in_size_M))
fi_asfloat32 = fi.astype(np.float32)

class Test_triple2grid_float64(ut.TestCase):
    def test_triple2grid_float64(self):
        fo = geocat.comp.triple2grid(x, y, fi, xgrid, ygrid)
        prepare_masked(fo_masked, fo.values)
        np.testing.assert_array_equal(fi, fo_masked)
        
class Test_triple2grid_float32(ut.TestCase):
    def test_triple2grid_float32(self):
        fo = geocat.comp.triple2grid(x.astype(np.float32), y.astype(np.float32), fi_asfloat32, xgrid.astype(np.float32), ygrid.astype(np.float32))
        prepare_masked(fo_masked, fo.values)
        np.testing.assert_array_equal(fi_asfloat32, fo_masked)
