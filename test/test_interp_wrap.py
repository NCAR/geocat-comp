import sys

import cftime
import numpy as np
import numpy.ma as ma
import unittest
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp import interp_wrap
else:
    from geocat.comp import interp_wrap

# input type = np array, data array
# dimension = 1d or 2d
# data type = int, float32, float64
# special cases = cyclic, missing_py, nan, masked
# dask = chunked or unchunked

# test np.ndarray input
# # with normal - 1d and 2d, 1d - nan and masked, 2d - nan and masked, dask - chunked and unchunked, cyclic, msg_py

# test xr.DataArray input
# with normal - 1d and 2d, 1d - nan and masked, 2d - nan and masked, dask - chunked and unchunked, cyclic, msg_py

# errors
# xr.DataArray w/o coordinates - 1d and 2d, empty arrays (coords/data), mismatched data sizes, chunking incorrectly

# generate data for testing
data_in = np.asarray([0.8273209 , 0.97402306, 0.49829999, 0.97088771, 0.14578397,
0.18818199, 0.69280023, 0.37583487, 0.72259201, 0.34248486])
lon_in = np.arange(0, 360, 36)
lon_out = np.arange(0, 360, 18)
data_out = [0.8273209 , 0.90067198, 0.97402306, 0.736161525, 0.49829999, 0.73459385, 0.97088771, 0.55833584, 0.14578397,
0.16698298, 0.18818199, 0.44049111, 0.69280023, 0.53431755, 0.37583487, 0.54921344, 0.72259201, 0.532538435, 0.34248486]

data_in_2d = np.asarray([0.8273209 , 0.97402306, 0.49829999, 0.97088771, 0.14578397,
0.18818199, 0.69280023, 0.37583487, 0.72259201, 0.34248486],
                [2.8273209 , 2.97402306, 2.49829999, 2.97088771, 2.14578397,
2.18818199, 2.69280023, 2.37583487, 2.72259201, 2.34248486])
lat_in = np.asarray([0, 2])
lat_out = np.asarray([0, 1, 2])
data_out_2d = np.asarray([[0.8273209 , 0.90067198, 0.97402306, 0.736161525, 0.49829999, 0.73459385, 0.97088771, 0.55833584, 0.14578397,
            0.16698298, 0.18818199, 0.44049111, 0.69280023, 0.53431755, 0.37583487, 0.54921344, 0.72259201, 0.532538435, 0.34248486],

            [1.8273209 , 1.90067198, 1.97402306, 1.736161525, 1.49829999, 1.73459385, 1.97088771, 1.55833584, 1.14578397,
            1.16698298, 1.18818199, 1.44049111, 1.69280023, 1.53431755, 1.37583487, 1.54921344, 1.72259201, 1.532538435, 1.34248486],

            [2.8273209 , 2.90067198, 2.97402306, 2.736161525, 2.49829999, 2.73459385, 2.97088771, 2.55833584, 2.14578397,
            2.16698298, 2.18818199, 2.44049111, 2.69280023, 2.53431755, 2.37583487, 2.54921344, 2.72259201, 2.532538435, 2.34248486]])

msg_py = 55.55

class Test_numpy_input(unittest.TestCase):

    def test_numpy_1d(self):
        np.testing.assert_array_equal(data_out, interp_wrap(data_in, lon_in=lon_in, lon_out=lon_out))

    def test_numpy_1d_cyclic(self):
        lon_out = np.append(lon_out, lon_in[-1] + 1)
        data_out = data_out.append(data_out, data_in[-1]+(data_in[0]*(lon_in[1]-lon_in[0]-1)))
        np.testing.assert_array_equal(data_out, interp_wrap(lon_out=lon_out, lon_in=lon_in, cyclic=True))

    def test_numpy_1d_msg(self):
        rand = (np.random.rand(1)*len(data_in)).astype(int)[0]
        data_in[rand] = msg_py 
        data_out[rand*2] = msg_py
        data_out[rand*2-1] = msg_py
        data_out[rand*2+1] = msg_py
        np.testing.assert_array_equal(data_out, interp_wrap(data_in, lon_out=lon_out, lon_in=lon_in, msg_py=msg_py))

    def test_numpy_1d_mask(self):
        rand = (np.random.rand(1)*len(data_in)).astype(int)[0]
        mask = np.zeroes(len(data_in))
        mask_out = np.zeroes(len(data_out))
        mask[rand] = 1
        mask_out[rand*2] = 1
        mask_out[rand*2-1] = 1
        mask_out[rand*2+1] = 1
        np.testing.assert_array_equal(ma.masked_array(data_out, mask_out), interp_wrap(data_in=ma.masked_array(data_in, mask), lon_out=lon_out, lon_in=lon_in))
    
    def test_numpy_1d_nan(self):
        rand = (np.random.rand(1)*len(data_in)).astype(int)[0]
        data_in[rand] = np.nan
        data_out[rand*2] = np.nan
        data_out[rand*2-1] = np.nan
        data_out[rand*2+1] = np.nan
        np.testing.assert_array_equal(data_out, interp_wrap(data_in=data_in, lat_in=lat_in, lat_out=lat_out, lon_out=lon_out, lon_in=lon_in))

    def test_numpy_2d_int(self):
        data_in = (data_in_2d*10).astype(int)
        data_out = (data_out_2d*10).astype(int)
        np.testing.assert_array_equal(data_out, interp_wrap(data_in, lon_in=lon_in, lat_in=lat_in, lon_out=lon_out, lat_out=lat_out))

    def test_numpy_2d_float32(self):
        data_in = (data_in_2d).astype(np.float32)
        data_out = (data_out_2d).astype(np.float32)
        np.testing.assert_array_equal(data_out, interp_wrap(data_in, lon_in=lon_in, lat_in=lat_in, lon_out=lon_out, lat_out=lat_out))

    def test_numpy_2d_float64(self):
        data_in = (data_in_2d).astype(np.float32)
        data_out = (data_out_2d).astype(np.float32)
        np.testing.assert_array_equal(data_out, interp_wrap(data_in, lon_in=lon_in, lat_in=lat_in, lon_out=lon_out, lat_out=lat_out))

    def test_numpy_2d_cyclic(self):
        lon_out = np.append(lon_out, lon_in[-1] + 1)
        for i in range(len(data_out)):
            data_out = data_out.append(data_out, data_in[i][-1]+(data_in[i][0]*(lon_in[1]-lon_in[0]-1)))
        np.testing.assert_array_equal(data_out, interp_wrap(data_in, lat_in=lat_in, lat_out=lat_out, lon_out=lon_out, lon_in=lon_in, cyclic=True))

    def test_numpy_2d_missing(self):
        randR = (np.random.rand(1)*(len(data_in)-1)).astype(int)[0]
        randC = (np.random.rand(1)*(len(data_in[0])-1)).astype(int)[0]
        data_in[randR][randC] = msg_py 
        data_out[randR*2][randC*2] = msg_py
        data_out[(randR*2)-1][randC*2] = msg_py
        data_out[randR*2][(randC*2)-1] = msg_py
        data_out[(randR*2)+1][randC*2] = msg_py
        data_out[randR*2][(randC*2)+1] = msg_py
        data_out[(randR*2)+1][(randC*2)+1] = msg_py
        data_out[(randR*2)-1][(randC*2)-1] = msg_py
        data_out[(randR*2)+1][(randC*2)-1] = msg_py
        data_out[(randR*2)-1][(randC*2)+1] = msg_py
        np.testing.assert_array_equal(data_out, interp_wrap(data_in, lon_out=lon_out, lon_in=lon_in, msg_py=msg_py))


    def test_numpy_2d_nan(self):
        randR = (np.random.rand(1)*(len(data_in)-1)).astype(int)[0]
        randC = (np.random.rand(1)*(len(data_in[0])-1)).astype(int)[0]
        data_in[randR][randC] = np.nan 
        data_out[randR*2][randC*2] = np.nan
        data_out[(randR*2)-1][randC*2] = np.nan
        data_out[randR*2][(randC*2)-1] = np.nan
        data_out[(randR*2)+1][randC*2] = np.nan
        data_out[randR*2][(randC*2)+1] = np.nan
        data_out[(randR*2)+1][(randC*2)+1] = np.nan
        data_out[(randR*2)-1][(randC*2)-1] = np.nan
        data_out[(randR*2)+1][(randC*2)-1] = np.nan
        data_out[(randR*2)-1][(randC*2)+1] = np.nan
        np.testing.assert_array_equal(data_out, interp_wrap(data_in, lat_in=lat_in, lat_out=lat_out, lon_out=lon_out, lon_in=lon_in))

    def test_numpy_2d_mask(self):
        randR = (np.random.rand(1)*(len(data_in)-1)).astype(int)[0]
        randC = (np.random.rand(1)*(len(data_in[0])-1)).astype(int)[0]
        mask = np.zeroes(len(data_in_2d))
        mask_out = np.zeroes(len(data_out_2d))
        mask[randR][randC] = 1 
        mask_out[randR*2][randC*2] = 1
        mask_out[(randR*2)-1][randC*2] = 1
        mask_out[randR*2][(randC*2)-1] = 1
        mask_out[(randR*2)+1][randC*2] = 1
        mask_out[randR*2][(randC*2)+1] = 1
        mask_out[(randR*2)+1][(randC*2)+1] = 1
        mask_out[(randR*2)-1][(randC*2)-1] = 1
        mask_out[(randR*2)+1][(randC*2)-1] = 1
        mask_out[(randR*2)-1][(randC*2)+1] = 1
        np.testing.assert_array_equal(ma.masked_array(data_out, mask_out), interp_wrap(ma.masked_array(data_in, mask), lat_in=lat_in, lat_out=lat_out, lon_out=lon_out, lon_in=lon_in))

"""
class Test_xarray_input(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # generate data for testing
        data_in = np.random.rand(96, 4, 64)
        lon_in = np.random.rand(64)

    def test_xarray_1d(self):


    def test_xarray_1d_cyclic(self):
    

    def test_xarray_1d_msg(self):


    def test_xarray_2d_int(self):
        

    def test_xarray_2d_float32(self):
        

    def test_xarray_2d_float64(self):
        

    def test_xarray_2d_cyclic(self):
        

    def test_xarray_2d_missing(self):
        

    def test_xarray_1d_nan(self):
    

    def test_xarray_2d_nan(self):
        

    def test_xarray_1d_mask(self):
    

    def test_xarray_2d_mask(self):




"""