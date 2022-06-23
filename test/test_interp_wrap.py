import unittest
import random

import numpy as np
import numpy.ma as ma
import xarray as xr

from src.geocat.comp import interp_wrap

class Test_manually_calc(unittest.TestCase):

    def setUpClass(cls):
        cls.data_in = np.asarray([0.8273209 , 0.97402306, 0.49829999, 0.97088771, 0.14578397, 0.18818199, 0.69280023, 0.37583487, 0.72259201, 0.34248486])
        cls.lon_in = np.arange(0, 360, 36)
        cls.lon_out = np.arange(0, 360-35, 18)

        cls.data_out = np.asarray([0.8273209, 0.90067198, 0.97402306, 0.736161525, 0.49829999, 0.73459385, 0.97088771, 0.55833584, 0.14578397, 0.16698298, 0.18818199, 0.44049111, 0.69280023, 0.53431755, 0.37583487, 0.54921344, 0.72259201, 0.532538435, 0.34248486])

        cls.data_in_2d = np.asarray([[0.8273209 , 0.97402306, 0.49829999, 0.97088771, 0.14578397, 0.18818199, 0.69280023, 0.37583487, 0.72259201, 0.34248486], [2.8273209 , 2.97402306, 2.49829999, 2.97088771, 2.14578397, 2.18818199, 2.69280023, 2.37583487, 2.72259201, 2.34248486], [4.8273209 , 4.97402306, 4.49829999, 4.97088771, 4.14578397, 4.18818199, 4.69280023, 4.37583487, 4.72259201, 4.34248486]])
        cls.lat_in = np.asarray([0, 2, 4])
        cls.lat_out = np.asarray([0, 1, 2, 3, 4])

        cls.data_out_2d = np.asarray([[0.8273209 , 0.90067198, 0.97402306, 0.736161525, 0.49829999, 0.73459385, 0.97088771, 0.55833584, 0.14578397, 0.16698298, 0.18818199, 0.44049111, 0.69280023, 0.53431755, 0.37583487, 0.54921344, 0.72259201, 0.532538435, 0.34248486], 
                    [1.8273209 , 1.90067198, 1.97402306, 1.736161525, 1.49829999, 1.73459385, 1.97088771, 1.55833584, 1.14578397, 1.16698298, 1.18818199, 1.44049111, 1.69280023, 1.53431755, 1.37583487, 1.54921344, 1.72259201, 1.532538435, 1.34248486],
                    [2.8273209 , 2.90067198, 2.97402306, 2.736161525, 2.49829999, 2.73459385, 2.97088771, 2.55833584, 2.14578397, 2.16698298, 2.18818199, 2.44049111, 2.69280023, 2.53431755, 2.37583487, 2.54921344, 2.72259201, 2.532538435, 2.34248486],
                    [3.8273209 , 3.90067198, 3.97402306, 3.736161525, 3.49829999, 3.73459385, 3.97088771, 3.55833584, 3.14578397, 3.16698298, 3.18818199, 3.44049111, 3.69280023, 3.53431755, 3.37583487, 3.54921344, 3.72259201, 3.532538435, 3.34248486],
                    [4.8273209 , 4.90067198, 4.97402306, 4.736161525, 4.49829999, 4.73459385, 4.97088771, 4.55833584, 4.14578397, 4.16698298, 4.18818199, 4.44049111, 4.69280023, 4.53431755, 4.37583487, 4.54921344, 4.72259201, 4.532538435, 4.34248486]])

        cls.data_in_1d_xr = xr.DataArray(cls.data_in,
                        dims=["lon"],
                        coords={
                            "lon": cls.lon_in
                        })

        cls.data_in_2d_xr = xr.DataArray(cls.data_in_2d,
                        dims=["x", "y"],
                        coords={
                            "x": cls.lat_in,
                            "y": cls.lon_in
                        })

        cls.msg_py = 55.55

    def test_numpy_1d(self):
        np.testing.assert_almost_equal(self.data_out, interp_wrap(self.data_in, lon_in=self.lon_in, lon_out=self.lon_out), 10)

    def test_numpy_1d_cyclic(self):
        x = np.append(self.lon_out, self.lon_in[-1] + 1)
        x_x1 = x[-1] - self.lon_in[-1]
        y2_y1 = self.data_in[0] - self.data_in[-1]
        x2_x1 = self.lon_in[-1]+(abs(self.lon_in[-1]-self.lon_in[-2])) - self.lon_in[-1]
        cyclic_point = self.data_in[-1] + (x_x1*y2_y1)/x2_x1
        odata = np.append(self.data_out, cyclic_point)
        np.testing.assert_almost_equal(odata, interp_wrap(self.data_in, lon_out=x, lon_in=self.lon_in, cyclic =True), 7)

    def test_numpy_1d_msg(self):
        rand = random.randint(1, len(self.data_in)-2)
        idata = self.data_in
        odata = self.data_out
        idata[rand] = self.msg_py 
        odata[rand*2] = self.msg_py
        odata[rand*2-1] = self.msg_py
        odata[rand*2+1] = self.msg_py
        np.testing.assert_almost_equal(odata, interp_wrap(idata, lon_out=self.lon_out, lon_in=self.lon_in, missing_val=self.msg_py), 10)

    def test_numpy_1d_mask(self):
        rand = random.randint(1, len(self.data_in)-2)
        mask = np.zeros(len(self.data_in))
        mask_out = np.zeros(len(self.data_out))
        mask[rand] = 1
        mask_out[rand*2] = 1
        mask_out[rand*2-1] = 1
        mask_out[rand*2+1] = 1
        np.testing.assert_almost_equal(ma.masked_array(self.data_out, mask_out), interp_wrap(data_in=ma.masked_array(self.data_in, mask), lon_out=self.lon_out, lon_in=self.lon_in), 10)
    
    def test_numpy_1d_nan(self):
        rand = random.randint(1, len(self.data_in)-2)
        idata = self.data_in
        odata = self.data_out
        idata[rand] = np.nan
        odata[rand*2] = np.nan
        odata[rand*2-1] = np.nan
        odata[rand*2+1] = np.nan
        np.testing.assert_almost_equal(odata, interp_wrap(idata, lon_out=self.lon_out, lon_in=self.lon_in), 10)

    def test_numpy_2d_float32(self):
        data_in = (self.data_in_2d).astype(np.float32)
        data_out = (self.data_out_2d).astype(np.float32)
        np.testing.assert_almost_equal(data_out, interp_wrap(data_in, lon_in=self.lon_in, lat_in=self.lat_in, lon_out=self.lon_out, lat_out=self.lat_out), 6)

    def test_numpy_2d_float64(self):
        data_in = (self.data_in_2d).astype(np.float64)
        data_out = (self.data_out_2d).astype(np.float64)
        np.testing.assert_almost_equal(data_out, interp_wrap(data_in, lon_in=self.lon_in, lat_in=self.lat_in, lon_out=self.lon_out, lat_out=self.lat_out), 6)

    def test_numpy_2d_cyclic(self):
        x = np.append(self.lon_out, self.lon_in[-1] + 1)
        odata = np.pad(self.data_out_2d, ((0, 0), (0, 1)))
        for i in [0, 1, 2]:
            x_x1 = x[-1] - self.lon_in[-1]
            y2_y1 = self.data_in_2d[i][0] - self.data_in_2d[i][-1]
            x2_x1 = self.lon_in[-1] + (abs(self.lon_in[-1]-self.lon_in[-2])) - self.lon_in[-1]
            cyclic_point = self.data_in[i][-1] + (x_x1*y2_y1)/x2_x1
            odata[i*2][-1] = cyclic_point
        np.testing.assert_almost_equal(odata, interp_wrap(self.data_in, lat_in=self.lat_in, lat_out=self.lat_out, lon_out=x, lon_in=self.lon_in, cyclic=True), 10)

    def test_numpy_2d_missing(self):
        randR = random.randint(1, len(self.data_in_2d)-2)
        randC = random.randint(1, len(self.data_in_2d[0])-2)
        idata = self.data_in_2d
        idata[randR][randC] = self.msg_py 
        odata = self.data_out_2d
        for i in [0, 1, 2, 3, 4]:
            for j in [(randC*2)-1, randC*2, (randC*2)+1, (randC*2)+2]:
                 odata[i][j] = self.msg_py
        np.testing.assert_almost_equal(odata, interp_wrap(idata, lon_out=self.lon_out, lat_out=self.lat_out, lat_in=self.lat_in, lon_in=self.lon_in, missing_val=self.msg_py), 7)

    def test_numpy_2d_nan(self):
        randR = random.randint(1, len(self.data_in_2d)-2)
        randC = random.randint(1, len(self.data_in_2d[0])-2)
        idata = self.data_in_2d
        idata[randR][randC] = np.nan 
        odata = self.data_out_2d
        for i in [0, 1, 2, 3, 4]:
            for j in [(randC*2)-1, randC*2, (randC*2)+1, (randC*2)+2]:
                odata[i][j] = np.nan
        np.testing.assert_almost_equal(odata, interp_wrap(idata, lat_in=self.lat_in, lat_out=self.lat_out, lon_out=self.lon_out, lon_in=self.lon_in), decimal=7)

    def test_numpy_2d_mask(self):
        randR = (np.random.rand(1)*(len(self.data_in_2d)-1)).astype(int)[0]
        randC = (np.random.rand(1)*(len(self.data_in_2d[0])-1)).astype(int)[0]
        mask = np.zeros(self.data_in_2d.shape)
        mask[randR][randC] = 1
        mask_out = np.zeros(self.data_out_2d.shape)
        for i in [0, 1, 2, 3, 4]:
            for j in [(randC*2)-1, randC*2, (randC*2)+1, (randC*2)+2]:
                 mask_out[i][j] = 1
        np.testing.assert_almost_equal(ma.masked_array(self.data_out_2d, mask_out), interp_wrap(ma.masked_array(self.data_in_2d, mask), lat_in=self.lat_in, lat_out=self.lat_out, lon_out=self.lon_out, lon_in=self.lon_in), decimal=10)

    def test_xarray_1d(self):
        np.testing.assert_almost_equal(self.data_out, interp_wrap(self.data_in_1d_xr, lon_out=self.lon_out), 10)
    
    def test_xarray_2d_float64(self):
        data_in = (self.data_in_2d).astype(np.float64)
        data_in_xr = xr.DataArray(data_in,
                                dims=["lat", "lon"],
                                coords={
                                    "lat": self.lat_in,
                                    "lon": self.lon_in
                                })
        data_out = (self.data_out_2d).astype(np.float64)
        np.testing.assert_almost_equal(data_out, interp_wrap(data_in_xr, lon_out=self.lon_out, lat_out=self.lat_out), 6)


class Test_larger_dataset(unittest.TestCase):

    def setUpClass(cls):
        input = xr.load_dataset("test/interpolation_test_data.nc")
        cls.input = input["__xarray_dataarray_variable__"]
        cls.two_x = xr.DataArray()
        cls.five_x = xr.DataArray()
        cls.ten_x = xr.DataArray()
        cls.input_chunked = xr.DataArray(np.random.rand(3, 96, 64, 64),
                                        dims=["time", "level", "lat", "lon"],
                                        coords={
                                            "lat": np.arange(0, 361, 10),
                                            "lon": np.arange(0, 181, 10)
                                        },
                                        chunks = {
                                            "time": 1,
                                            "level": 1,
                                            "lat": 36,
                                            "lon": 36
                                        })
    
    def test_2x_res(self):
        np.testing.assert_array_equal(self.two_x, interp_wrap(self.input, lon_out=np.arange(0, 361, 5), lat_out=np.arange(0, 181, 5)).values[::2])
    
    def test_5x_res(self):
        np.testing.assert_array_equal(self.five_x, interp_wrap(self.input, lon_out=np.arange(0, 361, 2), lat_out=np.arange(0, 181, 2)).values[::5])
    
    def test_10x_res(self):
        np.testing.assert_array_equal(self.ten_x, interp_wrap(self.input, lon_out=np.arange(0, 361, 1), lat_out=np.arange(0, 181, 1)).values[::10])
    
    def test_chunked(self):
        np.testing.assert_array_equal(self.input_chunked.values, interp_wrap(self.input_chunked, lat_in=np.arange(0, 361, 5), lon_in=np.arange(0, 181, 5)).values[::2])


class Test_errors(unittest.TestCase):
    
    def setUpClass(cls):
        cls.input_chunk = xr.DataArray(np.random.rand(3, 96, 64, 64))
        cls.data_in_1d = np.random.rand(10)
        cls.data_in_2d = np.random.rand((10, 10))
        cls.input_chunked = xr.DataArray(np.random.rand(3, 96, 64, 64),
                                dims=["time", "level", "lat", "lon"],
                                coords={
                                    "lat": np.arange(0, 361, 10),
                                    "lon": np.arange(0, 181, 10)
                                },
                                chunks = {
                                    "time": 1,
                                    "level": 1,
                                    "lat": 18,
                                    "lon": 18
                                })
    
    def test_coordinate_error_1d(self):
        np.testing.assert_raises(CoordinateError, 
                                interp_wrap, 
                                data_in=self.data_in_1d,
                                lon_out=np.arange(0, 10, 0.5))

    def test_coordinate_error_2d(self):
        np.testing.assert_raises(CoordinateError,
                                interp_wrap,
                                data_in=self.data_in_2d,
                                lon_out=np.arange(0, 10, 0.5),
                                lat_out=np.arange(0, 10, 0.5))

    def test_chunk_error(self):
        np.testing.assert_raises(ChunkError,
                                interp_wrap,
                                data_in=self.input_chunked,
                                lon_out=np.arange(0, 361, 5),
                                lat_out=np.arange(0, 361, 5))



# class Test_xarray_input(unittest.TestCase):

#     def test_xarray_1d(self):
#          np.testing.assert_almost_equal(self.data_out, interp_wrap(self.data_in_1d_xr, lon_out=self.lon_out), 10)

#     def test_xarray_1d_cyclic(self):
#         x = np.append(lon_out, lon_in[-1] + 1)
#         x_x1 = x[-1] - lon_in[-1]
#         y2_y1 = data_in[0] - data_in[-1]
#         x2_x1 = lon_in[-1]+(abs(lon_in[-1]-lon_in[-2])) - lon_in[-1]
#         cyclic_point = data_in[-1] + (x_x1*y2_y1)/x2_x1
#         odata = np.append(data_out, cyclic_point)
#         np.testing.assert_almost_equal(odata, interp_wrap(data_in_1d_xr, lon_out=x, cyclic =True), 7)

#     def test_xarray_1d_msg(self):
#         rand = random.randint(1, len(data_in)-2)
#         data_in[rand] = msg_py 
#         data_in_xr = xr.DataArray(data_in,
#                                 dims=["lon"],
#                                 coords={
#                                     "lon": lon_in
#                                 })
#         data_out[rand*2] = msg_py
#         data_out[rand*2-1] = msg_py
#         data_out[rand*2+1] = msg_py
#         np.testing.assert_almost_equal(data_out, interp_wrap(data_in_xr, lon_out=lon_out, missing_val=msg_py), 10)

#     def test_xarray_1d_nan(self):
#         rand = random.randint(1, len(data_in)-2)
#         data_in[rand] = np.nan
#         data_in_xr = xr.DataArray(data_in,
#                                 dims=["lon"],
#                                 coords={
#                                     "lon": lon_in
#                                 })
#         data_out[rand*2] = msg_py
#         data_out[rand*2] = np.nan
#         data_out[rand*2-1] = np.nan
#         data_out[rand*2+1] = np.nan
#         np.testing.assert_almost_equal(data_out, interp_wrap(data_in_xr, lon_out=lon_out), 10)

#     def test_xarray_1d_mask(self):
#         rand = random.randint(1, len(data_in)-2)
#         mask = np.zeros(len(data_in))
#         mask_out = np.zeros(len(data_out))
#         mask[rand] = 1
#         data_in_xr = xr.DataArray(ma.masked_array(data_in, mask),
#                                 dims=["lon"],
#                                 coords={
#                                     "lon": lon_in
#                                 })
#         data_out[rand*2] = msg_py
#         mask_out[rand*2] = 1
#         mask_out[rand*2-1] = 1
#         mask_out[rand*2+1] = 1
#         np.testing.assert_almost_equal(ma.masked_array(data_out, mask_out), interp_wrap(data_in_xr, lon_out=lon_out), 10)
    
#     def test_xarray_2d_float32(self):
#         data_in_xr = xr.DataArray((data_in_2d).astype(np.float32),
#                                 dims=["lat", "lon"],
#                                 coords={
#                                     "lat": lat_in,
#                                     "lon": lon_in
#                                 })
#         data_out = (data_out_2d).astype(np.float32)
#         np.testing.assert_almost_equal(data_out, interp_wrap(data_in_xr, lon_out=lon_out, lat_out=lat_out), 6)

#     def test_xarray_2d_float64(self):
#         data_in = (data_in_2d).astype(np.float64)
#         data_in_xr = xr.DataArray(data_in,
#                                 dims=["lat", "lon"],
#                                 coords={
#                                     "lat": lat_in,
#                                     "lon": lon_in
#                                 })
#         data_out = (data_out_2d).astype(np.float64)
#         np.testing.assert_almost_equal(data_out, interp_wrap(data_in_xr, lon_out=lon_out, lat_out=lat_out), 6)

#     def test_xarray_2d_cyclic(self):
#         x = np.append(lon_out, lon_in[-1] + 1)
#         odata = np.pad(data_out_2d, ((0, 0), (0, 1)))
#         for i in [0, 1, 2]:
#             #o_data = np.append(data_out_2d, data_in_2d[i][-1]+(data_in_2d[i][0]*(lon_in[1]-lon_in[0]-1)))
#             x_x1 = x[-1] - lon_in[-1]
#             y2_y1 = data_in_2d[i][0] - data_in_2d[i][-1]
#             x2_x1 = lon_in[-1] + (abs(lon_in[-1]-lon_in[-2])) - lon_in[-1]
#             cyclic_point = data_in_2d[i][-1] + (x_x1*y2_y1)/x2_x1
#             odata[i*2][-1] = cyclic_point
#         for i in [1, 3]:
#             x_x1 = lat_out[i] - lat_in[(int)(i/2)]
#             y2_y1 = data_in_2d[(int)(i/2)+1][-1] - data_in_2d[(int)(i/2)][-1]
#             x2_x1 = lat_in[(int)(i/2)+1] - lat_in[(int)(i/2)]
#             cyclic_point = data_in_2d[(int)(i/2)][-1] + (x_x1*y2_y1)/x2_x1
#             odata[i][-1] = cyclic_point
#         dataa = interp_wrap(data_in_2d_xr, lon_out=x, lon_in=lon_in, cyclic=True, lat_out=lat_out, lat_in=lat_in)
#         np.testing.assert_almost_equal(odata, interp_wrap(data_in_2d_xr, lon_out=x, lon_in=lon_in, cyclic=True, lat_out=lat_out, lat_in=lat_in), 10)

# #    def test_xarray_2d_missing(self):       
        

#     # def test_xarray_2d_nan(self):
        

#     # def test_xarray_2d_mask(self):

