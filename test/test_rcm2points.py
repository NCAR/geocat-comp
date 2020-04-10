import numpy as np
import xarray as xr
import geocat.comp

import sys
import time
import unittest as ut


'''
; comprative test case in ncl_stable
begin
    lat2d = (/(/ 1.1, 1.2, 1.3/),(/2.1, 2.2, 2.3/),(/3.1, 3.2, 3.3/)/)
    lon2d = (/(/ 1.1, 1.2, 1.3/),(/2.1, 2.2, 2.3/),(/3.1, 3.2, 3.3/)/)
        
    random_setallseed(1234567890, 123456789) ; set random seeds to default values for repeatability;

    fi = random_normal(1, 1, (/3, 3, 3/)) ; 27 point data cube normal distrobution about 1 with standard deviation of 1
    
    ; set centers of data cube and data cube faces to -99 (default missing value)
    fi(0,1,1)=-99
    fi(2,1,1)=-99
    fi(1,0,1)=-99
    fi(1,2,1)=-99
    fi(1,1,0)=-99
    fi(1,1,2)=-99
    fi(1,1,1)=-99
    
    lat = (/1.0, 2.0, 3.0/)
    lon = (/1.0, 2.0, 3.0/)

    fi@_FillValue = -99 ; I think that this sets the missing value message to -99

    print(dimsizes(fi))
    
    print(lat2d)
    print(lon2d)
    print(fi)
    print(lat)		
    print(lon)
    
    fo0 = rcm2points(lat2d, lon2d, fi, lat, lon, 0) ; inverse distance weighting
    fo2 = rcm2points(lat2d, lon2d, fi, lat, lon, 2) ; bilinear iterpolation

    ;print(dimsizes(fo))
    print(fo0)
    print(fo2)
end

Variable: lat2d
Type: float
Total Size: 36 bytes
            9 values
Number of Dimensions: 2
Dimensions and sizes:	[3] x [3]
Coordinates: 
(0,0)	1.1
(0,1)	1.2
(0,2)	1.3
(1,0)	2.1
(1,1)	2.2
(1,2)	2.3
(2,0)	3.1
(2,1)	3.2
(2,2)	3.3


Variable: lon2d
Type: float
Total Size: 36 bytes
            9 values
Number of Dimensions: 2
Dimensions and sizes:	[3] x [3]
Coordinates: 
(0,0)	1.1
(0,1)	1.2
(0,2)	1.3
(1,0)	2.1
(1,1)	2.2
(1,2)	2.3
(2,0)	3.1
(2,1)	3.2
(2,2)	3.3


Variable: fi
Type: float
Total Size: 108 bytes
            27 values
Number of Dimensions: 3
Dimensions and sizes:	[3] x [3] x [3]
Coordinates: 
Number Of Attributes: 1
  _FillValue :	-99
(0,0,0)	1.870327
(0,0,1)	1.872924
(0,0,2)	2.946794
(0,1,0)	1.98253
(0,1,1)	-99
(0,1,2)	0.8730035
(0,2,0)	0.1410671
(0,2,1)	1.877125
(0,2,2)	1.931963
(1,0,0)	-0.1676207
(1,0,1)	-99
(1,0,2)	1.735453
(1,1,0)	-99
(1,1,1)	-99
(1,1,2)	-99
(1,2,0)	1.754721
(1,2,1)	-99
(1,2,2)	0.381366
(2,0,0)	2.015617
(2,0,1)	0.4975608
(2,0,2)	2.169137
(2,1,0)	0.3293635
(2,1,1)	-99
(2,1,2)	2.691788
(2,2,0)	2.510986
(2,2,1)	1.027274
(2,2,2)	1.351906


Variable: lat
Type: float
Total Size: 12 bytes
            3 values
Number of Dimensions: 1
Dimensions and sizes:	[3]
Coordinates: 
(0)	 1
(1)	 2
(2)	 3


Variable: lon
Type: float
Total Size: 12 bytes
            3 values
Number of Dimensions: 1
Dimensions and sizes:	[3]
Coordinates: 
(0)	 1
(1)	 2
(2)	 3


Variable: fo0
Type: float
Total Size: 36 bytes
            9 values
Number of Dimensions: 2
Dimensions and sizes:	[3] x [3]
Coordinates: 
Number Of Attributes: 1
  _FillValue :	-99
(0,0)	1.95103
(0,1)	1.878375
(0,2)	0.6331819
(1,0)	0.02682439
(1,1)	1.067587
(1,2)	1.613317
(2,0)	1.744488
(2,1)	0.6278715
(2,2)	2.132033


Variable: fo2
Type: float
Total Size: 36 bytes
            9 values
Number of Dimensions: 2
Dimensions and sizes:	[3] x [3]
Coordinates: 
Number Of Attributes: 1
  _FillValue :	-99
(0,0)	1.95103
(0,1)	1.878375
(0,2)	0.6331819
(1,0)	0.02682439
(1,1)	1.067587
(1,2)	1.613317
(2,0)	1.744488
(2,1)	0.6278715
(2,2)	2.132033
'''

# create and fill the input 2D grid (lat2D, lon2D)

lat2d = np.asarray([1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3]).reshape((3, 3))

lon2d = np.asarray([1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3]).reshape((3, 3))

fi = np.asarray(
    [1.870327, 1.872924, 2.946794, 1.98253, - 99, 0.8730035, 0.1410671, 1.877125, 1.931963, - 0.1676207, - 99, 1.735453, - 99, - 99, - 99, 1.754721, - 99,
     0.381366, 2.015617, 0.4975608, 2.169137, 0.3293635, - 99, 2.691788, 2.510986, 1.027274, 1.351906]).reshape((3, 3, 3))

# EVERYTHINGT BELOW THIS LINE IS GOING TO BE GONE


'''
for i in range(in_size_M):
    lat2D[i, :] = float(i)
for j in range(in_size_N):
    lon2D[:, j] = float(j)

# create and fill input data array (fi)
fi = np.random.randn(1, in_size_M, in_size_N)

# create and fill the output 1D grid (lat1D, lon1D)
out_size_M = m + 1
out_size_N = n + 1

lat1D = np.zeros((out_size_M))
lon1D = np.zeros((out_size_N))
for i in range(out_size_M):
    lat1D[i] = float(i) * 0.5
for j in range(out_size_N):
    lon1D[j] = float(j) * 0.5

fo = geocat.comp.rcm2points(lat2D, lon2D, fi, lat1D, lon1D, 2)

fi_diag = np.asarray([np.diag(fi[0, :, :])])
fi_diag_asfloat32 = fi_diag.astype(np.float32)


class Test_rcm2points_float64(ut.TestCase):
    def test_rcm2points_float64(self):
        fo = geocat.comp.rcm2points(lat2D, lon2D, fi, lat1D, lon1D, 2)
        np.testing.assert_array_equal(fi_diag, fo[..., ::2, ::2].values)

    def test_rcm2points_msg_float64(self):
        fo = geocat.comp.rcm2points(lat2D, lon2D, fi, lat1D, lon1D, 0, msg=fi[0, 0, 0])
        np.testing.assert_array_equal(fi_diag, fo[..., ::2, ::2].values)

    def test_rcm2points_nan_float64(self):
        fi_np_copy = fi.copy()
        fi_np_copy[:, 0, 0] = np.nan
        fi_np_diag = np.asarray([np.diag(fi_np_copy[0, :, :])])
        fo = geocat.comp.rcm2points(lat2D, lon2D, fi_np_copy, lat1D, lon1D)
        np.testing.assert_array_equal(fi_diag[:, 1:], fo[..., 2::2].values)


class Test_rcm2points_float32(ut.TestCase):
    def test_rcm2points_float32(self):
        fi_asfloat32 = fi.astype(np.float32)
        fo = geocat.comp.rcm2points(lat2D.astype(np.float32), lon2D.astype(np.float32), fi_asfloat32, lat1D.astype(np.float32), lon1D.astype(np.float32))
        np.testing.assert_array_equal(fi_diag_asfloat32, fo[..., ::2, ::2].values)

    def test_rcm2points_msg_float32(self):
        fi_np_copy = fi.astype(np.float32)
        fo = geocat.comp.rcm2points(lat2D.astype(np.float32), lon2D.astype(np.float32), fi_np_copy, lat1D.astype(np.float32), lon1D.astype(np.float32), 0,
                                    msg=fi_np_copy[0, 0, 0])
        np.testing.assert_array_equal(fi_diag_asfloat32, fo[..., ::2, ::2].values)

    def test_rcm2points_nan_float32(self):
        fi_np_copy = fi.astype(np.float32)
        fi_np_copy[:, 0, 0] = np.nan
        fo = geocat.comp.rcm2points(lat2D, lon2D, fi_np_copy, lat1D, lon1D)
        np.testing.assert_array_equal(fi_diag_asfloat32[:, 1:], fo[..., 2::2].values)'''
