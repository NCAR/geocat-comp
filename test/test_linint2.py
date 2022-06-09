# import sys
# import time
# import unittest as ut
# import logging
# logging.basicConfig(level=logging.DEBUG)
# mylogger = logging.getLogger()

# import numpy as np
# import xarray as xr

# # Import from directory structure if coverage test, or from installed
# # packages otherwise
# # if "--cov" in str(sys.argv):
# #     from src.geocat.comp import ChunkError, CoordinateError, linint1
# # else:
# #     from geocat.comp import ChunkError, CoordinateError, linint1

# from src.geocat.comp import ChunkError, CoordinateError, linint1a, linint1

# n = 127

# xi = np.linspace(0, n, num=n // 2 + 1, dtype=np.float64)
# xo = np.linspace(xi.min(), xi.max(), num=xi.shape[0] * 2 - 1)
# fi_np = np.random.rand(96, 3, len(xi)).astype(np.float64)

# chunks = {
#     'time': fi_np.shape[0],
#     'level': fi_np.shape[1],
#     'lon': fi_np.shape[2]
# }

# #dummy data
# class Test_linint1(ut.TestCase):

#     def test_example(self):
#         xi = np.array([2, 4, 6, 8])
#         fi = np.array([1, 2, 3, 4])
#         xo = np.array([2, 3, 4, 5, 6, 7, 8])
#         fo = linint1a(fi, xo, xi)
#         np.testing.assert_array_equal([1, 1.5, 2, 2.5, 3, 3.5, 4], fo)
    
#     def test_cyclic(self):
#         xi = np.array([0, 50, 100, 350])
#         fi = np.array([1, 2, 3, 4])
#         xo = np.array([0, 40, 355])
#         fo = linint1a(fi, xo, xi, 1)
#         np.testing.assert_array_equal([1.0, 1.8, 3.94], fo)

#     def test_nonmcinc(self):
#         xi = np.array([1, 2, 4, 6, 3, 5])
#         fi = np.array([3, 4, 5, 7, 6, 5])
#         xo = np.array([1.5, 2, 2.5])
#         fo = linint1a(fi, xo, xi)
#         np.testing.assert_array_equal([3.5, 4.0, 5.0], fo)

#     def test_linint1(self):
#         fi = xr.DataArray(fi_np,
#                           dims=['time', 'level', 'lon'],
#                           coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint1a(fi, xo, xi)
#         # d = ~np.equal(fi.values, fo[..., ::2].values)
#         # where = np.where(d)
#         # for i in range(3):
#         #     for j in range(len(where[0])):
#         #         logging.warning("where[%i][%i] = %i", i, j, where[i][j])
#         # for i in range(12):
#         #     logging.warning("fi[%i][%i][%i] = %f", where[0][0], where[1][0], i, fi[where[0][0]][where[1][0]][i].values)
#         # for i in range(25):
#         #     logging.warning("fo[%i][%i][%i] = %f", where[0][0], where[1][0], i, fo[where[0][0]][where[1][0]][i].values)   
#         np.testing.assert_almost_equal(fi.values, fo[..., ::2].values, decimal=16)


# # TODO: All of these tests should be revisited since they indeed are not
# #  checking any actually interpolated values.
# class Test_linint1_float64(ut.TestCase):

#     def test_linint1(self):
#         fi = xr.DataArray(fi_np,
#                           dims=['time', 'level', 'lon'],
#                           coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint1a(fi, xo)
#         np.testing.assert_almost_equal(fi.values, fo[..., ::2].values, decimal=16)

#     def test_linint1_msg(self):
#         fi = xr.DataArray(fi_np,
#                           dims=['time', 'level', 'lon'],
#                           coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint1a(fi, xo, msg_py=fi_np[0, 0, 0])
#         np.testing.assert_almost_equal(fi.values, fo[..., ::2].values, decimal=16)

#     def test_linint2_nan(self):
#         fi_np_copy = fi_np.copy()
#         fi_np_copy[:, :, 0] = np.nan
#         fi_temp = np.asarray([np.nan, 8.53987625341092836273, np.nan, 4.3, 9.0192836584930282621, np.nan])
#         xi_temp = np.asarray([2, 4, 6,7, 8, 10])
#         xo_temp = np.asarray([2, 3, 4, 5, 6,6.5, 7,7.5, 8, 9, 10])
#         fi = xr.DataArray(fi_temp,
#                           dims=['lon'],
#                           coords={
#                               'lon': xi_temp
#                           })
#         fo = linint1a(fi, xo_temp)

#         #logging
#         # for x in range(3):
#         #     for i in range(63):
#         #         logging.debug("fi at %i = %f", i, fi[x,x,i])
#         #     for i in range(126):
#         #         logging.debug("fo at %i = %f", i, fo[x,x,i])
#         #np.testing.assert_almost_equal(fi.values, fo[..., ::2].values, decimal=16)

#         np.testing.assert_almost_equal(fi.values, fo[::2], decimal=16)

#     def test_linint1_masked(self):
#         fi_mask = np.zeros(fi_np.shape, dtype=bool)
#         fi_mask[:, :, 0] = True
#         fi_ma = np.ma.MaskedArray(fi_np, mask=fi_mask)
#         fi = xr.DataArray(fi_ma,
#                           dims=['time', 'level', 'lon'],
#                           coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint1a(fi, xo)
#         np.testing.assert_almost_equal(fi, fo[..., ::2].values, decimal=16)

# class Test_linint1_float32(ut.TestCase):

#     def test_linint1_float32(self):
#         fi = xr.DataArray(fi_np.astype(np.float32),
#                           dims=['time', 'level', 'lon'],
#                           coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint1a(fi, xo)
#         np.testing.assert_almost_equal(fi.values, fo[..., ::2].values, decimal=7)

#     def test_linint1_msg_float32(self):
#         fi_np_copy = fi_np.astype(np.float32)
#         fi = xr.DataArray(fi_np_copy,
#                           dims=['time', 'level', 'lon'],
#                           coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint1a(fi, xo, msg_py=fi_np_copy[0, 0, 0])
#         np.testing.assert_almost_equal(fi.values, fo[..., ::2].values, decimal=7)

#     def test_linint1_nan_float32(self):
#         fi_np_copy = fi_np.astype(np.float32)
#         fi_np_copy[:, :, 0] = np.nan
#         fi = xr.DataArray(fi_np_copy,
#                           dims=['time', 'level', 'lon'],
#                           coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint1a(fi, xo)
#         d = ~np.equal(fi.values, fo[..., ::2].values)
#         where = np.where(d)
#         for i in range(3):
#             for j in range(100):
#                 logging.warning("where[%i][%i] = %i", i, j, where[i][j])
#         # for i in range(12):
#         #     logging.warning("fi[%i][%i][%i] = %f", where[0][0], where[1][0], i, fi[where[0][0]][where[1][0]][i].values)
#         # for i in range(25):
#         #     logging.warning("fo[%i][%i][%i] = %f", where[0][0], where[1][0], i, fo[where[0][0]][where[1][0]][i].values)   
    
#         np.testing.assert_almost_equal(fi.values, fo[..., ::2].values, decimal=16)

#     def test_linint1_masked_float32(self):
#         fi_mask = np.zeros(fi_np.shape, dtype=bool)
#         fi_mask[:, :, 0] = True
#         fi_ma = np.ma.MaskedArray((fi_np * 100).astype(np.float32),
#                                   mask=fi_mask)
#         fi = xr.DataArray(fi_ma,
#                           dims=['time', 'level', 'lon'],
#                           coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint1a(fi, xo)
#         np.testing.assert_almost_equal(fi, fo[..., ::2].values, decimal=16)


# class Test_linint1_int32(ut.TestCase):

#     def test_linint1_int32(self):
#         fi = xr.DataArray((fi_np * 100).astype(np.int32),
#                           dims=['time', 'level', 'lon'],
#                           coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint1a(fi, xo)
#         np.testing.assert_array_equal(fi.values, fo[..., ::2].values)

#     def test_linint1_msg_int32(self):
#         fi_np_copy = (fi_np * 100).astype(np.int32)
#         fi = xr.DataArray(fi_np_copy,
#                           dims=['time', 'level', 'lon'],
#                           coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint1a(fi, xo, msg_py=fi_np_copy[0, 0, 0])
#         np.testing.assert_array_equal(fi.values, fo[..., ::2].values)

#     def test_linint1_masked_int32(self):
#         fi_mask = np.zeros(fi_np.shape, dtype=bool)
#         fi_mask[:, :, 0] = True
#         fi_ma = np.ma.MaskedArray((fi_np * 100).astype(np.int32), mask=fi_mask)
#         fi = xr.DataArray(fi_ma,
#                           dims=['time', 'level', 'lon'],
#                           coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint1a(fi, xo)
#         np.testing.assert_array_equal(fi, fo[..., ::2].values, decimal=16)


# class Test_linint1_int64(ut.TestCase):

#     def test_linint1_int64(self):
#         fi = xr.DataArray((fi_np * 100).astype(np.int64),
#                           dims=['time', 'level', 'lon'],
#                           coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint1a(fi, xo)
#         np.testing.assert_array_equal(fi.values, fo[..., ::2].values)
#         print("Good after assertion")

#     def test_linint1_msg_int64(self):
#         fi_np_copy = (fi_np * 100).astype(np.int64)
#         fi = xr.DataArray(fi_np_copy,
#                           dims=['time', 'level', 'lon'],
#                           coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint1a(fi, xo, msg_py=fi_np_copy[0, 0, 0])
#         np.testing.assert_array_equal(fi.values, fo[..., ::2].values)

#     def test_linint1_masked_int64(self):
#         fi_mask = np.zeros(fi_np.shape, dtype=bool)
#         fi_mask[:, :, 0] = True
#         fi_ma = np.ma.MaskedArray((fi_np * 100).astype(np.int64), mask=fi_mask)
#         fi = xr.DataArray(fi_ma,
#                           dims=['time', 'level', 'lon'],
#                           coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint1a(fi, xo)
#         np.testing.assert_array_equal(fi, fo[..., ::2].values, decimal=16)

# class Test_linint1_dask(ut.TestCase):

#     def test_linint1_chunked_leftmost(self):
#         # use 1 for time and level chunk sizes
#         chunks = {
#             'time': 1,
#             'level': 1,
#             'lon': fi_np.shape[2]
#         }
#         fi = xr.DataArray(fi_np,
#                            dims=['time', 'level', 'lon'],
#                            coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         fo = linint1a(fi, xo)
#         np.testing.assert_almost_equal(fi.values, fo[..., ::2].values, decimal=16)


# ## raising error correctly - assert not properly formatted
#     def test_linint1_chunked_interp(self):
#         # use 1 for interpolated dimension chunk sizes -- this should throw a ChunkError
#         chunks = {'time': 1, 'level': 1, 'lon': 1}
#         fi = xr.DataArray(fi_np,
#                           dims=['time', 'level', 'lon'],
#                           coords={
#                               'lon': xi
#                           }).chunk(chunks)
#         with self.assertRaises(ChunkError):
#             fo = linint1a(fi, xo)


# class Test_linint1_numpy(ut.TestCase):

#     def test_linint1_fi_np(self):
#         fo = linint1a(fi_np, xo, xi=xi)
#         np.testing.assert_almost_equal(fi_np, fo[..., ::2], decimal=16)

# ## raising error correctly - assert not properly formatted
#     def test_linint1_fi_np_no_xi(self):
#         self.assertRaises(CoordinateError, linint1a, fi_np, xo)
            

# # # class Test_linint2_non_monotonic(ut.TestCase):
# # #
# # #     def test_linint2_non_monotonic_xr(self):
# # #         fi = xr.DataArray(fi_np[:, :, ::-1, :],
# # #                           dims=['time', 'level', 'lat', 'lon'],
# # #                           coords={
# # #                               'lat': yi_reverse,
# # #                               'lon': xi
# # #                           }).chunk(chunks)
# # #         with self.assertWarns(_ncomp.NcompWarning):
# # #             linint2(fi, xo, yo).compute()
# # #
# # #     def test_linint2_non_monotonic_np(self):
# # #         with self.assertWarns(geocat.ncomp._ncomp.NcompWarning):
# # #             linint2(fi_np[:, :, ::-1, :],
# # #                                  xo,
# # #                                  yo,
# # #                                  xi=xi,
# # #                                  yi=yi_reverse)


# # class Test_linint1_non_contiguous(ut.TestCase):

# #     def test_linint1_non_contiguous_xr(self):
# #         fi = xr.DataArray(fi_np[:, :, ::-1, :],
# #                           dims=['time', 'level', 'lat', 'lon'],
# #                           coords={
# #                               'lat': yi_reverse,
# #                               'lon': xi
# #                           }).chunk(chunks)
# #         fo = linint2(fi[:, :, ::-1, :], xo, yo)
# #         np.testing.assert_array_equal(fi[:, :, ::-1, :].values,
# #                                       fo[..., ::2, ::2].values)

# #     def test_linint2_non_contiguous_np(self):
# #         fi = xr.DataArray(fi_np[:, :, ::-1, :],
# #                           dims=['time', 'level', 'lat', 'lon'],
# #                           coords={
# #                               'lat': yi_reverse,
# #                               'lon': xi
# #                           }).chunk(chunks)
# #         fo = linint2(fi[:, :, ::-1, :].values,
# #                      xo,
# #                      yo,
# #                      xi=xi,
# #                      yi=yi_reverse[::-1])
# #         np.testing.assert_array_equal(fi[:, :, ::-1, :], fo[..., ::2, ::2])

# from types import NoneType
# import typing
# import warnings

# from dask.array.core import map_blocks
# import numpy as np
# import xarray as xr
# from scipy import interpolate 
# from .errors import ChunkError, CoordinateError

# supported_types = typing.Union[xr.DataArray, np.ndarray]

# #add two cyclic points to beginning and end of xi and fi
# def dlincyc(xi, fi):
#     #find max and min
#     max = np.nanmax(xi)
#     maxIndx = np.where(xi == max)[0]
#     min = np.nanmin(xi)
#     minIndx = np.where(xi == min)[0]

#     #find 2nd to last max and min
#     temp = np.where(xi == max, np.nan, xi)
#     temp = np.where(temp == min, np.nan, temp)
#     max2 = np.nanmax(temp)
#     min2 = np.nanmin(temp)

#     #add points
#     xi = np.append(xi, min - abs(min - min2))
#     fi = np.append(fi, fi[maxIndx])

#     xi = np.append(xi, max + abs(max - max2))
#     fi = np.append(fi, fi[minIndx])

#     return xi, fi

# # Inner Function _<funcname>()
# # These are executed within dask processes (if any), and could/should
# # do anything that can benefit from parallel execution.

# def _linint1(xi, fi, xo, icycx, xmsg, shape):

#     if xmsg != None:
#         np.where(fi == xmsg, np.nan, fi)

#     fo = np.empty(len(xo))
#     fo.fill(xmsg)

#     #if cyclic - add cyclic points to beginning and end
#     if icycx:
#         xi, fi = dlincyc(xi, fi)

#     # rows = len(np.shape(fi))
#     # fi = fi[rows-1]
#     # fo = interpolate.interpolate_1d(xo, xi, fi)
#     f = interpolate.interp1d(xi, fi)
#     fo = f(xo)

#     if xmsg != None:
#         np.where(fo == np.nan, xmsg, fo)

#     # numpy and reshape
#     fo = fo.reshape(shape)

#     return fo

# # Outer Wrappers <funcname>()
# # These Wrappers are excecuted in the __main__ python process, and should be
# # used for any tasks which would not benefit from parallel execution.

# def linint1a(fi: supported_types,
#             xo: supported_types,
#             xi: supported_types = None,
#             icycx: np.number = 0,
#             msg_py: np.number = None) -> supported_types:
#     """Interpolates from one series to another using piecewise linear
#     interpolation across the rightmost dimension. The series may be cyclic in
#     the X direction.

#     If missing values are present, then linint1 will perform the piecewise linear interpolation at
#     all points possible, but will return missing values at coordinates which could not be used.

#     If any of the output coordinates `xo` are outside those of the input coordinates `xi`, the
#     `fo` values at those coordinates will be set to missing (i.e. no extrapolation is performed).

#     Parameters
#     ----------

#     fi : :class:`xarray.DataArray`, :class:`numpy.ndarray`:
#         An array of one or more dimensions. If `xi` is passed in as an argument, then the size of
#         the rightmost dimension of `fi` must match the rightmost dimension of `xi`.

#         If missing values are present, then `linint1` will perform the piecewise linear interpolation
#         at all points possible, but will return missing values at coordinates which could not be used.

#         Note:
#             This variable must be
#             supplied as a :class:`xarray.DataArray` in order to copy
#             the dimension names to the output. Otherwise, default
#             names will be used.

#     xo : :class:`xarray.DataArray`, :class:`numpy.ndarray`:
#         A one-dimensional array that specifies the X coordinates of
#         the return array. It doesn't need to be strictly monotonically
#         increasing or decreasing, or equally spaced.

#         If the output coordinates (xo) are outside those of the
#         input coordinates (xi), then the fo values at those
#         coordinates will be set to missing (i.e. no extrapolation is
#         performed).

#     xi : :class:`xarray.DataArray`, :class:`numpy.ndarray`:
#         An array that specifies the X coordinates of the fi array.
#         Most frequently, this array is one-dimensional.  It must contain
#         unique values, except for xmsg, and it doesn't need
#         to be monotonically increasing or decreasing or equally spaced.

#         If xi is multi-dimensional, then its rightmost dimensions must be the
#         same as fi's dimensions. If it is one-dimensional, its length
#         must be the same as the rightmost (fastest varying) dimension
#         of fi.

#         Note:
#             If fi is of type :class:`xarray.DataArray` and xi is
#             left unspecified, then the rightmost coordinate
#             dimension of fi will be used. If fi is not of type
#             :class:`xarray.DataArray`, then xi becomes a mandatory
#             parameter. This parameter must be specified as a keyword
#             argument.

#     icycx : :obj:`bool`:
#         An option to indicate whether the rightmost dimension of fi
#         is cyclic. This should be set to True only if you have
#         global data, but your longitude values don't quite wrap all
#         the way around the globe. For example, if your longitude
#         values go from, say, -179.75 to 179.75, or 0.5 to 359.5,
#         then you would set this to True.

#     msg_py : :obj:`numpy.number`:
#         A numpy scalar value that represent a missing value in fi.
#         This argument allows a user to use a missing value scheme
#         other than NaN or masked arrays, similar to what NCL allows.

#     Returns
#     -------
#     fo : :class:`xarray.DataArray`, :class:`numpy.ndarray`:
#         The interpolated series. The returned value will have the same
#         dimensions as fi, except for the rightmost dimension which
#         will have the same dimension size as the length of xo.
#         The return type will be double if fi is double, and float
#         otherwise.

#     Examples
#     --------

#     Example 1: Using linint1 with :class:`xarray.DataArray` input
#     .. code-block:: python
#         import numpy as np
#         import xarray as xr
#         import geocat.comp
#         fi_np = np.random.rand(80)  # random 80-element array
#         # xi does not have to be equally spaced, but it is
#         # in this example
#         xi = np.arange(80)
#         # create target coordinate array, in this case use the same
#         # min/max values as xi, but with different spacing
#         xo = np.linspace(xi.min(), xi.max(), 100)
#         # create :class:`xarray.DataArray` and chunk it using the
#         # full shape of the original array.
#         # note that xi is attached as a coordinate array
#         fi = xr.DataArray(fi_np,
#                           dims=['x'],
#                           coords={'x': xi}
#                          ).chunk(fi_np.shape)
#         fo = geocat.comp.linint1(fi, xo, icycx=0)
#     """

#     # ''' Start of boilerplate
#     is_input_xr = True
#     is_input_dask = False

#     # If the input is numpy.ndarray, convert it to xarray.DataArray
#     if not isinstance(fi, xr.DataArray):
#         if (xi is None):
#             raise CoordinateError(
#                 "linint2: Argument xi must be provided explicitly unless fi is an xarray.DataArray."
#             )

#         is_input_xr = False

#         fi = xr.DataArray(fi)
#         fi = fi.assign_coords({fi.dims[-1]: xi})

#     # xi should be coming as xarray input's associated coords or assigned
#     # as coords while xarray being initiated from numpy input above
#     xi = fi.coords[fi.dims[-1]].values

#     # If input data is already chunked
#     if fi.chunks is not None:
#         is_input_dask = True

#         # Ensure rightmost dimension of input is not chunked
#         if list(fi.chunks)[-1:] != [xi.shape]:
#             raise Exception(
#                 "linint1: fi must be unchunked along the last dimension")

#     # NOTE: Auto-chunking, regardless of what chunk sizes were given by the user, seems
#     # to be explicitly needed in this function because:
#     # The Fortran routine for this function is implemented assuming it would be looped
#     # across the leftmost dimensions of the input (`fi`), i.e. on one-dimensional
#     # chunks of size that is equal to the rightmost dimension of `fi`.

#     # Generate chunks of {'dim_0': 1, 'dim_1': 1, ..., 'dim_n': xi.shape}
#     fi_chunks = list(fi.dims)
#     fi_chunks[:-1] = [
#         (k, 1) for (k, v) in zip(list(fi.dims)[:-1],
#                                  list(fi.shape)[:-1])
#     ]
#     fi_chunks[-1:] = [
#         (k, v) for (k, v) in zip(list(fi.dims)[-1:],
#                                  list(fi.shape)[-1:])
#     ]
#     fi_chunks = dict(fi_chunks)
#     fi = fi.chunk(fi_chunks)

#     # fo data structure elements
#     fo_chunks = list(fi.chunks)
#     fo_chunks[-1:] = (xo.shape,)
#     fo_chunks = tuple(fo_chunks)
#     fo_shape = tuple(a[0] for a in list(fo_chunks))
#     fo_coords = {k: v for (k, v) in fi.coords.items()}
#     fo_coords[fi.dims[-1]] = xo
#     # ''' end of boilerplate

#     # Inner Fortran wrapper call
#     fo = map_blocks(
#         _linint1,
#         xi,
#         fi.data,
#         xo,
#         icycx,
#         msg_py,
#         fo_shape,
#         chunks=fo_chunks,
#         dtype=fi.dtype,
#         drop_axis=[fi.ndim - 1],
#         new_axis=[fi.ndim - 1],
#     )

#     # If input was xarray.DataArray, convert output to xarray.DataArray as well
#     if is_input_xr:
#         if is_input_dask:
#             fo = xr.DataArray(fo,
#                               attrs=fi.attrs,
#                               dims=fi.dims,
#                               coords=fo_coords)
#         else:
#             fo = xr.DataArray(fo,
#                               attrs=fi.attrs,
#                               dims=fi.dims,
#                               coords=fo_coords).compute()

#     # Else if input was numpy.ndarray, convert Dask output to numpy.ndarray with `.compute()
#     else:
#         fo = fo.compute()

#     return fo
