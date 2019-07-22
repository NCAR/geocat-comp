# cython: language_level=3, boundscheck=False
cimport ncomp
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

import cython
import numpy as np
cimport numpy as np
import functools

dtype_default_fill = {
             "DEFAULT_FILL":       ncomp.DEFAULT_FILL_DOUBLE,
             np.dtype(np.int8):    ncomp.DEFAULT_FILL_INT8,
             np.dtype(np.int16):   ncomp.DEFAULT_FILL_INT16,
             np.dtype(np.int32):   ncomp.DEFAULT_FILL_INT32,
             np.dtype(np.int64):   ncomp.DEFAULT_FILL_INT64,
             np.dtype(np.float32): ncomp.DEFAULT_FILL_FLOAT,
             np.dtype(np.float64): ncomp.DEFAULT_FILL_DOUBLE,
            }


dtype_to_ncomp = {np.dtype(np.bool):       ncomp.NCOMP_BOOL,
                  np.dtype(np.int8):       ncomp.NCOMP_BYTE,
                  np.dtype(np.uint8):      ncomp.NCOMP_UBYTE,
                  np.dtype(np.int16):      ncomp.NCOMP_SHORT,
                  np.dtype(np.uint16):     ncomp.NCOMP_USHORT,
                  np.dtype(np.int32):      ncomp.NCOMP_INT,
                  np.dtype(np.uint32):     ncomp.NCOMP_UINT,
                  np.dtype(np.int64):      ncomp.NCOMP_LONG,
                  np.dtype(np.uint64):     ncomp.NCOMP_ULONG,
                  np.dtype(np.longlong):   ncomp.NCOMP_LONGLONG,
                  np.dtype(np.ulonglong):  ncomp.NCOMP_ULONGLONG,
                  np.dtype(np.float32):    ncomp.NCOMP_FLOAT,
                  np.dtype(np.float64):    ncomp.NCOMP_DOUBLE,
                  np.dtype(np.float128):   ncomp.NCOMP_LONGDOUBLE,
                 }


def get_default_fill(arr):
    if isinstance(arr, type(np.dtype)):
        dtype = arr
    else:
        dtype = arr.dtype

    try:
        return dtype_default_fill[dtype]
    except KeyError:
        return dtype_default_fill['DEFAULT_FILL']


def get_ncomp_type(arr):
    try:
        return dtype_to_ncomp[arr.dtype]
    except KeyError:
        raise KeyError("dtype('{}') is not a valid NCOMP type".format(arr.dtype)) from None


cdef ncomp.ncomp_array* np_to_ncomp_array(np.ndarray nparr):
    cdef long long_addr = nparr.__array_interface__['data'][0]
    cdef void* addr = <void*> long_addr
    cdef int ndim = nparr.ndim
    cdef size_t* shape = <size_t*> nparr.shape
    cdef int np_type = nparr.dtype.num
    return <ncomp.ncomp_array*> ncomp.ncomp_array_alloc(addr, np_type, ndim, shape)


cdef np.ndarray ncomp_to_np_array(ncomp.ncomp_array* ncarr):
    np.import_array()
    nparr = np.PyArray_SimpleNewFromData(ncarr.ndim, <np.npy_intp *> ncarr.shape, ncarr.type, ncarr.addr)
    cdef extern from "numpy/arrayobject.h":
        void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    PyArray_ENABLEFLAGS(nparr, np.NPY_OWNDATA)
    return nparr


def _linint2(np.ndarray xi, np.ndarray yi, np.ndarray fi, np.ndarray xo, np.ndarray yo, int icycx, double xmsg=ncomp.DEFAULT_FILL_DOUBLE):
    cdef ncomp.ncomp_array* ncomp_xi = np_to_ncomp_array(xi)
    cdef ncomp.ncomp_array* ncomp_yi = np_to_ncomp_array(yi)
    cdef ncomp.ncomp_array* ncomp_fi = np_to_ncomp_array(fi)
    cdef ncomp.ncomp_array* ncomp_xo = np_to_ncomp_array(xo)
    cdef ncomp.ncomp_array* ncomp_yo = np_to_ncomp_array(yo)
    cdef ncomp.ncomp_array* ncomp_fo
    cdef int iopt = 0
    cdef long i
    if dtype_to_ncomp[fi.dtype] == ncomp.NCOMP_DOUBLE:
        fo_dtype = np.float64
    else:
        fo_dtype = np.float32
    cdef np.ndarray fo = np.zeros(tuple([fi.shape[i] for i in range(fi.ndim - 2)] + [yo.shape[0], xo.shape[0]]), dtype=fo_dtype)

    ncomp_fo = np_to_ncomp_array(fo)

#   release global interpreter lock
    cdef int ier
    with nogil:
        ier = ncomp.linint2(
            ncomp_xi, ncomp_yi, ncomp_fi,
            ncomp_xo, ncomp_yo, ncomp_fo,
            icycx, xmsg, iopt)
#   re-acquire interpreter lock
#   check errors ier

    return fo
