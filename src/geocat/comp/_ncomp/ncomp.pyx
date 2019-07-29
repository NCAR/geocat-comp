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
             np.dtype(np.int8):    np.int8(ncomp.DEFAULT_FILL_INT8),
             np.dtype(np.uint8):   np.uint8(ncomp.DEFAULT_FILL_UINT8),
             np.dtype(np.int16):   np.int16(ncomp.DEFAULT_FILL_INT16),
             np.dtype(np.uint16):  np.uint16(ncomp.DEFAULT_FILL_UINT16),
             np.dtype(np.int32):   np.int32(ncomp.DEFAULT_FILL_INT32),
             np.dtype(np.uint32):  np.uint32(ncomp.DEFAULT_FILL_UINT32),
             np.dtype(np.int64):   np.int64(ncomp.DEFAULT_FILL_INT64),
             np.dtype(np.uint64):  np.uint64(ncomp.DEFAULT_FILL_UINT64),
             np.dtype(np.float32): np.float32(ncomp.DEFAULT_FILL_FLOAT),
             np.dtype(np.float64): np.float64(ncomp.DEFAULT_FILL_DOUBLE),
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


ncomp_to_dtype = {ncomp.NCOMP_BOOL:         np.bool,
                  ncomp.NCOMP_BYTE:         np.int8,
                  ncomp.NCOMP_UBYTE:        np.uint8,
                  ncomp.NCOMP_SHORT:        np.int16,
                  ncomp.NCOMP_USHORT:       np.uint16,
                  ncomp.NCOMP_INT:          np.int32,
                  ncomp.NCOMP_UINT:         np.uint32,
                  ncomp.NCOMP_LONG:         np.int64,
                  ncomp.NCOMP_ULONG:        np.uint64,
                  ncomp.NCOMP_LONGLONG:     np.longlong,
                  ncomp.NCOMP_ULONGLONG:    np.ulonglong,
                  ncomp.NCOMP_FLOAT:        np.float32,
                  ncomp.NCOMP_DOUBLE:       np.float64,
                  ncomp.NCOMP_LONGDOUBLE:   np.float128,
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


cdef set_ncomp_msg(ncomp.ncomp_missing* ncomp_msg, num):
    ncomp_type = num.dtype.num
    if ncomp_type == ncomp.NCOMP_FLOAT:
        ncomp_msg.msg_float = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_DOUBLE:
        ncomp_msg.msg_double = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_BOOL:
        ncomp_msg.msg_bool = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_BYTE:
        ncomp_msg.msg_byte = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_UBYTE:
        ncomp_msg.msg_ubyte = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_SHORT:
        ncomp_msg.msg_short = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_USHORT:
        ncomp_msg.msg_ushort = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_INT:
        ncomp_msg.msg_int = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_UINT:
        ncomp_msg.msg_uint = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_LONG:
        ncomp_msg.msg_long = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_ULONG:
        ncomp_msg.msg_ulong = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_LONGLONG:
        ncomp_msg.msg_longlong = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_ULONGLONG:
        ncomp_msg.msg_ulonglong = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == ncomp.NCOMP_LONGDOUBLE:
        ncomp_msg.msg_longdouble = ncomp_to_dtype[ncomp_type](num)

def _linint2(np.ndarray xi, np.ndarray yi, np.ndarray fi, np.ndarray xo, np.ndarray yo, int icycx, msg=None):
    cdef ncomp.ncomp_array* ncomp_xi = np_to_ncomp_array(xi)
    cdef ncomp.ncomp_array* ncomp_yi = np_to_ncomp_array(yi)
    cdef ncomp.ncomp_array* ncomp_fi = np_to_ncomp_array(fi)
    cdef ncomp.ncomp_array* ncomp_xo = np_to_ncomp_array(xo)
    cdef ncomp.ncomp_array* ncomp_yo = np_to_ncomp_array(yo)
    cdef ncomp.ncomp_array* ncomp_fo
    cdef int iopt = 0
    cdef long i
    if ncomp_fi.type == ncomp.NCOMP_DOUBLE:
        fo_dtype = np.float64
    else:
        fo_dtype = np.float32
    cdef np.ndarray fo = np.zeros(tuple([fi.shape[i] for i in range(fi.ndim - 2)] + [yo.shape[0], xo.shape[0]]), dtype=fo_dtype)

    missing_inds_fi = None

    if msg is None or np.isnan(msg): # if no missing value specified, assume NaNs
        missing_inds_fi = np.isnan(fi)
        msg = get_default_fill(fi)
    else:
        missing_inds_fi = (fi == msg)

    set_ncomp_msg(&ncomp_fi.msg, msg) # always set missing on ncomp_fi

    if missing_inds_fi.any():
        ncomp_fi.has_missing = 1
        fi[missing_inds_fi] = msg

    ncomp_fo = np_to_ncomp_array(fo)

#   release global interpreter lock
    cdef int ier
    with nogil:
        ier = ncomp.linint2(
            ncomp_xi, ncomp_yi, ncomp_fi,
            ncomp_xo, ncomp_yo, ncomp_fo,
            icycx, iopt)
#   re-acquire interpreter lock
#   check errors ier

    if missing_inds_fi is not None and missing_inds_fi.any():
        fi[missing_inds_fi] = np.nan

    if ncomp_fo.type == ncomp.NCOMP_DOUBLE:
        fo_msg = ncomp_fo.msg.msg_double
    else:
        fo_msg = ncomp_fo.msg.msg_float

    fo[fo == fo_msg] = np.nan

    return fo
