# cython: language_level=3, boundscheck=False, embedsignature=True
cimport ncomp
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

import cython
import numpy as np
cimport numpy as np
import functools
import warnings

class NcompWarning(Warning):
    pass

class NcompError(Exception):
    pass

def carrayify(f):
    """
    A decorator that ensures that :class:`numpy.ndarray` arguments are
    C-contiguous in memory. The decorator function takes no arguments.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        new_args = list(args)
        for i, arg in enumerate(new_args):
            if isinstance(arg, np.ndarray) and not arg.flags.carray:
                new_args[i] = np.ascontiguousarray(arg)
        return f(*new_args, **kwargs)
    return wrapper


class GeocatArray(object):
    def __init__(self, data, attrs={}):
        self.data = data
        self.attrs = attrs

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

@carrayify
def _linint2(np.ndarray xi, np.ndarray yi, np.ndarray fi, np.ndarray xo, np.ndarray yo, int icycx, msg=None):
    """_linint2(xi, yi, fi, xo, yo, icycx, msg=None)

    Interpolates a regular grid to a rectilinear one using bi-linear
    interpolation.

    linint2 uses bilinear interpolation to interpolate from one
    rectilinear grid to another. The input grid may be cyclic in the x
    direction. The interpolation is first performed in the x direction,
    and then in the y direction.

    Args:

        xi (:class:`numpy.ndarray`):
            An array that specifies the X coordinates of the fi array.
            Most frequently, this is a 1D strictly monotonically
            increasing array that may be unequally spaced. In some
            cases, xi can be a multi-dimensional array (see next
            paragraph). The rightmost dimension (call it nxi) must have
            at least two elements, and is the last (fastest varying)
            dimension of fi.

            If xi is a multi-dimensional array, then each nxi subsection
            of xi must be strictly monotonically increasing, but may be
            unequally spaced. All but its rightmost dimension must be
            the same size as all but fi's rightmost two dimensions.

            For geo-referenced data, xi is generally the longitude
            array.

        yi (:class:`numpy.ndarray`):
            An array that specifies the Y coordinates of the fi array.
            Most frequently, this is a 1D strictly monotonically
            increasing array that may be unequally spaced. In some
            cases, yi can be a multi-dimensional array (see next
            paragraph). The rightmost dimension (call it nyi) must have
            at least two elements, and is the second-to-last dimension
            of fi.

            If yi is a multi-dimensional array, then each nyi subsection
            of yi must be strictly monotonically increasing, but may be
            unequally spaced. All but its rightmost dimension must be
            the same size as all but fi's rightmost two dimensions.

            For geo-referenced data, yi is generally the latitude array.

        fi (:class:`numpy.ndarray`):
            An array of two or more dimensions. If xi is passed in as an
            argument, then the size of the rightmost dimension of fi
            must match the rightmost dimension of xi. Similarly, if yi
            is passed in as an argument, then the size of the second-
            rightmost dimension of fi must match the rightmost dimension
            of yi.

            If missing values are present, then linint2 will perform the
            bilinear interpolation at all points possible, but will
            return missing values at coordinates which could not be
            used.

        xo (:class:`numpy.ndarray`):
            A one-dimensional array that specifies the X coordinates of
            the return array. It must be strictly monotonically
            increasing, but may be unequally spaced.

            For geo-referenced data, xo is generally the longitude
            array.

            If the output coordinates (xo) are outside those of the
            input coordinates (xi), then the fo values at those
            coordinates will be set to missing (i.e. no extrapolation is
            performed).

        yo (:class:`numpy.ndarray`):
            A one-dimensional array that specifies the Y coordinates of
            the return array. It must be strictly monotonically
            increasing, but may be unequally spaced.

            For geo-referenced data, yo is generally the latitude array.

            If the output coordinates (yo) are outside those of the
            input coordinates (yi), then the fo values at those
            coordinates will be set to missing (i.e. no extrapolation is
            performed).

        icycx (:obj:`bool`):
            An option to indicate whether the rightmost dimension of fi
            is cyclic. This should be set to True only if you have
            global data, but your longitude values don't quite wrap all
            the way around the globe. For example, if your longitude
            values go from, say, -179.75 to 179.75, or 0.5 to 359.5,
            then you would set this to True.

        msg (:obj:`numpy.number`):
            A numpy scalar value that represent a missing value in fi.
            This argument allows a user to use a missing value scheme
            other than NaN or masked arrays, similar to what NCL allows.

    Returns:
        :class:`numpy.ndarray`: The interpolated grid. The returned
        value will have the same dimensions as fi, except for the
        rightmost two dimensions which will have the same dimension
        sizes as the lengths of yo and xo. The return type will be
        double if fi is double, and float otherwise.

    """

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
    if ier:
        warnings.warn("linint2: {}: xi, yi, xo, and yo must be monotonically increasing".format(ier),
                      NcompWarning)

    if missing_inds_fi is not None and missing_inds_fi.any():
        fi[missing_inds_fi] = np.nan

    if ncomp_fo.type == ncomp.NCOMP_DOUBLE:
        fo_msg = ncomp_fo.msg.msg_double
    else:
        fo_msg = ncomp_fo.msg.msg_float

    fo[fo == fo_msg] = np.nan

    return fo

@cython.embedsignature(True)
def _eofunc(np.ndarray np_input, int neval, opt={}):
    """Computes empirical orthogonal functions (EOFs, aka: Principal Component
    Analysis).

    Args:

      data (:class:`numpy.ndarray`):
        A multi-dimensional array in which the rightmost dimension is the number
        of observations. Generally, this is the time dimension. If your rightmost
        dimension is not time, then see eofunc_n. Commonly, the data array
        contains anomalies from some base climatology, however, this is not
        required.

      neval (:obj:`int`):
        A scalar integer that specifies the number of eigenvalues and
        eigenvectors to be returned. This is usually less than or equal to the
        minimum number of observations or number of variables.

      options (:obj:`dict`):
        - "jopt"        : both routines
        - "return_eval" : both routines (unadvertised)
        - "return_trace": return trace
        - "return_pcrit": return pcrit
        - "pcrit"       : transpose routine only
        - "anomalies"   : If True, anomalies have already been calculated by
                        user, and this interface shouldn't remove them.
        - "transpose"   : If True, call transpose routine no matter what
                      : If False, don't call transpose routine no matter what
        - "oldtranspose": If True, call Dennis' old transpose routine.
        - "debug"       : turn on debug

      Returns:
        A multi-dimensional array containing normalized EOFs. The returned
        array will be of the same size as data with the rightmost dimension
        removed and an additional leftmost dimension of the same size as neval
        added. Double if data is double, float otherwise.

        The return variable will have associated with it the following
        attributes:

          - eval: a one-dimensional array of size neval that contains the
                  eigenvalues.
          - pcvar: a one-dimensional float array of size neval equal to the
                   percent variance associated with each eigenvalue.
          - pcrit: The same value and type of options["pcrit"] if the user
                   changed the default.
          - matrix: A string indicating the type of matrix used,
                    "correlation" or "covariance".
          - method: A string indicating if the input array, data, was/was-not
                    transposed for the purpose of computing the eigenvalues and
                    eigenvectors. The string can have two values: "transpose"
                    or "no transpose"
          - eval_transpose: This attribute is returned only if
                            method="transpose". eval_transpose will contain the
                            eigenvalues of the transposed covariance matrix.
                            These eigenvalues are then scaled such that they are
                            consistent with the original input data.


    """
    # convert np_input to ncomp_array
    cdef ncomp.ncomp_array* ncomp_input = np_to_ncomp_array(np_input)

    # convert opt dict to ncomp_attributes struct
    cdef ncomp.ncomp_attributes* attrs = dict_to_ncomp_attributes(opt)

    # allocate output ncomp_array and ncomp_attributes
    cdef ncomp.ncomp_array ncomp_output
    cdef ncomp.ncomp_attributes attrs_output

    cdef int ier
    with nogil:
        ier = ncomp.eofunc(ncomp_input, neval, attrs, &ncomp_output, &attrs_output)

    # convert ncomp_output to np.ndarray
    np_output = ncomp_to_np_array(&ncomp_output)

    # convert attrs_output to dict
    # do something here
    np_attrs_dict = ncomp_attributes_to_dict(attrs_output)

    return (np_output, np_attrs_dict)

cdef ncomp.ncomp_single_attribute* np_to_ncomp_single_attribute(char* name, np.ndarray nparr):
    cdef long long_addr = nparr.__array_interface__['data'][0]
    cdef void* addr = <void*> long_addr
    cdef int ndim = nparr.ndim
    cdef size_t* shape = <size_t*> nparr.shape
    cdef int np_type = nparr.dtype.num
    return <ncomp.ncomp_single_attribute*> ncomp.create_ncomp_single_attribute(name, addr, np_type, ndim, shape)

cdef ncomp.ncomp_attributes* dict_to_ncomp_attributes(d):
    nAttribute = len(d)
    cdef ncomp.ncomp_attributes* out_attrs = ncomp.ncomp_attributes_allocate(nAttribute)
    print(out_attrs.nAttribute)
    for i, k in enumerate(d):
        v = d[k]
        out_attrs.attribute_array[i] = np_to_ncomp_single_attribute(k, v)
    return out_attrs

cdef ncomp_attributes_to_dict(ncomp.ncomp_attributes attrs):
    d = {}
    cdef ncomp.ncomp_single_attribute* attr
    for i in range(attrs.nAttribute):
        attr = (attrs.attribute_array)[i]
        d[attr.name] = ncomp_to_np_array(attr.value)
    return d
