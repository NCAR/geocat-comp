# cython: language_level=3, boundscheck=False, embedsignature=True
cimport ncomp as libncomp
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


cdef class Array:
    cdef libncomp.ncomp_array* ncomp
    cdef np.ndarray            numpy
    cdef int                   ndim
    cdef int                   type
    cdef void*                 addr
    cdef size_t*               shape

    def __init__(self):
        raise NotImplementedError("_ncomp.Array must be instantiated using the from_np or from_ncomp methods.")

    cdef libncomp.ncomp_array* np_to_ncomp_array(self):
        return <libncomp.ncomp_array*> libncomp.ncomp_array_alloc(self.addr, self.type, self.ndim, self.shape)

    cdef np.ndarray ncomp_to_np_array(self):
        np.import_array()
        nparr = np.PyArray_SimpleNewFromData(self.ndim, <np.npy_intp *> self.shape, self.type, self.addr)
        cdef extern from "numpy/arrayobject.h":
            void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
        PyArray_ENABLEFLAGS(nparr, np.NPY_OWNDATA)
        return nparr

    @staticmethod
    cdef Array from_np(np.ndarray nparr):
        cdef Array a = Array.__new__(Array)
        a.numpy = nparr
        a.ndim = nparr.ndim
        a.shape = <size_t*>nparr.shape
        a.type = nparr.dtype.num
        a.addr = <void*> (<unsigned long> nparr.__array_interface__['data'][0])
        a.ncomp = a.np_to_ncomp_array()
        return a

    @staticmethod
    cdef Array from_ncomp(libncomp.ncomp_array* ncarr):
        cdef Array a = Array.__new__(Array)
        a.ncomp = ncarr
        a.ndim = ncarr.ndim
        a.shape = ncarr.shape
        a.type = ncarr.type
        a.addr = ncarr.addr
        a.numpy = a.ncomp_to_np_array()
        return a

    def __dealloc__(self):
        if self.ncomp is not NULL:
            libncomp.ncomp_array_free(self.ncomp, 1)


dtype_default_fill = {
             "DEFAULT_FILL":       libncomp.DEFAULT_FILL_DOUBLE,
             np.dtype(np.int8):    np.int8(libncomp.DEFAULT_FILL_INT8),
             np.dtype(np.uint8):   np.uint8(libncomp.DEFAULT_FILL_UINT8),
             np.dtype(np.int16):   np.int16(libncomp.DEFAULT_FILL_INT16),
             np.dtype(np.uint16):  np.uint16(libncomp.DEFAULT_FILL_UINT16),
             np.dtype(np.int32):   np.int32(libncomp.DEFAULT_FILL_INT32),
             np.dtype(np.uint32):  np.uint32(libncomp.DEFAULT_FILL_UINT32),
             np.dtype(np.int64):   np.int64(libncomp.DEFAULT_FILL_INT64),
             np.dtype(np.uint64):  np.uint64(libncomp.DEFAULT_FILL_UINT64),
             np.dtype(np.float32): np.float32(libncomp.DEFAULT_FILL_FLOAT),
             np.dtype(np.float64): np.float64(libncomp.DEFAULT_FILL_DOUBLE),
            }


dtype_to_ncomp = {np.dtype(np.bool):       libncomp.NCOMP_BOOL,
                  np.dtype(np.int8):       libncomp.NCOMP_BYTE,
                  np.dtype(np.uint8):      libncomp.NCOMP_UBYTE,
                  np.dtype(np.int16):      libncomp.NCOMP_SHORT,
                  np.dtype(np.uint16):     libncomp.NCOMP_USHORT,
                  np.dtype(np.int32):      libncomp.NCOMP_INT,
                  np.dtype(np.uint32):     libncomp.NCOMP_UINT,
                  np.dtype(np.int64):      libncomp.NCOMP_LONG,
                  np.dtype(np.uint64):     libncomp.NCOMP_ULONG,
                  np.dtype(np.longlong):   libncomp.NCOMP_LONGLONG,
                  np.dtype(np.ulonglong):  libncomp.NCOMP_ULONGLONG,
                  np.dtype(np.float32):    libncomp.NCOMP_FLOAT,
                  np.dtype(np.float64):    libncomp.NCOMP_DOUBLE,
                  np.dtype(np.float128):   libncomp.NCOMP_LONGDOUBLE,
                 }


ncomp_to_dtype = {libncomp.NCOMP_BOOL:         np.bool,
                  libncomp.NCOMP_BYTE:         np.int8,
                  libncomp.NCOMP_UBYTE:        np.uint8,
                  libncomp.NCOMP_SHORT:        np.int16,
                  libncomp.NCOMP_USHORT:       np.uint16,
                  libncomp.NCOMP_INT:          np.int32,
                  libncomp.NCOMP_UINT:         np.uint32,
                  libncomp.NCOMP_LONG:         np.int64,
                  libncomp.NCOMP_ULONG:        np.uint64,
                  libncomp.NCOMP_LONGLONG:     np.longlong,
                  libncomp.NCOMP_ULONGLONG:    np.ulonglong,
                  libncomp.NCOMP_FLOAT:        np.float32,
                  libncomp.NCOMP_DOUBLE:       np.float64,
                  libncomp.NCOMP_LONGDOUBLE:   np.float128,
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


cdef libncomp.ncomp_array* np_to_ncomp_array(np.ndarray nparr):
    cdef void* addr = <void*> (<unsigned long> nparr.__array_interface__['data'][0])
    cdef int ndim = nparr.ndim
    cdef size_t* shape = <size_t*> nparr.shape
    cdef int np_type = nparr.dtype.num

    return <libncomp.ncomp_array*> libncomp.ncomp_array_alloc(addr, np_type, ndim, shape)

cdef np.ndarray ncomp_to_np_array(libncomp.ncomp_array* ncarr):
    np.import_array()
    nparr = np.PyArray_SimpleNewFromData(ncarr.ndim, <np.npy_intp *> ncarr.shape, ncarr.type, ncarr.addr)
    cdef extern from "numpy/arrayobject.h":
        void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    PyArray_ENABLEFLAGS(nparr, np.NPY_OWNDATA)
    return nparr

cdef set_ncomp_msg(libncomp.ncomp_missing* ncomp_msg, num):
    ncomp_type = num.dtype.num
    if ncomp_type == libncomp.NCOMP_FLOAT:
        ncomp_msg.msg_float = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == libncomp.NCOMP_DOUBLE:
        ncomp_msg.msg_double = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == libncomp.NCOMP_BOOL:
        ncomp_msg.msg_bool = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == libncomp.NCOMP_BYTE:
        ncomp_msg.msg_byte = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == libncomp.NCOMP_UBYTE:
        ncomp_msg.msg_ubyte = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == libncomp.NCOMP_SHORT:
        ncomp_msg.msg_short = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == libncomp.NCOMP_USHORT:
        ncomp_msg.msg_ushort = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == libncomp.NCOMP_INT:
        ncomp_msg.msg_int = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == libncomp.NCOMP_UINT:
        ncomp_msg.msg_uint = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == libncomp.NCOMP_LONG:
        ncomp_msg.msg_long = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == libncomp.NCOMP_ULONG:
        ncomp_msg.msg_ulong = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == libncomp.NCOMP_LONGLONG:
        ncomp_msg.msg_longlong = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == libncomp.NCOMP_ULONGLONG:
        ncomp_msg.msg_ulonglong = ncomp_to_dtype[ncomp_type](num)
    elif ncomp_type == libncomp.NCOMP_LONGDOUBLE:
        ncomp_msg.msg_longdouble = ncomp_to_dtype[ncomp_type](num)

@carrayify
def _linint2(np.ndarray xi_np, np.ndarray yi_np, np.ndarray fi_np, np.ndarray xo_np, np.ndarray yo_np, int icycx, msg=None):
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

    xi = Array.from_np(xi_np)
    yi = Array.from_np(yi_np)
    fi = Array.from_np(fi_np)
    xo = Array.from_np(xo_np)
    yo = Array.from_np(yo_np)

    cdef int iopt = 0
    cdef long i
    if fi.type == libncomp.NCOMP_DOUBLE:
        fo_dtype = np.float64
    else:
        fo_dtype = np.float32
    cdef np.ndarray fo_np = np.zeros(tuple([fi.shape[i] for i in range(fi.ndim - 2)] + [yo.shape[0], xo.shape[0]]), dtype=fo_dtype)

    missing_inds_fi = None

    if msg is None or np.isnan(msg): # if no missing value specified, assume NaNs
        missing_inds_fi = np.isnan(fi.numpy)
        msg = get_default_fill(fi.numpy)
    else:
        missing_inds_fi = (fi.numpy == msg)

    set_ncomp_msg(&(fi.ncomp.msg), msg) # always set missing on fi.ncomp

    if missing_inds_fi.any():
        fi.ncomp.has_missing = 1
        fi.numpy[missing_inds_fi] = msg

    fo = Array.from_np(fo_np)

#   release global interpreter lock
    cdef int ier
    with nogil:
        ier = libncomp.linint2(
            xi.ncomp, yi.ncomp, fi.ncomp,
            xo.ncomp, yo.ncomp, fo.ncomp,
            icycx, iopt)
#   re-acquire interpreter lock
#   check errors ier
    if ier:
        warnings.warn("linint2: {}: xi, yi, xo, and yo must be monotonically increasing".format(ier),
                      NcompWarning)

    if missing_inds_fi is not None and missing_inds_fi.any():
        fi.numpy[missing_inds_fi] = np.nan

    if fo.type == libncomp.NCOMP_DOUBLE:
        fo_msg = fo.ncomp.msg.msg_double
    else:
        fo_msg = fo.ncomp.msg.msg_float

    fo.numpy[fo.numpy == fo_msg] = np.nan

    return fo.numpy

cdef adjust_for_missing_values(np.ndarray np_input, libncomp.ncomp_array* ncomp_input, dict kwargs):
    missing_value = kwargs.get("missing_value", np.nan)

    missing_mask = None
    if np.isnan(missing_value):
        # print("No Missing value provided or it was already set to NaN.")
        missing_mask = np.isnan(np_input)
        missing_value = get_default_fill(np_input)
        np_input[missing_mask] = missing_value
    else:
        # print(f"Using provided Missing value: {missing_value}")
        if np.isnan(np_input).any():
            raise ValueError(
                "The missing value is set to a non-NaN value but the data still contains some NaN. "
                "Either change all the NaN numbers to your provided missing_value or "
                "change all the missing values to NaN and do not specify the missing_values or specify it as NaN"
            )
        if isinstance(missing_value, np.number):
            if missing_value.dtype != np_input.dtype:
                missing_value = missing_value.astype(np_input.dtype)
        else:
            missing_value = np.asarray([missing_value])[0].astype(np_input.dtype)
            # alternatively we could do:
            # missing_value = np.float128(missing_value).astype(np_input.dtype)
            # however, that's assuming the cating issing_calue to float128 doesn't change anything
            # prefer the asarray lines, because we let numpy to choose the proper type.

        missing_mask = (np_input == missing_value)

    if missing_mask.any():
        ncomp_input.has_missing = 1
        if isinstance(missing_value, np.number):
            set_ncomp_msg(&ncomp_input.msg, missing_value)
            if np_input.dtype != missing_value.dtype:
                raise TypeError(
                    "This should never be raised at this point. "
                    "By now the missing_value should have the proper type"
                )
        else:
            set_ncomp_msg(&ncomp_input.msg, np.float128(missing_value).astype(np_input.dtype))

    return missing_mask

cdef reverse_missing_values_adjustments(np.ndarray np_input, np.ndarray missing_mask, dict kwargs):
    missing_value = kwargs.get("missing_value", np.nan)

    if np.isnan(missing_value) and missing_mask.any():
        missing_value = get_default_fill(np_input)
        np_input[missing_mask] = np.nan

@carrayify
def _eofunc(np.ndarray np_input, int neval, opt={}, **kwargs):
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

      opt (:obj:`dict`):
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

      kwargs:
        extra parameters to control the behavior of the function, such as missing_value

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
    input = Array.from_np(np_input)
    missing_mask = adjust_for_missing_values(input.numpy, input.ncomp, kwargs)

    # convert opt dict to ncomp_attributes struct
    cdef libncomp.ncomp_attributes* attrs = dict_to_ncomp_attributes(opt)

    # allocate output ncomp_array and ncomp_attributes
    cdef libncomp.ncomp_array* ncomp_output = NULL
    cdef libncomp.ncomp_attributes attrs_output

    cdef int ier
    with nogil:
        ier = libncomp.eofunc(input.ncomp, neval, attrs, &ncomp_output, &attrs_output)

    # convert ncomp_output to np.ndarray
    output = Array.from_ncomp(ncomp_output)

    # making sure that output missing values is NaN
    output_missing_value = output.ncomp.msg.msg_double \
            if output.ncomp.type == libncomp.NCOMP_DOUBLE \
            else output.ncomp.msg.msg_float

    output.numpy[output.numpy == output_missing_value] = np.nan

    # convert attrs_output to dict
    np_attrs_dict = ncomp_attributes_to_dict(attrs_output)

    # Reversing the changed values
    reverse_missing_values_adjustments(input.numpy, missing_mask, kwargs)

    return (output.numpy, np_attrs_dict)

@carrayify
def _eofunc_n(np.ndarray np_input, int neval, int t_dim, opt={}, **kwargs):
    # convert np_input to ncomp_array
    input = Array.from_np(np_input)
    missing_mask = adjust_for_missing_values(input.numpy, input.ncomp, kwargs)

    # convert opt dict to ncomp_attributes struct
    cdef libncomp.ncomp_attributes* attrs = dict_to_ncomp_attributes(opt)

    # allocate output ncomp_array and ncomp_attributes
    cdef libncomp.ncomp_array* ncomp_output = NULL
    cdef libncomp.ncomp_attributes attrs_output

    cdef int ier
    with nogil:
        ier = libncomp.eofunc_n(input.ncomp, neval, t_dim, attrs, &ncomp_output, &attrs_output)

    # convert ncomp_output to np.ndarray
    output = Array.from_ncomp(ncomp_output)

    # making sure that output missing values is NaN
    output_missing_value = output.ncomp.msg.msg_double \
            if output.ncomp.type == libncomp.NCOMP_DOUBLE \
            else output.ncomp.msg.msg_float

    output.numpy[output.numpy == output_missing_value] = np.nan

    # convert attrs_output to dict
    np_attrs_dict = ncomp_attributes_to_dict(attrs_output)

    # Reversing the changed values
    reverse_missing_values_adjustments(input.numpy, missing_mask, kwargs)

    return (output.numpy, np_attrs_dict)

cdef libncomp.ncomp_single_attribute* np_to_ncomp_single_attribute(char* name, np.ndarray nparr):
    cdef long long_addr = nparr.__array_interface__['data'][0]
    cdef void* addr = <void*> long_addr
    cdef int ndim = nparr.ndim
    cdef size_t* shape = <size_t*> nparr.shape
    cdef int np_type = nparr.dtype.num
    return <libncomp.ncomp_single_attribute*> libncomp.create_ncomp_single_attribute(name, addr, np_type, ndim, shape)

cdef libncomp.ncomp_attributes* dict_to_ncomp_attributes(d):
    nAttribute = len(d)
    cdef libncomp.ncomp_attributes* out_attrs = libncomp.ncomp_attributes_allocate(nAttribute)
    for i, k in enumerate(d):
        v = d[k]
        out_attrs.attribute_array[i] = np_to_ncomp_single_attribute(k, v)
    return out_attrs

cdef ncomp_attributes_to_dict(libncomp.ncomp_attributes attrs):
    d = {}
    cdef libncomp.ncomp_single_attribute* attr
    for i in range(attrs.nAttribute):
        attr = (attrs.attribute_array)[i]
        d[attr.name] = ncomp_to_np_array(attr.value)
    return d


@carrayify
def _moc_globe_atl(np.ndarray lat_aux_grid, np.ndarray a_wvel, np.ndarray a_bolus, np.ndarray a_submeso, np.ndarray tlat, np.ndarray rmlak, msg=None):
    """Facilitates calculating the meridional overturning circulation for the globe and Atlantic.
    Args:
    lat_aux_grid (:class:`numpy.ndarray`):
        Latitude grid for transport diagnostics.

    a_wvel (:class:`numpy.ndarray`):
        Area weighted Eulerian-mean vertical velocity [TAREA*WVEL].

    a_bolus (:class:`numpy.ndarray`):
        Area weighted Eddy-induced (bolus) vertical velocity [TAREA*WISOP].

    a_submeso (:class:`numpy.ndarray`):
        Area weighted submeso vertical velocity [TAREA*WSUBM].

    tlat (:class:`numpy.ndarray`):
        Array of t-grid latitudes.

    rmlak (:class:`numpy.ndarray`):
        Basin index number: [0]=Globe, [1]=Atlantic

    msg (:obj:`numpy.number`):
        A numpy scalar value that represent a missing value in a_wvel.
        This argument allows a user to use a missing value scheme
        other than NaN or masked arrays, similar to what NCL allows.

    Returns:
        :class:`numpy.ndarray`: A multi-dimensional array of size [moc_comp] x
        [n_transport_reg] x [kdepth] x [nyaux] where:

        - moc_comp refers to the three components returned
        - n_transport_reg refers to the Globe and Atlantic
        - kdepth is the the number of vertical levels of the work arrays
        - nyaux is the size of the lat_aux_grid

        The type of the output data will be double only if a_wvel or a_bolus or
        a_submesa is of type double. Otherwise, the return type will be float.
    """

    # Convert np_input to ncomp_array
    cdef libncomp.ncomp_array* ncomp_lat_aux_grid = np_to_ncomp_array(lat_aux_grid)
    cdef libncomp.ncomp_array* ncomp_a_wvel = np_to_ncomp_array(a_wvel)
    cdef libncomp.ncomp_array* ncomp_a_bolus = np_to_ncomp_array(a_bolus)
    cdef libncomp.ncomp_array* ncomp_a_submeso = np_to_ncomp_array(a_submeso)
    cdef libncomp.ncomp_array* ncomp_tlat = np_to_ncomp_array(tlat)
    cdef libncomp.ncomp_array* ncomp_rmlak = np_to_ncomp_array(rmlak)

    # Handle missing values
    missing_inds_a_wvel = None

    if msg is None or np.isnan(msg):    # if no missing value specified, assume NaNs
        missing_inds_a_wvel = np.isnan(a_wvel)
        msg = get_default_fill(a_wvel)
    else:
        missing_inds_a_wvel = (a_wvel == msg)

    set_ncomp_msg(&ncomp_a_wvel.msg, msg)    # always set missing on ncomp_a_wvel

    if missing_inds_a_wvel.any():
        ncomp_a_wvel.has_missing = 1
        a_wvel[missing_inds_a_wvel] = msg

    # Allocate output ncomp_array
    cdef libncomp.ncomp_array* ncomp_output = NULL

    cdef int ier
    with nogil:
        ier = libncomp.moc_globe_atl(ncomp_lat_aux_grid, ncomp_a_wvel, ncomp_a_bolus,
                                  ncomp_a_submeso, ncomp_tlat, ncomp_rmlak,
                                  &ncomp_output)

    # Check errors ier
    if ier:
      raise NcompError(f"moc_globe_atl: There is an error: {ier}")

    # Convert ncomp_output to np.ndarray
    output = Array.from_ncomp(ncomp_output)

    # Make sure output missing values are NaN
    output_missing_value = ncomp_output.msg.msg_double

    if ncomp_output.type != libncomp.NCOMP_DOUBLE:
        output_missing_value = ncomp_output.msg.msg_float

    # TODO: May need to revisit for output missing value
    # output.numpy[output.numpy == output_missing_value] = np.nan

    return output.numpy

def _rcm2rgrid(np.ndarray lat2d, np.ndarray lon2d, np.ndarray fi, np.ndarray lat1d, np.ndarray lon1d, msg=None):
    """Interpolates data on a curvilinear grid (i.e. RCM, WRF, NARR) to a rectilinear grid.

    Args:

        lat2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the latitudes locations
	    of fi. Because this array is two-dimensional it is not an associated
	    coordinate variable of `fi`. The latitude order must be south-to-north.

        lon2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the longitude locations
	    of fi. Because this array is two-dimensional it is not an associated
	    coordinate variable of `fi`. The latitude order must be west-to-east.

        fi (:class:`numpy.ndarray`):
	    A multi-dimensional array to be interpolated. The rightmost two
	    dimensions (latitude, longitude) are the dimensions to be interpolated.

        lat1d (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the latitude coordinates of
	    the regular grid. Must be monotonically increasing.

        lon1d (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the longitude coordinates of
	    the regular grid. Must be monotonically increasing.

        msg (:obj:`numpy.number`):
            A numpy scalar value that represent a missing value in fi.
            This argument allows a user to use a missing value scheme
            other than NaN or masked arrays, similar to what NCL allows.

    Returns:
        :class:`numpy.ndarray`: The interpolated grid. A multi-dimensional array
	of the same size as fi except that the rightmost dimension sizes have been
	replaced by the sizes of lat1d and lon1d respectively.
	Double if fi is double, otherwise float.

    Description:
        Interpolates RCM (Regional Climate Model), WRF (Weather Research and Forecasting) and
    	NARR (North American Regional Reanalysis) grids to a rectilinear grid. Actually, this
	function will interpolate most grids that use curvilinear latitude/longitude grids.
	No extrapolation is performed beyond the range of the input coordinates. Missing values
	are allowed but ignored.

	The weighting method used is simple inverse distance squared. Missing values are allowed
	but ignored.

	The code searches the input curvilinear grid latitudes and longitudes for the four
	grid points that surround a specified output grid coordinate. Because one or more of
	these input points could contain missing values, fewer than four points
	could be used in the interpolation.

	Curvilinear grids which have two-dimensional latitude and longitude coordinate axes present
	some issues because the coordinates are not necessarily monotonically increasing. The simple
	search algorithm used by rcm2rgrid is not capable of handling all cases. The result is that,
	sometimes, there are small gaps in the interpolated grids. Any interior points not
	interpolated in the initial interpolation pass will be filled using linear interpolation.
        In some cases, edge points may not be filled.

    """
    cdef libncomp.ncomp_array* ncomp_lat2d = np_to_ncomp_array(lat2d)
    cdef libncomp.ncomp_array* ncomp_lon2d = np_to_ncomp_array(lon2d)
    cdef libncomp.ncomp_array* ncomp_fi    = np_to_ncomp_array(fi)
    cdef libncomp.ncomp_array* ncomp_lat1d = np_to_ncomp_array(lat1d)
    cdef libncomp.ncomp_array* ncomp_lon1d = np_to_ncomp_array(lon1d)

    cdef libncomp.ncomp_array* ncomp_fo
    cdef long i

    # Set output type and dimensions. Double if `fi` is double, otherwise float.
    if ncomp_fi.type == libncomp.NCOMP_DOUBLE:
        fo_dtype = np.float64
    else:
        fo_dtype = np.float32
    cdef np.ndarray fo = np.zeros(tuple([fi.shape[i] for i in range(fi.ndim - 2)] + [lat1d.shape[0], lon1d.shape[0]]), dtype=fo_dtype)

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
        ier = libncomp.rcm2rgrid(
            ncomp_lat2d, ncomp_lon2d, ncomp_fi,
            ncomp_lat1d, ncomp_lon1d, ncomp_fo)

#   re-acquire interpreter lock
    # Check errors ier
    if ier:
      raise NcompError(f"rcm2rgrid: There is an error: {ier}")

    if missing_inds_fi is not None and missing_inds_fi.any():
        fi[missing_inds_fi] = np.nan

    if ncomp_fo.type == libncomp.NCOMP_DOUBLE:
        fo_msg = ncomp_fo.msg.msg_double
    else:
        fo_msg = ncomp_fo.msg.msg_float

    fo[fo == fo_msg] = np.nan

    return fo

def _rgrid2rcm(np.ndarray lat1d, np.ndarray lon1d, np.ndarray fi, np.ndarray lat2d, np.ndarray lon2d, msg=None):
    """Interpolates data on a rectilinear lat/lon grid to a curvilinear grid like those used by the RCM, WRF and NARR models/datasets.

    Args:

        lat1d (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the latitude coordinates of
	    the regular grid. Must be monotonically increasing.

        lon1d (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the longitude coordinates of
	    the regular grid. Must be monotonically increasing.

        fi (:class:`numpy.ndarray`):
	    A multi-dimensional array to be interpolated. The rightmost two
	    dimensions (latitude, longitude) are the dimensions to be interpolated.

        lat2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the latitude locations of `fi`.
	    Because this array is two-dimensional, it is not an associated
	    coordinate variable of `fi`.

        lon2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the longitude locations of `fi`.
	    Because this array is two-dimensional, it is not an associated
	    coordinate variable of `fi`.

        msg (:obj:`numpy.number`):
            A numpy scalar value that represent a missing value in fi.
            This argument allows a user to use a missing value scheme
            other than NaN or masked arrays, similar to what NCL allows.

    Returns:
        :class:`numpy.ndarray`: The interpolated grid. A multi-dimensional array of the
	same size as `fi` except that the rightmost dimension sizes have been replaced
	by the sizes of `lat2d` and `lon2d` respectively. Double if `fi` is double,
	otherwise float.

    Description:
        Interpolates data on a rectilinear lat/lon grid to a curvilinear grid, such as those
	used by the RCM (Regional Climate Model), WRF (Weather Research and Forecasting) and
	NARR (North American Regional Reanalysis) models/datasets. No extrapolation is
	performed beyond the range of the input coordinates. The method used is simple inverse
	distance weighting. Missing values are allowed but ignored.

    """
    cdef libncomp.ncomp_array* ncomp_lat1d = np_to_ncomp_array(lat1d)
    cdef libncomp.ncomp_array* ncomp_lon1d = np_to_ncomp_array(lon1d)
    cdef libncomp.ncomp_array* ncomp_fi    = np_to_ncomp_array(fi)
    cdef libncomp.ncomp_array* ncomp_lat2d = np_to_ncomp_array(lat2d)
    cdef libncomp.ncomp_array* ncomp_lon2d = np_to_ncomp_array(lon2d)

    cdef libncomp.ncomp_array* ncomp_fo
    cdef long i

    # Set output type and dimensions. Double if `fi` is double, otherwise float.
    if ncomp_fi.type == libncomp.NCOMP_DOUBLE:
        fo_dtype = np.float64
    else:
        fo_dtype = np.float32
    cdef np.ndarray fo = np.zeros(tuple([fi.shape[i] for i in range(fi.ndim - 2)] + [lat2d.shape[0], lon2d.shape[0]]), dtype=fo_dtype)

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
        ier = libncomp.rgrid2rcm(
            ncomp_lat1d, ncomp_lon1d, ncomp_fi,
            ncomp_lat2d, ncomp_lon2d, ncomp_fo)

#   re-acquire interpreter lock
    # Check errors ier
    if ier:
      raise NcompError(f"rgrid2rcm: There is an error: {ier}")

    if missing_inds_fi is not None and missing_inds_fi.any():
        fi[missing_inds_fi] = np.nan

    if ncomp_fo.type == libncomp.NCOMP_DOUBLE:
        fo_msg = ncomp_fo.msg.msg_double
    else:
        fo_msg = ncomp_fo.msg.msg_float

    fo[fo == fo_msg] = np.nan

    return fo
