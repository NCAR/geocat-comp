# cython: language_level=3

cimport numpy as np
from . cimport libncomp

cdef class Array:
    cdef libncomp.ncomp_array* ncomp
    cdef np.ndarray            numpy
    cdef int                   ndim
    cdef int                   type
    cdef void*                 addr
    cdef size_t*               shape

    @staticmethod
    cdef Array from_np(np.ndarray nparr)

    @staticmethod
    cdef Array from_ncomp(libncomp.ncomp_array* ncarr)
