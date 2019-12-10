from geocat.comp cimport _ncomp

import unittest
import numpy as np

class TestNCompCython(unittest.TestCase):
    def test_numpy_array(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        a_Array = _ncomp.Array.from_np(a_np)
        np.testing.assert_array_equal(a_np, a_Array.numpy)

        np.testing.assert_equal(a_np.ndim, a_Array.ndim)
        np.testing.assert_equal(a_np.dtype.num, a_Array.type)
        for dim in np.arange(a_np.ndim):
            np.testing.assert_equal(a_np.shape[dim], a_Array.shape[dim])

        np.testing.assert_equal(a_np.ndim, a_Array.ncomp.ndim)
        np.testing.assert_equal(a_np.dtype.num, a_Array.ncomp.type)
        for dim in np.arange(a_np.ndim):
            np.testing.assert_equal(a_np.shape[dim], a_Array.ncomp.shape[dim])

        np.testing.assert_equal(a_np[0,0], (<double*>a_Array.ncomp.addr)[0])
