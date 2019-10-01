import numpy as np
import xarray as xr
from geocat.comp._ncomp import carrayify

import sys
import time
import unittest as ut

class Test_carrayify(ut.TestCase):
    def test_carrayify_one_arg(self):
        @carrayify
        def new_func(a):
            return a
        a = np.array([1, 2, 3])
        b = a[::-1]
        c = new_func(a)
        self.assertTrue(a.flags.carray)
        self.assertFalse(b.flags.carray)
        self.assertTrue(c.flags.carray)
