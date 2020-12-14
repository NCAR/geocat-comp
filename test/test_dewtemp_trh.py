import unittest
import pytest

import numpy as np
from geocat.comp.dewtemp_trh import dewtemp_trh

rh_td_1 = 6.3


class Ttest_relhum(unittest.TestCase):
    def test_single_run(self):
        tk = 18. + 273.15
        rh = 46.5

        assert dewtemp_trh(tk, rh) - 273.15 == pytest.approx(rh_td_1, 0.1)