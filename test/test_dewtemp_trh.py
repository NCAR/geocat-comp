import unittest
import pytest

import numpy as np
from geocat.comp.dewtemp_trh import dewtemp_trh

dt_td_1 = 6.3

dt_td_2 = [24.38342, 19.55563, 15.53281, 16.64218, 16.81433,
           14.22482, 9.401337, 6.149719, -4.1604, -5.096619,
           -6.528168, -12.61957, -19.38332, -25.00714, -28.9841,
           -33.34853, -46.51273, -58.18289]


class Ttest_relhum(unittest.TestCase):
    def test_single_run(self):
        tk = 18. + 273.15
        rh = 46.5

        assert dewtemp_trh(tk, rh) - 273.15 == pytest.approx(dt_td_1, 0.1)

    def test_array_run(self):
        t = np.asarray([29.3, 28.1, 23.5, 20.9, 18.4, 15.9, 13.1, 10.1, 6.7, 3.1, -0.5, -4.5, -9.0, -14.8, -21.5, -29.7,
                        -40.0, -52.4])
        rh = np.asarray(
            [75.0, 60.0, 61.1, 76.7, 90.5, 89.8, 78.3, 76.5, 46.0, 55.0, 63.8, 53.2, 42.9, 41.7, 51.0, 70.6, 50.0,
             50.0])

        tk = t + 273.15

        assert dewtemp_trh(tk, rh) - 273.15 == pytest.approx(dt_td_2, 0.1)
