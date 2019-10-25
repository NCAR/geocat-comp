from unittest import TestCase
from geocat.comp._ncomp import _eofunc_n
import numpy as np


class Test_eofunc_n(TestCase):
    def test_eofunc_n_pyx_01(self):
        data = np.arange(64, dtype='double').reshape((4, 4, 4))

        response = _eofunc_n(data, 1, 1)

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        for e in np.nditer(result):
            self.assertAlmostEqual(0.25, e, 2)

        properties = response[1]
        self.assertEqual(5, len(properties))

        self.assertAlmostEqual(85.33333, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(100.0, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(426.66666, properties[b'eval'][0], 4)

        self.assertEqual("covariance", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])