from unittest import TestCase
import numpy as np

from geocat.comp._ncomp import _eofunc


class Test_eofunc(TestCase):
    def test_eofunc_pyx_01(self):
        data = np.arange(64, dtype='double').reshape((4, 4, 4))

        response = _eofunc(data, 1)

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        for e in np.nditer(result):
            self.assertEqual(0.25, e)

        properties = response[1]
        self.assertEqual(5, len(properties))

        self.assertAlmostEqual(5.33333, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(100.0, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(26.66666, properties[b'eval'][0], 4)


