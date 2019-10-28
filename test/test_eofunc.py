from unittest import TestCase
import numpy as np

from geocat.comp._ncomp import _eofunc
from geocat.comp._ncomp import _eofunc_n


class Test_eofunc(TestCase):
    def test_eofunc_pyx_01(self):
        data = np.arange(64, dtype='double').reshape((4, 4, 4))

        response = _eofunc(data, 1)

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        for e in np.nditer(result):
            self.assertAlmostEqual(0.25, e, 2)

        properties = response[1]
        self.assertEqual(5, len(properties))

        self.assertAlmostEqual(5.33333, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(100.0, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(26.66666, properties[b'eval'][0], 4)

        self.assertEqual("covariance", properties[b'matrix'].tostring().decode('ascii')[:-1])
        print(properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])

    def test_eofunc_pyx_02(self):
        data = np.arange(64, dtype='double').reshape((4, 4, 4))

        options = {b'jopt': np.asarray([1])}
        response = _eofunc(data, 1, options)

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        for e in np.nditer(result):
            self.assertAlmostEqual(0.25, e, 2)

        properties = response[1]
        self.assertEqual(5, len(properties))

        self.assertAlmostEqual(3.20000, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(100.0, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(16.00000, properties[b'eval'][0], 4)

        self.assertEqual("correlation", properties[b'matrix'].tostring().decode('ascii')[:-1])
        print(properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])

    def test_eofunc_pyx_03(self):
        data = np.arange(64, dtype='double').reshape((4, 4, 4))

        options = {
            b'jopt': np.asarray(1.0),
            b'pcrit': np.asarray(32.0)
        }
        response = _eofunc(data, 1, options)

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        for e in np.nditer(result):
            self.assertAlmostEqual(0.25, e, 2)

        properties = response[1]
        self.assertEqual(6, len(properties))

        self.assertAlmostEqual(3.20000, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(100.0, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(16.00000, properties[b'eval'][0], 4)
        self.assertAlmostEqual(32.00000, properties[b'pcrit'][0], 4)
        self.assertEqual("correlation", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])

    def test_eofunc_pyx_03_1(self):
        data = np.arange(64, dtype='double').reshape((4, 4, 4))

        options = {
            b'jopt': np.asarray(1.0),
            b'pcrit': np.asarray(32)
        }
        response = _eofunc(data, 1, options)

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        for e in np.nditer(result):
            self.assertAlmostEqual(0.25, e, 2)

        properties = response[1]
        self.assertEqual(6, len(properties))

        self.assertAlmostEqual(3.20000, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(100.0, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(16.00000, properties[b'eval'][0], 4)
        self.assertAlmostEqual(32.00000, properties[b'pcrit'][0], 4)
        self.assertEqual("correlation", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])

    def test_eofunc_pyx_04(self):
        data = np.asarray(
            [0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99, 18, -99, 20, 21, 22, -99, 24, 25, 26, 27, 28, -99, 30, -99, 32, 33, 34, 35, 36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
            dtype='double'
        ).reshape((4, 4, 4))

        data[data == -99] = np.nan

        options = {
            b'jopt': np.asarray(1.0),
            b'pcrit': np.asarray(32.0)
        }

        response = _eofunc(data, 1, options)

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        expected_results = [0.0600, 0.1257, 0.1778, 0.2675, 0.1257, 0.1778, 0.3404, 0.1257, 0.3404, 0.2675, 0.1257, 0.1778, 0.3404, 0.3404, 0.3404, 0.3404]
        for e in zip(expected_results, result.reshape((16, 1)).tolist()):
            self.assertAlmostEqual(e[0], e[1][0], 4)

        properties = response[1]
        self.assertEqual(6, len(properties))

        self.assertAlmostEqual(2.9852, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(98.71625, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(14.9260, properties[b'eval'][0], 4)

        self.assertEqual("correlation", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])
        self.assertAlmostEqual(32.00000, properties[b'pcrit'][0], 4)

        self.assertTrue(np.isnan(data[0, 0, 3]))

    def test_eofunc_pyx_05(self):
        data = np.asarray(
            [0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99, 18, -99, 20, 21, 22, -99, 24, 25, 26, 27, 28, -99, 30, -99, 32, 33, 34, 35, 36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
            dtype='double'
        ).reshape((4, 4, 4))

        data[data == -99] = np.nan

        options = {
            b'jopt': np.asarray(1.0),
            b'pcrit': np.asarray(32.0)
        }

        response = _eofunc(data, 1, options, missing_value=np.nan)

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        expected_results = [0.0600, 0.1257, 0.1778, 0.2675, 0.1257, 0.1778, 0.3404, 0.1257, 0.3404, 0.2675, 0.1257, 0.1778, 0.3404, 0.3404, 0.3404, 0.3404]
        for e in zip(expected_results, result.reshape((16, 1)).tolist()):
            self.assertAlmostEqual(e[0], e[1][0], 4)

        properties = response[1]
        self.assertEqual(6, len(properties))

        self.assertAlmostEqual(2.9852, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(98.71625, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(14.9260, properties[b'eval'][0], 4)

        self.assertEqual("correlation", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])
        self.assertAlmostEqual(32.00000, properties[b'pcrit'][0], 4)

        self.assertTrue(np.isnan(data[0, 0, 3]))

    def test_eofunc_pyx_06(self):
        data = np.asarray(
            [0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99, 18, -99, 20, 21, 22, -99, 24, 25, 26, 27, 28, -99, 30, -99, 32, 33, 34, 35, 36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
            dtype='double'
        ).reshape((4, 4, 4))

        data[data == -99] = np.nan

        options = {
            b'jopt': np.asarray(1.0),
            b'pcrit': np.asarray(32.0)
        }

        with self.assertRaises(ValueError):
            _eofunc(data, 1, options, missing_value=42)

    def test_eofunc_pyx_07(self):
        data = np.asarray(
            [0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99, 18, -99, 20, 21, 22, -99, 24, 25, 26, 27, 28, -99, 30, -99, 32, 33, 34, 35, 36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
            dtype='double'
        ).reshape((4, 4, 4))

        options = {
            b'jopt': np.asarray(1.0),
            b'pcrit': np.asarray(32.0)
        }

        response = _eofunc(data, 1, options, missing_value=-99.0)

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        expected_results = [0.0600, 0.1257, 0.1778, 0.2675, 0.1257, 0.1778, 0.3404, 0.1257, 0.3404, 0.2675, 0.1257, 0.1778, 0.3404, 0.3404, 0.3404, 0.3404]
        for e in zip(expected_results, result.reshape((16, 1)).tolist()):
            self.assertAlmostEqual(e[0], e[1][0], 4)

        properties = response[1]
        self.assertEqual(6, len(properties))

        self.assertAlmostEqual(2.9852, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(98.71625, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(14.9260, properties[b'eval'][0], 4)

        self.assertEqual("correlation", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])
        self.assertAlmostEqual(32.00000, properties[b'pcrit'][0], 4)

        self.assertAlmostEqual(-99.0, data[0, 0, 3], 1)

    def test_eofunc_pyx_08(self):
        data = np.asarray(
            [0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99, 18, -99, 20, 21, 22, -99, 24, 25, 26, 27, 28, -99, 30, -99, 32, 33, 34, 35, 36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
            dtype='double'
        ).reshape((4, 4, 4))

        options = {
            b'jopt': np.asarray(1.0),
            b'pcrit': np.asarray(32.0)
        }

        response = _eofunc(data, 1, options, missing_value=-99) #None-double

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        expected_results = [0.0600, 0.1257, 0.1778, 0.2675, 0.1257, 0.1778, 0.3404, 0.1257, 0.3404, 0.2675, 0.1257, 0.1778, 0.3404, 0.3404, 0.3404, 0.3404]
        for e in zip(expected_results, result.reshape((16, 1)).tolist()):
            self.assertAlmostEqual(e[0], e[1][0], 4)

        properties = response[1]
        self.assertEqual(6, len(properties))

        self.assertAlmostEqual(2.9852, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(98.71625, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(14.9260, properties[b'eval'][0], 4)

        self.assertEqual("correlation", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])
        self.assertAlmostEqual(32.00000, properties[b'pcrit'][0], 4)

        self.assertAlmostEqual(-99.0, data[0, 0, 3], 1)

    def test_eofunc_pyx_09(self):
        data = np.asarray(
            [0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99, 18, -99, 20, 21, 22, -99, 24, 25, 26, 27, 28, -99, 30, -99, 32, 33, 34, 35, 36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
            dtype='double'
        ).reshape((4, 4, 4))

        options = {
            b'jopt': np.asarray(1.0),
            b'pcrit': np.asarray(32.0)
        }
        response = _eofunc(data, 1, options, missing_value=np.int8(-99))  # None-double np.number

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        expected_results = [0.0600, 0.1257, 0.1778, 0.2675, 0.1257, 0.1778, 0.3404, 0.1257, 0.3404, 0.2675, 0.1257, 0.1778, 0.3404, 0.3404, 0.3404, 0.3404]
        for e in zip(expected_results, result.reshape((16, 1)).tolist()):
            self.assertAlmostEqual(e[0], e[1][0], 4)

        properties = response[1]
        self.assertEqual(6, len(properties))

        self.assertAlmostEqual(2.9852, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(98.71625, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(14.9260, properties[b'eval'][0], 4)

        self.assertEqual("correlation", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])
        self.assertAlmostEqual(32.00000, properties[b'pcrit'][0], 4)

        self.assertAlmostEqual(-99.0, data[0, 0, 3], 1)

    def test_scratch_00(self):
        data = np.arange(64, dtype='double').reshape((4, 4, 4))

        data[0, 0, 3] = np.nan

        response = _eofunc(data, 1, {}, missing_value=np.nan)


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

    def test_eofunc_n_pyx_02(self):
        data = np.arange(64, dtype='double').reshape((4, 4, 4))

        options = {
            b'jopt': np.asarray(1)
        }

        response = _eofunc_n(data, 1, 1, options)

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        for e in np.nditer(result):
            self.assertAlmostEqual(0.25, e, 2)

        properties = response[1]
        self.assertEqual(5, len(properties))

        self.assertAlmostEqual(3.2000, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(100.0, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(16.0000, properties[b'eval'][0], 4)

        self.assertEqual("correlation", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])

    def test_eofunc_n_pyx_03(self):
        data = np.arange(64, dtype='double').reshape((4, 4, 4))

        options = {
            b'jopt': np.asarray(0),
            b'pcrit': np.asarray(32.0)
        }

        response = _eofunc_n(data, 1, 0, options)

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        for e in np.nditer(result):
            self.assertAlmostEqual(0.25, e, 2)

        properties = response[1]
        self.assertEqual(6, len(properties))

        self.assertAlmostEqual(1365.3333, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(100.0, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(6826.6667, properties[b'eval'][0], 4)
        self.assertAlmostEqual(32.00000, properties[b'pcrit'][0], 4)

        self.assertEqual("covariance", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])

    def test_eofunc_n_pyx_03_1(self):
        data = np.arange(64, dtype='double').reshape((4, 4, 4))

        options = {
            b'jopt': np.asarray(0),
            b'pcrit': np.asarray(32)
        }

        response = _eofunc_n(data, 1, 0, options)

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        for e in np.nditer(result):
            self.assertAlmostEqual(0.25, e, 2)

        properties = response[1]
        self.assertEqual(6, len(properties))

        self.assertAlmostEqual(1365.3333, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(100.0, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(6826.6667, properties[b'eval'][0], 4)
        self.assertAlmostEqual(32.00000, properties[b'pcrit'][0], 4)

        self.assertEqual("covariance", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])

    def test_eofunc_n_pyx_04(self):
        data = np.asarray(
            [0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99, 18, -99, 20, 21, 22, -99, 24, 25,
             26, 27, 28, -99, 30, -99, 32, 33, 34, 35, 36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
            dtype='double'
        ).reshape((4, 4, 4))

        data[data == -99] = np.nan

        options = {
            b'jopt': np.asarray(0),
            b'pcrit': np.asarray(32.0)
        }

        response = _eofunc_n(data, 1, 1, options)

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        expected_results = [0.3139, 0.1243, 0.1274, -99.0, 0.3139, 0.0318, 0.3139, -99.0, 0.3139, 0.2821, 0.3139, 0.0303, 0.3139, 0.3139, 0.3139, 0.3139]
        for e in zip(expected_results, result.reshape((16, 1)).tolist()):
            if (e[0] != -99):
                self.assertAlmostEqual(e[0], e[1][0], 4)
            else:
                self.assertTrue(np.isnan(e[1]))

        properties = response[1]
        self.assertEqual(6, len(properties))

        self.assertAlmostEqual(84.75415, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(102.4951, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(339.0166, properties[b'eval'][0], 4)
        self.assertAlmostEqual(32.00000, properties[b'pcrit'][0], 4)

        self.assertEqual("covariance", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])

        self.assertTrue(np.isnan(data[0, 0, 3]))

    def test_eofunc_n_pyx_05(self):
        data = np.asarray(
            [0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99, 18, -99, 20, 21, 22, -99, 24, 25,
             26, 27, 28, -99, 30, -99, 32, 33, 34, 35, 36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
            dtype='double'
        ).reshape((4, 4, 4))

        data[data == -99] = np.nan

        options = {
            b'jopt': np.asarray(0),
            b'pcrit': np.asarray(32.0)
        }

        response = _eofunc_n(data, 1, 1, options, missing_value=np.nan)

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        expected_results = [0.3139, 0.1243, 0.1274, -99.0, 0.3139, 0.0318, 0.3139, -99.0, 0.3139, 0.2821, 0.3139, 0.0303, 0.3139, 0.3139, 0.3139, 0.3139]
        for e in zip(expected_results, result.reshape((16, 1)).tolist()):
            if (e[0] != -99):
                self.assertAlmostEqual(e[0], e[1][0], 4)
            else:
                self.assertTrue(np.isnan(e[1]))

        properties = response[1]
        self.assertEqual(6, len(properties))

        self.assertAlmostEqual(84.75415, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(102.4951, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(339.0166, properties[b'eval'][0], 4)
        self.assertAlmostEqual(32.00000, properties[b'pcrit'][0], 4)

        self.assertEqual("covariance", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])

        self.assertTrue(np.isnan(data[0, 0, 3]))

    def test_eofunc_n_pyx_06(self):
        data = np.asarray(
            [0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99, 18, -99, 20, 21, 22, -99, 24, 25,
             26, 27, 28, -99, 30, -99, 32, 33, 34, 35, 36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
            dtype='double'
        ).reshape((4, 4, 4))

        data[data == -99] = np.nan

        options = {
            b'jopt': np.asarray(0),
            b'pcrit': np.asarray(32.0)
        }

        with self.assertRaises(ValueError):
            response = _eofunc_n(data, 1, 1, options, missing_value=42)

    def test_eofunc_n_pyx_07(self):
        data = np.asarray(
            [0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99, 18, -99, 20, 21, 22, -99, 24, 25,
             26, 27, 28, -99, 30, -99, 32, 33, 34, 35, 36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
            dtype='double'
        ).reshape((4, 4, 4))

        options = {
            b'jopt': np.asarray(0),
            b'pcrit': np.asarray(32.0)
        }

        response = _eofunc_n(data, 1, 1, options, missing_value=np.float64(-99.0))

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        expected_results = [0.3139, 0.1243, 0.1274, -99.0, 0.3139, 0.0318, 0.3139, -99.0, 0.3139, 0.2821, 0.3139, 0.0303, 0.3139, 0.3139, 0.3139, 0.3139]
        for e in zip(expected_results, result.reshape((16, 1)).tolist()):
            if (e[0] != -99):
                self.assertAlmostEqual(e[0], e[1][0], 4)
            else:
                self.assertTrue(np.isnan(e[1]))

        properties = response[1]
        self.assertEqual(6, len(properties))

        self.assertAlmostEqual(84.75415, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(102.4951, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(339.0166, properties[b'eval'][0], 4)
        self.assertAlmostEqual(32.00000, properties[b'pcrit'][0], 4)

        self.assertEqual("covariance", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])

        self.assertEqual(-99, data[0, 0, 3])

    def test_eofunc_n_pyx_08(self):
        data = np.asarray(
            [0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99, 18, -99, 20, 21, 22, -99, 24, 25,
             26, 27, 28, -99, 30, -99, 32, 33, 34, 35, 36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
            dtype='double'
        ).reshape((4, 4, 4))

        options = {
            b'jopt': np.asarray(0),
            b'pcrit': np.asarray(32.0)
        }

        response = _eofunc_n(data, 1, 1, options, missing_value=-99)

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        expected_results = [0.3139, 0.1243, 0.1274, -99.0, 0.3139, 0.0318, 0.3139, -99.0, 0.3139, 0.2821, 0.3139, 0.0303, 0.3139, 0.3139, 0.3139, 0.3139]
        for e in zip(expected_results, result.reshape((16, 1)).tolist()):
            if (e[0] != -99):
                self.assertAlmostEqual(e[0], e[1][0], 4)
            else:
                self.assertTrue(np.isnan(e[1]))

        properties = response[1]
        self.assertEqual(6, len(properties))

        self.assertAlmostEqual(84.75415, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(102.4951, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(339.0166, properties[b'eval'][0], 4)
        self.assertAlmostEqual(32.00000, properties[b'pcrit'][0], 4)

        self.assertEqual("covariance", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])

        self.assertEqual(-99, data[0, 0, 3])

    def test_eofunc_n_pyx_09(self):
        data = np.asarray(
            [0, 1, -99, -99, 4, -99, 6, -99, 8, 9, 10, -99, 12, -99, 14, 15, 16, -99, 18, -99, 20, 21, 22, -99, 24, 25,
             26, 27, 28, -99, 30, -99, 32, 33, 34, 35, 36, -99, 38, 39, 40, -99, 42, -99, 44, 45, 46, -99, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
            dtype='double'
        ).reshape((4, 4, 4))

        options = {
            b'jopt': np.asarray(0),
            b'pcrit': np.asarray(32.0)
        }

        response = _eofunc_n(data, 1, 1, options, missing_value=np.int8(-99))

        result = response[0]

        self.assertEqual((1, 4, 4), result.shape)

        expected_results = [0.3139, 0.1243, 0.1274, -99.0, 0.3139, 0.0318, 0.3139, -99.0, 0.3139, 0.2821, 0.3139, 0.0303, 0.3139, 0.3139, 0.3139, 0.3139]
        for e in zip(expected_results, result.reshape((16, 1)).tolist()):
            if (e[0] != -99):
                self.assertAlmostEqual(e[0], e[1][0], 4)
            else:
                self.assertTrue(np.isnan(e[1]))

        properties = response[1]
        self.assertEqual(6, len(properties))

        self.assertAlmostEqual(84.75415, properties[b'eval_transpose'][0], 4)
        self.assertAlmostEqual(102.4951, properties[b'pcvar'][0], 1)
        self.assertAlmostEqual(339.0166, properties[b'eval'][0], 4)
        self.assertAlmostEqual(32.00000, properties[b'pcrit'][0], 4)

        self.assertEqual("covariance", properties[b'matrix'].tostring().decode('ascii')[:-1])
        self.assertEqual("transpose", properties[b'method'].tostring().decode('ascii')[:-1])

        self.assertEqual(-99, data[0, 0, 3])
