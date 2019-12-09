from unittest import TestCase
from geocat.comp.polynomial import _ndpolyfit, ndpolyfit

import numpy as np
import xarray as xr


class test_internal_ndpolyfit(TestCase):
    def test_01(self):
        x = np.arange(10).astype(dtype=np.float)
        y = np.arange(10).astype(dtype=np.float)

        p = _ndpolyfit(x, y)

        np.testing.assert_almost_equal(p, [1.0, 0.0])

    def test_02(self):
        x = np.arange(10).astype(dtype=np.float).reshape((-1, 1))
        y = np.arange(10).astype(dtype=np.float)

        p = _ndpolyfit(x, y)

        np.testing.assert_almost_equal(p, [1.0, 0.0])

    def test_03(self):
        x = np.arange(10).astype(dtype=np.float).reshape((1, -1))
        y = np.arange(10).astype(dtype=np.float)

        p = _ndpolyfit(x, y)

        np.testing.assert_almost_equal(p, [1.0, 0.0])

    def test_04(self):
        x = np.arange(10).astype(dtype=np.float).reshape((2, 5))
        y = np.arange(10).astype(dtype=np.float)

        with self.assertRaises(ValueError):
            p = _ndpolyfit(x, y)

    def test_05(self):
        x = np.arange(10).astype(dtype=np.float)
        for i in range(50):
            expected_p = np.random.randint(-10, 10, size=2)
            while expected_p[0] == 0:
                expected_p = np.random.randint(-10, 10, size=2)

            y = expected_p[0] * x + expected_p[1]

            actual_p = _ndpolyfit(x, y)

            np.testing.assert_almost_equal(expected_p, actual_p)

    def test_06(self):
        x = np.arange(-10, 10).astype(dtype=np.float)
        y = np.arange(-10, 10).astype(dtype=np.float)

        y = np.moveaxis(np.tile(y, (4, 3, 1)), 2, 0)

        actual_p = _ndpolyfit(x, y)

        expected_p = np.moveaxis(np.tile(np.asarray([1.0, 0.0]), (4, 3, 1)), 2, 0)

        np.testing.assert_almost_equal(actual_p, expected_p)

    def test_07(self):
        x = np.arange(-10, 10).astype(dtype=np.float)
        y = np.arange(-10, 10).astype(dtype=np.float)

        axis = 1
        y = np.moveaxis(np.tile(y, (4, 3, 1)), 2, axis)

        actual_p = _ndpolyfit(x, y, axis=axis)

        expected_p = np.moveaxis(np.tile(np.asarray([1.0, 0.0]), (4, 3, 1)), 2, axis)

        np.testing.assert_almost_equal(actual_p, expected_p)

    def test_08(self):
        x = np.arange(-10, 10).astype(dtype=np.float)
        y = np.arange(-10, 10).astype(dtype=np.float)

        axis = 1
        y = np.moveaxis(np.tile(y, (4, 3, 1)), 2, axis)

        actual_p = _ndpolyfit(x, y, axis=axis)

        expected_p = np.moveaxis(np.tile(np.asarray([1.0, 0.0]), (4, 3, 1)), 2, axis)

        np.testing.assert_almost_equal(actual_p, expected_p)

    def test_09(self):
        x = np.arange(-10, 10).astype(dtype=np.float)
        max_dim = 6
        max_dim_size = 11

        for i in range(50):
            expected_p = np.random.randint(-10, 10, size=2)
            while expected_p[0] == 0:
                expected_p = np.random.randint(-10, 10, size=2)

            y = expected_p[0] * x + expected_p[1]

            other_dims = np.random.randint(1, max_dim_size, np.random.randint(1, max_dim))
            axis = np.random.randint(0, other_dims.ndim + 1)
            y = np.moveaxis(np.tile(y, (*other_dims, 1)), -1, axis)
            expected_p = np.moveaxis(np.tile(expected_p, (*other_dims, 1)), -1, axis)

            actual_p = _ndpolyfit(x, y, axis=axis)

            np.testing.assert_almost_equal(expected_p, actual_p)

    def test_10(self):
        x = np.arange(10).astype(dtype=np.float)
        y = np.arange(10).astype(dtype=np.float)

        y[5] = np.nan
        p = _ndpolyfit(x, y)

        np.testing.assert_almost_equal(p, [1.0, 0.0])

    def test_11(self):
        x = np.arange(10).astype(dtype=np.float)
        y = np.arange(10).astype(dtype=np.float)

        for i in range(20):
            idx = np.random.randint(0, 10)
            y[idx] = np.nan
            p = _ndpolyfit(x, y)
            y[idx] = idx
            np.testing.assert_almost_equal(p, [1.0, 0.0])

    def test_12(self):
        x = np.arange(10).astype(dtype=np.float)
        for i in range(50):
            expected_p = np.random.randint(-10, 10, size=2)
            while expected_p[0] == 0:
                expected_p = np.random.randint(-10, 10, size=2)

            y = expected_p[0] * x + expected_p[1]
            y[np.random.randint(0, 10)] = np.nan
            actual_p = _ndpolyfit(x, y)

            np.testing.assert_almost_equal(expected_p, actual_p)

    def test_13(self):
        x = np.arange(10).astype(dtype=np.float)
        y = np.concatenate(
            (
                np.arange(10).astype(dtype=np.float).reshape((-1, 1)),
                np.arange(10).astype(dtype=np.float).reshape((-1, 1))
            ),
            axis=1
        )

        y[4, 0] = np.nan
        y[2, 1] = np.nan
        p = _ndpolyfit(x, y)
        np.testing.assert_almost_equal(p, [[1.0, 1.0], [0.0, 0.0]])

    def test_14(self):
        x = np.arange(10).astype(dtype=np.float)
        y = np.concatenate(
            (
                np.arange(10).astype(dtype=np.float).reshape((-1, 1)),
                np.arange(10).astype(dtype=np.float).reshape((-1, 1))
            ),
            axis=1
        )

        y[4, :] = np.nan
        y[2, :] = np.nan
        p = _ndpolyfit(x, y)
        np.testing.assert_almost_equal(p, [[1.0, 1.0], [0.0, 0.0]])

    def test_15(self):
        x = np.arange(-10, 10).astype(dtype=np.float)
        max_dim = 6
        max_dim_size = 11

        for i in range(50):
            expected_p = np.random.randint(-10, 10, size=2)
            while expected_p[0] == 0:
                expected_p = np.random.randint(-10, 10, size=2)

            y = expected_p[0] * x + expected_p[1]
            y[np.random.randint(0, 10, size=2)] = np.nan

            other_dims = np.random.randint(1, max_dim_size, np.random.randint(1, max_dim))
            axis = np.random.randint(0, other_dims.ndim + 1)
            y = np.moveaxis(np.tile(y, (*other_dims, 1)), -1, axis)
            expected_p = np.moveaxis(np.tile(expected_p, (*other_dims, 1)), -1, axis)

            actual_p = _ndpolyfit(x, y, axis=axis)

            np.testing.assert_almost_equal(expected_p, actual_p)

    def test_16(self):
        x = np.arange(10).astype(dtype=np.float)
        y = np.concatenate(
            (
                np.arange(10).astype(dtype=np.float).reshape((-1, 1)),
                np.arange(10).astype(dtype=np.float).reshape((-1, 1))
            ),
            axis=1
        )

        p = _ndpolyfit(x, y, missing_value=5)
        np.testing.assert_almost_equal(p, [[1.0, 1.0], [0.0, 0.0]])


class test_ndpolyfit(TestCase):
    def test_00(self):
        x = np.arange(10).astype(dtype=np.float).tolist()
        y = np.arange(10).astype(dtype=np.float).tolist()

        p = ndpolyfit(x, y, deg=1)

        np.testing.assert_almost_equal(p, [1.0, 0.0])

    def test_01(self):
        x = np.arange(10).astype(dtype=np.float)
        y = np.arange(10).astype(dtype=np.float)

        p = ndpolyfit(x, y, deg=1)

        np.testing.assert_almost_equal(p, [1.0, 0.0])

    def test_02(self):
        x = xr.DataArray(np.arange(10).astype(dtype=np.float))
        y = xr.DataArray(np.arange(10).astype(dtype=np.float))

        p = ndpolyfit(x, y, deg=1)

        np.testing.assert_almost_equal(p, [1.0, 0.0])

    def test_03(self):
        x = xr.DataArray(np.arange(10).astype(dtype=np.float))
        y = xr.DataArray(np.arange(10).astype(dtype=np.float))
        y.attrs["attr1"] = 1
        y.attrs["attr2"] = 2

        p = ndpolyfit(x, y, deg=1)

        np.testing.assert_almost_equal(p, [1.0, 0.0])

        self.assertTrue("attr1" in p.attrs)
        self.assertTrue("attr2" in p.attrs)
        self.assertEqual(1, p.attrs["attr1"])
        self.assertEqual(2, p.attrs["attr2"])

    def test_04(self):
        x = xr.DataArray(np.arange(10).astype(dtype=np.float))
        y = xr.DataArray(np.arange(10).astype(dtype=np.float))
        y.attrs["attr1"] = 1
        y.attrs["attr2"] = 2

        p = ndpolyfit(x, y, deg=1, meta=False)

        np.testing.assert_almost_equal(p, [1.0, 0.0])

        self.assertFalse("attr1" in p.attrs)
        self.assertFalse("attr2" in p.attrs)

    def test_05(self):
        x = np.arange(10).astype(dtype=np.float)
        y = np.arange(10).astype(dtype=np.float)

        p = ndpolyfit(x, y, deg=1, meta=False)

        np.testing.assert_almost_equal(p, [1.0, 0.0])

    def test_6(self):
        x = np.arange(10).astype(dtype=np.float)
        y = np.concatenate(
            (
                np.arange(10).astype(dtype=np.float).reshape((-1, 1)),
                np.arange(10).astype(dtype=np.float).reshape((-1, 1))
            ),
            axis=1
        )

        y[5, :] = 999

        p = ndpolyfit(x, y, deg=1, missing_value=999)
        np.testing.assert_almost_equal(p, [[1.0, 1.0], [0.0, 0.0]])

    def test_7(self):
        x = xr.DataArray(np.arange(-10, 10).astype(dtype=np.float))
        max_dim = 6
        max_dim_size = 11

        for i in range(50):
            expected_p = np.random.randint(-10, 10, size=2)
            while expected_p[0] == 0:
                expected_p = np.random.randint(-10, 10, size=2)

            y = expected_p[0] * x + expected_p[1]
            y[np.random.randint(0, 10, size=2)] = np.nan

            other_dims = np.random.randint(1, max_dim_size, np.random.randint(1, max_dim))
            axis = np.random.randint(0, other_dims.ndim + 1)
            y = xr.DataArray(np.moveaxis(np.tile(y, (*other_dims, 1)), -1, axis))
            expected_p = np.moveaxis(np.tile(expected_p, (*other_dims, 1)), -1, axis)

            actual_p = ndpolyfit(x, y, deg=1, axis=axis)

            np.testing.assert_almost_equal(expected_p, actual_p)


























