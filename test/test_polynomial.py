from unittest import TestCase
from geocat.comp.polynomial import _ndpolyfit, ndpolyfit, _ndpolyval, ndpolyval, detrend

import numpy as np
import xarray as xr
import dask.array as da


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

    def test_8(self):
        x = xr.DataArray(np.arange(-10, 10).astype(dtype=np.float))
        max_dim = 3
        max_dim_size = 5

        for i in range(3):
            expected_p = np.random.randint(-10, 10, size=2)
            while expected_p[0] == 0:
                expected_p = np.random.randint(-10, 10, size=2)

            y = expected_p[0] * x + expected_p[1]

            other_dims = np.random.randint(1, max_dim_size, np.random.randint(1, max_dim))
            axis = np.random.randint(0, other_dims.ndim + 1)
            y = xr.DataArray(np.moveaxis(np.tile(y, (*other_dims, 1)), -1, axis))
            expected_p = np.moveaxis(np.tile(expected_p, (*other_dims, 1)), -1, axis)

            y = da.from_array(y, chunks=np.ones((y.ndim,)))
            actual_p = ndpolyfit(x, y, deg=1, axis=axis)

            np.testing.assert_almost_equal(expected_p, actual_p)


class test_internal_ndpolyval(TestCase):
    def test_01(self):
        p = np.asarray([1.0, 0.0])

        x = np.arange(5)

        y = _ndpolyval(p, x)

        np.testing.assert_almost_equal(
            y,
            p[0] * x + p[1]
        )

    def test_02(self):
        p = np.moveaxis(np.asarray([1.0, 0.0]*15).reshape((3, 5, 2)), -1, 0)

        x = np.arange(5)

        y = _ndpolyval(p, x)

        expected = np.moveaxis(np.asarray(list(range(5))*15).reshape((3, 5, 5)), -1, 0)
        np.testing.assert_almost_equal(
            y,
            expected
        )

    def test_03(self):
        p = np.asarray([1.0, 0.0]*15).reshape((3, 5, 2))

        x = np.arange(5)

        y = _ndpolyval(p, x, axis=2)

        expected = np.asarray(list(range(5))*15).reshape((3, 5, 5))
        np.testing.assert_almost_equal(
            y,
            expected
        )

    def test_04(self):
        p = np.moveaxis(np.asarray([1.0, 0.0]*15).reshape((3, 5, 2)), -1, 1)

        x = np.arange(5)

        y = _ndpolyval(p, x, axis=1)

        expected = np.moveaxis(np.asarray(list(range(5))*15).reshape((3, 5, 5)), -1, 1)
        np.testing.assert_almost_equal(
            y,
            expected
        )

    def test_05(self):
        p = np.moveaxis(np.asarray([1.0, 0.0]*15).reshape((3, 5, 2)), -1, 1)

        x = np.moveaxis(np.asarray(list(range(5))*15).reshape((3, 5, 5)), -1, 1)

        y = _ndpolyval(p, x, axis=1)

        expected = np.moveaxis(np.asarray(list(range(5))*15).reshape((3, 5, 5)), -1, 1)
        np.testing.assert_almost_equal(
            y,
            expected
        )

    def test_06(self):
        for i in range(50):
            # these limits are just to limit the time it takes to test.
            deg = np.random.randint(0, 4)  # Maximum polynomial degree = 3
            ndim = np.random.randint(1, 6)  # Maximim 5-Dimensional array
            axis = np.random.randint(0, ndim)
            max_dim_size = 11  # The maximum number of elements along one dimension

            if ndim > 1:
                tmp_shape = np.random.randint(1, max_dim_size, size=ndim)
                data_shape = tmp_shape.copy()
                data_shape[axis] = np.random.randint(1, max_dim_size)
                data_shape = tuple(data_shape)

                p_shape = tmp_shape.copy()
                p_shape[axis] = deg + 1
                p_shape = tuple(p_shape)
            else:
                data_shape = (np.random.randint(1, max_dim_size), )
                p_shape = (deg + 1, )

            p = np.random.random(size=p_shape)
            x = np.random.random(size=data_shape)

            y_expected = np.zeros(data_shape)

            for i in range(deg+1):
                y_expected += p.take([i], axis=axis) * np.power(x, deg - i)

            y_actual = _ndpolyval(p, x, axis=axis)

            np.testing.assert_almost_equal(
                y_actual,
                y_expected
            )


class test_ndpolyval(TestCase):
    def test_01(self):
        for i in range(50):
            # these limits are just to limit the time it takes to test.
            deg = np.random.randint(0, 4)  # Maximum polynomial degree = 3
            ndim = np.random.randint(1, 6)  # Maximim 5-Dimensional array
            axis = np.random.randint(0, ndim)
            max_dim_size = 11  # The maximum number of elements along one dimension

            if ndim > 1:
                tmp_shape = np.random.randint(1, max_dim_size, size=ndim)
                data_shape = tmp_shape.copy()
                data_shape[axis] = np.random.randint(1, max_dim_size)
                data_shape = tuple(data_shape)

                p_shape = tmp_shape.copy()
                p_shape[axis] = deg + 1
                p_shape = tuple(p_shape)
            else:
                data_shape = (np.random.randint(1, max_dim_size), )
                p_shape = (deg + 1, )

            p = np.random.random(size=p_shape)
            x = np.random.random(size=data_shape)

            y_expected = np.zeros(data_shape)

            for i in range(deg+1):
                y_expected += p.take([i], axis=axis) * np.power(x, deg - i)

            y_actual = ndpolyval(p, x, axis=axis)

            np.testing.assert_almost_equal(
                y_actual,
                y_expected
            )

    def test_02(self):
        for i in range(50):
            # these limits are just to limit the time it takes to test.
            deg = np.random.randint(0, 4)  # Maximum polynomial degree = 3
            ndim = np.random.randint(1, 6)  # Maximim 5-Dimensional array
            axis = np.random.randint(0, ndim)
            max_dim_size = 11  # The maximum number of elements along one dimension

            if ndim > 1:
                tmp_shape = np.random.randint(1, max_dim_size, size=ndim)
                data_shape = tmp_shape.copy()
                data_shape[axis] = np.random.randint(1, max_dim_size)
                data_shape = tuple(data_shape)

                p_shape = tmp_shape.copy()
                p_shape[axis] = deg + 1
                p_shape = tuple(p_shape)
            else:
                data_shape = (np.random.randint(1, max_dim_size), )
                p_shape = (deg + 1, )

            p = np.random.random(size=p_shape)
            x = xr.DataArray(np.random.random(size=data_shape))

            y_expected = np.zeros(data_shape)

            for i in range(deg+1):
                y_expected += p.take([i], axis=axis) * np.power(x, deg - i)

            y_actual = ndpolyval(p, x, axis=axis)

            np.testing.assert_almost_equal(
                y_actual.data,
                y_expected.data
            )

    def test_03(self):
        for i in range(5):
            # these limits are just to limit the time it takes to test.
            deg = np.random.randint(0, 4)  # Maximum polynomial degree = 3
            ndim = np.random.randint(1, 6)  # Maximim 5-Dimensional array
            axis = np.random.randint(0, ndim)
            max_dim_size = 11  # The maximum number of elements along one dimension

            if ndim > 1:
                tmp_shape = np.random.randint(1, max_dim_size, size=ndim)
                data_shape = tmp_shape.copy()
                data_shape[axis] = np.random.randint(1, max_dim_size)
                data_shape = tuple(data_shape)

                p_shape = tmp_shape.copy()
                p_shape[axis] = deg + 1
                p_shape = tuple(p_shape)
            else:
                data_shape = (np.random.randint(1, max_dim_size), )
                p_shape = (deg + 1, )

            p = np.random.random(size=p_shape)
            x_nparr = np.random.random(size=data_shape)
            x = da.from_array(x_nparr, chunks=np.ones((ndim, )))

            y_expected = np.zeros(data_shape)

            for i in range(deg+1):
                y_expected += p.take([i], axis=axis) * np.power(x_nparr, deg - i)

            y_actual = ndpolyval(p, x, axis=axis)
            np.testing.assert_almost_equal(
                y_actual.data,
                y_expected.data
            )


class test_detrend(TestCase):
    def test_01(self):
        # Creating synthetic data
        x = np.linspace(-8*np.pi, 8 * np.pi, 33, dtype=np.float64)
        y0 = 1.0 * x
        y1 = np.sin(x)
        y = y0 + y1

        p = ndpolyfit(x, y, deg=1)
        y_trend = ndpolyval(p, x)

        y_detrended = detrend(y, x=x)

        np.testing.assert_almost_equal(y_detrended + y_trend, y)

    def test_02(self):
        # Creating synthetic data
        x = np.linspace(-8*np.pi, 8 * np.pi, 33, dtype=np.float64)
        y0 = 1.0 * x
        y1 = np.sin(x)
        y = y0 + y1

        p = ndpolyfit(np.arange(x.size), y, deg=1)
        y_trend = ndpolyval(p, np.arange(x.size))

        y_detrended = detrend(y)

        np.testing.assert_almost_equal(y_detrended + y_trend, y)

    def test_03(self):
        # Creating synthetic data
        x = np.linspace(-8*np.pi, 8 * np.pi, 33, dtype=np.float64)
        y0 = 1.0 * x
        y1 = np.sin(x)
        y = y0 + y1

        p = ndpolyfit(x, y, deg=1)
        y_trend = ndpolyval(p, x)

        y_detrended = detrend(y, x=x, return_info=False)

        np.testing.assert_almost_equal(y_detrended + y_trend, y)

        np.testing.assert_equal({}, y_detrended.attrs)

























