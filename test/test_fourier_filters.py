import math as m
import sys

import numpy as np

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp import (fourier_band_block, fourier_band_pass,
                                 fourier_high_pass, fourier_low_pass)
else:
    from geocat.comp import (fourier_band_block, fourier_band_pass,
                             fourier_high_pass, fourier_low_pass)


def test_one_low_pass():
    freq = 1000
    t = np.arange(1000) / freq
    t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
              np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
              np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
              np.sin(100 * t * m.tau) / 10)
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau))
    t_result = fourier_low_pass(t_data, freq, 15)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_one_high_pass():
    freq = 1000
    t = np.arange(1000) / freq
    t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
              np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
              np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
              np.sin(100 * t * m.tau) / 10)
    t_expected_result = (np.sin(20 * t * m.tau) / 2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_result = fourier_high_pass(t_data, freq, 15)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_one_band_pass():
    freq = 1000
    t = np.arange(1000) / freq
    t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
              np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
              np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
              np.sin(100 * t * m.tau) / 10)
    t_expected_result = (np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
                         np.sin(20 * t * m.tau) / 2)
    t_result = fourier_band_pass(t_data, freq, 3, 30)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_one_band_block():
    freq = 10000
    t = np.arange(30000) / freq
    t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
              np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
              np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
              np.sin(100 * t * m.tau) / 10)
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_result = fourier_band_block(t_data, freq, 3, 30)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_two_low_pass():
    freq = 1000
    t = np.arange(1000) / freq
    t = t[:, None] + t
    t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
              np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
              np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
              np.sin(100 * t * m.tau) / 10)
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau))
    t_result = fourier_low_pass(t_data, freq, 15, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_two_high_pass():
    freq = 1000
    t = np.arange(1000) / freq
    t = t[:, None] + t
    t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
              np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
              np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
              np.sin(100 * t * m.tau) / 10)
    t_expected_result = (np.sin(20 * t * m.tau) / 2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_result = fourier_high_pass(t_data, freq, 15, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_two_band_pass():
    freq = 1000
    t = np.arange(1000) / freq
    t = t[:, None] + t
    t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
              np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
              np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
              np.sin(100 * t * m.tau) / 10)
    t_expected_result = (np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
                         np.sin(20 * t * m.tau) / 2)
    t_result = fourier_band_pass(t_data, freq, 3, 30, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_two_band_block():
    freq = 1000
    t = np.arange(1000) / freq
    t = t[:, None] + t
    t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
              np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
              np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
              np.sin(100 * t * m.tau) / 10)
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_result = fourier_band_block(t_data, freq, 3, 30, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_three_low_pass():
    freq = 200
    t = np.arange(200) / freq
    t = t[:, None] + t
    t = t[:, :, None] + t
    t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
              np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
              np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
              np.sin(100 * t * m.tau) / 10)
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau))
    t_result = fourier_low_pass(t_data, freq, 15, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_three_high_pass():
    freq = 200
    t = np.arange(200) / freq
    t = t[:, None] + t
    t = t[:, :, None] + t
    t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
              np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
              np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
              np.sin(100 * t * m.tau) / 10)
    t_expected_result = (np.sin(20 * t * m.tau) / 2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_result = fourier_high_pass(t_data, freq, 15, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_three_band_pass():
    freq = 200
    t = np.arange(200) / freq
    t = t[:, None] + t
    t = t[:, :, None] + t
    t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
              np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
              np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
              np.sin(100 * t * m.tau) / 10)
    t_expected_result = (np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
                         np.sin(20 * t * m.tau) / 2)
    t_result = fourier_band_pass(t_data, freq, 3, 30, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_three_band_block():
    freq = 200
    t = np.arange(200) / freq
    t = t[:, None] + t
    t = t[:, :, None] + t
    t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
              np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
              np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
              np.sin(100 * t * m.tau) / 10)
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_result = fourier_band_block(t_data, freq, 3, 30, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_three_band_block_t1():
    freq = 200
    t = np.arange(200) / freq
    t = t[:, None] + t
    t = t[:, :, None] + t
    t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
              np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
              np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
              np.sin(100 * t * m.tau) / 10)
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_result = fourier_band_block(t_data, freq, 3, 30, time_axis=1)
    np.testing.assert_almost_equal(t_result, t_expected_result)
