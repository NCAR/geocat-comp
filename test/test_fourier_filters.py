import math as m
import sys

import numpy as np
import xarray as xr

from geocat.comp import (fourier_band_block, fourier_band_pass,
                         fourier_high_pass, fourier_low_pass)

freq = 1000
t = np.arange(1000) / freq
t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
          np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
          np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
          np.sin(100 * t * m.tau) / 10)


def test_one_low_pass():
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau))
    t_result = fourier_low_pass(t_data, freq, 15)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_one_high_pass():
    t_expected_result = (np.sin(20 * t * m.tau) / 2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_result = fourier_high_pass(t_data, freq, 15)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_one_band_pass():
    t_expected_result = (np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
                         np.sin(20 * t * m.tau) / 2)
    t_result = fourier_band_pass(t_data, freq, 3, 30)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_one_band_block():
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_result = fourier_band_block(t_data, freq, 3, 30)
    np.testing.assert_almost_equal(t_result, t_expected_result)


freq = 1000
t = np.arange(1000) / freq
t = t[:, None] + t
t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
          np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
          np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
          np.sin(100 * t * m.tau) / 10)


def test_two_low_pass():
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau))
    t_result = fourier_low_pass(t_data, freq, 15, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_two_high_pass():
    t_expected_result = (np.sin(20 * t * m.tau) / 2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_result = fourier_high_pass(t_data, freq, 15, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_two_band_pass():
    t_expected_result = (np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
                         np.sin(20 * t * m.tau) / 2)
    t_result = fourier_band_pass(t_data, freq, 3, 30, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_two_band_block():
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_result = fourier_band_block(t_data, freq, 3, 30, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


freq = 200
t = np.arange(200) / freq
t = t[:, None] + t
t = t[:, :, None] + t
t_data = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
          np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
          np.sin(20 * t * m.tau) / 2 + np.sin(50 * t * m.tau) / 5 +
          np.sin(100 * t * m.tau) / 10)


def test_three_low_pass():
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau))
    t_result = fourier_low_pass(t_data, freq, 15, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_three_high_pass():
    t_expected_result = (np.sin(20 * t * m.tau) / 2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_result = fourier_high_pass(t_data, freq, 15, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_three_band_pass():
    t_expected_result = (np.sin(5 * t * m.tau) / 0.5 + np.sin(10 * t * m.tau) +
                         np.sin(20 * t * m.tau) / 2)
    t_result = fourier_band_pass(t_data, freq, 3, 30, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_three_band_block():
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_result = fourier_band_block(t_data, freq, 3, 30, time_axis=0)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_three_band_block_t1():
    t_data_ = np.swapaxes(t_data, 1, 0)
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_expected_result = np.swapaxes(t_expected_result, 1, 0)
    t_result = fourier_band_block(t_data_, freq, 3, 30, time_axis=1)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_three_band_block_t2():
    t_data_ = np.swapaxes(t_data, 2, 0)
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_expected_result = np.swapaxes(t_expected_result, 2, 0)
    t_result = fourier_band_block(t_data_, freq, 3, 30, time_axis=2)
    np.testing.assert_almost_equal(t_result, t_expected_result)


def test_three_band_block_xr():
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_data_ = xr.DataArray(t_data)
    t_expected_result = xr.DataArray(t_expected_result)
    t_result = fourier_band_block(t_data_, freq, 3, 30, time_axis=0)
    np.testing.assert_almost_equal(t_result.data, t_expected_result)


def test_three_band_block_t1_xr():
    t_data_ = np.swapaxes(t_data, 1, 0)
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_expected_result = np.swapaxes(t_expected_result, 1, 0)
    t_data_ = xr.DataArray(t_data_)
    t_expected_result = xr.DataArray(t_expected_result)
    t_result = fourier_band_block(t_data_, freq, 3, 30, time_axis=1)
    np.testing.assert_almost_equal(t_result.data, t_expected_result)


def test_three_band_block_t2_xr():
    t_data_ = np.swapaxes(t_data, 2, 0)
    t_expected_result = (np.sin(t * m.tau) / 0.1 + np.sin(2 * t * m.tau) / 0.2 +
                         np.sin(50 * t * m.tau) / 5 +
                         np.sin(100 * t * m.tau) / 10)
    t_expected_result = np.swapaxes(t_expected_result, 2, 0)
    t_data_ = xr.DataArray(t_data_)
    t_expected_result = xr.DataArray(t_expected_result)
    t_result = fourier_band_block(t_data_, freq, 3, 30, time_axis=2)
    np.testing.assert_almost_equal(t_result.data, t_expected_result)
