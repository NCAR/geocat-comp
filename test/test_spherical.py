import scipy.special as ss
import numpy as np
import math as ma
import xarray as xr
import sys
from typing import Union

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp import harmonic_decomposition, harmonic_recomposition
else:
    from geocat.comp import harmonic_decomposition, harmonic_recomposition

max_harm = 23
num_phi = 360
num_theta = 720

theta = np.linspace(0, ma.tau - ma.tau / num_theta, num_theta)
phi = np.linspace(ma.pi / (2 * num_phi), ma.pi - ma.pi / (2 * num_phi), num_phi)
theta_np, phi_np = np.meshgrid(theta, phi)
theta_xr = xr.DataArray(theta_np, dims=['lat', 'lon'])
phi_xr = xr.DataArray(phi_np, dims=['lat', 'lon'])
test_scale_np = np.sin(phi_np)
test_scale_xr = xr.DataArray(test_scale_np, dims=['lat', 'lon'])

test_data = np.zeros(theta_np.shape)
test_results = []
test_harmonics = []
for n in range(max_harm + 1):
    for m in range(n + 1):
        test_harmonics.append([m, n])
        test_results.append(0)
        if n in [0, 2, 3, 5, 7, 11, 13, 17, 19, 23
                ] and m in [0, 2, 3, 5, 7, 11, 13, 17, 19, 23]:
            if m in [2, 5, 11, 17, 23]:
                test_data += ss.sph_harm(m, n, theta_np, phi_np).imag
                test_results[-1] = 1j
            else:
                test_data += ss.sph_harm(m, n, theta_np, phi_np).real
                test_results[-1] = 1

test_harmonics_np = np.array(test_harmonics)
test_harmonics_xr = xr.DataArray(test_harmonics_np, dims=['har',
                                                          'm,n']).compute()
test_data_np = test_data
test_data_xr = xr.DataArray(test_data_np, dims=['lat', 'lon']).compute()
test_results_np = np.array(test_results)
test_results_xr = xr.DataArray(test_results_np, dims=['har']).compute()


def test_decomposition_np():
    results_np = decomposition(test_data_np, test_scale_np, theta_np, phi_np)
    np.testing.assert_almost_equal(results_np, test_results_np, decimal=3)


def test_decomposition_xr():
    results_xr = decomposition(test_data_xr, test_scale_xr, theta_xr, phi_xr)
    np.testing.assert_almost_equal(results_xr.to_numpy(),
                                   test_results_xr.to_numpy(),
                                   decimal=3)


def test_recomposition_np():
    data_np = recomposition(test_results_np, theta_np, phi_np)
    np.testing.assert_almost_equal(data_np, test_data_np)


def test_recomposition_xr():
    data_xr = recomposition(test_results_xr, theta_xr, phi_xr)
    np.testing.assert_almost_equal(data_xr.to_numpy(), test_data_xr.to_numpy())


def test_scale_voronoi_np():
    scale_np = scale_voronoi(theta_np, phi_np)
    np.testing.assert_almost_equal(
        scale_np / np.sum(scale_np, axis=(0, 1)),
        test_scale_np / np.sum(test_scale_np, axis=(0, 1)))


def test_scale_voronoi_xr():
    scale_xr = scale_voronoi(theta_xr, phi_xr)
    np.testing.assert_almost_equal(
        scale_xr / np.sum(scale_xr, axis=(0, 1)),
        test_scale_xr / np.sum(test_scale_xr, axis=(0, 1)))
