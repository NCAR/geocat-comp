import scipy.special as ss
import numpy as np
import math as ma
import xarray as xr
import sys
from typing import Union

from src.geocat.comp import harmonic_decomposition, harmonic_recomposition
# Import from directory structure if coverage test, or from installed
# packages otherwise
# if "--cov" in str(sys.argv):
#     from src.geocat.comp import harmonic_decomposition, harmonic_recomposition
# else:
#     from geocat.comp import harmonic_decomposition, harmonic_recomposition

max_harm = 23
num_phi = 1000
num_theta = 2000
chunkshape = (500, 500)

phi = np.linspace(ma.pi / (2 * num_phi), ma.pi - ma.pi / (2 * num_phi), num_phi)
theta = np.linspace(0, ma.tau - ma.tau / num_theta, num_theta)
theta_np, phi_np = np.meshgrid(theta, phi)
# phi_coord = np.linspace(90-90/num_phi, 90/num_phi-90, num_phi)
# theta_coord = np.linspace(0, 360-360/num_theta, num_theta)
theta_xr = xr.DataArray(theta_np, dims=['lat', 'lon']).chunk(chunkshape)
phi_xr = xr.DataArray(phi_np, dims=['lat', 'lon']).chunk(chunkshape)
scale_phi_np = np.sin(phi_np)  # area weighting for data points.
scale_phi_xr = xr.DataArray(scale_phi_np, dims=['lat', 'lon']).chunk(chunkshape)

test_data = np.zeros(theta_np.shape)
test_results = []

for n in range(max_harm + 1):
    for m in range(n + 1):
        test_results.append(0)
        if n in [0, 2, 3, 5, 7, 11, 13, 17, 19, 23
                ] and m in [0, 2, 3, 5, 7, 11, 13, 17, 19, 23]:
            if m in [2, 5, 11, 17, 23]:
                test_data += ss.sph_harm(m, n, theta_np, phi_np).imag
                test_results[-1] = 1j
            else:
                test_data += ss.sph_harm(m, n, theta_np, phi_np).real
                test_results[-1] = 1

test_data_np = test_data
test_data_xr = xr.DataArray(test_data_np, dims=['lat', 'lon']).chunk(chunkshape)
test_results_np = np.asarray(test_results)
test_results_xr = xr.DataArray(test_results_np).chunk((100,))


def test_decomposition_xr():
    results_xr = harmonic_decomposition(test_data_xr, scale_phi_xr, theta_xr,
                                        phi_xr)
    np.testing.assert_almost_equal(results_xr, test_result_xr)


def test_decomposition_np():
    results_np = harmonic_decomposition(test_data_np, scale_phi_np, theta_np,
                                        phi_np)
    np.testing.assert_almost_equal(results_np, test_result_np)
