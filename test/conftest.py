from math import pi, tau
import numpy as np
import pytest
import xarray as xr

from packaging.version import Version
from scipy import __version__ as scipy_version

# import scipy shp_harm[_y] function depending on scipy version
scipy_version = Version(scipy_version)
if scipy_version < Version('1.15.0'):
    from scipy.special import sph_harm as sph_harm

    def sph_harm_y(n, m, theta, phi):
        return sph_harm(m, n, phi, theta)
else:
    from scipy.special import sph_harm_y


@pytest.fixture
def spherical_data():
    max_harm = 23
    num_phi = 90
    num_theta = 180

    theta = np.linspace(0, tau - tau / num_theta, num_theta)
    phi = np.linspace(
        pi / (2 * num_phi),
        pi - pi / (2 * num_phi),
        num_phi,
    )
    theta_np, phi_np = np.meshgrid(theta, phi)
    theta_xr = xr.DataArray(theta_np, dims=['lat', 'lon'])
    phi_xr = xr.DataArray(phi_np, dims=['lat', 'lon'])
    test_scale_np = np.sin(phi_np)
    test_scale_xr = xr.DataArray(
        test_scale_np,
        dims=['lat', 'lon'],
    )

    count = 0
    primes = [0, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    test_data = np.zeros(phi_np.shape)
    test_results = []
    test_harmonics = []
    for n in range(max_harm + 1):
        for m in range(n + 1):
            test_harmonics.append([n, m])
            test_results.append(0)
            if n in primes and m in primes:
                if m in primes[1::2]:
                    test_data += sph_harm_y(
                        n,
                        m,
                        phi_np,
                        theta_np,
                    ).imag
                    test_results[-1] = 1j
                    count += 1
                else:
                    test_data += sph_harm_y(
                        n,
                        m,
                        phi_np,
                        theta_np,
                    ).real
                    test_results[-1] = 1
                    count += 1

    test_data_np = test_data
    test_data_xr = xr.DataArray(
        test_data_np,
        dims=['lat', 'lon'],
    )
    test_results_np = np.array(test_results)
    test_results_xr = xr.DataArray(
        test_results_np,
        dims=['har'],
    )

    return {
        'max_harm': max_harm,
        'num_phi': num_phi,
        'num_theta': num_theta,
        'theta_np': theta_np,
        'phi_np': phi_np,
        'theta_xr': theta_xr,
        'phi_xr': phi_xr,
        'test_scale_np': test_scale_np,
        'test_scale_xr': test_scale_xr,
        'test_data_np': test_data_np,
        'test_data_xr': test_data_xr,
        'test_results_np': test_results_np,
        'test_results_xr': test_results_xr,
    }
