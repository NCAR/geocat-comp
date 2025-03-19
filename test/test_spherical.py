import math as ma

import numpy as np
import scipy.special as ss
import xarray as xr

from geocat.comp import decomposition, recomposition, scale_voronoi


class Test_Spherical:
    max_harm = 23
    num_phi = 90
    num_theta = 180

    theta = np.linspace(0, ma.tau - ma.tau / num_theta, num_theta)
    phi = np.linspace(
        ma.pi / (2 * num_phi),
        ma.pi - ma.pi / (2 * num_phi),
        num_phi,
    )
    theta_np, phi_np = np.meshgrid(theta, phi)
    theta_xr = xr.DataArray(theta_np, dims=['lat', 'lon'])
    phi_xr = xr.DataArray(phi_np, dims=['lat', 'lon'])
    test_scale_np = np.sin(phi_np)
    test_scale_xr = xr.DataArray(
        test_scale_np,
        dims=['lat', 'lon'],
    ).compute()

    test_data = np.zeros(theta_np.shape)
    test_results = []
    test_harmonics = []
    for n in range(max_harm + 1):
        for m in range(n + 1):
            test_harmonics.append([m, n])
            test_results.append(0)
            if n in [0, 2, 3, 5, 7, 11, 13, 17, 19, 23] and m in [
                    0,
                    2,
                    3,
                    5,
                    7,
                    11,
                    13,
                    17,
                    19,
                    23,
            ]:
                if m in [2, 5, 11, 17, 23]:
                    test_data += ss.sph_harm(
                        m,
                        n,
                        theta_np,
                        phi_np,
                    ).imag
                    test_results[-1] = 1j
                else:
                    test_data += ss.sph_harm(
                        m,
                        n,
                        theta_np,
                        phi_np,
                    ).real
                    test_results[-1] = 1

    test_harmonics_np = np.array(test_harmonics)
    test_harmonics_xr = xr.DataArray(
        test_harmonics_np,
        dims=['har', 'm,n'],
    ).compute()
    test_data_np = test_data
    test_data_xr = xr.DataArray(
        test_data_np,
        dims=['lat', 'lon'],
    ).compute()
    test_results_np = np.array(test_results)
    test_results_xr = xr.DataArray(
        test_results_np,
        dims=['har'],
    ).compute()

    def test_decomposition_np(self) -> None:
        results_np = decomposition(
            self.test_data_np,
            self.test_scale_np,
            self.theta_np,
            self.phi_np,
        )
        np.testing.assert_almost_equal(
            results_np,
            self.test_results_np,
            decimal=2,
        )

    def test_decomposition_xr(self) -> None:
        results_xr = decomposition(
            self.test_data_xr,
            self.test_scale_xr,
            self.theta_xr,
            self.phi_xr,
        )
        np.testing.assert_almost_equal(
            results_xr.to_numpy(),
            self.test_results_xr.to_numpy(),
            decimal=2,
        )

    def test_recomposition_np(self) -> None:
        data_np = recomposition(
            self.test_results_np,
            self.theta_np,
            self.phi_np,
        )
        np.testing.assert_almost_equal(
            data_np,
            self.test_data_np,
        )

    def test_recomposition_xr(self) -> None:
        data_xr = recomposition(
            self.test_results_xr,
            self.theta_xr,
            self.phi_xr,
        )
        np.testing.assert_almost_equal(
            data_xr.to_numpy(),
            self.test_data_xr.to_numpy(),
        )

    def test_scale_voronoi_np(self) -> None:
        scale_np = scale_voronoi(
            self.theta_np,
            self.phi_np,
        )
        np.testing.assert_almost_equal(
            scale_np / np.sum(scale_np, axis=(0, 1)),
            self.test_scale_np / np.sum(self.test_scale_np, axis=(0, 1)),
        )

    def test_scale_voronoi_xr(self) -> None:
        scale_xr = scale_voronoi(
            self.theta_xr,
            self.phi_xr,
        )
        np.testing.assert_almost_equal(
            scale_xr / np.sum(scale_xr, axis=(0, 1)),
            self.test_scale_xr / np.sum(self.test_scale_xr, axis=(0, 1)),
        )
