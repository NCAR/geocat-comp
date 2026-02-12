import numpy as np

from geocat.comp import decomposition, recomposition, scale_voronoi


class Test_Spherical:
    def test_decomposition_np(self, spherical_data) -> None:
        results_np = decomposition(
            spherical_data['test_data_np'],
            spherical_data['test_scale_np'],
            spherical_data['theta_np'],
            spherical_data['phi_np'],
            spherical_data['max_harm'],
        )
        np.testing.assert_almost_equal(
            results_np,
            spherical_data['test_results_np'],
            decimal=2,
        )

    def test_decomposition_xr(self, spherical_data) -> None:
        results_xr = decomposition(
            spherical_data['test_data_xr'],
            spherical_data['test_scale_xr'],
            spherical_data['theta_xr'],
            spherical_data['phi_xr'],
        )
        np.testing.assert_almost_equal(
            results_xr.to_numpy(),
            spherical_data['test_results_xr'].to_numpy(),
            decimal=2,
        )

    def test_recomposition_np(self, spherical_data) -> None:
        data_np = recomposition(
            spherical_data['test_results_np'],
            spherical_data['theta_np'],
            spherical_data['phi_np'],
        )
        np.testing.assert_almost_equal(
            data_np,
            spherical_data['test_data_np'],
        )

    def test_recomposition_xr(self, spherical_data) -> None:
        data_xr = recomposition(
            spherical_data['test_results_xr'],
            spherical_data['theta_xr'],
            spherical_data['phi_xr'],
        )
        np.testing.assert_almost_equal(
            data_xr.to_numpy(),
            spherical_data['test_data_xr'].to_numpy(),
        )

    def test_scale_voronoi_np(self, spherical_data) -> None:
        scale_np = scale_voronoi(
            spherical_data['theta_np'],
            spherical_data['phi_np'],
        )
        np.testing.assert_almost_equal(
            scale_np / np.sum(scale_np, axis=(0, 1)),
            spherical_data['test_scale_np']
            / np.sum(spherical_data['test_scale_np'], axis=(0, 1)),
        )

    def test_scale_voronoi_xr(self, spherical_data) -> None:
        scale_xr = scale_voronoi(
            spherical_data['theta_xr'],
            spherical_data['phi_xr'],
        )
        np.testing.assert_almost_equal(
            scale_xr / np.sum(scale_xr, axis=(0, 1)),
            spherical_data['test_scale_xr']
            / np.sum(spherical_data['test_scale_xr'], axis=(0, 1)),
        )
