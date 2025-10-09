from math import pi, tau
import numpy as np
from packaging.version import Version
from scipy import __version__ as scipy_version
import xarray as xr

# import scipy shp_harm[_y] function depending on scipy version
scipy_version = Version(scipy_version)
if scipy_version < Version('1.15.0'):
    from scipy.special import sph_harm as sph_harm

    def sph_harm_y(n, m, theta, phi):
        return sph_harm(m, n, phi, theta)
else:
    from scipy.special import sph_harm_y

from geocat.comp import decomposition, recomposition, scale_voronoi


class Bench_spherical:
    def setup(self):
        max_harm = 23
        num_phi = 90
        num_theta = 180

        theta = np.linspace(0, tau - tau / num_theta, num_theta)
        phi = np.linspace(
            pi / (2 * num_phi),
            pi - pi / (2 * num_phi),
            num_phi,
        )
        self.theta_grid, self.phi_grid = np.meshgrid(theta, phi)
        self.test_scale = np.sin(self.phi_grid)

        count = 0
        primes = [0, 2, 3, 5, 7, 11, 13, 17, 19, 23]
        self.test_data = np.zeros(self.phi_grid.shape)
        self.test_results = []
        test_harmonics = []
        for n in range(max_harm + 1):
            for m in range(n + 1):
                test_harmonics.append([n, m])
                self.test_results.append(0)
                if n in primes and m in primes:
                    if m in primes[1::2]:
                        self.test_data += sph_harm_y(
                            n,
                            m,
                            self.phi_grid,
                            self.theta_grid,
                        ).imag
                        self.test_results[-1] = 1j
                        count += 1
                    else:
                        self.test_data += sph_harm_y(
                            n,
                            m,
                            self.phi_grid,
                            self.theta_grid,
                        ).real
                        self.test_results[-1] = 1
                        count += 1

    def time_decomposition(self):
        decomposition(self.test_data, self.test_scale, self.theta_grid, self.phi_grid)

    def time_recomposition(self):
        recomposition(self.test_results, self.theta_grid, self.phi_grid)

    def time_scale_voronoi(self):
        scale_voronoi(self.theta_grid, self.phi_grid)

    def peakmem_decomposition(self):
        decomposition(self.test_data, self.test_scale, self.theta_grid, self.phi_grid)

    def peakmem_recomposition(self):
        recomposition(self.test_results, self.theta_grid, self.phi_grid)

    def peakmem_scale_voronoi(self):
        scale_voronoi(self.theta_grid, self.phi_grid)
