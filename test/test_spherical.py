import scipy.special as ss
import numpy as np
import math as ma
import xarray as xr
from typing import Union

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp import harmonic_decomposition, harmonic_recomposition
else:
    from geocat.comp import harmonic_decomposition, harmonic_recomposition

start = time.perf_counter()
num_phi = 2000
num_theta = 4000
phi = np.linspace(ma.pi / (2 * num_phi), ma.pi - ma.pi / (2 * num_phi),
                  num_phi)  #[1:-1]
theta = np.linspace(0, ma.tau - ma.tau / num_theta, num_theta)  #[:-1]
chunkshape = (1000, 1000)
theta, phi = np.meshgrid(theta, phi)
phi_coord = np.linspace(90 - 90 / (num_phi), 90 / (num_phi) - 90, num_phi)
theta_coord = np.linspace(0, 360 - 360 / num_theta, num_theta)
theta = xr.DataArray(theta, dims=['lat', 'lon']).chunk(chunkshape)
phi = xr.DataArray(phi, dims=['lat', 'lon']).chunk(chunkshape)

#area weighting for data points.
scale_phi = np.sin(phi)

demo_data = xr.DataArray(np.zeros(theta.shape), dims=['lat',
                                                      'lon']).chunk(chunkshape)

harms = []
for n in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
    for m in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
        if m <= n:
            harms.append([m, n])

for m, n in harms:
    if m in [5, 11, 17, 23]:
        demo_data += ss.sph_harm(m, n, theta, phi).imag
    else:
        demo_data += ss.sph_harm(m, n, theta, phi).real

demo_data = demo_data.compute().chunk(chunkshape)
