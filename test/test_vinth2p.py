import pytest
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import geocat.comp
import geocat.datafiles as gdf


# Open a netCDF data file using xarray default engine and
# load the data into xarrays
ds = xr.open_dataset(gdf.get("netcdf_files/atmos.nc"), decode_times=False)

u = ds.U[0,:,:,:]

# convert to pressure levels
hyam   = ds.hyam
hybm   = ds.hybm
ps     = ds.PS
p0     = 1000. * 100   # Pa
pres3d = np.asarray([1000,950,800,700,600,500,400,300,200])   # mb
pres3d = pres3d * 100   # mb to Pa

#u_int = vinth2p(u, hyam, hybm, pres3d, ps(0,:,:), 2, p0, 2, False)

u_int = geocat.comp.lev_to_plev(u, ps[0,:,:], hyam, hybm, P0=p0, new_levels=pres3d, parallel=False)

#lev_to_plev(data, ps, hyam, hybm, P0=100000., new_levels=None, parallel=False):
"""Interpolate data from hybrid-sigma levels to isobaric levels.

data : DataArray with a 'lev' coordinate
ps   : DataArray of surface pressure (Pa), same time/space shape as data
hyam, hybm : hybrid coefficients, size of len(lev)
P0 : reference pressure
new_levels : the output pressure levels (Pa)
parallel : if True, use the Numba version to parallelize interpolation step.
"""

# uzon = u_int[:,:,0]
#uzon = dim_avg(u_int)
uzon = u_int.mean(dim='lon')

# Plot:

# Generate figure (set its size (width, height) in inches) and axes
plt.figure(figsize=(12, 12))
ax = plt.gca()

# Draw filled contours
colors = uzon.plot.contourf(ax=ax,
                         levels=np.arange(-12, 44, 4),
                         add_colorbar=False,
                         add_labels=False)
# Draw contour lines
lines = uzon.plot.contour(ax=ax,
                       colors='black',
                       levels=np.arange(-12, 44, 4),
                       linewidths=0.5,
                       linestyles='solid',
                       add_labels=False)

# Create horizontal colorbar
cbar = plt.colorbar(colors,
                    ticks=np.arange(-12, 44, 4),
                    orientation='horizontal',
                    drawedges=True,
                    aspect=12,
                    shrink=0.8,
                    pad=0.075)

# Show the plot
plt.tight_layout()
plt.show()


# def test_climatology_invalid_freq():
#     with pytest.raises(ValueError):
#         geocat.comp.climatology(dset_a, "hourly")

