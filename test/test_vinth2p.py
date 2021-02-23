import pytest
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import geocat.comp
import geocat.datafiles as gdf

import time


# Replicates NCL's conwomap_5 (almost)

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

start_time = time.time()

u_int = geocat.comp.interp_hybrid_to_pressure(u, ps[0,:,:], hyam, hybm, p0=p0, new_levels=pres3d)

print("--- %s seconds ---" % (time.time() - start_time))

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

