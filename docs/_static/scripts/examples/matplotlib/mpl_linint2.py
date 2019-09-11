#!/usr/bin/env python
# coding: utf-8

import numpy as np
import xarray as xr
import geocat.comp
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid

# Open a netCDF data file using xarray default engine and load the data into xarrays
ds = xr.open_dataset("sst.nc")
sst = ds.TEMP[0,0,:,:]
lat = ds.LAT[:]
lon = ds.LON[:]

# Provide interpolation grid and call $linint2$ function from $geocat-comp$
newlat = np.linspace(min(lat), max(lat), 128)
newlon = np.linspace(min(lon), max(lon), 360)
newsst = geocat.comp.linint2(sst, newlon, newlat, 0)

# Visualize the interpolated grid $newsst$ using matplotlib and cartopy
#### -----  Graphics using cartopy and matplotlib ----- ####
projection = ccrs.PlateCarree()
axes_class = (GeoAxes, dict(map_projection=projection))
fig = plt.figure(figsize=(10,8))
axgr = AxesGrid(fig, 111, axes_class=axes_class,
                nrows_ncols=(2, 1),
                axes_pad=0.7,
                cbar_location='bottom',
                cbar_mode='single',
                cbar_pad=0.1,
                cbar_size='10%',
                label_mode='')  # note the empty label_mode

for i, ax in enumerate(axgr):
    ax.coastlines()
    ax.set_xticks(np.linspace(-180, 180, 13), crs=projection)
    ax.set_yticks(np.linspace(-60, 60, 5), crs=projection)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    # Plot contours for both the subplots
    if( i==0 ):
        X, Y = np.meshgrid(lon, lat)
        p=ax.contourf(X, Y, sst, levels=16, transform=projection, cmap=cm.jet)
        ax.set_title('Original Grid', fontsize=14, fontweight='bold')
    else:
        newX, newY = np.meshgrid(newlon, newlat)
        p=ax.contourf(newX, newY, newsst, levels=16, transform=projection, cmap=cm.jet)
        ax.set_title('ReGrid - linint2', fontsize=14, fontweight='bold')

# Add color bar and label details (title, size, etc.)
cax=axgr.cbar_axes[0]
cax.colorbar(p)
axis=cax.axis[cax.orientation]
axis.label.set_text('Temperature ($^{\circ} C$)')
axis.label.set_size(16)
axis.major_ticklabels.set_size(10)

# Save figure and show
plt.savefig('linint2', dpi=300)
plt.show()
