"""
mpl_linint2points.py
==============
This script illustrates the following concepts:
   - Usage of GeoCAT's linint2_points function
   - Bilinear interpolation from a rectilinear grid to an unstructured grid or locations

See following GitHub repositories to see further information about the function and to access data:
    - linint2_points function: https://github.com/NCAR/geocat-comp
    - "sst.nc" data file: https://github.com/NCAR/geocat-datafiles/tree/master/netcdf_files
"""

# Import packages:
import geocat.comp
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import cm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid

# Open a netCDF data file (Sea surface temperature) using xarray default engine and load the data into xarrays
ds = xr.open_dataset("sst.nc")
sst = ds.TEMP[0,0,:,:]
lat = ds.LAT[:]
lon = ds.LON[:]

# Provide (output) interpolation locations. This script uses 3000 arbitrary locations world-wide in order to demonstrate
# an extensive comparison of the linint2_points outputs to the original grid throughout the globe. The function can be
# even used for a single location though.
newlat = np.random.uniform(low=min(lat), high=max(lat), size=(3000,))
newlon = np.random.uniform(low=min(lon), high=max(lon), size=(3000,))

# Call `linint2_points` from `geocat-comp`
newsst = geocat.comp.linint2_points(sst, newlon, newlat, False)

# Generate figure and set its size (width, height) in inches
fig = plt.figure(figsize=(10,8))

# Generate Axes grid using a Cartopy projection
projection = ccrs.PlateCarree()
axes_class = (GeoAxes, dict(map_projection=projection))
axgr = AxesGrid(fig, 111, axes_class=axes_class,
                nrows_ncols=(2, 1),
                axes_pad=0.7,
                cbar_location='right',
                cbar_mode='single',
                cbar_pad=0.5,
                cbar_size='3%',
                label_mode='')  # note the empty label_mode

# Create a dictionary for common plotting options for both subplots
common_options = dict(vmin=-30, vmax=30, cmap=cm.jet)

# Plot original grid and linint2_points interpolations as two subplots within the figure
for i, ax in enumerate(axgr):

    # Plot original grid and linint2_points interpolations within the subplots
    if( i==0 ):
        p = sst.plot.contourf(ax=ax, **common_options,
                              transform=projection, levels=16, extend='neither', add_colorbar=False)
        ax.set_title('Sea Surface Temperature - Original Grid', fontsize=14, fontweight='bold')
    else:
        ax.scatter(newlon, newlat, c=newsst, **common_options, s=25)
        ax.set_title('linint2_points - Bilinear interpolation for 3000 random locations', fontsize=14, fontweight='bold')

    # Add coastlines to the subplots
    ax.coastlines()

    # Set axis tick values and limits
    ax.set_xticks(np.linspace(-180, 180, 13), crs=projection)
    ax.set_yticks(np.linspace(-60, 60, 5), crs=projection)
    ax.set_ylim(-60, 60)

    # Use Cartopy latitude, longitude tick labels
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)

# Add color bar and label details (title, size, etc.)
cax = axgr.cbar_axes[0]
cax.colorbar(p)
axis = cax.axis[cax.orientation]
axis.label.set_text('Temperature ($^{\circ} C$)')
axis.label.set_size(16)
axis.major_ticklabels.set_size(10)

# Save figure and show
plt.savefig('linint2_points.png', dpi=300)
plt.show()
