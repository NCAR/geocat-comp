"""
mpl_linint2points.py
==============
This script illustrates the following concepts:
   - Usage of geocat-comp's linint2_points function
   - Bilinear interpolation from a rectilinear grid to an unstructured grid or locations
   - Usage of geocat-datafiles for accessing NetCDF files
   - Usage of geocat-viz plotting convenience functions

See following GitHub repositories to see further information about the function and to access data:
    - linint2_points function: https://github.com/NCAR/geocat-comp
    - "sst.nc" data file: https://github.com/NCAR/geocat-datafiles/tree/master/netcdf_files

Dependencies:
    - geocat.comp
    - geocat.datafiles (Not necessary but for NetCDF data file access convenience)
    - geocat.viz (Not necessary but for plotting convenience)
    - Numpy
    - XArray
    - Cartopy
    - Matplotlib
    - Mpl_toolkits
"""

###############################################################################
# Import packages:
import geocat.comp
import geocat.datafiles as gdf
import geocat.viz.util as gvutil

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import cm
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid

###############################################################################
# Read in data:

# Open a netCDF data file (Sea surface temperature) using xarray default
# engine and load the data into xarrays
ds = xr.open_dataset(gdf.get('netcdf_files/sst.nc'))
sst = ds.TEMP[0, 0, :, :]
lat = ds.LAT[:]
lon = ds.LON[:]

###############################################################################
# Invoke geocat.comp.linint2_points():

# Provide (output) interpolation locations. This script uses 3000 arbitrary locations world-wide in order to demonstrate
# an extensive comparison of the linint2_points outputs to the original grid throughout the globe. The function can be
# even used for a single location though.
newlat = np.random.uniform(low=min(lat), high=max(lat), size=(3000,))
newlon = np.random.uniform(low=min(lon), high=max(lon), size=(3000,))

# Call `linint2_points` from `geocat-comp`
newsst = geocat.comp.linint2_points(sst, newlon, newlat, False)

###############################################################################
# Plot:

# Generate figure and set its size (width, height) in inches
fig = plt.figure(figsize=(10, 8))

# Generate Axes grid using a Cartopy projection
projection = ccrs.PlateCarree()
axes_class = (GeoAxes, dict(map_projection=projection))
axgr = AxesGrid(
    fig,
    111,
    axes_class=axes_class,
    nrows_ncols=(2, 1),
    axes_pad=0.7,
    cbar_location='right',
    cbar_mode='single',
    cbar_pad=0.5,
    cbar_size='3%',
    label_mode='')

# Create a dictionary for common plotting options for both subplots
common_options = dict(vmin=-30, vmax=30, cmap=cm.jet)

# Plot original grid and linint2_points interpolations as two subplots
# within the figure
for i, ax in enumerate(axgr):

  # Plot original grid and linint2_points interpolations within the subplots
  if (i == 0):
    p = sst.plot.contourf(
        ax=ax,
        **common_options,
        transform=projection,
        levels=16,
        extend='neither',
        add_colorbar=False,
        add_labels=False)
    ax.set_title(
        'Sea Surface Temperature - Original Grid',
        fontsize=14,
        fontweight='bold',
        y=1.04)
  else:
    ax.scatter(newlon, newlat, c=newsst, **common_options, s=25)
    ax.set_title(
        'linint2_points - Bilinear interpolation for 3000 random locations',
        fontsize=14,
        fontweight='bold',
        y=1.04)

  # Add coastlines to the subplots
  ax.coastlines()

  # Use geocat.viz.util convenience function to add minor and major tick
  # lines
  gvutil.add_major_minor_ticks(ax)

  # Use geocat.viz.util convenience function to set axes limits & tick
  # values without calling several matplotlib functions
  gvutil.set_axes_limits_and_ticks(
      ax,
      ylim=(-60, 60),
      xticks=np.linspace(-180, 180, 13),
      yticks=np.linspace(-60, 60, 5))

  # Use geocat.viz.util convenience function to make plots look like NCL
  # plots by using latitude, longitude tick labels
  gvutil.add_lat_lon_ticklabels(ax, zero_direction_label=True)

# Add color bar and label details (title, size, etc.)
cax = axgr.cbar_axes[0]
cax.colorbar(p)
axis = cax.axis[cax.orientation]
axis.label.set_text(r'Temperature ($^{\circ} C$)')
axis.label.set_size(16)
axis.major_ticklabels.set_size(10)

# Save figure and show
plt.savefig('linint2_points.png', dpi=300)
plt.show()
