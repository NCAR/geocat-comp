##############################################################################
# Import packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from itertools import chain

import geocat.viz.util as gvutil
import geocat.datafiles as gdf

###############################################################################
# Read in data:

# Open a CSV data file using xarray default engine and load the data into xarrays
da = pd.read_csv(gdf.get('ascii_files/sounding.testdata'), delimiter='\\s+', header=None)

# Extract the data
p = da[1].values*units.hPa   # Pressure [mb/hPa]

tc = (da[5].values + 2)*units.degC # Temperature [C]
# print(tc[11])

tdc = (da[9].values + 2)*units.degC  # Dew pt temp  [C]
ta = mpcalc.parcel_profile(p, tc[0], tdc[0]) # Parcel profile
tac = (ta.magnitude - 273.15)*units.degC # Parcel temp in C

# Create dummy wind data
wspd = np.linspace(0, 150, len(p))*units.knots    # Wind speed   [knots or m/s]
wdir = np.linspace(0, 360, len(p))*units.degrees    # Meteorological wind dir
u, v = mpcalc.wind_components(wspd, wdir)   # Calculate wind components

##############################################################################
# Create function to write NCL style str for later implementation

def get_skewt_vars(ds_name, p, tc, tdc, tac, pres_pos = 1, envT_loc = 5):
    
    """
    This function processes the dataset values and returns a string element which
    can be used as a subtitle to replicate the styles of NCL Skew-T Diagrams
    
    Args:
        ds_name (:class: `pandas.core.frame.DataFrame`, :class: `netCDF4.Dataset`
        p (:class: `pint.quantity.build_quantity_class.<locals>.Quantity`):
            Pressure level input from dataset
        tc (:class: `pint.quantity.build_quantity_class.<locals>.Quantity`):
            Temperature for parcel from dataset
        
        tdc (:class: `pint.quantity.build_quantity_class.<locals>.Quantity`):
            Dew point temperature for parcel from dataset
        
        tac (:class: `pint.quantity.build_quantity_class.<locals>.Quantity`):
            Parcel profile temperature converted to degC
            
        pres_pos (:class: `int`):
            Position of column in dataset that contains pressure data
            
        envT_loc (:class: `int`):
            Position of column in dataset that contains environmental 
            temperature data
    
    Returns:
        :class: 'str'
        
    """

    # CAPE 
    cape = mpcalc.cape_cin(p, tc, tdc, ta)
    cape = cape[0].magnitude 
    # print(cape)
    
    # Precipitable Water
    pwat = mpcalc.precipitable_water(p, tdc)
    pwat = (pwat.magnitude/10)*units.cm # Convert mm to cm 
    pwat = pwat.magnitude
    # print(precp) 
    
    # Pressure and temperature of lcl
    lcl = mpcalc.lcl(p[0], tc[0], tdc[0])
    plcl = lcl[0].magnitude
    tlcl = lcl[1].magnitude
    # print(lcl)

    # Shox/Stability
    # Calculate parcel temp when raised dry adiabatically from surface to lcl
    # Define a 500mb height for second part of shox calculation
    p_end = 500*units.hPa
    dl = mpcalc.dry_lapse(lcl[0], tac[0], p[0])
    dl = (dl.magnitude - 273.15)*units.degC # Change units to C
    
    # Calculate parcel temp when raised moist adiabatically from lcl to 500mb
    ml = mpcalc.moist_lapse(p_end, dl, lcl[0]) 
   
    # Define environmental temp at 500mb
    # print(ds.loc[ds[1] == 500])
    ttop = ds_name.loc[ds_name[pres_pos] == 500] 
    ttop = (ttop[envT_loc].values + 2)*units.degC
    # print("The environmental temp at 500mb is", ttop)

    # Calculate the Showalter index
    shox = ttop - ml
    shox = int(shox.magnitude)
    # print("The calculated value for Showalter index is", shox)
    
    # Place calculated values in iterable list 
    vals = [ plcl, tlcl, shox, pwat, cape]
    vals = [round(num) for num in vals]
    
    # Define variable names for calculated values
    names = ['Plcl=', 'Tlcl[C]=', 'Shox=', 'Pwat[cm]=', 'Cape[J]=']
    
    # Combine the list of values with their corresponding labels
    lst = list(chain.from_iterable(zip(names, vals)))
    lst = map(str,lst)
    
    # Create one large string for later plotting use
    joined = ' '.join(lst)
    
    return joined

##############################################################################
# Plot:

# Note that MetPy forces the x axis scale to be in Celsius and the y axis
# scale to be in hectoPascals. Once data is plotted, then the axes labels are
# automatically added
fig = plt.figure(figsize=(12, 12))

# The rotation keyword changes how skewed the temperature lines are. MetPy has
# a default skew of 30 degrees
skew = SkewT(fig, rotation=45)
ax = skew.ax

# Plot temperature and dew point
skew.plot(p, tc, color='black')
skew.plot(p, tdc, color='blue')

# Draw parcel path
parcel_prof = mpcalc.parcel_profile(p, tc[0], tdc[0]).to('degC')
skew.plot(p, parcel_prof, color='red', linestyle='--')
u = np.where(p>=100*units.hPa, u, np.nan)
v = np.where(p>=100*units.hPa, v, np.nan)
p = np.where(p>=100*units.hPa, p, np.nan)

# Add wind barbs
skew.plot_barbs(pressure=p[::2],
                u=u[::2],
                v=v[::2],
                xloc=1.05,
                fill_empty=True,
                sizes=dict(emptybarb=0.075, width=0.1, height=0.2))

# Draw line underneath wind barbs
line = mlines.Line2D([1.05, 1.05], [0, 1],
                      color='gray',
                      linewidth=0.5,
                      transform=ax.transAxes,
                      clip_on=False,
                      zorder=1)
ax.add_line(line)

# Shade every other section between isotherms
x1 = np.linspace(-100, 40, 8)  # The starting x values for the shaded regions
x2 = np.linspace(-90, 50, 8)  # The ending x values for the shaded regions
y = [1050, 100]  # The range of y values that the shades regions should cover
for i in range(0, 8):
    skew.shade_area(y=y,
                    x1=x1[i],
                    x2=x2[i],
                    color='limegreen',
                    alpha=0.25,
                    zorder=1)

# Choose starting temperatures in Kelvin for the dry adiabats
t0 = units.K * np.arange(243.15, 444.15, 10)
skew.plot_dry_adiabats(t0=t0, linestyles='solid', colors='tan', linewidths=1.5)

# Choose starting temperatures in Kelvin for the moist adiabats
t0 = units.K * np.arange(281.15, 306.15, 4)
skew.plot_moist_adiabats(t0=t0,
                          linestyles='solid',
                          colors='lime',
                          linewidth=1.5)

# Choose mixing ratios
w = np.array([0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.020]).reshape(-1, 1)

# Choose the range of pressures that the mixing ratio lines are drawn over
p_levs = units.hPa * np.linspace(1000, 400, 7)

# Plot mixing ratio lines
skew.plot_mixing_lines(w,
                        p_levs,
                        linestyle='dashed',
                        colors='lime',
                        linewidths=1)

# Use geocat.viz utility functions to set axes limits and ticks
gvutil.set_axes_limits_and_ticks(
    ax=ax,
    xlim=[-32, 38],
    yticks=[1000, 850, 700, 500, 400, 300, 250, 200, 150, 100])

# Use geocat.viz utility function to change the look of ticks and ticklabels
gvutil.add_major_minor_ticks(ax=ax,
                              x_minor_per_major=1,
                              y_minor_per_major=1,
                              labelsize=14)
# The utility function draws tickmarks all around the plot. We only need ticks
# on the left and bottom edges
ax.tick_params('both', which='both', top=False, right=False)

# Use geocat.viz utility functions to add a main title
gvutil.set_titles_and_labels(ax=ax,
                              maintitle="Raob; [Wind Reports]",
                              maintitlefontsize=22,
                              xlabel='Temperature (C)',
                              ylabel='P (hPa)',
                              labelfontsize=14)

# Change the style of the gridlines
plt.grid(True,
          which='major',
          axis='both',
          color='tan',
          linewidth=1.5,
          alpha=0.5)

# Create subtitle var for plotting
title_var = get_skewt_vars(da, p, tc, tdc, tac, )

# Add subtitle to plot
fig.text(.30, .89, title_var , size=12)

plt.show()