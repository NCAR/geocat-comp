##############################################################################
# Import packages:

import pandas as pd
from metpy.units import units
import metpy.calc as mpcalc

import geocat.datafiles as gdf

###############################################################################
# Read in data:

# Open a CSV data file using xarray default engine and load the data into xarrays
ds = pd.read_csv(gdf.get('ascii_files/sounding.testdata'),
                 delimiter='\\s+',
                 header=None)

# Extract the data
p = ds[1].values * units.hPa  # Pressure [mb/hPa]
# p = np.atleast_1d(p)
# print(p[2])

tc = (ds[5].values + 2) * units.degC  # Temperature [C]
# print(tc[11])

tdc = (ds[9].values + 2) * units.degC  # Dew pt temp  [C]
ta = mpcalc.parcel_profile(p, tc[0], tdc[0])  # Parcel profile
tac = (ta.magnitude - 273.15) * units.degC  # Parcel temp in C
print("The parcel temp at 500mb is", tac[11])
##############################################################################

# CAPE and CIN
cape = mpcalc.cape_cin(p, tc, tdc, ta)
# print(cape[0].magnitude)

# Precipitable Water
precp = mpcalc.precipitable_water(p, tdc)
precp = precp / 10  # Convert mm to cm
# print(precp.magnitude)

# Pressure and temperature of lcl
lcl = mpcalc.lcl(p[0], tc[0], tdc[0])

# Shox/Stability
# Calculate parcel temp when raised dry adiabatically from surface to lcl
dl = mpcalc.dry_lapse(p[2], tac[0], p[0])
dl = (dl.magnitude - 273.15) * units.degC  # Change units to C
# print((dl))

# Calculate parcel temp when raised moist adiabatically from lcl to 500mb
ml = mpcalc.moist_lapse(p[11], dl, p[2])
# print(ml)

# Define environmental temp at 500mb
ttop = tc[11]
print("The environmental temp at 500mb is", ttop)

shox = ttop - ml
print("The calculated value for Showalter index is", shox)
