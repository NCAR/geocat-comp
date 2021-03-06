#!/usr/bin/env python
# coding: utf-8
"""This example script demonstrates how to computes POP MOC field offline from
POP netcdf history files.  This routine is designed for the CESM4 ocean
component.

Example netCDF file used below can be found under https://github.com/NCAR/geocat-datafiles/tree/master/netcdf_files
"""

import numpy as np
import xarray as xr
import geocat.comp

# Open a netCDF data file using xarray default engine and load the data
# into xarrays
ds = xr.open_dataset("tavg_downsized.nc")
lat_aux_grid = ds.lat_aux_grid[:].values.astype(np.double)
w_e = ds.WVEL[:].values.astype(np.double)
w_i = ds.WISOP[:].values.astype(np.double)
w_sm = ds.WSUBM[:].values.astype(np.double)
tarea = ds.TAREA[:]
rmask = ds.REGION_MASK[:]
kmt = ds.KMT[:]
tlat = ds.TLAT[:].values.astype(np.double)

# Read important parameters from input data
nyaux = lat_aux_grid.shape[0]    # 395
km = np.max(kmt.values).astype(np.int)
ny = tarea.shape[0]
nx = tarea.shape[1]

# Generate rmlak: region_mask_lat_aux
rmlak = np.tile(rmask, (2, 1, 1)).astype(np.int32)
rmlak[0, :, :] = np.where(rmask > 0, 1, 0)
rmlak[1, :, :] = np.where(np.logical_and(rmask >= 6, rmask <= 11), 1, 0)
# todo Convert rmlak to xArray

# Generate a_wvel, a_bolus, and a_submeso
k3d = np.repeat(np.repeat(np.arange(0, km, 1).reshape(km, 1), ny,
                          axis=1)[:, :, np.newaxis],
                nx,
                axis=2)
kmt3d = np.repeat(kmt.values[np.newaxis, :, :], km, axis=0)
tarea3d = np.repeat(tarea.values[np.newaxis, :, :], km, axis=0)
ocean = k3d <= kmt3d

a_wvel = np.where(ocean, w_e[0, :, :, :] * tarea3d, 0.0)
a_bolus = np.where(ocean, w_i[0, :, :, :] * tarea3d, 0.0)
a_submeso = np.where(ocean, w_sm[0, :, :, :] * tarea3d, 0.0)

# Invoke `moc_globe_atl` from `geocat-comp`
result = geocat.comp.moc_globe_atl(lat_aux_grid,
                                   a_wvel,
                                   a_bolus,
                                   a_submeso,
                                   tlat,
                                   rmlak,
                                   msg=None,
                                   meta=False)

print("moc_globe_atl successfully generated output.")
