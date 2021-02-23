from pathlib import Path
import os
import numpy as np
import xarray as xr
from numba import njit, prange

run_parallel = False

def _pressure_from_hybrid(psfc, hya, hyb, p0=100000.):
    # p = a(k)*p0 + b(k)*ps.
    # this will be in Pa
    return hya*p0 + hyb*psfc


@njit(parallel=run_parallel)
def _vertical_remap(x_mdl, p_mdl, plev):
    """Apply simple 1-d interpolation to a field, x
       given the pressure p and the new pressures plev.
       x_mdl, p_mdl are numpy arrays of shape (nlevel, spacetime).
    """
    out_shape = (plev.shape[0], x_mdl.shape[1])
    output = np.full(out_shape, np.nan)
    for i in prange(out_shape[1]):
        output[:,i] = np.interp(plev, p_mdl[:,i], x_mdl[:,i])
    return output


def lev_to_plev(data, ps, hyam, hybm, P0=100000., new_levels=None, parallel=False):
    """Interpolate data from hybrid-sigma levels to isobaric levels.

    data : DataArray with a 'lev' coordinate
    ps   : DataArray of surface pressure (Pa), same time/space shape as data
    hyam, hybm : hybrid coefficients, size of len(lev)
    P0 : reference pressure
    new_levels : the output pressure levels (Pa)
    parallel : if True, use the Numba version to parallelize interpolation step.
    """
    pressure = _pressure_from_hybrid(ps, hyam, hybm, P0)  # Pa
    if new_levels is None:
        pnew = 100.0 * np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1])  # mandatory levels, converted to Pa
    else:
        pnew = new_levels
    # reshape data and pressure assuming "lev" is the name of the coordinate
    zdims = [i for i in data.dims if i != 'lev']
    dstack = data.stack(z=zdims)
    pstack = pressure.stack(z=zdims)


    # if parallel:
    #     output = vert_remap2(dstack.values, pstack.values, pnew)
    # else:
    #     output = vert_remap(dstack.values, pstack.values, pnew)

    run_parallel = parallel
    output = _vertical_remap(dstack.values, pstack.values, pnew)

    output = xr.DataArray(output, dims=("plev", "z"), coords={"plev":pnew, "z":pstack['z']})
    output = output.unstack()
    return output