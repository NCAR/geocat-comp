import numpy as np
import xarray as xr

__pres_lev_mandatory__ = np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1])  # Mandatory pressure levels (mb)
__pres_lev_mandatory__ = __pres_lev_mandatory__ * 100.0  # Convert mb to Pa

def _pressure_from_hybrid(psfc, hya, hyb, p0=100000.):
    """
    Calculate pressure at the hybrid levels
    """
    # This will be in Pa
    return hya * p0 + hyb * psfc


def _vertical_remap(x_mdl, p_mdl, plev):
    """
    Apply simple 1-d interpolation to a field, x given the pressure p.

    Parameters
    ----------
    x_mdl: `numpy.ndarray`:
        Hybrid-sigma levels of shape (nlevel, spacetime).

    p_mdl: `numpy.ndarray`:
        Data of shape (nlevel, spacetime).

    plev: `numpy.ndarray`:
        New pressure levels.
    """
    out_shape = (plev.shape[0], x_mdl.shape[1])
    output = np.full(out_shape, np.nan)

    for i in range(out_shape[1]):
        output[:,i] = np.interp(plev, p_mdl[:,i], x_mdl[:,i])

    return output


def interp_hybrid_to_pressure(data, ps, hyam, hybm, p0=100000., new_levels=__pres_lev_mandatory__):
    """
    Interpolate data from hybrid-sigma levels to isobaric levels.

    Parameters
    ----------
    data : `xarray.DataArray`:
        Multidimensional data array, which holds hybrid-sigma levels and has a 'lev' coordinate.

    ps : `xarray.DataArray`:
        A multi-dimensional array of surface pressures (Pa), same time/space shape as data.

    hyam, hybm : `xarray.DataArray`:
        One-dimensional arrays containing the hybrid A and B coefficients. Must have the same
        dimension size as the 'lev' dimension of data.

    p0 :
        Scalar numeric value equal to surface reference pressure (Pa).

    new_levels : the output pressure levels (Pa)
        A one-dimensional array of output pressure levels (Pa). If not given, the mandatory
        list of 21 pressure levels is used.

    """
    pressure = _pressure_from_hybrid(ps, hyam, hybm, p0)  # Pa

    if new_levels is not None:
        pnew = new_levels

    # reshape data and pressure assuming "lev" is the name of the coordinate
    zdims = [i for i in data.dims if i != 'lev']
    dstack = data.stack(z=zdims)
    pstack = pressure.stack(z=zdims)

    # Apply vertical interpolation
    # Apply Dask parallelization with xarray.apply_ufunc
    output = xr.apply_ufunc(_vertical_remap,
                            dstack,
                            pstack,
                            pnew,
                            exclude_dims=set(("lev",)),  # dimensions allowed to change size. Must be set!
                            input_core_dims=[["lev", "z"], ["lev", "z"], ["plev"]],  # Set "lev" as core dimension in both dstack and pstack
                            output_core_dims=[["plev", "z"]],  # Specify output dimensions
                            vectorize=True,  # loop over non-core dims
                            dask="parallelized",  # Dask parallelization
                            output_dtypes=[dstack.dtype],
                            )

    # Adjust dims and coords
    output = xr.DataArray(output, dims=("plev", "z"), coords={"plev":pnew, "z":pstack['z']})
    output = output.unstack()

    return output
