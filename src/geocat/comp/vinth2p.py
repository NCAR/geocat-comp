import metpy.interpolate
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


def interp_hybrid_to_pressure(data, ps, hyam, hybm, p0=100000., new_levels=__pres_lev_mandatory__, lev_dim='lev', method='linear'):
    """
    Interpolate data from hybrid-sigma levels to isobaric levels.

    ACKNOWLEDGEMENT: We'd like to thank to Brian Medeiros (https://github.com/brianpm), Matthew Long
    (https://github.com/matt-long), and Deepak Cherian (https://github.com/dcherian) at NCAR for their
    great contributions since the code implemented here is mostly based on their work.

    Parameters
    ----------
    data : `xarray.DataArray`:
        Multidimensional data array, which holds hybrid-sigma levels and has a `lev_dim` coordinate.

    ps : `xarray.DataArray`:
        A multi-dimensional array of surface pressures (Pa), same time/space shape as data.

    hyam, hybm : `xarray.DataArray`:
        One-dimensional arrays containing the hybrid A and B coefficients. Must have the same
        dimension size as the `lev_dim` dimension of data.

    p0 :
        Scalar numeric value equal to surface reference pressure (Pa).

    new_levels : the output pressure levels (Pa)
        A one-dimensional array of output pressure levels (Pa). If not given, the mandatory
        list of 21 pressure levels is used.

    lev_dim : `string`:
        String that is the name of level dimension in data. Defaults to "lev".

    method : `string`:
        String that is the interpolation method; can be either "linear" or "log". Defaults to "linear".

    """

    pressure = _pressure_from_hybrid(ps, hyam, hybm, p0)  # Pa

    interp_axis = data.dims.index(lev_dim)

    # Define interpolation function
    if method == 'linear':
        func_interpolate = metpy.interpolate.interpolate_1d
    elif method == 'log':
        func_interpolate = metpy.interpolate.log_interpolate_1d
    else:
        raise ValueError(f'Unknown interpolation method: {method}')

    def _vertical_remap(plev, pressure, data, interp_axis):
        """Define interpolation function."""

        return func_interpolate(plev, pressure, data, axis=interp_axis)

    # Apply vertical interpolation
    # Apply Dask parallelization with xarray.apply_ufunc
    output = xr.apply_ufunc(_vertical_remap,
                            new_levels,
                            pressure.values,
                            data.values,
                            interp_axis,
                            exclude_dims=set((lev_dim,)),  # dimensions allowed to change size. Must be set!
                            input_core_dims=[[lev_dim], [lev_dim], ["plev"], []],  # Set lev_dim as core dimension in both dstack and pstack
                            output_core_dims=[["plev"]],  # Specify output dimensions
                            vectorize=True,  # loop over non-core dims
                            dask="parallelized",  # Dask parallelization
                            output_dtypes=[data.dtype],
                            )

    # Set output dims and coords
    dims = ["plev"] + [data.dims[i] for i in range(data.ndim) if i != interp_axis]

    coords = {}
    for (k, v) in data.coords.items():
        if k != lev_dim:
            coords.update({k: v})
        else:
            coords.update({"plev": new_levels})

    output = xr.DataArray(output, dims=dims, coords=coords)

    return output
