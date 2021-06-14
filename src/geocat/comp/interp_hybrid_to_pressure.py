import cf_xarray
import metpy.interpolate
import numpy as np
import xarray as xr

__pres_lev_mandatory__ = np.array([
    1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10,
    7, 5, 3, 2, 1
]).astype(np.float32)  # Mandatory pressure levels (mb)
__pres_lev_mandatory__ = __pres_lev_mandatory__ * 100.0  # Convert mb to Pa


def _pressure_from_hybrid(psfc, hya, hyb, p0=100000.):
    """Calculate pressure at the hybrid levels."""
    # This will be in Pa
    return hya * p0 + hyb * psfc


def interp_hybrid_to_pressure(data,
                              ps,
                              hyam,
                              hybm,
                              p0=100000.,
                              new_levels=__pres_lev_mandatory__,
                              lev_dim=None,
                              method='linear'):
    """Interpolate data from hybrid-sigma levels to isobaric levels.

    Notes
    -----
    ACKNOWLEDGEMENT: We'd like to thank to Brian Medeiros (https://github.com/brianpm), Matthew Long
    (https://github.com/matt-long), and Deepak Cherian (https://github.com/dcherian) at NCAR for their
    great contributions since the code implemented here is mostly based on their work.

    Parameters
    ----------
    data : xarray.DataArray
        Multidimensional data array, which holds hybrid-sigma levels and has a `lev_dim` coordinate.

    ps : xarray.DataArray
        A multi-dimensional array of surface pressures (Pa), same time/space shape as data.

    hyam, hybm : xarray.DataArray
        One-dimensional arrays containing the hybrid A and B coefficients. Must have the same
        dimension size as the `lev_dim` dimension of data.

    p0 :
        Scalar numeric value equal to surface reference pressure (Pa).

    new_levels : np.ndarray
        A one-dimensional array of output pressure levels (Pa). If not given, the mandatory
        list of 21 pressure levels is used.

    lev_dim : str
        String that is the name of level dimension in data. Defaults to "lev".

    method : str
        String that is the interpolation method; can be either "linear" or "log". Defaults to "linear".
    """

    # Determine the level dimension and then the interpolation axis
    if lev_dim is None:
        try:
            lev_dim = data.cf["vertical"].name
        except Exception:
            raise ValueError(
                "Unable to determine vertical dimension name. Please specify the name via `lev_dim` argument.'"
            )

    interp_axis = data.dims.index(lev_dim)

    # Calculate pressure levels at the hybrid levels
    pressure = _pressure_from_hybrid(ps, hyam, hybm, p0)  # Pa

    # Define interpolation function
    if method == 'linear':
        func_interpolate = metpy.interpolate.interpolate_1d
    elif method == 'log':
        func_interpolate = metpy.interpolate.log_interpolate_1d
    else:
        raise ValueError(f'Unknown interpolation method: {method}. '
                         f'Supported methods are: "log" and "linear".')

    def _vertical_remap(data, pressure):
        """Define interpolation function."""

        return func_interpolate(new_levels, pressure, data, axis=interp_axis)

    # Apply vertical interpolation
    # Apply Dask parallelization with xarray.apply_ufunc
    output = xr.apply_ufunc(
        _vertical_remap,
        data,
        pressure,
        exclude_dims=set((lev_dim,)),  # Set dimensions allowed to change size
        input_core_dims=[[lev_dim], [lev_dim]],  # Set core dimensions
        output_core_dims=[["plev"]],  # Specify output dimensions
        vectorize=True,  # loop over non-core dims
        dask="parallelized",  # Dask parallelization
        output_dtypes=[data.dtype],
        dask_gufunc_kwargs={"output_sizes": {
            "plev": len(new_levels)
        }},
    )

    # Set output dims and coords
    dims = ["plev"
           ] + [data.dims[i] for i in range(data.ndim) if i != interp_axis]

    coords = {}
    for (k, v) in data.coords.items():
        if k != lev_dim:
            coords.update({k: v})
        else:
            coords.update({"plev": new_levels})

    output = output.transpose(*dims).assign_coords(coords)

    return output
