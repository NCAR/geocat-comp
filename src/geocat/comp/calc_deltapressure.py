import numpy as np
import xarray as xr
import warnings


def _calc_deltapressure_1D(pressure_lev, surface_pressure):
    """Helper function for `calc_deltapressure1. Calculates the pressure layer
    thickness (delta pressure) of a one-dimensional pressure level array.

    Returns an array of length matching `pressure_lev`.

    Parameters
    ----------
    pressure_lev : :class:`np.Array`
        The pressure level array. May be in ascending or descending order.
        Must have the same units as `surface_pressure`.

    surface_pressure : :class:`float`
        The scalar surface pressure. Must have the same units as
        pressure_lev`.

    Returns
    -------
    delta_pressure : :class:`np.Array`
        The pressure layer thickness array. Shares dimensions and units of
        `pressure_lev`.
    """
    pressure_top = min(pressure_lev)

    # Safety checks
    if not surface_pressure:
        warnings.warn("'surface_pressure` can't equal a missing value.")
    if pressure_top <= 0:
        warnings.warn("'pressure_lev` values must all be positive.")
    if pressure_top > surface_pressure:
        warnings.warn(
            "`surface_pressure` must be greater than minimum `pressure_lev` value."
        )

    # Sort so pressure increases (array goes from top of atmosphere to bottom)
    is_pressuredecreasing = pressure_lev[1] < pressure_lev[0]
    if is_pressuredecreasing:
        pressure_lev = np.flip(pressure_lev)

    # Calculate delta pressure
    delta_pressure = np.empty_like(pressure_lev)

    delta_pressure[0] = (pressure_lev[0] +
                         pressure_lev[1]) / 2 - pressure_top  # top level
    for i in (np.arange(1, len(pressure_lev) - 1)):  # middle levels
        delta_pressure[i] = (pressure_lev[i + 1] - pressure_lev[i - 1]) / 2
        i += 1
    delta_pressure[-1] = surface_pressure - (
        pressure_lev[-1] + pressure_lev[-2]) / 2  # bottom level

    # Return delta_pressure to original order
    if is_pressuredecreasing:
        delta_pressure = np.flip(delta_pressure)

    return delta_pressure


def calc_deltapressure(pressure_lev, surface_pressure):
    """Calculates the pressure layer thickness (delta pressure) of a constant
    pressure level coordinate system.

    Returns an array of length matching `pressure_lev`.

    Parameters
    ----------
    pressure_lev : :class:`np.Array`, :class:'xr.DataArray`
        The pressure level array. May be in ascending or descending order.
        Must have the same units as `surface_pressure`.
    surface_pressure : :class:`np.Array`, :class:'xr.DataArray`
        The scalar or N-dimensional surface pressure array. Must have the same
        units as `pressure_lev`. Cannot exceed 3 dimensions.

    Returns
    -------
    delta_pressure : :class:`np.Array`, :class:'xr.DataArray`
        The pressure layer thickness array. Shares units with `pressure_lev`.
        If `surface_pressure` is scalar, shares dimensions with
        `pressure_level`. If `surface_pressure` is an array than the returned
        array will have an additional dimension [e.g. (lat, lon, time) becomes
        (lat, lon, time, lev)].
    """
    # Get original array types
    type_surface_pressure = type(
        surface_pressure
    )  # save type for delta_pressure to same type as surface_pressure at end
    type_pressure_level = type(pressure_lev)

    # Preserve attributes for Xarray
    if type_surface_pressure == xr.DataArray:
        coords = surface_pressure.coords
        attrs = surface_pressure.attrs
        dims = surface_pressure.dims
        name = surface_pressure.name
    if type_pressure_level == xr.DataArray:
        attrs = pressure_lev.attrs  # Overwrite attributes to match pressure_lev

    # Convert inputs to numpy arrays
    try:
        pressure_lev = np.asarray(pressure_lev)
    except AttributeError:
        print("`pressure_lev` must be array-like.")

    try:
        surface_pressure = np.asarray(surface_pressure)
    except AttributeError:
        print("`surface_pressure` must be array-like.")

    # Get dimensions of `surface_pressure`
    try:
        dims = len(surface_pressure.shape)
    except:
        dims = 0

    # Safety check
    if dims > 3:
        warnings.warn("`surface_pressure` cannot have more than 3 dimensions.")

    # If Xarray save attributes
    type_surface_pressure = type(
        surface_pressure)  # save type for promoting back to Xarray at end
    if type_surface_pressure == xr.core.dataarray.DataArray:
        coords = surface_pressure.coords
        attrs = surface_pressure.attrs
        dims = surface_pressure.dims
        name = surface_pressure.name

    # Convert to floats to prevent integer division rounding errors
    pressure_lev = [float(i) for i in pressure_lev]

    # Calculate delta pressure
    if dims == 0:  # scalar case
        delta_pressure = _calc_deltapressure_1D(pressure_lev, surface_pressure)
    else:  # 1, 2, and 3 dimensional cases
        shape = surface_pressure.shape
        delta_pressure_shape = shape + (len(pressure_lev),
                                       )  # preserve shape for reshaping

        surface_pressure_flattened = np.ravel(
            surface_pressure)  # flatten to avoid nested for loops
        delta_pressure = [
            _calc_deltapressure_1D(pressure_lev, float(e))
            for e in surface_pressure_flattened
        ]

        delta_pressure = np.array(delta_pressure).reshape(delta_pressure_shape)

    # If passed in an Xarray array, return an Xarray array
    # Change this to return a dataset that has both surface pressure and delta pressure?
    if type_surface_pressure == xr.core.dataarray.DataArray:
        coords['lev'] = pressure_lev
        dims["lev"] = "lev"
        attrs["long name"] = "pressure layer thickness"
        delta_pressure = xr.DataArray(delta_pressure,
                                      coords=coords,
                                      dims=dims,
                                      attrs=attrs,
                                      name=name)

    return delta_pressure
