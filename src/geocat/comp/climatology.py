import xarray as xr
import cf_xarray
import numpy as np
import typing

xr.set_options(keep_attrs=True)

_FREQUENCIES = {"day", "month", "year", "season"}


def _find_time_invariant_vars(dset, time_coord_name):
    if isinstance(dset, xr.Dataset):
        return [
            v for v in dset.variables if time_coord_name not in dset[v].dims
        ]
    return


def _contains_datetime_like_objects(d_arr):
    """Check if a variable contains datetime like objects (either
    np.datetime64, or cftime.datetime)
    """
    return np.issubdtype(
        d_arr.dtype,
        np.datetime64) or xr.core.common.contains_cftime_datetimes(d_arr)


def _validate_freq(freq):
    if freq not in _FREQUENCIES:
        raise ValueError(
            f"{freq} is not among supported frequency aliases={list(_FREQUENCIES)}"
        )


def climatology(
        dset: typing.Union[xr.DataArray, xr.Dataset],
        freq: str,
        time_coord_name: str = None) -> typing.Union[xr.DataArray, xr.Dataset]:
    """
    Parameters
    ----------
    dset : xr.Dataset, xr.DataArray
        The data on which to operate

    freq : str
        Climatology frequency alias. Accepted alias:

            - 'day': for daily climatologies
            - 'month': for monthly climatologies
            - 'year': for annual climatologies
            - 'season': for seasonal climatologies

    time_coord_name: str, Optional
         Name for time coordinate to use

    Returns
    -------
    computed_dset : xr.Dataset, xr.DataArray
       The computed climatology data
    """
    _validate_freq(freq)

    if time_coord_name:
        time = dset[time_coord_name].copy()
    else:
        time = dset.cf["time"].copy()
        time_coord_name = time.name

    if not _contains_datetime_like_objects(time):
        # TODO: Do we want to support encoded time values in the future?
        raise ValueError(
            f"The {time_coord_name} coordinate should be either `np.datetime64` or `cftime.datetime`"
        )

    time_invariant_vars = _find_time_invariant_vars(dset, time_coord_name)

    if time_invariant_vars:
        data = dset.drop(time_invariant_vars)
    else:
        data = dset

    time_dot_freq = ".".join([time_coord_name, freq])
    grouped = data.groupby(time_dot_freq)
    # TODO: Compute weighted climatologies when `time_bounds` are available
    clim = grouped.mean(time_coord_name)
    if time_invariant_vars:
        return xr.concat([dset[time_invariant_vars], clim], dim=time_coord_name)
    else:
        return clim
