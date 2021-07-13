import typing

import cf_xarray
import cftime
import numpy as np
import xarray as xr
import pandas as pd
import warnings

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
    np.datetime64, or cftime.datetime)"""
    return np.issubdtype(
        d_arr.dtype,
        np.datetime64) or xr.core.common.contains_cftime_datetimes(d_arr)


def _validate_freq(freq):
    if freq not in _FREQUENCIES:
        raise ValueError(
            f"{freq} is not among supported frequency aliases={list(_FREQUENCIES)}"
        )


def _get_time_coordinate_info(dset, time_coord_name):
    if time_coord_name:
        time = dset[time_coord_name]
    else:
        time = dset.cf["time"]
        time_coord_name = time.name

    if not _contains_datetime_like_objects(time):
        raise ValueError(
            f"The {time_coord_name} coordinate should be either `np.datetime64` or `cftime.datetime`"
        )

    return time_coord_name


def _setup_clim_anom_input(dset, freq, time_coord_name):
    _validate_freq(freq)

    time_coord_name = _get_time_coordinate_info(dset, time_coord_name)
    time_invariant_vars = _find_time_invariant_vars(dset, time_coord_name)
    if time_invariant_vars:
        data = dset.drop_vars(time_invariant_vars)
    else:
        data = dset
    time_dot_freq = ".".join([time_coord_name, freq])

    return data, time_invariant_vars, time_coord_name, time_dot_freq


def _calculate_center_of_time_bounds(dset,
                                     time_dim,
                                     frequency,
                                     calendar,
                                     start=None,
                                     end=None):
    """Helper function to determine the time bounds based on the given dataset
    and frequency and then calculate the averages of them.

    Returns the dataset with the time coordinate changed to the center
    of the time bounds.
    """
    if sum(x is not None for x in [start, end]) == 1:
        raise ValueError(
            "Both `start` and `end` must be specified or both left unspecified."
        )
    if start is None and end is None:
        start = dset[time_dim].values[0]
        end = dset[time_dim].values[-1]
    time_bounds = xr.cftime_range(start, end, freq=frequency, calendar=calendar)
    time_bounds = time_bounds.append(time_bounds[-1:].shift(1, freq=frequency))
    time =  xr.DataArray(np.vstack((time_bounds[:-1], time_bounds[1:])).T,
                         dims=[time_dim, 'nbd']) \
        .mean(dim='nbd')
    return dset.assign_coords({time_dim: time})


def climatology(
        dset: typing.Union[xr.DataArray, xr.Dataset],
        freq: str,
        time_coord_name: str = None) -> typing.Union[xr.DataArray, xr.Dataset]:
    """Compute climatologies for a specified time frequency.

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

    Examples
    --------
    >>> import xarray as xr
    >>> import pandas as pd
    >>> import numpy as np
    >>> import geocat.comp
    >>> dates = pd.date_range(start="2000/01/01", freq="M", periods=24)
    >>> ts = xr.DataArray(np.arange(24).reshape(24, 1, 1), dims=["time", "lat", "lon"], coords={"time": dates})
    >>> ts
    <xarray.DataArray (time: 24, lat: 1, lon: 1)>
    array([[[ 0]],

        [[ 1]],

        [[ 2]],
    ...
        [[21]],

        [[22]],

        [[23]]])
    Coordinates:
    * time     (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2001-12-31
    Dimensions without coordinates: lat, lon
    >>> geocat.comp.climatology(ts, 'year')
    <xarray.DataArray (year: 2, lat: 1, lon: 1)>
    array([[[ 5.5]],

        [[17.5]]])
    Coordinates:
    * year     (year) int64 2000 2001
    Dimensions without coordinates: lat, lon
    >>> geocat.comp.climatology(ts, 'season')
    <xarray.DataArray (season: 4, lat: 1, lon: 1)>
    array([[[10.]],

        [[12.]],

        [[ 9.]],

        [[15.]]])
    Coordinates:
    * season   (season) object 'DJF' 'JJA' 'MAM' 'SON'
    Dimensions without coordinates: lat, lon
    """
    data, time_invariant_vars, time_coord_name, time_dot_freq = _setup_clim_anom_input(
        dset, freq, time_coord_name)

    grouped = data.groupby(time_dot_freq)
    # TODO: Compute weighted climatologies when `time_bounds` are available
    clim = grouped.mean(time_coord_name)
    if time_invariant_vars:
        return xr.concat([dset[time_invariant_vars], clim], dim=time_coord_name)
    else:
        return clim


def anomaly(
        dset: typing.Union[xr.DataArray, xr.Dataset],
        freq: str,
        time_coord_name: str = None) -> typing.Union[xr.DataArray, xr.Dataset]:
    """Compute anomalies for a specified time frequency.

    Parameters
    ----------
    dset : xr.Dataset, xr.DataArray
        The data on which to operate

    freq : str
        Anomaly frequency alias. Accepted alias:

            - 'day': for daily anomalies
            - 'month': for monthly anomalies
            - 'year': for annual anomalies
            - 'season': for seasonal anomalies

    time_coord_name: str, Optional
         Name for time coordinate to use

    Returns
    -------
    computed_dset : xr.Dataset, xr.DataArray
       The computed anomaly data

    Examples
    --------
    >>> import xarray as xr
    >>> import pandas as pd
    >>> import numpy as np
    >>> import geocat.comp
    >>> dates = pd.date_range(start="2000/01/01", freq="M", periods=24)
    >>> ts = xr.DataArray(np.arange(24).reshape(24, 1, 1), dims=["time", "lat", "lon"], coords={"time": dates})
    >>> ts
    <xarray.DataArray (time: 24, lat: 1, lon: 1)>
    array([[[ 0]],

        [[ 1]],

        [[ 2]],

    ...

        [[21]],

        [[22]],

        [[23]]])
    Coordinates:
    * time     (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2001-12-31
    Dimensions without coordinates: lat, lon
    >>> geocat.comp.anomaly(ts, 'season')
    <xarray.DataArray (time: 24, lat: 1, lon: 1)>
    array([[[-10.]],

        [[ -9.]],

        [[ -7.]],

    ...

        [[  6.]],

        [[  7.]],

        [[ 13.]]])
    Coordinates:
    * time     (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2001-12-31
        season   (time) <U3 'DJF' 'DJF' 'MAM' 'MAM' ... 'SON' 'SON' 'SON' 'DJF'
    Dimensions without coordinates: lat, lon
    """

    data, time_invariant_vars, time_coord_name, time_dot_freq = _setup_clim_anom_input(
        dset, freq, time_coord_name)

    clim = climatology(data, freq, time_coord_name)
    anom = data.groupby(time_dot_freq) - clim
    if time_invariant_vars:
        return xr.merge([dset[time_invariant_vars], anom])
    else:
        return anom


def month_to_season(
    dset: typing.Union[xr.Dataset, xr.DataArray],
    season: str,
    time_coord_name: str = None,
) -> typing.Union[xr.Dataset, xr.DataArray]:
    """Computes a user-specified three-month seasonal mean.

    This function takes an xarray dataset containing monthly data spanning years and
    returns a dataset with one sample per year, for a specified three-month season.

    Parameters
    ----------
    dset : xr.Dataset, xr.DataArray
        The data on which to operate
    season : str
        A string representing the season to calculate: e.g., "JFM", "JJA".
        Valid values are:

         - DJF {December, January, February}
         - JFM {January, February, March}
         - FMA {February, March, April}
         - MAM {March, April, May}
         - AMJ {April, May, June}
         - MJJ {May, June, July}
         - JJA {June, July, August}
         - JAS {July, August, September}
         - ASO {August, September, October}
         - SON {September, October, November}
         - OND {October, November, Decmber}
         - NDJ {November, Decmber, January}
    time_coord_name: str, Optional
        Name for time coordinate to use

    Returns
    -------
    computed_dset : xr.Dataset, xr.DataArray
       The computed data

    Notes
    -----
    This function requires the number of months to be a multiple of 12, i.e. full years must be provided.
    Time stamps are centered on the season. For example, seasons='DJF' returns January timestamps.
    If a calculated season's timestamp falls outside the original range of monthly values, then the calculated mean
    is dropped.  For example, if the monthly data's time range is [Jan-2000, Dec-2003] and the season is "DJF", the
    seasonal mean computed from the single month of Dec-2003 is dropped.
    """

    time_coord_name = _get_time_coordinate_info(dset, time_coord_name)
    mod = 12
    if dset[time_coord_name].size % mod != 0:
        raise ValueError(
            f"The {time_coord_name} axis length must be a multiple of {mod}.")

    seasons_pd = {
        "DJF": ([12, 1, 2], 'QS-DEC'),
        "JFM": ([1, 2, 3], 'QS-JAN'),
        "FMA": ([2, 3, 4], 'QS-FEB'),
        "MAM": ([3, 4, 5], 'QS-MAR'),
        "AMJ": ([4, 5, 6], 'QS-APR'),
        "MJJ": ([5, 6, 7], 'QS-MAY'),
        "JJA": ([6, 7, 8], 'QS-JUN'),
        "JAS": ([7, 8, 9], 'QS-JUL'),
        "ASO": ([8, 9, 10], 'QS-AUG'),
        "SON": ([9, 10, 11], 'QS-SEP'),
        "OND": ([10, 11, 12], 'QS-OCT'),
        "NDJ": ([11, 12, 1], 'QS-NOV'),
    }
    try:
        (months, quarter) = seasons_pd[season]
    except KeyError:
        raise KeyError(
            f"contributed: month_to_season: bad season: SEASON = {season}. Valid seasons include: {list(seasons_pd.keys())}"
        )

    # Filter data to only contain the months of interest
    data_filter = dset.sel(
        {time_coord_name: dset[time_coord_name].dt.month.isin(months)})

    if season == 'DJF':  # For this season, the last "mean" will be the value for Dec so we drop the last month
        data_filter = data_filter.isel({time_coord_name: slice(None, -1)})
    elif season == 'NDJ':  # For this season, the first "mean" will be the value for Jan so we drop the first month
        data_filter = data_filter.isel({time_coord_name: slice(1, None)})

    # Group the months into three and take the mean
    means = data_filter.resample({
        time_coord_name: quarter
    }, loffset='MS').mean()

    # The line above tries to take the mean for all quarters even if there is not data for some of them
    # Therefore, we must filter out the NaNs
    return means.sel(
        {time_coord_name: means[time_coord_name].dt.month == months[1]})


def calendar_average(
        dset: typing.Union[xr.DataArray, xr.Dataset],
        freq: str,
        time_dim: str = None,
        climatology: bool = False,
        calendar: str = None) -> typing.Union[xr.DataArray, xr.Dataset]:
    """This function computes averages according to a given time frequency.

    Parameters
    ----------
    dset : :class:`xarray.Dataset`, :class:`xarray.DataArray`
        The data on which to operate. It must be evenly spaced in the time dimension.

    freq : :class:`str`
        Frequency alias. Accepted alias:
            - 'hour': for hourly averages
            - 'day': for daily averages
            - 'month': for monthly averages
            - 'season': for meteorological seasonal averages (DJF, MAM, JJA, and SON)
            - 'yearly': for yearly averages

    time_dim : :class:`str`, Optional
        Name of the time coordinate for `xarray` objects

    climatology : :class:`bool`, Optional
        Default False. If False, the average for each period (day, month, etc.)
        will be calculated for it's given year (i.e. the average for Jan-2000
        will be independent of the average for Jan-2001). If True,
        climatological averages will be calculated across all years provided.

    calendar : :class:`str`, Optional
        Alias for the kind of calendar the time dimension of the data is in.
        Defaults to `None`, in which case the calendar is infered from the data.
        A list of accepted aliases can be found `here <http://xarray.pydata.org/en/stable/generated/xarray.cftime_range.html>`_

    Returns
    -------
    computed_dset: same type as dset
        The computed data

    Notes
    -----
    Seasonal averages are weighted based on the number of days in each month.
    This means that the given data must be monotonic (i.e. data every 6 hours,
    every two days, every month, etc.) and must not cross month boundaries
    (i.e. don't use weekly averages where the week falls in two
    different months)
    """
    # TODO: add functionality for users to select specific seasons or hours for avgs/clims
    freq_dict = {
        'hour': ('%m-%d %H', 'H', '30min'),
        'day': ('%m-%d', 'D', '12H'),
        'month': ('%m', 'MS', 'SMS'),
        'season': (None, 'QS-DEC', 'MS'),
        'year': (None, 'YS', '6MS')
    }

    try:
        freq_dict[freq]
    except KeyError:
        raise KeyError(
            f"contributed: calendar_average: bad period: PERIOD = {freq}. Valid periods include: {list(freq_dict.keys())}"
        )

    # If freq is 'season', key is set to monthly in order to calculate monthly
    # averages which are then used to calculate seasonal averages
    key = 'month' if freq in {'season', 'year'} else freq

    (format, frequency, offset) = freq_dict[key]

    time_dim = _get_time_coordinate_info(dset, time_dim)

    # Retrieve calendar name
    if calendar is None:
        calendar = xr.coding.times.infer_calendar_name(dset[time_dim])

    if freq == 'year' and climatology:
        climatology = False
        warnings.warn(
            'Cannot compute yearly climatology data since climatologies are averaged over years. To remove warning, set `climatology` to False'
        )
    # Average data across years
    if climatology:
        if freq == 'season':
            # Calculate monthly average before calculating seasonal climatologies
            dset = dset.resample({time_dim: frequency}).mean().dropna(time_dim)

            # Compute the weights for the months in each season so that the
            # seasonal averages account for months being of different lengths
            month_length = dset[time_dim].dt.days_in_month.groupby(
                f"{time_dim}.season")
            weights = month_length / month_length.sum()
            dset = (dset * weights).groupby(f"{time_dim}.season")
            dset = dset.sum(dim=time_dim)
        else:
            # Retrieve floor of median year
            median_yr = np.median(dset[time_dim].dt.year.values)

            # Group data by format then calculate average of groups
            dset = dset.groupby(
                dset[time_dim].dt.strftime(format)).mean().rename(
                    {'strftime': time_dim})

            # Create array of datetimes to set as time coordinate of returned data
            # Offsets are used to ensure the time coordinate of the returned climatology is centered on the period
            start_time = dset[time_dim].values[0]
            end_time = dset[time_dim].values[-1]

            dset = _calculate_center_of_time_bounds(
                dset,
                time_dim,
                frequency,
                calendar,
                start=f'{median_yr:.0f}-{start_time}',
                end=f'{median_yr:.0f}-{end_time}')

    # Average data for each period considering the year of the period
    else:
        # Resample data using given frequency which preserves the year of the data
        dset = dset.resample({time_dim: frequency}).mean().dropna(time_dim)
        if freq == 'season' or freq == 'year':
            key = freq
            (format, frequency, offset) = freq_dict[key]
            # Compute the weights for the months in each season so that the
            # seasonal averages account for months being of different lengths
            month_length = dset[time_dim].dt.days_in_month.resample(
                {time_dim: frequency})
            weights = month_length.map(lambda group: group / group.sum())
            dset = (dset * weights).resample({time_dim: frequency}).sum()

        # Set time coordinates to center of time bounds
        dset[time_dim] = dset[time_dim].dt.strftime('%Y-%m-%d %H:%M:%S')
        dset = _calculate_center_of_time_bounds(dset, time_dim, frequency,
                                                calendar)
    return dset
