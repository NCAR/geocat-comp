import cf_xarray
import cftime
import numpy as np
import typing
import xarray as xr

_FREQUENCIES = {"day", "month", "year", "season"}


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


def _calculate_center_of_time_bounds(
        dset: typing.Union[xr.Dataset, xr.DataArray],
        time_dim: str,
        freq: str,
        calendar: str,
        start: typing.Union[str, cftime.datetime],
        end: typing.Union[str, cftime.datetime],
        keep_attrs: bool = None) -> typing.Union[xr.Dataset, xr.DataArray]:
    """Helper function to determine the time bounds based on the given dataset
    and frequency and then calculate the averages of them.

    Returns the dataset with the time coordinate changed to the center
    of the time bounds.

    Parameters
    ----------
    dset : :class:`xarray.Dataset`, :class:`xarray.DataArray`
        The data on which to operate. It must be uniformly spaced in the time
        dimension.

    time_dim : str
        Name of the time coordinate for `xarray` objects

    freq : str
        Alias for the frequency of the adjusted timestamps (i.e. daily, monthly)

    calendar : str
        Alias for the calendar the time stamps are in (i.e. gregorian, no leap years)

    start : str, :class:`cftime.datetime`
        The starting date of the data. The string representation must be in ISO format

    end : str, :class:`cftime.datetime`
        The ending date of the data. The string representation must be in ISO format

    keep_attrs : bool, optional
        If True, attrs will be copied from the original object to the new one.
        If False, the new object will be returned without attributes.
        Defaults to None which means the attrs will only be kept in unambiguous circumstances.

    Returns
    -------
    computed_dset: :class:`xarray.Dataset`, :class:`xarray.DataArray`
        The data with adjusted time coordinate

    Notes
    -----
    See `xarray.cftime_range <http://xarray.pydata.org/en/stable/generated/xarray.cftime_range.html>`__ for accepted values for `freq` and `calendar`.
    """

    time_bounds = xr.cftime_range(start, end, freq=freq, calendar=calendar)
    time_bounds = time_bounds.append(time_bounds[-1:].shift(1, freq=freq))
    time =  xr.DataArray(np.vstack((time_bounds[:-1], time_bounds[1:])).T,
                         dims=[time_dim, 'nbd']) \
        .mean(dim='nbd', keep_attrs=keep_attrs)
    return dset.assign_coords({time_dim: time})


def _infer_calendar_name(dates):
    """Given an array of datetimes, infer the CF calendar name.

    This code was taken from `xarray/coding/times.py <https://github.com/pydata/xarray/blob/75eefb8e95a70f1667623a8418d81da3f3148a40/xarray/coding/times.py>`_
    as the function is considered internal by xarray and could change at
    anytime. It was copied to preserve the version that is compatible with
    functions in climatologies.py
    """
    if np.asarray(dates).dtype == "datetime64[ns]":
        return "proleptic_gregorian"
    else:
        return np.asarray(dates).ravel()[0].calendar


def climate_anomaly(
        dset: typing.Union[xr.DataArray, xr.Dataset],
        freq: str,
        time_dim: str = None,
        keep_attrs: bool = 'default') -> typing.Union[xr.DataArray, xr.Dataset]:
    """This function calculates climate anomalies by subtracting the long term
    mean of each ``freq`` period (day, month, season, or year) from each
    datapoint.

    Parameters
    ----------
    dset : :class:`xarray.Dataset`, :class:`xarray.DataArray`
        The data on which to operate. It must be uniformly spaced in the time
        dimension.

    freq : str
        Frequency alias. When the ``'year'`` allias is used, the yearly average is
        subtracted from each data point. Multiyear climatologies are not yet possible
        with this function. Accepted aliases:

        - `day`: for anomalies from the daily climatology
        - `month`: for anomalies from the monthly climatology
        - `season`: for anomalies from the seasonal climatology (seasons are DJF, MAM, JJA, and SON)
        - `year`: for anomalies from the yearly average

    time_dim : str, optional
        Name of the time coordinate for ``xarray`` objects. Defaults to ``None`` and
        infers the name from the data.

    keep_attrs : bool, optional
        If True, attrs will be copied from the original object to the new one.
        If False, the new object will be returned without attributes.
        Defaults to None which means the attrs will only be kept in unambiguous circumstances.

    Returns
    -------
    computed_dset : :class:`xarray.Dataset`, :class:`xarray.DataArray`
        The computed anomalies

    Note
    ----
    Seasonal averages are weighted based on the number of days in each month.
    This means that the given data must be uniformly spaced (i.e. data every 6
    hours, every two days, every month, etc.) and must not cross month
    boundaries (i.e. don't use weekly averages where the week falls in two
    different months)

    See Also
    --------
    Related GeoCAT Functions:
    :func:`climatology_average`
    :func:`calendar_average`

    Related NCL Functions:
    `calcDayAnomTLL <https://www.ncl.ucar.edu/Document/Functions/Contributed/calcDayAnomTLL.shtml>`__
    `calcMonAnomLLLT <https://www.ncl.ucar.edu/Document/Functions/Contributed/calcMonAnomLLLT.shtml>`__
    `calcMonAnomLLT <https://www.ncl.ucar.edu/Document/Functions/Contributed/calcMonAnomLLT.shtml>`__
    `calcMonAnomTLL <https://www.ncl.ucar.edu/Document/Functions/Contributed/calcMonAnomTLL.shtml>`__
    """
    # TODO add support for user specified seasons
    time_dim = _get_time_coordinate_info(dset, time_dim)
    attrs = {}
    if keep_attrs or keep_attrs is None:
        attrs = dset.attrs

    freq_dict = {
        'day': ('%m-%d', 'D'),
        'month': ('%m', 'MS'),
        'season': (None, 'QS-DEC'),
        'year': ('%y', 'Y')
    }

    if freq not in freq_dict:
        raise KeyError(
            f"Received bad period {freq!r}. Expected one of {list(freq_dict.keys())!r}"
        )
    format, frequency = freq_dict[freq]

    if freq == 'year':
        clim = calendar_average(dset, freq, time_dim, keep_attrs)
    else:
        clim = climatology_average(dset, freq, time_dim, keep_attrs)
    if freq == 'season':
        anom = dset.groupby(f"{time_dim}.season") - clim
        return anom.assign_attrs(attrs)
    else:
        anom = dset.groupby(dset[time_dim].dt.strftime(format)) - clim.groupby(
            clim[time_dim].dt.strftime(format)).sum()
        return anom.drop_vars('strftime').assign_attrs(attrs)


def month_to_season(
        dset: typing.Union[xr.Dataset, xr.DataArray],
        season: str,
        time_coord_name: str = None,
        keep_attrs: bool = None) -> typing.Union[xr.Dataset, xr.DataArray]:
    """Computes a user-specified three-month seasonal mean.

    This function takes an xarray dataset containing monthly data spanning years and
    returns a dataset with one sample per year, for a specified three-month season.

    Parameters
    ----------
    dset : :class:`xarray.Dataset`, :class:`xarray.DataArray`
        The data on which to operate

    season : str
        A string representing the season to calculate: e.g., "JFM", "JJA". Valid values are:
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
         - OND {October, November, December}
         - NDJ {November, December, January}

    time_coord_name : str, optional
        Name for time coordinate to use. Defaults to ``None`` and infers the name
        from the data.

    keep_attrs : bool, optional
        If True, attrs will be copied from the original object to the new one.
        If False, the new object will be returned without attributes.
        Defaults to None which means the attrs will only be kept in unambiguous circumstances.

    Returns
    -------
    computed_dset : :class:`xarray.Dataset`, :class:`xarray.DataArray`
       The computed data

    Note
    ----
    This function requires the number of months to be a multiple of 12, i.e. full years must be provided.
    Time stamps are centered on the season. For example, seasons='DJF' returns January timestamps.
    If a calculated season's timestamp falls outside the original range of monthly values, then the calculated mean
    is dropped.  For example, if the monthly data's time range is [Jan-2000, Dec-2003] and the season is "DJF", the
    seasonal mean computed from the single month of Dec-2003 is dropped.


    See Also
    --------
    Related NCL Functions:
    `month_to_season <https://www.ncl.ucar.edu/Document/Functions/Contributed/month_to_season.shtml>`__
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
    }, loffset='MS').mean(keep_attrs=keep_attrs)

    # The line above tries to take the mean for all quarters even if there is not data for some of them
    # Therefore, we must filter out the NaNs
    return means.sel(
        {time_coord_name: means[time_coord_name].dt.month == months[1]})


def calendar_average(
        dset: typing.Union[xr.DataArray, xr.Dataset],
        freq: str,
        time_dim: str = None,
        keep_attrs: bool = 'default') -> typing.Union[xr.DataArray, xr.Dataset]:
    """This function divides the data into time periods (months, seasons, etc)
    and computes the average for the data in each one.

    Parameters
    ----------
    dset : :class:`xarray.Dataset`, :class:`xarray.DataArray`
        The data on which to operate. It must be uniformly spaced in the time
        dimension.

    freq : str
        Frequency alias. Accepted alias:

        - `hour`: for hourly averages
        - `day`: for daily averages
        - `month`: for monthly averages
        - `season`: for meteorological seasonal averages (DJF, MAM, JJA, and SON)
        - `year`: for yearly averages

    time_dim : str, optional
        Name of the time coordinate for `xarray` objects. Defaults to ``None`` and
        infers the name from the data.

    Returns
    -------
    computed_dset : :class:`xarray.Dataset`, :class:`xarray.DataArray`
        The computed data with the same type as `dset`

    keep_attrs : bool, optional
        If True, attrs will be copied from the original object to the new one.
        If False, the new object will be returned without attributes.
        Defaults to None which means the attrs will only be kept in unambiguous circumstances.

    Examples
    --------
    See this `example notebook <https://geocat-comp.readthedocs.io/en/latest/examples/calendar_average.html>`__.

    Note
    ----
    Seasonal averages are weighted based on the number of days in each month.
    This means that the given data must be uniformly spaced (i.e. data every 6
    hours, every two days, every month, etc.) and must not cross month
    boundaries (i.e. don't use weekly averages where the week falls in two
    different months)


    See Also
    --------
    :func:`climatology_average`
    """
    # TODO: add functionality for users to select specific seasons or hours for averages
    freq_dict = {
        'hour': ('%m-%d %H', 'H'),
        'day': ('%m-%d', 'D'),
        'month': ('%m', 'MS'),
        'season': (None, 'QS-DEC'),
        'year': (None, 'YS')
    }

    if freq not in freq_dict:
        raise KeyError(
            f"Received bad period {freq!r}. Expected one of {list(freq_dict.keys())!r}"
        )

    # If freq is 'season' or 'year', key is set to monthly in order to
    # calculate monthly averages which are then used to calculate seasonal averages
    key = 'month' if freq in {'season', 'year'} else freq

    format, frequency = freq_dict[key]

    # If time_dim is None, infer time dimension name. Confirm dset[time_dim] contain datetimes
    time_dim = _get_time_coordinate_info(dset, time_dim)

    # Check if data is uniformly spaced
    if xr.infer_freq(dset[time_dim]) is None:
        raise ValueError(
            f"Data needs to be uniformly spaced in the {time_dim!r} dimension.")

    # Retrieve calendar name
    calendar = _infer_calendar_name(dset[time_dim])

    # Group data
    dset = dset.resample({
        time_dim: frequency
    }).mean(keep_attrs=keep_attrs).dropna(time_dim)

    # Weight the data by the number of days in each month
    if freq in ['season', 'year']:
        attrs = {}
        if keep_attrs or keep_attrs is None:
            attrs = dset.attrs

        key = freq
        format, frequency = freq_dict[key]
        # Compute the weights for the months in each season so that the
        # seasonal/yearly averages account for months being of different lengths
        month_length = dset[time_dim].dt.days_in_month.resample(
            {time_dim: frequency})
        weights = month_length.map(lambda group: group / group.sum())

        dset_weighted = dset * weights
        dset = (dset_weighted).resample({time_dim: frequency}).sum()

    # Center the time coordinate by inferring and then averaging the time bounds
    dset = _calculate_center_of_time_bounds(dset,
                                            time_dim,
                                            frequency,
                                            calendar,
                                            start=dset[time_dim].values[0],
                                            end=dset[time_dim].values[-1],
                                            keep_attrs=keep_attrs)
    if freq in ['season', 'year']:
        return dset.assign_attrs(attrs)
    else:
        return dset


def climatology_average(
        dset: typing.Union[xr.DataArray, xr.Dataset],
        freq: str,
        time_dim: str = None,
        keep_attrs: bool = None) -> typing.Union[xr.DataArray, xr.Dataset]:
    """This function calculates long term hourly, daily, monthly, or seasonal
    averages across all years in the given dataset.

    Parameters
    ----------
    dset : :class:`xarray.Dataset`, :class:`xarray.DataArray`
        The data on which to operate. It must be uniformly spaced in the time
        dimension.

    freq : str
        Frequency alias. Accepted alias:

        - `hour`: for hourly averages
        - `day`: for daily averages
        - `month`: for monthly averages
        - `season`: for meteorological seasonal averages (DJF, MAM, JJA, and SON)

    time_dim : str, optional
        Name of the time coordinate for `xarray` objects. Defaults to ``None`` and
        infers the name from the data.

    keep_attrs : bool, optional
        If True, attrs will be copied from the original object to the new one.
        If False, the new object will be returned without attributes.
        Defaults to None which means the attrs will only be kept in unambiguous circumstances.

    Returns
    -------
    computed_dset : :class:`xarray.Dataset`, :class:`xarray.DataArray`
        The computed data

    Note
    ----
    Seasonal averages are weighted based on the number of days in each month.
    This means that the given data must be uniformly spaced (i.e. data every 6
    hours, every two days, every month, etc.) and must not cross month
    boundaries (i.e. don't use weekly averages where the week falls in two
    different months)


    See Also
    --------
    Related GeoCAT Functions:
    :func:`calendar_average`

    Related NCL Functions:
    `clmDayHourTLL <https://www.ncl.ucar.edu/Document/Functions/Contributed/clmDayHourTLL.shtml>`__,
    `clmDauHourTLLL <https://www.ncl.ucar.edu/Document/Functions/Contributed/clmDayHourTLLL.shtml>`__,
    `clmDayTLL <https://www.ncl.ucar.edu/Document/Functions/Contributed/clmDayTLL.shtml>`__,
    `clmDayTLLL <https://www.ncl.ucar.edu/Document/Functions/Contributed/clmDayTLLL.shtml>`__,
    `clmMonLLLT <https://www.ncl.ucar.edu/Document/Functions/Contributed/clmMonLLLT.shtml>`__,
    `clmMonLLT <https://www.ncl.ucar.edu/Document/Functions/Contributed/clmMonLLT.shtml>`__,
    `clmMonTLL <https://www.ncl.ucar.edu/Document/Functions/Contributed/clmMonTLL.shtml>`__,
    `clmMonTLLL <https://www.ncl.ucar.edu/Document/Functions/Contributed/clmMonTLLL.shtml>`__,
    `month_to_season <https://www.ncl.ucar.edu/Document/Functions/Contributed/month_to_season.shtml>`__
    """
    # TODO: add functionality for users to select specific seasons or hours for climatologies
    freq_dict = {
        'hour': ('%m-%d %H', 'H'),
        'day': ('%m-%d', 'D'),
        'month': ('%m', 'MS'),
        'season': (None, 'QS-DEC')
    }

    if freq not in freq_dict:
        raise KeyError(
            f"Received bad period {freq!r}. Expected one of {list(freq_dict.keys())!r}"
        )

    # If freq is 'season', key is set to monthly in order to calculate monthly
    # averages which are then used to calculate seasonal averages
    key = 'month' if freq == 'season' else freq

    format, frequency = freq_dict[key]

    # If time_dim is None, infer time dimension name. Confirm dset[time_dim] contain datetimes
    time_dim = _get_time_coordinate_info(dset, time_dim)

    # Check if data is uniformly spaced
    if xr.infer_freq(dset[time_dim]) is None:
        raise ValueError(
            f"Data needs to be uniformly spaced in the {time_dim!r} dimension.")

    # Retrieve calendar name
    calendar = _infer_calendar_name(dset[time_dim])

    if freq == 'season':
        attrs = {}
        if keep_attrs or keep_attrs is None:
            attrs = dset.attrs
        if xr.infer_freq(dset[time_dim]) != 'MS':
            # Calculate monthly average before calculating seasonal climatologies
            dset = dset.resample({
                time_dim: frequency
            }).mean(keep_attrs=keep_attrs).dropna(time_dim)

        # Compute the weights for the months in each season so that the
        # seasonal averages account for months being of different lengths
        month_length = dset[time_dim].dt.days_in_month.groupby(
            f"{time_dim}.season")
        weights = month_length / month_length.sum(keep_attrs=keep_attrs)
        dset = (dset * weights).groupby(f"{time_dim}.season")
        dset = dset.sum(dim=time_dim, keep_attrs=keep_attrs)
        return dset.assign_attrs(attrs)
    else:
        # Retrieve floor of median year
        median_yr = np.median(dset[time_dim].dt.year.values)

        # Group data by format then calculate average of groups
        dset = dset.groupby(dset[time_dim].dt.strftime(format)).mean(
            keep_attrs=keep_attrs).rename({'strftime': time_dim})

        # Center the time coordinate by inferring and then averaging the time bounds
        start_time = dset[time_dim].values[0]
        end_time = dset[time_dim].values[-1]
        dset = _calculate_center_of_time_bounds(
            dset,
            time_dim,
            frequency,
            calendar,
            start=f'{median_yr:.0f}-{start_time}',
            end=f'{median_yr:.0f}-{end_time}',
            keep_attrs=keep_attrs)

        return dset
