

def month_to_season(xMon, season, time_coord_name='time'):
    """ This function takes an xarray dataset containing monthly data spanning years and
        returns a dataset with one sample per year, for a specified three-month season.

        This function requires the number of months to be a multiple of 12, i.e. full years must be provided.

        Time stamps are centered on the season. For example, seasons='DJF' returns January timestamps.

        If a calculated season's timestamp falls outside the original range of monthly values, then the calculated mean
        is dropped.  For example, if the monthly data's time range is [Jan-2000, Dec-2003] and the season is "DJF", the
        seasonal mean computed from the single month of Dec-2003 is dropped.
    """
    mod_check(xMon[time_coord_name].size, 12)

    startDate = xMon[time_coord_name][0]
    endDate = xMon[time_coord_name][-1]
    seasons_pd = {'DJF': ('QS-DEC', 1), 'JFM': ('QS-JAN', 2), 'FMA': ('QS-FEB', 3), 'MAM': ('QS-MAR', 4),
                  'AMJ': ('QS-APR', 5), 'MJJ': ('QS-MAY', 6), 'JJA': ('QS-JUN', 7), 'JAS': ('QS-JUL', 8),
                  'ASO': ('QS-AUG', 9), 'SON': ('QS-SEP', 10), 'OND': ('QS-OCT', 11), 'NDJ': ('QS-NOV', 12)}
    try:
        (season_pd, season_sel) = seasons_pd[season]
    except KeyError:
        raise KeyError(f"contributed: month_to_season: bad season: SEASON = {season}. Valid seasons include: {list(seasons_pd.keys())}")

    # Compute the three-month means, moving time labels ahead to the middle month.
    month_offset = 'MS'
    xSeasons = xMon.resample({time_coord_name: season_pd}, loffset=month_offset).mean()

    # Filter just the desired season, and trim to the desired time range.
    xSea = xSeasons.sel({time_coord_name: xSeasons[time_coord_name].dt.month == season_sel})
    xSea = xSea.sel({time_coord_name: slice(startDate, endDate)})
    return xSea


def mod_check(value, mod):
    if value % mod != 0:
        raise ValueError(f'Expected a multiple of {mod} values')
