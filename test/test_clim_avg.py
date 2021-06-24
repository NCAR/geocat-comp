import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geocat.comp.climatology import clim_avg


def _get_dummy_data(start_date, end_date, freq, nlats, nlons):
    """Returns a simple xarray dataset to test with.

    Data can be hourly, daily, or monthly.
    """
    # Coordinates
    time = pd.date_range(start=start_date, end=end_date, freq=freq)
    lats = np.linspace(start=-90, stop=90, num=nlats, dtype='float32')
    lons = np.linspace(start=-180, stop=180, num=nlons, dtype='float32')

    # Create data variable
    values = np.expand_dims(np.arange(len(time)), axis=(1, 2))
    data = np.tile(values, (1, nlats, nlons))
    ds = xr.Dataset(data_vars={'data': (('time', 'lat', 'lon'), data)},
                    coords={
                        'time': time,
                        'lat': lats,
                        'lon': lons
                    })
    return ds


# Test Datasets
hourly_data = np.arange(24 * 62).reshape(1488, 1, 1)
hourly_2020 = _get_dummy_data('01-01-2020', '1-31-2020T23:00:00', 'H', 1, 1)
hourly_2021 = _get_dummy_data('01-01-2021', '1-31-2021T23:00:00', 'H', 1, 1)
hourly = xr.concat([hourly_2020, hourly_2021], dim='time')\
           .update({'data': (('time', 'lat', 'lon'), hourly_data)})

daily = _get_dummy_data('01-01-2020', '12-31-2021', 'D', 1, 1)

monthly = _get_dummy_data('01-01-2020', '12-01-2021', 'MS', 1, 1)

# Tests w/ expected outputs
year_avg_time = pd.to_datetime(['07-01-2020', '07-01-2021'])
day_2_year_avg = [[[182.5]], [[548]]]
day_2_year_avg = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), day_2_year_avg)},
    coords={
        'time': year_avg_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })
month_2_year_avg = [[[5.513661202]], [[17.5260274]]]
month_2_year_avg = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), month_2_year_avg)},
    coords={
        'time': year_avg_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(daily, day_2_year_avg),
                                            (monthly, month_2_year_avg)])
def test_yearly_avg(dset, expected):
    result = clim_avg(dset, freq='year')
    xr.testing.assert_allclose(result, expected)


day_avg = np.arange(11.5, 1499.5, 24).reshape(62, 1, 1)
day_avg_time = np.concatenate([
    pd.date_range('01-01-2020T12:00:00', '01-31-2020T12:00:00', freq='24H'),
    pd.date_range('01-01-2021T12:00:00', '01-31-2021T12:00:00', freq='24H')
])
hour_2_day_avg = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), day_avg)},
    coords={
        'time': day_avg_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(hourly, hour_2_day_avg)])
def test_hourly_to_daily_avg(dset, expected):
    result = clim_avg(dset, freq='day', climatology=False)
    xr.testing.assert_equal(result, expected)


month_avg = np.array([
    15, 45, 75, 105.5, 136, 166.5, 197, 228, 258.5, 289, 319.5, 350, 381, 410.5,
    440, 470.5, 501, 531.5, 562, 593, 623.5, 654, 684.5, 715
]).reshape(24, 1, 1)
month_avg_time = pd.date_range('01-01-2020', '12-31-2021',
                               freq='MS') + pd.offsets.SemiMonthBegin()
day_2_month_avg = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), month_avg)},
    coords={
        'time': month_avg_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(daily, day_2_month_avg)])
def test_daily_to_monthly_avg(dset, expected):
    result = clim_avg(dset, freq='month', climatology=False)
    xr.testing.assert_equal(result, expected)


season_avg = np.array([29.5, 105.5, 197.5, 289, 379.5, 470.5, 562.5, 654,
                       715]).reshape(9, 1, 1)
season_avg_time = pd.date_range('01-01-2020', '01-01-2022', freq='3MS')
day_2_season_avg = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), season_avg)},
    coords={
        'time': season_avg_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(daily, day_2_season_avg)])
def test_daily_to_seasonal_avg(dset, expected):
    result = clim_avg(dset, freq='season', climatology=False)
    xr.testing.assert_equal(result, expected)


season_avg = np.array(
    [0.483333333, 3, 6.010869565, 9, 11.96666667, 15, 18.01086957, 21,
     23]).reshape(9, 1, 1)
season_avg_time = pd.date_range('01-01-2020', '01-01-2022', freq='3MS')
month_2_season_avg = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), season_avg)},
    coords={
        'time': season_avg_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(monthly, month_2_season_avg)])
def test_monthly_to_seasonal_avg(dset, expected):
    result = clim_avg(dset, freq='season', climatology=False)
    xr.testing.assert_allclose(result, expected)


# Climatology Computational Tests
day_clim = np.arange(383.5, 1127.5, 24).reshape(31, 1, 1)
day_clim_time = np.concatenate(
    [pd.date_range('01-01-2020T12:00:00', '01-31-2020T12:00:00', freq='24H')])

hour_2_day_clim = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), day_clim)},
    coords={
        'time': day_clim_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(hourly, hour_2_day_clim)])
def test_hourly_to_daily_clim(dset, expected):
    result = clim_avg(dset, freq='day', climatology=True)
    xr.testing.assert_equal(result, expected)


month_clim = np.array([
    198, 224.5438596, 257.5, 288, 318.5, 349, 379.5, 410.5, 441, 471.5, 502,
    532.5
]).reshape(12, 1, 1)
month_clim_time = pd.date_range('01-01-2020', '12-31-2020',
                                freq='MS') + pd.offsets.SemiMonthBegin()
day_2_month_clim = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), month_clim)},
    coords={
        'time': month_clim_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(daily, day_2_month_clim)])
def test_daily_to_monthly_clim(dset, expected):
    result = clim_avg(dset, freq='month', climatology=True)
    xr.testing.assert_allclose(result, expected)


season_clim = np.array([320.9392265, 380, 288, 471.5]).reshape(4, 1, 1)
season_clim_time = ['DJF', 'JJA', 'MAM', 'SON']
day_2_season_clim = xr.Dataset(
    data_vars={'data': (('season', 'lat', 'lon'), season_clim)},
    coords={
        'season': season_clim_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(daily, day_2_season_clim)])
def test_daily_to_seasonal_clim(dset, expected):
    result = clim_avg(dset, freq='season', climatology=True)
    xr.testing.assert_allclose(result, expected)


season_clim = np.array([10.04972376, 12.01086957, 9, 15]).reshape(4, 1, 1)
season_clim_time = ['DJF', 'JJA', 'MAM', 'SON']
month_2_season_clim = xr.Dataset(
    data_vars={'data': (('season', 'lat', 'lon'), season_clim)},
    coords={
        'season': season_clim_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(monthly, month_2_season_clim)])
def test_monthly_to_seasonal_clim(dset, expected):
    result = clim_avg(dset, freq='season', climatology=True)
    xr.testing.assert_allclose(result, expected)


# Argument Tests
@pytest.mark.parametrize('freq', ['TEST', None])
def test_invalid_freq(freq):
    with pytest.raises(KeyError):
        clim_avg(monthly, freq=freq)


time_dim = 'my_time'
custom_time = daily.rename({'time': time_dim})
custom_time_expected = day_2_month_avg.rename({'time': time_dim})


@pytest.mark.parametrize('dset, expected, time_dim',
                         [(custom_time, custom_time_expected, time_dim)])
def test_custom_time_coord(dset, expected, time_dim):
    result = clim_avg(dset, freq='month', time_dim=time_dim, climatology=False)
    xr.testing.assert_allclose(result, expected)


array = daily['data']
array_expected = day_2_month_avg['data']


@pytest.mark.parametrize('da, expected', [(array, array_expected)])
def test_xr_DataArray_support(da, expected):
    result = clim_avg(da, freq='month', climatology=False)
    xr.testing.assert_equal(result, expected)
