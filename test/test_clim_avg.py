import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geocat.comp.climatology import clim_avg

def _get_dummy_data(start_date, end_date, freq, nlats, nlons):
    """
    Returns a simple xarray dataset to test with.
    Data can be hourly, daily, or monthly.
    """
    # Coordinates
    time = pd.date_range(start=start_date, end=end_date, freq=freq)
    lats = np.linspace(start=-90, stop=90, num=nlats, dtype='float32')
    lons = np.linspace(start=-180, stop=180, num=nlons, dtype='float32')

    # Create data variable
    values = np.expand_dims(np.arange(len(time)), axis=(1,2))
    data = np.tile(values, (1, nlats, nlons))
    ds = xr.Dataset(data_vars={'data': (('time', 'lat', 'lon'), data)},
                    coords={'time':time,
                            'lat':lats,
                            'lon':lons})
    return ds


# Test Data
hourly_2020 = _get_dummy_data('01-01-2020', '1-31-2020T23:00:00', 'H', 1, 1)
hourly_2021 = _get_dummy_data('01-01-2021', '1-31-2021T23:00:00', 'H', 1, 1)
hourly = xr.concat([hourly_2020, hourly_2021], dim='time')

daily = _get_dummy_data('01-01-2020', '12-01-2021', 'D', 2, 2)

monthly = _get_dummy_data('01-01-2020', '12-01-2021', 'MS', 2, 2)


day_avg = np.arange(11.5, 755.5, 24).reshape(31, 1, 1)
day_avg = np.concatenate([day_avg, day_avg])
day_avg_time = np.concatenate([pd.date_range('01-01-2020T12:00:00',
                                             '01-31-2020T12:00:00',
                                             freq='24H').date,
                               pd.date_range('01-01-2021T12:00:00',
                                             '01-31-2021T12:00:00',
                                             freq='24H').date
                               ])
hour_2_day_avg = xr.Dataset(data_vars={'data': (('time', 'lat', 'lon'), day_avg)},
                            coords={'time':day_avg_time,
                                    'lat':[-90],
                                    'lon':[-180]})
@pytest.mark.parametrize('dset, expected', [(hourly, hour_2_day_avg)])
def test_hourly_to_daily_avg(dset, expected):
    result = clim_avg(dset, freq='day', climatology=False)
    np.testing.assert_equal(result.data, expected)

'''
def test_daily_to_monthly_avg(dset, expected):
    assert False

def test_daily_to_seasonal_avg():
    assert False

def test_monthly_to_seasonal_avg():
    assert False

# Climatology Computational Tests
def test_hourly_to_daily_clim():
    assert False

def test_daily_to_monthly_clim():
    assert False

def test_daily_to_seasonal_clim():
    assert False

def test_monthly_to_seasonal_clim():
    assert False

# Argument Tests
def test_invalid_freq():
    assert False

def test_custom_time_coord():
    assert False

def test_xr_Dataset_support():
    assert False

def test_xr_DataArray_support():
    assert False
'''