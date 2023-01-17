import sys

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp import anomaly, climatology, month_to_season, calendar_average, climatology_average
else:
    from geocat.comp import anomaly, climatology, month_to_season, calendar_average, climatology_average

dset_a = xr.tutorial.open_dataset("rasm")
dset_b = xr.tutorial.open_dataset("air_temperature")
dset_c = dset_a.copy().rename({"time": "Times"})
dset_encoded = xr.tutorial.open_dataset("rasm", decode_cf=False)


def get_fake_dataset(start_month, nmonths, nlats, nlons):
    """Returns a very simple xarray dataset for testing.

    Data values are equal to "month of year" for monthly time steps.
    """
    # Create coordinates
    months = pd.date_range(start=pd.to_datetime(start_month),
                           periods=nmonths,
                           freq="MS")
    lats = np.linspace(start=-90, stop=90, num=nlats, dtype="float32")
    lons = np.linspace(start=-180, stop=180, num=nlons, dtype="float32")

    # Create data variable. Construct a 3D array with time as the first
    # dimension.
    month_values = np.expand_dims(np.arange(start=1, stop=nmonths + 1),
                                  axis=(1, 2))
    var_values = np.tile(month_values, (1, nlats, nlons))

    ds = xr.Dataset(
        data_vars={
            "my_var": (("time", "lat", "lon"), var_values.astype("float32")),
        },
        coords={
            "time": months,
            "lat": lats,
            "lon": lons
        },
        attrs={'Description': 'This is dummy data for testing.'}
    )
    return ds


def _get_dummy_data(start_date,
                    end_date,
                    freq,
                    nlats,
                    nlons,
                    calendar='standard'):
    """Returns a simple xarray dataset to test with.

    Data can be hourly, daily, or monthly.
    """
    # Coordinates
    time = xr.cftime_range(start=start_date,
                           end=end_date,
                           freq=freq,
                           calendar=calendar)
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
                    },
                    attrs={'Description': 'This is dummy data for testing.'})
    return ds

@pytest.mark.parametrize("dataset", [dset_a, dset_b, dset_c["Tair"]])
@pytest.mark.parametrize("freq", ["day", "month", "year", "season"])
@pytest.mark.parametrize("keep_attrs", [None, True, False])
def test_climatology_keep_attrs(dataset, freq, keep_attrs):
    computed_dset = climatology(dataset, freq, keep_attrs=keep_attrs)
    if keep_attrs or keep_attrs==None:
        assert computed_dset.attrs == dataset.attrs
    elif not keep_attrs:
        assert computed_dset.attrs == {}


def test_climatology_invalid_freq():
    with pytest.raises(ValueError):
        climatology(dset_a, "hourly")


def test_climatology_encoded_time():
    with pytest.raises(ValueError):
        climatology(dset_encoded, "monthly")


@pytest.mark.parametrize("dataset", [dset_a, dset_b, dset_c["Tair"]])
@pytest.mark.parametrize("freq", ["day", "month", "year", "season"])
def test_climatology_setup(dataset, freq):
    computed_dset = climatology(dataset, freq)
    assert type(dataset) == type(computed_dset)


@pytest.mark.parametrize("dataset", [dset_a, dset_b, dset_c["Tair"]])
@pytest.mark.parametrize("freq", ["day", "month", "year", "season"])
def test_anomaly_setup(dataset, freq):
    computed_dset = anomaly(dataset, freq)
    assert type(dataset) == type(computed_dset)


ds1 = get_fake_dataset(start_month="2000-01", nmonths=12, nlats=1, nlons=1)

# Create another dataset for the year 2001.
ds2 = get_fake_dataset(start_month="2001-01", nmonths=12, nlats=1, nlons=1)

# Create a dataset that combines the two previous datasets, for two
# years of data.
ds3 = xr.concat([ds1, ds2], dim="time")

# Create a dataset with the wrong number of months.
partial_year_dataset = get_fake_dataset(start_month="2000-01",
                                        nmonths=13,
                                        nlats=1,
                                        nlons=1)

# Create a dataset with a custom time coordinate.
custom_time_dataset = get_fake_dataset(start_month="2000-01",
                                       nmonths=12,
                                       nlats=1,
                                       nlons=1)
custom_time_dataset = custom_time_dataset.rename({"time": "my_time"})

# Create a more complex dataset just to verify that get_fake_dataset()
# is generally working.
complex_dataset = get_fake_dataset(start_month="2001-01",
                                   nmonths=12,
                                   nlats=10,
                                   nlons=10)

@pytest.mark.parametrize("keep_attrs", [None, True, False])
def test_month_to_season_keep_attrs(keep_attrs):
    season_ds = month_to_season(ds1, 'JFM', keep_attrs=keep_attrs)
    if keep_attrs or keep_attrs == None:
        assert season_ds.attrs == ds1.attrs
    elif not keep_attrs:
        assert season_ds.attrs == {}

@pytest.mark.parametrize("dataset, season, expected", [(ds1, "JFM", 2.0),
                                                       (ds1, "JJA", 7.0)])
def test_month_to_season_returns_middle_month_value(dataset, season, expected):
    season_ds = month_to_season(dataset, season)
    np.testing.assert_equal(season_ds["my_var"].data, expected)


def test_month_to_season_bad_season_exception():
    with pytest.raises(KeyError):
        month_to_season(ds1, "TEST")


def test_month_to_season_partial_years_exception():
    with pytest.raises(ValueError):
        month_to_season(partial_year_dataset, "JFM")


@pytest.mark.parametrize("dataset, season, expected", [(ds1, "NDJ", 11.5)])
def test_month_to_season_final_season_returns_2month_average(
        dataset, season, expected):
    season_ds = month_to_season(dataset, season)
    np.testing.assert_equal(season_ds["my_var"].data, expected)


@pytest.mark.parametrize(
    "season",
    [
        "DJF",
        "JFM",
        "FMA",
        "MAM",
        "AMJ",
        "MJJ",
        "JJA",
        "JAS",
        "ASO",
        "SON",
        "OND",
        "NDJ",
    ],
)
def test_month_to_season_returns_one_point_per_year(season):
    nyears_of_data = ds3.sizes["time"] / 12
    season_ds = month_to_season(ds3, season)
    assert season_ds["my_var"].size == nyears_of_data


@pytest.mark.parametrize(
    "dataset, time_coordinate, var_name, expected",
    [
        (custom_time_dataset, "my_time", "my_var", 2.0),
        (dset_c.isel(x=110, y=200), None, "Tair", [-10.56, -8.129, -7.125]),
    ],
)
def test_month_to_season_custom_time_coordinate(dataset, time_coordinate,
                                                var_name, expected):
    season_ds = month_to_season(dataset, "JFM", time_coord_name=time_coordinate)
    np.testing.assert_almost_equal(season_ds[var_name].data,
                                   expected,
                                   decimal=1)


# Test Datasets For calendar_average() and climatology_average()
minute = _get_dummy_data('2020-01-01', '2021-12-31 23:30:00', '30min', 1, 1)

hourly = _get_dummy_data('2020-01-01', '2021-12-31 23:00:00', 'H', 1, 1)

daily = _get_dummy_data('2020-01-01', '2021-12-31', 'D', 1, 1)

monthly = _get_dummy_data('2020-01-01', '2021-12-01', 'MS', 1, 1)

# Computational Tests for calendar_average()
hour_avg = np.arange(0.5, 35088.5, 2).reshape((365 + 366) * 24, 1, 1)
hour_avg_time = xr.cftime_range('2020-01-01 00:30:00',
                                '2021-12-31 23:30:00',
                                freq='H')
min_2_hour_avg = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), hour_avg)},
    coords={
        'time': hour_avg_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(minute, min_2_hour_avg)])
def test_30min_to_hourly_calendar_average(dset, expected):
    result = calendar_average(dset, freq='hour')
    xr.testing.assert_equal(result, expected)


day_avg = np.arange(11.5, 17555.5, 24).reshape(366 + 365, 1, 1)
day_avg_time = xr.cftime_range('2020-01-01 12:00:00',
                               '2021-12-31 12:00:00',
                               freq='D')
hour_2_day_avg = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), day_avg)},
    coords={
        'time': day_avg_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(hourly, hour_2_day_avg)])
def test_hourly_to_daily_calendar_average(dset, expected):
    result = calendar_average(dset, freq='day')
    xr.testing.assert_equal(result, expected)


month_avg = np.array([
    15, 45, 75, 105.5, 136, 166.5, 197, 228, 258.5, 289, 319.5, 350, 381, 410.5,
    440, 470.5, 501, 531.5, 562, 593, 623.5, 654, 684.5, 715
]).reshape(24, 1, 1)
month_avg_time = xr.cftime_range('2020-01-01', '2022-01-01', freq='MS')
month_avg_time = xr.DataArray(np.vstack((month_avg_time[:-1], month_avg_time[1:])).T,
                              dims=['time', 'nbd']) \
                    .mean(dim='nbd')
day_2_month_avg = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), month_avg)},
    coords={
        'time': month_avg_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(daily, day_2_month_avg)])
def test_daily_to_monthly_calendar_average(dset, expected):
    result = calendar_average(dset, freq='month')
    xr.testing.assert_equal(result, expected)


season_avg = np.array([29.5, 105.5, 197.5, 289, 379.5, 470.5, 562.5, 654,
                       715]).reshape(9, 1, 1)
season_avg_time = xr.cftime_range('2019-12-01', '2022-03-01', freq='QS-DEC')
season_avg_time = xr.DataArray(np.vstack((season_avg_time[:-1], season_avg_time[1:])).T,
                               dims=['time', 'nbd']) \
                    .mean(dim='nbd')
day_2_season_avg = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), season_avg)},
    coords={
        'time': season_avg_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })

season_avg = np.array(
    [0.483333333, 3, 6.010869565, 9, 11.96666667, 15, 18.01086957, 21,
     23]).reshape(9, 1, 1)
month_2_season_avg = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), season_avg)},
    coords={
        'time': season_avg_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(daily, day_2_season_avg),
                                            (monthly, month_2_season_avg)])
def test_daily_monthly_to_seasonal_calendar_average(dset, expected):
    result = calendar_average(dset, freq='season')
    xr.testing.assert_allclose(result, expected)


year_avg_time = [
    cftime.datetime(2020, 7, 2),
    cftime.datetime(2021, 7, 2, hour=12)
]
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
def test_daily_monthly_to_yearly_calendar_average(dset, expected):
    result = calendar_average(dset, freq='year')
    xr.testing.assert_allclose(result, expected)


# Computational Tests for climatology_average()
hour_clim = np.concatenate([np.arange(8784.5, 11616.5, 2),
                            np.arange(2832.5, 2880.5, 2),
                            np.arange(11640.5, 26328.5, 2)])\
              .reshape(8784, 1, 1)
hour_clim_time = xr.cftime_range('2020-01-01 00:30:00',
                                 '2020-12-31 23:30:00',
                                 freq='H')
min_2_hourly_clim = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), hour_clim)},
    coords={
        'time': hour_clim_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(minute, min_2_hourly_clim)])
def test_30min_to_hourly_climatology_average(dset, expected):
    result = climatology_average(dset, freq='hour')
    xr.testing.assert_allclose(result, expected)


day_clim = np.concatenate([np.arange(4403.5, 5819.5, 24),
                           [1427.5],
                           np.arange(5831.5, 13175.5, 24)]) \
             .reshape(366, 1, 1)
day_clim_time = xr.cftime_range('2020-01-01 12:00:00',
                                '2020-12-31 12:00:00',
                                freq='24H')

hour_2_day_clim = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), day_clim)},
    coords={
        'time': day_clim_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(hourly, hour_2_day_clim)])
def test_hourly_to_daily_climatology_average(dset, expected):
    result = climatology_average(dset, freq='day')
    xr.testing.assert_equal(result, expected)


month_clim = np.array([
    198, 224.5438596, 257.5, 288, 318.5, 349, 379.5, 410.5, 441, 471.5, 502,
    532.5
]).reshape(12, 1, 1)
month_clim_time = xr.cftime_range('2020-01-01', '2021-01-01', freq='MS')
month_clim_time = xr.DataArray(np.vstack(
    (month_clim_time[:-1], month_clim_time[1:])).T,
                               dims=['time', 'nbd']).mean(dim='nbd')
day_2_month_clim = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), month_clim)},
    coords={
        'time': month_clim_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(daily, day_2_month_clim)])
def test_daily_to_monthly_climatology_average(dset, expected):
    result = climatology_average(dset, freq='month')
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

season_clim = np.array([10.04972376, 12.01086957, 9, 15]).reshape(4, 1, 1)
month_2_season_clim = xr.Dataset(
    data_vars={'data': (('season', 'lat', 'lon'), season_clim)},
    coords={
        'season': season_clim_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected', [(daily, day_2_season_clim),
                                            (monthly, month_2_season_clim)])
def test_daily_monthly_to_seasonal_climatology_average(dset, expected):
    result = climatology_average(dset, freq='season')
    xr.testing.assert_allclose(result, expected)


# Argument Tests for climatology_average() and calendar_average()
@pytest.mark.parametrize('freq', ['TEST', None])
def test_invalid_freq_climatology_average(freq):
    with pytest.raises(KeyError):
        climatology_average(monthly, freq=freq)


@pytest.mark.parametrize('freq', ['TEST', None])
def test_invalid_freq_calendar_average(freq):
    with pytest.raises(KeyError):
        calendar_average(monthly, freq=freq)


time_dim = 'my_time'
custom_time = daily.rename({'time': time_dim})
custom_time_expected = day_2_month_clim.rename({'time': time_dim})


@pytest.mark.parametrize('dset, expected, time_dim',
                         [(custom_time, custom_time_expected, time_dim)])
def test_custom_time_coord_climatology_average(dset, expected, time_dim):
    result = climatology_average(dset, freq='month', time_dim=time_dim)
    xr.testing.assert_allclose(result, expected)


custom_time_expected = day_2_month_avg.rename({'time': time_dim})


@pytest.mark.parametrize('dset, expected, time_dim',
                         [(custom_time, custom_time_expected, time_dim)])
def test_custom_time_coord_calendar_average(dset, expected, time_dim):
    result = calendar_average(dset, freq='month', time_dim=time_dim)
    xr.testing.assert_allclose(result, expected)


array = daily['data']
array_expected = day_2_month_clim['data']


@pytest.mark.parametrize('da, expected', [(array, array_expected)])
def test_xr_DataArray_support_climatology_average(da, expected):
    result = climatology_average(da, freq='month')
    xr.testing.assert_allclose(result, expected)


array_expected = day_2_month_avg['data']


@pytest.mark.parametrize('da, expected', [(array, array_expected)])
def test_xr_DataArray_support_calendar_average(da, expected):
    result = calendar_average(da, freq='month')
    xr.testing.assert_equal(result, expected)


dset_encoded = xr.tutorial.open_dataset("air_temperature", decode_cf=False)


def test_non_datetime_like_objects_climatology_average():
    with pytest.raises(ValueError):
        climatology_average(dset_encoded, 'month')


def test_non_datetime_like_objects_calendar_average():
    with pytest.raises(ValueError):
        calendar_average(dset_encoded, 'month')


time = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-04'])
non_uniform = xr.Dataset(data_vars={'data': (('time'), np.arange(3))},
                         coords={'time': time})


def test_non_uniformly_spaced_data_climatology_average():
    with pytest.raises(ValueError):
        climatology_average(non_uniform, freq='day')


def test_non_uniformly_spaced_data_calendar_average():
    with pytest.raises(ValueError):
        calendar_average(non_uniform, freq='day')


julian_daily = _get_dummy_data('2020-01-01',
                               '2021-12-31',
                               'D',
                               1,
                               1,
                               calendar='julian')
noleap_daily = _get_dummy_data('2020-01-01',
                               '2021-12-31',
                               'D',
                               1,
                               1,
                               calendar='noleap')
all_leap_daily = _get_dummy_data('2020-01-01',
                                 '2021-12-31',
                                 'D',
                                 1,
                                 1,
                                 calendar='all_leap')
day_360_daily = _get_dummy_data('2020-01-01',
                                '2021-12-30',
                                'D',
                                1,
                                1,
                                calendar='360_day')

# Daily -> Monthly Climatologies for Julian Calendar
julian_month_clim = np.array([198, 224.54385965, 257.5, 288, 318.5, 349,
                              379.5, 410.5, 441, 471.5, 502, 532.5])\
                      .reshape(12, 1, 1)
julian_month_clim_time = xr.cftime_range('2020-01-01',
                                         '2021-01-01',
                                         freq='MS',
                                         calendar='julian')
julian_month_clim_time = xr.DataArray(np.vstack((julian_month_clim_time[:-1], julian_month_clim_time[1:])).T,
                                     dims=['time', 'nbd']) \
    .mean(dim='nbd')
julian_day_2_month_clim = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), julian_month_clim)},
    coords={
        'time': julian_month_clim_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })
# Daily -> Monthly Climatologies for NoLeap Calendar
noleap_month_clim = np.array([197.5, 227, 256.5, 287, 317.5, 348,
                              378.5, 409.5, 440, 470.5, 501, 531.5])\
                      .reshape(12, 1, 1)
noleap_month_clim_time = xr.cftime_range('2020-01-01',
                                         '2021-01-01',
                                         freq='MS',
                                         calendar='noleap')
noleap_month_clim_time = xr.DataArray(np.vstack((noleap_month_clim_time[:-1], noleap_month_clim_time[1:])).T,
                                     dims=['time', 'nbd']) \
    .mean(dim='nbd')
noleap_day_2_month_clim = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), noleap_month_clim)},
    coords={
        'time': noleap_month_clim_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })
# Daily -> Monthly Climatologies for AllLeap Calendar
all_leap_month_clim = np.array([198, 228, 258, 288.5, 319, 349.5,
                                380, 411, 441.5, 472, 502.5, 533])\
                        .reshape(12, 1, 1)
all_leap_month_clim_time = xr.cftime_range('2020-01-01',
                                           '2021-01-01',
                                           freq='MS',
                                           calendar='all_leap')
all_leap_month_clim_time = xr.DataArray(np.vstack((all_leap_month_clim_time[:-1], all_leap_month_clim_time[1:])).T,
                                       dims=['time', 'nbd']) \
    .mean(dim='nbd')
all_leap_day_2_month_clim = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), all_leap_month_clim)},
    coords={
        'time': all_leap_month_clim_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })
# Daily -> Monthly Climatologies for 360 Day Calendar
day_360_leap_month_clim = np.arange(194.5, 554.5, 30).reshape(12, 1, 1)
day_360_leap_month_clim_time = xr.cftime_range('2020-01-01',
                                               '2021-01-01',
                                               freq='MS',
                                               calendar='360_day')
day_360_leap_month_clim_time = xr.DataArray(np.vstack((day_360_leap_month_clim_time[:-1], day_360_leap_month_clim_time[1:])).T,
                                           dims=['time', 'nbd']) \
    .mean(dim='nbd')
day_360_leap_day_2_month_clim = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), day_360_leap_month_clim)},
    coords={
        'time': day_360_leap_month_clim_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected',
                         [(julian_daily, julian_day_2_month_clim),
                          (noleap_daily, noleap_day_2_month_clim),
                          (all_leap_daily, all_leap_day_2_month_clim),
                          (day_360_daily, day_360_leap_day_2_month_clim)])
def test_non_standard_calendars_climatology_average(dset, expected):
    result = climatology_average(dset, freq='month')
    xr.testing.assert_allclose(result, expected)


# Daily -> Monthly Means for Julian Calendar
julian_month_avg = np.array([
    15, 45, 75, 105.5, 136, 166.5, 197, 228, 258.5, 289, 319.5, 350, 381, 410.5,
    440, 470.5, 501, 531.5, 562, 593, 623.5, 654, 684.5, 715
]).reshape(24, 1, 1)
julian_month_avg_time = xr.cftime_range('2020-01-01',
                                        '2022-01-01',
                                        freq='MS',
                                        calendar='julian')
julian_month_avg_time = xr.DataArray(np.vstack((julian_month_avg_time[:-1], julian_month_avg_time[1:])).T,
                                     dims=['time', 'nbd']) \
                          .mean(dim='nbd')
julian_day_2_month_avg = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), julian_month_avg)},
    coords={
        'time': julian_month_avg_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })
# Daily -> Monthly Means for NoLeap Calendar
noleap_month_avg = np.array([
    15, 44.5, 74, 104.5, 135, 165.5, 196, 227, 257.5, 288, 318.5, 349, 380,
    409.5, 439, 469.5, 500, 530.5, 561, 592, 622.5, 653, 683.5, 714
]).reshape(24, 1, 1)
noleap_month_avg_time = xr.cftime_range('2020-01-01',
                                        '2022-01-01',
                                        freq='MS',
                                        calendar='noleap')
noleap_month_avg_time = xr.DataArray(np.vstack((noleap_month_avg_time[:-1], noleap_month_avg_time[1:])).T,
                                     dims=['time', 'nbd']) \
                          .mean(dim='nbd')
noleap_day_2_month_avg = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), noleap_month_avg)},
    coords={
        'time': noleap_month_avg_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })
# Daily -> Monthly Means for AllLeap Calendar
all_leap_month_avg = np.array([
    15, 45, 75, 105.5, 136, 166.5, 197, 228, 258.5, 289, 319.5, 350, 381, 411,
    441, 471.5, 502, 532.5, 563, 594, 624.5, 655, 685.5, 716
]).reshape(24, 1, 1)
all_leap_month_avg_time = xr.cftime_range('2020-01-01',
                                          '2022-01-01',
                                          freq='MS',
                                          calendar='all_leap')
all_leap_month_avg_time = xr.DataArray(np.vstack((all_leap_month_avg_time[:-1], all_leap_month_avg_time[1:])).T,
                                     dims=['time', 'nbd']) \
    .mean(dim='nbd')
all_leap_day_2_month_avg = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), all_leap_month_avg)},
    coords={
        'time': all_leap_month_avg_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })
# Daily -> Monthly Means for 360 Day Calendar
day_360_leap_month_avg = np.arange(14.5, 734.5, 30).reshape(24, 1, 1)
day_360_leap_month_avg_time = xr.cftime_range('2020-01-01',
                                              '2022-01-01',
                                              freq='MS',
                                              calendar='360_day')
day_360_leap_month_avg_time = xr.DataArray(np.vstack((day_360_leap_month_avg_time[:-1], day_360_leap_month_avg_time[1:])).T,
                                       dims=['time', 'nbd']) \
    .mean(dim='nbd')
day_360_leap_day_2_month_avg = xr.Dataset(
    data_vars={'data': (('time', 'lat', 'lon'), day_360_leap_month_avg)},
    coords={
        'time': day_360_leap_month_avg_time,
        'lat': [-90.0],
        'lon': [-180.0]
    })


@pytest.mark.parametrize('dset, expected',
                         [(julian_daily, julian_day_2_month_avg),
                          (noleap_daily, noleap_day_2_month_avg),
                          (all_leap_daily, all_leap_day_2_month_avg),
                          (day_360_daily, day_360_leap_day_2_month_avg)])
def test_non_standard_calendars_calendar_average(dset, expected):
    result = calendar_average(dset, freq='month')
    xr.testing.assert_equal(result, expected)
