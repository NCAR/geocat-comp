import sys
import unittest
import cftime
import numpy as np
import pandas as pd
from parameterized import parameterized
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp import anomaly, climatology, month_to_season, calendar_average, climatology_average
else:
    from geocat.comp import anomaly, climatology, month_to_season, calendar_average, climatology_average

##### Helper Functions #####
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
                    })
    return ds
##### End Helper Functions #####


class test_climatology(unittest.TestCase):
    dset_a = xr.tutorial.open_dataset("rasm")
    dset_b = xr.tutorial.open_dataset("air_temperature")
    dset_c = dset_a.copy().rename({"time": "Times"})
    dset_encoded = xr.tutorial.open_dataset("rasm", decode_cf=False)

    def test_climatology_invalid_freq(self):
        with self.assertRaises(ValueError):
            climatology(self.dset_a, 'hourly')

    def test_climatology_encoded_time(self):
        with self.assertRaises(ValueError):
            climatology(self.dset_encoded, 'monthly')

    @parameterized.expand([
        ('dset_a, day', dset_a, 'day'),
        ('dset_a, month', dset_a, 'month'),
        ('dset_a, season', dset_a, 'season'),
        ('dset_a, year', dset_a, 'year'),
        ('dset_b, day', dset_b, 'day'),
        ('dset_b, month', dset_b, 'month'),
        ('dset_b, season', dset_b, 'season'),
        ('dset_b, year', dset_b, 'year'),
        ('dset_c[\'Tair\'], day', dset_c['Tair'], 'day'),
        ('dset_c[\'Tair\'], month', dset_c['Tair'], 'month'),
        ('dset_c[\'Tair\'], season', dset_c['Tair'], 'season'),
        ('dset_c[\'Tair\'], year', dset_c['Tair'], 'year'),
    ])
    def test_climatology_setup(self, name, dataset, freq):
        computed_dset = climatology(dataset, freq)
        assert type(dataset) == type(computed_dset)


class test_anomaly(unittest.TestCase):
    dset_a = xr.tutorial.open_dataset("rasm")
    dset_b = xr.tutorial.open_dataset("air_temperature")
    dset_c = dset_a.copy().rename({"time": "Times"})
    dset_encoded = xr.tutorial.open_dataset("rasm", decode_cf=False)

    @parameterized.expand([
        ('dset_a, day', dset_a, 'day'),
        ('dset_a, month', dset_a, 'month'),
        ('dset_a, season', dset_a, 'season'),
        ('dset_a, year', dset_a, 'year'),
        ('dset_b, day', dset_b, 'day'),
        ('dset_b, month', dset_b, 'month'),
        ('dset_b, season', dset_b, 'season'),
        ('dset_b, year', dset_b, 'year'),
        ('dset_c[\'Tair\'], day', dset_c['Tair'], 'day'),
        ('dset_c[\'Tair\'], month', dset_c['Tair'], 'month'),
        ('dset_c[\'Tair\'], season', dset_c['Tair'], 'season'),
        ('dset_c[\'Tair\'], year', dset_c['Tair'], 'year'),
    ])
    def test_anomaly_setup(self, name, dataset, freq):
        computed_dset = anomaly(dataset, freq)
        assert type(dataset) == type(computed_dset)


class test_month_to_season(unittest.TestCase):
    ds1 = get_fake_dataset(start_month="2000-01", nmonths=12, nlats=1, nlons=1)

    # Create another dataset for the year 2001.
    ds2 = get_fake_dataset(start_month="2001-01", nmonths=12, nlats=1, nlons=1)

    # Create a dataset that combines the two previous datasets, for two
    # years of data.
    ds3 = xr.concat([ds1, ds2], dim="time")

    ds4 = xr.tutorial.open_dataset("rasm").rename({"time": "Times"})

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

    @parameterized.expand([
        ('ds1, JFM', ds1, 'JFM', 2.0),
        ('ds2, JAA', ds1, 'JJA', 7.0)
    ])
    def test_month_to_season_returns_middle_month_value(self, name, dset, season, expected):
        season_ds = month_to_season(dset, season)
        np.testing.assert_equal(season_ds["my_var"].data, expected)

    def test_month_to_season_bad_season_exception(self):
        with self.assertRaises(KeyError):
            month_to_season(self.ds1, "TEST")

    def test_month_to_season_partial_years_exception(self):
        with self.assertRaises(ValueError):
            month_to_season(self.partial_year_dataset, "JFM")

    def test_month_to_season_final_season_returns_2month_average(self):
        season_ds = month_to_season(self.ds1, 'NDJ')
        np.testing.assert_equal(season_ds["my_var"].data, 11.5)

    @parameterized.expand([
        ('DJF', 'DJF'),
        ('JFM', 'JFM'),
        ('FMA', 'FMA'),
        ('MAM', 'MAM'),
        ('AMJ', 'AMJ'),
        ('MJJ', 'MJJ'),
        ('JJA', 'JJA'),
        ('JAS', 'JAS'),
        ('ASO', 'ASO'),
        ('SON', 'SON'),
        ('OND', 'OND'),
        ('NDJ', 'NDJ')
    ])
    def test_month_to_season_returns_one_point_per_year(self, name, season):
        nyears_of_data = self.ds3.sizes["time"] / 12
        season_ds = month_to_season(self.ds3, season)
        assert season_ds["my_var"].size == nyears_of_data

    @parameterized.expand([
        ('custom_time_dataset', custom_time_dataset, "my_time", "my_var", 2.0),
        ('ds4', ds4.isel(x=110, y=200), None, "Tair", [-10.56, -8.129, -7.125]),
    ])
    def test_month_to_season_custom_time_coordinate(self, name, dataset, time_coordinate,
                                                    var_name, expected):
        season_ds = month_to_season(dataset, "JFM",
                                    time_coord_name=time_coordinate)
        np.testing.assert_almost_equal(season_ds[var_name].data,
                                       expected,
                                       decimal=1)


class test_calendar_average(unittest.TestCase):
    minute = _get_dummy_data('2020-01-01', '2021-12-31 23:30:00', '30min', 1, 1)
    hourly = _get_dummy_data('2020-01-01', '2021-12-31 23:00:00', 'H', 1, 1)
    daily = _get_dummy_data('2020-01-01', '2021-12-31', 'D', 1, 1)
    monthly = _get_dummy_data('2020-01-01', '2021-12-01', 'MS', 1, 1)

    def test_30min_to_hourly_calendar_average(self):
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

        result = calendar_average(self.minute, freq='hour')
        xr.testing.assert_equal(result, min_2_hour_avg)

    def test_hourly_to_daily_calendar_average(self):
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
        result = calendar_average(self.hourly, freq='day')
        xr.testing.assert_equal(result, hour_2_day_avg)

    def test_daily_to_monthly_calendar_average(self):
        month_avg = np.array([
            15, 45, 75, 105.5, 136, 166.5, 197, 228, 258.5, 289, 319.5, 350,
            381,
            410.5,
            440, 470.5, 501, 531.5, 562, 593, 623.5, 654, 684.5, 715
        ]).reshape(24, 1, 1)
        month_avg_time = xr.cftime_range('2020-01-01', '2022-01-01', freq='MS')
        month_avg_time = xr.DataArray(
            np.vstack((month_avg_time[:-1], month_avg_time[1:])).T,
            dims=['time', 'nbd']) \
            .mean(dim='nbd')
        day_2_month_avg = xr.Dataset(
            data_vars={'data': (('time', 'lat', 'lon'), month_avg)},
            coords={
                'time': month_avg_time,
                'lat': [-90.0],
                'lon': [-180.0]
            })


        result = calendar_average(self.daily, freq='month')
        xr.testing.assert_equal(result, day_2_month_avg)

    season_avg = np.array(
        [29.5, 105.5, 197.5, 289, 379.5, 470.5, 562.5, 654,
         715]).reshape(9, 1, 1)
    season_avg_time = xr.cftime_range('2019-12-01', '2022-03-01',
                                      freq='QS-DEC')
    season_avg_time = xr.DataArray(
        np.vstack((season_avg_time[:-1], season_avg_time[1:])).T,
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
    @parameterized.expand([('daily to seasonal', daily, day_2_season_avg),
                           ('monthly to seasonal', monthly, month_2_season_avg)])
    def test_daily_monthly_to_seasonal_calendar_average(self, name, dset, expected):
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
    @parameterized.expand([('daily to yearly', daily, day_2_year_avg),
                           ('monthly to yearly', monthly, month_2_year_avg)])
    def test_daily_monthly_to_yearly_calendar_average(self, name, dset, expected):
        result = calendar_average(dset, freq='year')
        xr.testing.assert_allclose(result, expected)