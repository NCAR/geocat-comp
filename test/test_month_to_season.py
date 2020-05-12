import unittest

import numpy as np
import pandas as pd
import xarray as xr
from geocat.comp.month_to_season import month_to_season


def get_fake_dataset(start_month, nmonths, nlats, nlons):
    """ Returns a very simple xarray dataset for testing.
        Data values are equal to "month of year" for monthly time steps.
    """
    # Create coordinates
    months = pd.date_range(start=pd.to_datetime(start_month), periods=nmonths, freq='MS')
    lats = np.linspace(start=-90, stop=90, num=nlats, dtype='float32')
    lons = np.linspace(start=-180, stop=180, num=nlons, dtype='float32')

    # Create data variable. Construct a 3D array with time as the first dimension.
    month_values = np.expand_dims(np.arange(start=1, stop=nmonths + 1), axis=(1, 2))
    var_values = np.tile(month_values, (1, nlats, nlons))

    ds = xr.Dataset(
        data_vars={
            'my_var': (('time', 'lat', 'lon'), var_values.astype('float32')),
        },
        coords={'time': months, 'lat': lats, 'lon': lons},
    )
    return ds


class Test_month_to_season(unittest.TestCase):

    def setUp(self):
        # Create a dataset for the year 2000.
        self.ds1 = get_fake_dataset(start_month='2000-01', nmonths=12, nlats=1, nlons=1)

        # Create another dataset for the year 2001.
        self.ds2 = get_fake_dataset(start_month='2001-01', nmonths=12, nlats=1, nlons=1)

        # Create a dataset that combines the two previous datasets, for two years of data.
        self.ds3 = xr.concat([self.ds1, self.ds2], dim='time')

        # Create a dataset with the wrong number of months.
        self.partial_year_dataset = get_fake_dataset(start_month='2000-01', nmonths=13, nlats=1, nlons=1)

        # Create a dataset with a custom time coordinate.
        custom_time_dataset = get_fake_dataset(start_month='2000-01', nmonths=12, nlats=1, nlons=1)
        self.custom_time_dataset = custom_time_dataset.rename({'time': 'my_time'})

        # Create a more complex dataset just to verify that get_fake_dataset() is generally working.
        self.complex_dataset = get_fake_dataset(start_month='2001-01', nmonths=12, nlats=10, nlons=10)

        # Check all possible season choices for some tests.
        self.all_seasons = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']

    def test_m2s_returns_middle_month_value(self):
        season_ds = month_to_season(self.ds1, 'JFM')
        season_value_array = season_ds['my_var'].data

        # Should equal the average of [1.0, 2.0, 3.0]
        self.assertEqual(season_value_array[0, 0, 0], 2.0)

        season_ds = month_to_season(self.ds1, 'JJA')
        season_value_array = season_ds['my_var'].data

        # Should equal the average of [6.0, 7.0, 8.0]
        self.assertEqual(season_value_array[0, 0, 0], 7.0)

    def test_bad_season_returns_exception(self):
        with self.assertRaises(ValueError):
            season_ds = month_to_season(self.ds1, 'XXX')

    def test_partial_years_returns_exception(self):
        with self.assertRaises(ValueError):
            season_ds = month_to_season(self.partial_year_dataset, 'JFM')

    def test_final_season_returns_2month_average(self):
        season_ds = month_to_season(self.ds1, 'NDJ')
        season_value_array = season_ds['my_var'].data
        self.assertEqual(season_value_array[0, 0, 0], 11.5)

    def test_each_season_returns_one_point_per_year(self):
        nyears_of_data = self.ds3.sizes['time'] / 12
        for season in self.all_seasons:
            season_ds = month_to_season(self.ds3, season)
            season_value_array = season_ds['my_var'].data
            self.assertEqual(season_value_array.size, nyears_of_data)

    def test_custom_time_coordinate(self):
        season_ds = month_to_season(self.custom_time_dataset, 'JFM', time_coord_name='my_time')
        season_value_array = season_ds['my_var'].data

        # Should equal the average of [1.0, 2.0, 3.0]
        self.assertEqual(season_value_array[0, 0, 0], 2.0)


if __name__ == '__main__':
    unittest.main()
