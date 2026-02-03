import numpy as np
import xarray as xr
import pandas as pd

from geocat.comp import (
    climate_anomaly,
    month_to_season,
    calendar_average,
    climatology_average,
)


def get_fake_dataset(start_month, nmonths, nlats, nlons):
    """Returns a very simple xarray dataset for testing. Data values are equal to "month of year" for monthly time steps."""
    # Create coordinates
    months = pd.date_range(
        start=pd.to_datetime(start_month), periods=nmonths, freq="MS"
    )
    lats = np.linspace(start=-90, stop=90, num=nlats, dtype="float32")
    lons = np.linspace(start=-180, stop=180, num=nlons, dtype="float32")

    # Create data variable. Construct a 3D array with time as the first
    # dimension.
    month_values = np.expand_dims(np.arange(start=1, stop=nmonths + 1), axis=(1, 2))
    var_values = np.tile(month_values, (1, nlats, nlons))

    ds = xr.Dataset(
        data_vars={
            "my_var": (("time", "lat", "lon"), var_values.astype("float32")),
        },
        coords={"time": months, "lat": lats, "lon": lons},
        attrs={'Description': 'This is dummy data for testing.'},
    )
    return ds


def _get_dummy_data(start_date, end_date, freq, nlats, nlons, calendar='standard'):
    """Returns a simple xarray dataset to test with. Data can be hourly, daily, or monthly."""
    # Coordinates
    time = xr.date_range(
        start=start_date, end=end_date, freq=freq, calendar=calendar, use_cftime=True
    )
    lats = np.linspace(start=-90, stop=90, num=nlats, dtype='float32')
    lons = np.linspace(start=-180, stop=180, num=nlons, dtype='float32')

    # Create data variable
    values = np.expand_dims(np.arange(len(time)), axis=(1, 2))
    data = np.tile(values, (1, nlats, nlons))
    ds = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), data)},
        coords={'time': time, 'lat': lats, 'lon': lons},
        attrs={'Description': 'This is dummy data for testing.'},
    )
    return ds


class Bench_climate_anomaly:
    def setup(self):
        self.daily = _get_dummy_data('2020-01-01', '2021-12-31', 'D', 1, 1)

    def time_climate_anomaly(self):
        # monthly anomaly
        climate_anomaly(self.daily, 'month')


class Bench_month_to_season:
    def setup(self):
        self.ds = get_fake_dataset(start_month="2000-01", nmonths=12, nlats=3, nlons=1)

    def time_month_to_season(self):
        month_to_season(self.ds, "JJA")


class Bench_calendar_average:
    def setup(self):
        self.hourly = _get_dummy_data('2020-01-01', '2021-12-31 23:00:00', 'h', 1, 1)

    def time_calendar_average(self):
        # hourly to daily
        calendar_average(self.hourly, freq='day')


class Bench_climatology_average:
    def setup(self):
        self.monthly = _get_dummy_data('2020-01-01', '2021-12-01', 'MS', 1, 1)

    def time_climatology_average(self):
        # custom season
        climatology_average(
            self.monthly, freq='season', custom_seasons=['DJF', 'JJA', 'MAM', 'SON']
        )
