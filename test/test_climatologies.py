import cftime
import numpy as np
from packaging.version import Version
import pandas as pd
import pytest
import xarray.testing
import xarray as xr
from xarray import __version__ as xarray_version

from geocat.comp import (
    climate_anomaly,
    month_to_season,
    calendar_average,
    climatology_average,
)
from geocat.comp.climatologies import _infer_calendar_name


##### Helper Functions #####
def get_fake_dataset(start_month, nmonths, nlats, nlons):
    """Returns a very simple xarray dataset for testing.

    Data values are equal to "month of year" for monthly time steps.
    """
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
    """Returns a simple xarray dataset to test with.

    Data can be hourly, daily, or monthly.
    """
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


##### End Helper Functions #####


class Test_Climate_Anomaly:
    daily = _get_dummy_data('2020-01-01', '2021-12-31', 'D', 1, 1)

    def test_daily_anomaly(self) -> None:
        expected_anom = np.concatenate(
            [
                np.full(59, -183),
                [0],
                np.full(306, -182.5),
                np.full(59, 183),
                np.full(306, 182.5),
            ]
        )
        expected_anom = xr.Dataset(
            data_vars={
                'data': (('time', 'lat', 'lon'), np.reshape(expected_anom, (731, 1, 1)))
            },
            coords={
                'time': xr.date_range(
                    start='2020-01-01', end='2021-12-31', freq='D', use_cftime=True
                ),
                'lat': [-90],
                'lon': [-180],
            },
            attrs={'Description': 'This is dummy data for testing.'},
        )
        anom = climate_anomaly(self.daily, 'day')
        xarray.testing.assert_allclose(anom, expected_anom)

    def test_monthly_anomaly(self) -> None:
        expected_anom = np.concatenate(
            [
                np.arange(-198, -167),
                np.arange(-193.54386, -165),
                np.arange(-197.5, -166.5),
                np.arange(-197, -167),
                np.arange(-197.5, -166.5),
                np.arange(-197, -167),
                np.arange(-197.5, -166.5),
                np.arange(-197.5, -166.5),
                np.arange(-197, -167),
                np.arange(-197.5, -166.5),
                np.arange(-197, -167),
                np.arange(-197.5, -166.5),
                np.arange(168, 199),
                np.arange(172.4561404, 200),
                np.arange(167.5, 198.5),
                np.arange(168, 198),
                np.arange(167.5, 198.5),
                np.arange(168, 198),
                np.arange(167.5, 198.5),
                np.arange(167.5, 198.5),
                np.arange(168, 198),
                np.arange(167.5, 198.5),
                np.arange(168, 198),
                np.arange(167.5, 198.5),
            ]
        )
        expected_anom = xr.Dataset(
            data_vars={
                'data': (('time', 'lat', 'lon'), np.reshape(expected_anom, (731, 1, 1)))
            },
            coords={
                'time': xr.date_range(
                    start='2020-01-01', end='2021-12-31', freq='D', use_cftime=True
                ),
                'lat': [-90],
                'lon': [-180],
            },
            attrs={'Description': 'This is dummy data for testing.'},
        )
        anom = climate_anomaly(self.daily, 'month')
        xarray.testing.assert_allclose(anom, expected_anom)

    def test_seasonal_anomaly(self) -> None:
        expected_anom = np.concatenate(
            [
                np.arange(-320.9392265, -261),
                np.arange(-228, -136),
                np.arange(-228, -136),
                np.arange(-227.5, -137),
                np.arange(14.06077348, 104),
                np.arange(137, 229),
                np.arange(137, 229),
                np.arange(137.5, 228),
                np.arange(379.0607735, 410),
            ]
        )
        seasons = (
            ['DJF'] * 60
            + ['MAM'] * 92
            + ['JJA'] * 92
            + ['SON'] * 91
            + ['DJF'] * 90
            + ['MAM'] * 92
            + ['JJA'] * 92
            + ['SON'] * 91
            + ['DJF'] * 31
        )

        expected_anom = xr.Dataset(
            data_vars={
                'data': (('time', 'lat', 'lon'), np.reshape(expected_anom, (731, 1, 1)))
            },
            coords={
                'time': xr.date_range(
                    start='2020-01-01', end='2021-12-31', freq='D', use_cftime=True
                ),
                'lat': [-90],
                'lon': [-180],
                'season': ('time', seasons),
            },
            attrs={'Description': 'This is dummy data for testing.'},
        )
        anom = climate_anomaly(self.daily, 'season')
        xarray.testing.assert_allclose(anom, expected_anom)

    def test_yearly_anomaly(self) -> None:
        expected_anom = np.concatenate([np.arange(-182.5, 183), np.arange(-182, 183)])
        expected_anom = xr.Dataset(
            data_vars={
                'data': (('time', 'lat', 'lon'), np.reshape(expected_anom, (731, 1, 1)))
            },
            coords={
                'time': xr.date_range(
                    start='2020-01-01', end='2021-12-31', freq='D', use_cftime=True
                ),
                'lat': [-90],
                'lon': [-180],
            },
            attrs={'Description': 'This is dummy data for testing.'},
        )
        anom = climate_anomaly(self.daily, 'year')
        xarray.testing.assert_allclose(anom, expected_anom)

    @pytest.mark.parametrize(
        "name, dset, freq, keep_attrs",
        [
            ('daily, "month", None', daily, 'month', None),
            ('daily, "month", True', daily, 'month', True),
            ('daily, "month", False', daily, 'month', False),
            ('daily, "season", None', daily, 'season', None),
            ('daily, "season", True', daily, 'season', True),
            ('daily, "season", False', daily, 'season', False),
            ('daily, "year", None', daily, 'year', None),
            ('daily, "year", True', daily, 'year', True),
            ('daily, "year", False', daily, 'year', False),
        ],
    )
    def test_keep_attrs(self, name, dset, freq, keep_attrs) -> None:
        result = climate_anomaly(dset, freq, keep_attrs=keep_attrs)
        if keep_attrs or keep_attrs is None:
            assert result.attrs == dset.attrs
        elif not keep_attrs:
            assert result.attrs == {}

    def test_custom_time_dim(self) -> None:
        time_dim = 'my_time'
        expected_anom = np.concatenate(
            [
                np.arange(-198, -167),
                np.arange(-193.54386, -165),
                np.arange(-197.5, -166.5),
                np.arange(-197, -167),
                np.arange(-197.5, -166.5),
                np.arange(-197, -167),
                np.arange(-197.5, -166.5),
                np.arange(-197.5, -166.5),
                np.arange(-197, -167),
                np.arange(-197.5, -166.5),
                np.arange(-197, -167),
                np.arange(-197.5, -166.5),
                np.arange(168, 199),
                np.arange(172.4561404, 200),
                np.arange(167.5, 198.5),
                np.arange(168, 198),
                np.arange(167.5, 198.5),
                np.arange(168, 198),
                np.arange(167.5, 198.5),
                np.arange(167.5, 198.5),
                np.arange(168, 198),
                np.arange(167.5, 198.5),
                np.arange(168, 198),
                np.arange(167.5, 198.5),
            ]
        )
        expected_anom = xr.Dataset(
            data_vars={
                'data': (
                    (time_dim, 'lat', 'lon'),
                    np.reshape(expected_anom, (731, 1, 1)),
                )
            },
            coords={
                time_dim: xr.date_range(
                    start='2020-01-01', end='2021-12-31', freq='D', use_cftime=True
                ),
                'lat': [-90],
                'lon': [-180],
            },
            attrs={'Description': 'This is dummy data for testing.'},
        )

        anom = climate_anomaly(
            self.daily.rename({'time': time_dim}), freq='month', time_dim=time_dim
        )
        xr.testing.assert_allclose(anom, expected_anom)


class Test_Month_to_Season:
    ds1 = get_fake_dataset(start_month="2000-01", nmonths=12, nlats=1, nlons=1)

    # Create another dataset for the year 2001.
    ds2 = get_fake_dataset(start_month="2001-01", nmonths=12, nlats=1, nlons=1)

    # Create a dataset that combines the two previous datasets, for two
    # years of data.
    ds3 = xr.concat([ds1, ds2], dim="time")

    ds4 = xr.tutorial.open_dataset("rasm").rename({"time": "Times"})

    # Create a dataset with the wrong number of months.
    partial_year_dataset = get_fake_dataset(
        start_month="2000-01", nmonths=13, nlats=1, nlons=1
    )

    # Create a dataset with a custom time coordinate.
    custom_time_dataset = get_fake_dataset(
        start_month="2000-01", nmonths=12, nlats=1, nlons=1
    )
    custom_time_dataset = custom_time_dataset.rename({"time": "my_time"})

    # Create a more complex dataset just to verify that get_fake_dataset()
    # is generally working.
    complex_dataset = get_fake_dataset(
        start_month="2001-01", nmonths=12, nlats=10, nlons=10
    )

    @pytest.mark.parametrize(
        "name, keep_attrs", [('None', None), ('True', True), ('False', False)]
    )
    def test_month_to_season_keep_attrs(self, name, keep_attrs) -> None:
        season_ds = month_to_season(self.ds1, 'JFM', keep_attrs=keep_attrs)
        if keep_attrs or keep_attrs is None:
            assert season_ds.attrs == self.ds1.attrs
        elif not keep_attrs:
            assert season_ds.attrs == {}

    @pytest.mark.parametrize(
        "name, dset, season, expected",
        [('ds1, JFM', ds1, 'JFM', 2.0), ('ds2, JAA', ds1, 'JJA', 7.0)],
    )
    def test_month_to_season_returns_middle_month_value(
        self, name, dset, season, expected
    ) -> None:
        season_ds = month_to_season(dset, season)
        np.testing.assert_equal(season_ds["my_var"].data, expected)

    def test_month_to_season_bad_season_exception(self) -> None:
        with pytest.raises(KeyError):
            month_to_season(self.ds1, "TEST")

    def test_month_to_season_partial_years_exception(self) -> None:
        with pytest.raises(ValueError):
            month_to_season(self.partial_year_dataset, "JFM")

    def test_month_to_season_final_season_returns_2month_average(self) -> None:
        season_ds = month_to_season(self.ds1, 'NDJ')
        np.testing.assert_equal(season_ds["my_var"].data, 11.5)

    @pytest.mark.parametrize(
        "name, season",
        [
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
            ('NDJ', 'NDJ'),
        ],
    )
    def test_month_to_season_returns_one_point_per_year(self, name, season) -> None:
        nyears_of_data = self.ds3.sizes["time"] / 12
        season_ds = month_to_season(self.ds3, season)
        assert season_ds["my_var"].size == nyears_of_data

    @pytest.mark.parametrize(
        "name, dataset, time_coordinate, var_name, expected",
        [
            ('custom_time_dataset', custom_time_dataset, "my_time", "my_var", 2.0),
            ('ds4', ds4.isel(x=110, y=200), None, "Tair", [-10.56, -8.129, -7.125]),
        ],
    )
    def test_month_to_season_custom_time_coordinate(
        self, name, dataset, time_coordinate, var_name, expected
    ) -> None:
        season_ds = month_to_season(dataset, "JFM", time_coord_name=time_coordinate)
        np.testing.assert_almost_equal(season_ds[var_name].data, expected, decimal=1)


class Test_Calendar_Average:
    minute = _get_dummy_data('2020-01-01', '2021-12-31 23:30:00', '30min', 1, 1)
    hourly = _get_dummy_data('2020-01-01', '2021-12-31 23:00:00', 'h', 1, 1)
    daily = _get_dummy_data('2020-01-01', '2021-12-31', 'D', 1, 1)
    monthly = _get_dummy_data('2020-01-01', '2021-12-01', 'MS', 1, 1)

    month_avg = np.array(
        [
            15,
            45,
            75,
            105.5,
            136,
            166.5,
            197,
            228,
            258.5,
            289,
            319.5,
            350,
            381,
            410.5,
            440,
            470.5,
            501,
            531.5,
            562,
            593,
            623.5,
            654,
            684.5,
            715,
        ]
    ).reshape(24, 1, 1)
    month_avg_time = xr.date_range(
        '2020-01-01', '2022-01-01', freq='MS', use_cftime=True
    )
    month_avg_time = xr.DataArray(
        np.vstack((month_avg_time[:-1], month_avg_time[1:])).T, dims=['time', 'nbd']
    ).mean(dim='nbd')
    day_2_month_avg = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), month_avg)},
        coords={'time': month_avg_time, 'lat': [-90.0], 'lon': [-180.0]},
    )

    season_avg = np.array(
        [29.5, 105.5, 197.5, 289, 379.5, 470.5, 562.5, 654, 715]
    ).reshape(9, 1, 1)
    season_avg_time = xr.date_range(
        '2019-12-01', '2022-03-01', freq='QS-DEC', use_cftime=True
    )
    season_avg_time = xr.DataArray(
        np.vstack((season_avg_time[:-1], season_avg_time[1:])).T, dims=['time', 'nbd']
    ).mean(dim='nbd')
    day_2_season_avg = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), season_avg)},
        coords={'time': season_avg_time, 'lat': [-90.0], 'lon': [-180.0]},
    )

    season_avg = np.array(
        [0.483333333, 3, 6.010869565, 9, 11.96666667, 15, 18.01086957, 21, 23]
    ).reshape(9, 1, 1)
    month_2_season_avg = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), season_avg)},
        coords={'time': season_avg_time, 'lat': [-90.0], 'lon': [-180.0]},
    )

    year_avg_time = [cftime.datetime(2020, 7, 2), cftime.datetime(2021, 7, 2, hour=12)]
    day_2_year_avg = [[[182.5]], [[548]]]
    day_2_year_avg = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), day_2_year_avg)},
        coords={'time': year_avg_time, 'lat': [-90.0], 'lon': [-180.0]},
    )
    month_2_year_avg = [[[5.513661202]], [[17.5260274]]]
    month_2_year_avg = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), month_2_year_avg)},
        coords={'time': year_avg_time, 'lat': [-90.0], 'lon': [-180.0]},
    )

    time_dim = 'my_time'
    custom_time = daily.rename({'time': time_dim})
    custom_time_expected = day_2_month_avg.rename({'time': time_dim})

    julian_daily = _get_dummy_data(
        '2020-01-01', '2021-12-31', 'D', 1, 1, calendar='julian'
    )
    noleap_daily = _get_dummy_data(
        '2020-01-01', '2021-12-31', 'D', 1, 1, calendar='noleap'
    )
    all_leap_daily = _get_dummy_data(
        '2020-01-01', '2021-12-31', 'D', 1, 1, calendar='all_leap'
    )
    day_360_daily = _get_dummy_data(
        '2020-01-01', '2021-12-30', 'D', 1, 1, calendar='360_day'
    )
    # Daily -> Monthly Means for Julian Calendar
    julian_month_avg = np.array(
        [
            15,
            45,
            75,
            105.5,
            136,
            166.5,
            197,
            228,
            258.5,
            289,
            319.5,
            350,
            381,
            410.5,
            440,
            470.5,
            501,
            531.5,
            562,
            593,
            623.5,
            654,
            684.5,
            715,
        ]
    ).reshape(24, 1, 1)
    julian_month_avg_time = xr.date_range(
        '2020-01-01', '2022-01-01', freq='MS', calendar='julian', use_cftime=True
    )
    julian_month_avg_time = xr.DataArray(
        np.vstack((julian_month_avg_time[:-1], julian_month_avg_time[1:])).T,
        dims=['time', 'nbd'],
    ).mean(dim='nbd')
    julian_day_2_month_avg = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), julian_month_avg)},
        coords={'time': julian_month_avg_time, 'lat': [-90.0], 'lon': [-180.0]},
    )
    # Daily -> Monthly Means for NoLeap Calendar
    noleap_month_avg = np.array(
        [
            15,
            44.5,
            74,
            104.5,
            135,
            165.5,
            196,
            227,
            257.5,
            288,
            318.5,
            349,
            380,
            409.5,
            439,
            469.5,
            500,
            530.5,
            561,
            592,
            622.5,
            653,
            683.5,
            714,
        ]
    ).reshape(24, 1, 1)
    noleap_month_avg_time = xr.date_range(
        '2020-01-01', '2022-01-01', freq='MS', calendar='noleap', use_cftime=True
    )
    noleap_month_avg_time = xr.DataArray(
        np.vstack((noleap_month_avg_time[:-1], noleap_month_avg_time[1:])).T,
        dims=['time', 'nbd'],
    ).mean(dim='nbd')
    noleap_day_2_month_avg = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), noleap_month_avg)},
        coords={'time': noleap_month_avg_time, 'lat': [-90.0], 'lon': [-180.0]},
    )
    # Daily -> Monthly Means for AllLeap Calendar
    all_leap_month_avg = np.array(
        [
            15,
            45,
            75,
            105.5,
            136,
            166.5,
            197,
            228,
            258.5,
            289,
            319.5,
            350,
            381,
            411,
            441,
            471.5,
            502,
            532.5,
            563,
            594,
            624.5,
            655,
            685.5,
            716,
        ]
    ).reshape(24, 1, 1)
    all_leap_month_avg_time = xr.date_range(
        '2020-01-01', '2022-01-01', freq='MS', calendar='all_leap', use_cftime=True
    )
    all_leap_month_avg_time = xr.DataArray(
        np.vstack((all_leap_month_avg_time[:-1], all_leap_month_avg_time[1:])).T,
        dims=['time', 'nbd'],
    ).mean(dim='nbd')
    all_leap_day_2_month_avg = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), all_leap_month_avg)},
        coords={'time': all_leap_month_avg_time, 'lat': [-90.0], 'lon': [-180.0]},
    )
    # Daily -> Monthly Means for 360 Day Calendar
    day_360_leap_month_avg = np.arange(14.5, 734.5, 30).reshape(24, 1, 1)
    day_360_leap_month_avg_time = xr.date_range(
        '2020-01-01', '2022-01-01', freq='MS', calendar='360_day', use_cftime=True
    )
    day_360_leap_month_avg_time = xr.DataArray(
        np.vstack(
            (day_360_leap_month_avg_time[:-1], day_360_leap_month_avg_time[1:])
        ).T,
        dims=['time', 'nbd'],
    ).mean(dim='nbd')
    day_360_leap_day_2_month_avg = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), day_360_leap_month_avg)},
        coords={'time': day_360_leap_month_avg_time, 'lat': [-90.0], 'lon': [-180.0]},
    )

    @pytest.mark.parametrize(
        "name, dset, freq, keep_attrs",
        [
            ('daily, "month", None', daily, 'month', None),
            ('daily, "month", True', daily, 'month', True),
            ('daily, "month", False', daily, 'month', False),
            ('monthly, "season", None', monthly, 'season', None),
            ('monthly, "season", True', monthly, 'season', True),
            ('monthly, "season", False', monthly, 'season', False),
            ('monthly, "year", None', monthly, 'year', None),
            ('monthly, "year", True', monthly, 'year', True),
            ('monthly, "year", False', monthly, 'year', False),
        ],
    )
    def test_calendar_average_keep_attrs(self, name, dset, freq, keep_attrs) -> None:
        result = calendar_average(dset, freq, keep_attrs=keep_attrs)
        if keep_attrs or keep_attrs is None:
            assert result.attrs == dset.attrs
        elif not keep_attrs:
            assert result.attrs == {}

    def test_30min_to_hourly_calendar_average(self) -> None:
        hour_avg = np.arange(0.5, 35088.5, 2).reshape((365 + 366) * 24, 1, 1)
        hour_avg_time = xr.date_range(
            '2020-01-01 00:30:00', '2021-12-31 23:30:00', freq='h', use_cftime=True
        )
        min_2_hour_avg = xr.Dataset(
            data_vars={'data': (('time', 'lat', 'lon'), hour_avg)},
            coords={'time': hour_avg_time, 'lat': [-90.0], 'lon': [-180.0]},
        )

        result = calendar_average(self.minute, freq='hour')
        xr.testing.assert_equal(result, min_2_hour_avg)

    def test_hourly_to_daily_calendar_average(self) -> None:
        day_avg = np.arange(11.5, 17555.5, 24).reshape(366 + 365, 1, 1)
        day_avg_time = xr.date_range(
            '2020-01-01 12:00:00', '2021-12-31 12:00:00', freq='D', use_cftime=True
        )
        hour_2_day_avg = xr.Dataset(
            data_vars={'data': (('time', 'lat', 'lon'), day_avg)},
            coords={'time': day_avg_time, 'lat': [-90.0], 'lon': [-180.0]},
        )
        result = calendar_average(self.hourly, freq='day')
        xr.testing.assert_equal(result, hour_2_day_avg)

    def test_daily_to_monthly_calendar_average(self) -> None:
        month_avg = np.array(
            [
                15,
                45,
                75,
                105.5,
                136,
                166.5,
                197,
                228,
                258.5,
                289,
                319.5,
                350,
                381,
                410.5,
                440,
                470.5,
                501,
                531.5,
                562,
                593,
                623.5,
                654,
                684.5,
                715,
            ]
        ).reshape(24, 1, 1)
        month_avg_time = xr.date_range(
            '2020-01-01', '2022-01-01', freq='MS', use_cftime=True
        )
        month_avg_time = xr.DataArray(
            np.vstack((month_avg_time[:-1], month_avg_time[1:])).T, dims=['time', 'nbd']
        ).mean(dim='nbd')
        day_2_month_avg = xr.Dataset(
            data_vars={'data': (('time', 'lat', 'lon'), month_avg)},
            coords={'time': month_avg_time, 'lat': [-90.0], 'lon': [-180.0]},
        )

        result = calendar_average(self.daily, freq='month')
        xr.testing.assert_equal(result, day_2_month_avg)

    @pytest.mark.parametrize(
        "name, dset, expected",
        [
            ('daily to seasonal', daily, day_2_season_avg),
            ('monthly to seasonal', monthly, month_2_season_avg),
        ],
    )
    def test_daily_monthly_to_seasonal_calendar_average(
        self, name, dset, expected
    ) -> None:
        result = calendar_average(dset, freq='season')
        xr.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "name, dset, expected",
        [
            ('daily to yearly', daily, day_2_year_avg),
            ('monthly to yearly', monthly, month_2_year_avg),
        ],
    )
    def test_daily_monthly_to_yearly_calendar_average(
        self, name, dset, expected
    ) -> None:
        result = calendar_average(dset, freq='year')
        xr.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("name, freq", [('freq=TEST', 'TEST'), ('freq=None', None)])
    def test_invalid_freq_calendar_average(self, name, freq) -> None:
        with pytest.raises(ValueError):
            calendar_average(self.monthly, freq=freq)

    def test_custom_time_coord_calendar_average(self) -> None:
        result = calendar_average(
            self.custom_time, freq='month', time_dim=self.time_dim
        )
        xr.testing.assert_allclose(result, self.custom_time_expected)

    def test_xr_DataArray_support_calendar_average(self) -> None:
        array = self.daily['data']
        array_expected = self.day_2_month_avg['data']
        result = calendar_average(array, freq='month')
        xr.testing.assert_equal(result, array_expected)

    def test_non_datetime_like_objects_calendar_average(self) -> None:
        dset_encoded = xr.tutorial.open_dataset("air_temperature", decode_cf=False)
        with pytest.raises(ValueError):
            calendar_average(dset_encoded, 'month')

    def test_non_uniformly_spaced_data_calendar_average(self) -> None:
        time = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-04'])
        non_uniform = xr.Dataset(
            data_vars={'data': (('time'), np.arange(3))}, coords={'time': time}
        )
        with pytest.raises(ValueError):
            calendar_average(non_uniform, freq='day')

    @pytest.mark.parametrize(
        "name, dset, expected",
        [
            ('julian_calendar', julian_daily, julian_day_2_month_avg),
            ('no_leap_calendar', noleap_daily, noleap_day_2_month_avg),
            ('all_leap_calendar', all_leap_daily, all_leap_day_2_month_avg),
            ('day_360_calendar', day_360_daily, day_360_leap_day_2_month_avg),
        ],
    )
    def test_non_standard_calendars_calendar_average(
        self, name, dset, expected
    ) -> None:
        result = calendar_average(dset, freq='month')
        xr.testing.assert_equal(result, expected)


class Test_Climatology_Average:
    minute = _get_dummy_data('2020-01-01', '2021-12-31 23:30:00', '30min', 1, 1)

    hourly = _get_dummy_data('2020-01-01', '2021-12-31 23:00:00', 'h', 1, 1)

    daily = _get_dummy_data('2020-01-01', '2021-12-31', 'D', 1, 1)

    monthly = _get_dummy_data('2020-01-01', '2021-12-01', 'MS', 1, 1)

    hour_clim = np.concatenate(
        [
            np.arange(8784.5, 11616.5, 2),
            np.arange(2832.5, 2880.5, 2),
            np.arange(11640.5, 26328.5, 2),
        ]
    ).reshape(8784, 1, 1)
    hour_clim_time = xr.date_range(
        '2020-01-01 00:30:00', '2020-12-31 23:30:00', freq='h', use_cftime=True
    )
    min_2_hourly_clim = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), hour_clim)},
        coords={'time': hour_clim_time, 'lat': [-90.0], 'lon': [-180.0]},
    )
    day_clim = np.concatenate(
        [np.arange(4403.5, 5819.5, 24), [1427.5], np.arange(5831.5, 13175.5, 24)]
    ).reshape(366, 1, 1)
    day_clim_time = xr.date_range(
        '2020-01-01 12:00:00', '2020-12-31 12:00:00', freq='24h', use_cftime=True
    )

    hour_2_day_clim = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), day_clim)},
        coords={'time': day_clim_time, 'lat': [-90.0], 'lon': [-180.0]},
    )
    month_clim = np.array(
        [198, 224.5438596, 257.5, 288, 318.5, 349, 379.5, 410.5, 441, 471.5, 502, 532.5]
    ).reshape(12, 1, 1)
    month_clim_time = xr.date_range(
        '2020-01-01', '2021-01-01', freq='MS', use_cftime=True
    )
    month_clim_time = xr.DataArray(
        np.vstack((month_clim_time[:-1], month_clim_time[1:])).T, dims=['time', 'nbd']
    ).mean(dim='nbd')
    day_2_month_clim = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), month_clim)},
        coords={'time': month_clim_time, 'lat': [-90.0], 'lon': [-180.0]},
    )

    season_clim = np.array([320.9392265, 380, 288, 471.5]).reshape(4, 1, 1)
    season_clim_time = ['DJF', 'JJA', 'MAM', 'SON']
    day_2_season_clim = xr.Dataset(
        data_vars={'data': (('season', 'lat', 'lon'), season_clim)},
        coords={
            'season': np.array(season_clim_time).astype(object),
            'lat': [-90.0],
            'lon': [-180.0],
        },
    )

    season_clim = np.array([10.04972376, 12.01086957, 9, 15]).reshape(4, 1, 1)
    month_2_season_clim = xr.Dataset(
        data_vars={'data': (('season', 'lat', 'lon'), season_clim)},
        coords={
            'season': np.array(season_clim_time).astype(object),
            'lat': [-90.0],
            'lon': [-180.0],
        },
    )

    julian_daily = _get_dummy_data(
        '2020-01-01', '2021-12-31', 'D', 1, 1, calendar='julian'
    )
    noleap_daily = _get_dummy_data(
        '2020-01-01', '2021-12-31', 'D', 1, 1, calendar='noleap'
    )
    all_leap_daily = _get_dummy_data(
        '2020-01-01', '2021-12-31', 'D', 1, 1, calendar='all_leap'
    )
    day_360_daily = _get_dummy_data(
        '2020-01-01', '2021-12-30', 'D', 1, 1, calendar='360_day'
    )

    # Daily -> Monthly Climatologies for Julian Calendar
    julian_month_clim = np.array(
        [
            198,
            224.54385965,
            257.5,
            288,
            318.5,
            349,
            379.5,
            410.5,
            441,
            471.5,
            502,
            532.5,
        ]
    ).reshape(12, 1, 1)
    julian_month_clim_time = xr.date_range(
        '2020-01-01', '2021-01-01', freq='MS', calendar='julian', use_cftime=True
    )
    julian_month_clim_time = xr.DataArray(
        np.vstack((julian_month_clim_time[:-1], julian_month_clim_time[1:])).T,
        dims=['time', 'nbd'],
    ).mean(dim='nbd')
    julian_day_2_month_clim = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), julian_month_clim)},
        coords={'time': julian_month_clim_time, 'lat': [-90.0], 'lon': [-180.0]},
    )
    # Daily -> Monthly Climatologies for NoLeap Calendar
    noleap_month_clim = np.array(
        [197.5, 227, 256.5, 287, 317.5, 348, 378.5, 409.5, 440, 470.5, 501, 531.5]
    ).reshape(12, 1, 1)
    noleap_month_clim_time = xr.date_range(
        '2020-01-01', '2021-01-01', freq='MS', calendar='noleap', use_cftime=True
    )
    noleap_month_clim_time = xr.DataArray(
        np.vstack((noleap_month_clim_time[:-1], noleap_month_clim_time[1:])).T,
        dims=['time', 'nbd'],
    ).mean(dim='nbd')
    noleap_day_2_month_clim = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), noleap_month_clim)},
        coords={'time': noleap_month_clim_time, 'lat': [-90.0], 'lon': [-180.0]},
    )
    # Daily -> Monthly Climatologies for AllLeap Calendar
    all_leap_month_clim = np.array(
        [198, 228, 258, 288.5, 319, 349.5, 380, 411, 441.5, 472, 502.5, 533]
    ).reshape(12, 1, 1)
    all_leap_month_clim_time = xr.date_range(
        '2020-01-01', '2021-01-01', freq='MS', calendar='all_leap', use_cftime=True
    )
    all_leap_month_clim_time = xr.DataArray(
        np.vstack((all_leap_month_clim_time[:-1], all_leap_month_clim_time[1:])).T,
        dims=['time', 'nbd'],
    ).mean(dim='nbd')
    all_leap_day_2_month_clim = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), all_leap_month_clim)},
        coords={'time': all_leap_month_clim_time, 'lat': [-90.0], 'lon': [-180.0]},
    )
    # Daily -> Monthly Climatologies for 360 Day Calendar
    day_360_leap_month_clim = np.arange(194.5, 554.5, 30).reshape(12, 1, 1)
    day_360_leap_month_clim_time = xr.date_range(
        '2020-01-01', '2021-01-01', freq='MS', calendar='360_day', use_cftime=True
    )
    day_360_leap_month_clim_time = xr.DataArray(
        np.vstack(
            (day_360_leap_month_clim_time[:-1], day_360_leap_month_clim_time[1:])
        ).T,
        dims=['time', 'nbd'],
    ).mean(dim='nbd')
    day_360_leap_day_2_month_clim = xr.Dataset(
        data_vars={'data': (('time', 'lat', 'lon'), day_360_leap_month_clim)},
        coords={'time': day_360_leap_month_clim_time, 'lat': [-90.0], 'lon': [-180.0]},
    )

    @pytest.mark.parametrize(
        "name, dset, freq, custom_seasons, keep_attrs",
        [
            ('daily, "month", None', daily, 'month', [], None),
            ('daily, "month", True', daily, 'month', [], True),
            ('daily, "month", False', daily, 'month', [], False),
            ('monthly, "season", None', monthly, 'season', [], None),
            ('monthly, "season", True', monthly, 'season', [], True),
            ('monthly, "season", False', monthly, 'season', [], False),
            (
                'monthly, "season", None',
                monthly,
                'season',
                ['DJF', 'MAM', 'JJA', 'SON'],
                None,
            ),
            (
                'monthly, "season", True',
                monthly,
                'season',
                ['DJF', 'MAM', 'JJA', 'SON'],
                True,
            ),
            (
                'monthly, "season", False',
                monthly,
                'season',
                ['DJF', 'MAM', 'JJA', 'SON'],
                False,
            ),
        ],
    )
    def test_climatology_average_keep_attrs(
        self, name, dset, freq, custom_seasons, keep_attrs
    ) -> None:
        result = climatology_average(
            dset, freq=freq, custom_seasons=custom_seasons, keep_attrs=keep_attrs
        )
        if keep_attrs or keep_attrs is None:
            assert result.attrs == dset.attrs
        elif not keep_attrs:
            assert result.attrs == {}

    def test_30min_to_hourly_climatology_average(self) -> None:
        result = climatology_average(self.minute, freq='hour')
        xr.testing.assert_allclose(result, self.min_2_hourly_clim)

    def test_hourly_to_daily_climatology_average(self) -> None:
        result = climatology_average(self.hourly, freq='day')
        xr.testing.assert_equal(result, self.hour_2_day_clim)

    def test_daily_to_monthly_climatology_average(self) -> None:
        result = climatology_average(self.daily, freq='month')
        xr.testing.assert_allclose(result, self.day_2_month_clim)

    def test_custom_season_climatology_average(self) -> None:
        result = climatology_average(
            self.monthly, freq='season', custom_seasons=['DJF', 'JJA', 'MAM', 'SON']
        )
        expected = climatology_average(self.monthly, freq='season')
        xr.testing.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "name, dset, expected",
        [
            ('daily to seasonal', daily, day_2_season_clim),
            ('monthly to seasonal', monthly, month_2_season_clim),
        ],
    )
    def test_daily_monthly_to_seasonal_climatology_average(
        self, name, dset, expected
    ) -> None:
        result = climatology_average(dset, freq='season')
        xr.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("name, freq", [('freq=TEST', 'TEST'), ('freq=None', None)])
    def test_invalid_freq_climatology_average(self, name, freq) -> None:
        with pytest.raises(ValueError):
            climatology_average(self.monthly, freq=freq)

    def test_custom_time_coord_climatology_average(self) -> None:
        time_dim = 'my_time'
        custom_time = self.daily.rename({'time': time_dim})

        custom_time_expected = self.day_2_month_clim.rename({'time': time_dim})

        result = climatology_average(custom_time, freq='month', time_dim=time_dim)
        xr.testing.assert_allclose(result, custom_time_expected)

    def test_xr_DataArray_support_climatology_average(self) -> None:
        array = self.daily['data']
        array_expected = self.day_2_month_clim['data']

        result = climatology_average(array, freq='month')
        xr.testing.assert_allclose(result, array_expected)

    def test_datetime_climatology_average(self) -> None:
        array = self.daily['data']
        if Version(xarray_version) < Version('2025.01.2'):
            array['time'] = array.time.to_index().to_datetimeindex()
        else:
            array['time'] = array.time.to_index().to_datetimeindex(time_unit='ns')
        array_expected = self.day_2_month_clim['data']
        result = climatology_average(array, freq='month')
        xr.testing.assert_allclose(result, array_expected)

    def test_non_datetime_like_objects_climatology_average(self) -> None:
        dset_encoded = xr.tutorial.open_dataset("air_temperature", decode_cf=False)
        with pytest.raises(ValueError):
            climatology_average(dset_encoded, freq='month')

    def test_non_uniformly_spaced_data_climatology_average(self) -> None:
        time = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-04'])
        non_uniform = xr.Dataset(
            data_vars={'data': (('time'), np.arange(3))}, coords={'time': time}
        )
        with pytest.raises(ValueError):
            climatology_average(non_uniform, freq='day')

    @pytest.mark.parametrize(
        "name, dset, expected",
        [
            ('julian_calendar', julian_daily, julian_day_2_month_clim),
            ('no_leap_calendar', noleap_daily, noleap_day_2_month_clim),
            ('all_leap_calendar', all_leap_daily, all_leap_day_2_month_clim),
            ('day_360_calendar', day_360_daily, day_360_leap_day_2_month_clim),
        ],
    )
    def test_non_standard_calendars_climatology_average(
        self, name, dset, expected
    ) -> None:
        result = climatology_average(dset, freq='month')
        xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("units", ("s", "ms", "us", "ns"))
def test_units_infer_calendar_name(units):
    time = pd.date_range("2000-01-01", periods=10, freq="1D", unit=units)
    assert _infer_calendar_name(time) == "proleptic_gregorian"
