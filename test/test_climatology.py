import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp.climatology import anomaly, climatology, month_to_season, month_to_season12
else:
    from geocat.comp.climatology import anomaly, climatology, month_to_season, month_to_season12

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
    )
    return ds


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


# month_to_season12() tests
output = np.array([1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11.5])
output = output.reshape(12, 1, 1)

Tair_output = [
    7.46595, 2.146251, -3.792110333, -10.33093667, -11.091882, -10.55811833,
    -4.627381667, 2.964090333, 10.493255, 15.52151533, 17.46814333, 16.09427167,
    10.80445333, 4.354372, -2.675233667, -7.786983, -10.875342, -8.128558,
    -1.981713667, 5.608445667, 12.14102933, 16.326123, 17.559428, 14.78849933,
    8.127865333, -0.221743333, -7.366629667, -9.816663333, -9.657381, -7.124741,
    -2.536883667, 4.460225667, 11.641407, 16.05071, 17.40207767, 17.2052265
]

non_datetime = ds1.assign_coords(time=[
    '2000-01', '2000-02', '2000-03', '2000-04', '2000-05', '2000-06', '2000-07',
    '2000-08', '2000-09', '2000-10', '2000-11', '2000-12'
])


@pytest.mark.parametrize("dataset, expected", [(ds1, output)])
def test_month_to_season12_returns_correct_vals(dataset, expected):
    season_ds = month_to_season12(dataset)
    np.testing.assert_equal(season_ds["my_var"].data, expected)


@pytest.mark.parametrize("dataset, dims", [(ds1, ds1.dims)])
def test_month_to_season12_dimension(dataset, dims):
    season_ds = month_to_season12(dataset)
    assert season_ds.dims == dims


def test_month_to_season12_partial_years_exception():
    with pytest.raises(ValueError):
        month_to_season12(partial_year_dataset)


@pytest.mark.parametrize(
    "dataset, time_coordinate, var_name, expected",
    [
        (custom_time_dataset, "my_time", "my_var", output),
        (dset_c.isel(x=110, y=200), None, "Tair", Tair_output),
    ],
)
def test_month_to_season12_custom_time_coordinate(dataset, time_coordinate,
                                                  var_name, expected):
    season_ds = month_to_season12(dataset, time_coord_name=time_coordinate)
    np.testing.assert_almost_equal(season_ds[var_name].data,
                                   expected,
                                   decimal=5)


def test_month_to_season12_not_datetimelike():
    with pytest.raises(ValueError):
        month_to_season12(non_datetime)
