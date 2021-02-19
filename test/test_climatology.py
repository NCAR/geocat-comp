import pytest
import xarray as xr
import numpy as np
import pandas as pd
import geocat.comp

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
        geocat.comp.climatology(dset_a, "hourly")


def test_climatology_encoded_time():
    with pytest.raises(ValueError):
        geocat.comp.climatology(dset_encoded, "monthly")


@pytest.mark.parametrize("dataset", [dset_a, dset_b, dset_c["Tair"]])
@pytest.mark.parametrize("freq", ["day", "month", "year", "season"])
def test_climatology_setup(dataset, freq):
    computed_dset = geocat.comp.climatology(dataset, freq)
    assert type(dataset) == type(computed_dset)


@pytest.mark.parametrize("dataset", [dset_a, dset_b, dset_c["Tair"]])
@pytest.mark.parametrize("freq", ["day", "month", "year", "season"])
def test_anomaly_setup(dataset, freq):
    computed_dset = geocat.comp.anomaly(dataset, freq)
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
    season_ds = geocat.comp.month_to_season(dataset, season)
    np.testing.assert_equal(season_ds["my_var"].data, expected)


def test_month_to_season_bad_season_exception():
    with pytest.raises(KeyError):
        geocat.comp.month_to_season(ds1, "TEST")


def test_month_to_season_partial_years_exception():
    with pytest.raises(ValueError):
        geocat.comp.month_to_season(partial_year_dataset, "JFM")


@pytest.mark.parametrize("dataset, season, expected", [(ds1, "NDJ", 11.5)])
def test_month_to_season_final_season_returns_2month_average(
        dataset, season, expected):
    season_ds = geocat.comp.month_to_season(dataset, season)
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
    season_ds = geocat.comp.month_to_season(ds3, season)
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
    season_ds = geocat.comp.month_to_season(dataset,
                                            "JFM",
                                            time_coord_name=time_coordinate)
    np.testing.assert_almost_equal(season_ds[var_name].data,
                                   expected,
                                   decimal=1)
