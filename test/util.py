import xarray as xr
import numpy as np
import pandas as pd


def make_toy_temp_dataset(
    nlat=10, nlon=30, lat=None, lon=None, nt=3, nans=False, cf=True
):
    """Makes a toy xarray dataset with two temperature variables, 't' and 't2'.

    Parameters
    ----------
    nlat : int
        optional, number of latitude points (default 20 unless lat specified)

    nlon : int
        optional, number of longitude points (default 30 unless lon specified)

    lat : list
        optional, list of latitude points. Cannot be specified with nlat. (default None)

    lon : list
        optional, list of longitude points. Cannot be specified with nlon. (default None)

    nt : int
        optional, number of time points (default 5)

    nans : bool
        optional, if True, randomly insert nans into the data. will always
        insert at least one nan (default False)

    cf : bool
        optional, adds standard_name to attrs for latitude, longitude, and time dimensions

    Returns
    -------
    xr.Dataset
    """
    time = pd.date_range('2023-01-01', periods=nt)
    if lat is None:
        lat = np.linspace(-90, 90, nlat)
    else:
        nlat = len(lat)

    if lon is None:
        lon = np.linspace(-180, 180, nlon)
    else:
        nlon = len(lon)

    # Create some random temperature data
    t = 15 + 8 * np.random.randn(nt, nlat, nlon)
    t2 = 15 + 8 * np.random.randn(nt, nlat, nlon)

    # Define the data in an xarray dataset
    ds = xr.Dataset(
        {
            "t": (["time", "lat", "lon"], t),
            "t2": (["time", "lat", "lon"], t2),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "description": "Sample temperature data",
            "units": "Celsius",
        },
    )

    if nans:
        for var in ds.data_vars:
            # make a mask for n nans, where n is 10% of the data, or 1, whichever is larger
            n = int(0.1 * ds[var].size) if int(0.1 * ds[var].size) > 0 else 1
            index = np.random.choice(ds[var].size, n, replace=False)
            ds[var].values.ravel()[index] = np.nan

    if cf:
        ds.lat.attrs["standard_name"] = "latitude"
        ds.lon.attrs["standard_name"] = "longitude"
        ds.time.attrs["standard_name"] = "time"

    return ds


def get_fake_climatology_dataset(start_month, nmonths, nlats, nlons):
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


def _get_toy_climatology_data(
    start_date, end_date, freq, nlats, nlons, calendar='standard'
):
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
