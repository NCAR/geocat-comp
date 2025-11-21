import xarray as xr
import numpy as np
import pandas as pd


def make_toy_temp_dataset(nlat=10, nlon=30, nt=3, nans=False, cf=True):
    """Makes a toy xarray dataset with two temperature variables, 't' and 't2'.

    Parameters
    ----------
    nlat : int
        optional, number of latitude points (default 20)

    nlon : int
        optional, number of longitude points (default 30)

    nt : int
        optional, number of time points (default 5)

    nans : bool
        optional, if True, randomly insert nans into the data. will always
        insert at least one nan (default False)

    Returns
    -------
    xr.Dataset
    """
    time = pd.date_range('2023-01-01', periods=nt)
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(-180, 180, nlon)

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
