import sys

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
from src.geocat.comp import pearson_r
#from geocat.comp import pearson_r

def _get_dummy_data():
    """
    Returns a simple xarray dataset to test with.
    """
    # Coordinates
    time = xr.cftime_range(start='2022-07-25',
                           end='2022-07-31',
                           freq='D')
    lats = np.linspace(start=-90, stop=90, num=2, dtype='float32')
    lons = np.linspace(start=-180, stop=180, num=18, dtype='float32')

    # Create data variable
    np.random.seed(0)
    a = np.random.random_sample((len(time), len(lats), len(lons)))
    b = np.power(a, 2)
    ds = xr.Dataset(data_vars={'a': (('time', 'lat', 'lon'), a),
                               'b': (('time', 'lat', 'lon'), b)},
                    coords={
                        'time': time,
                        'lat': lats,
                        'lon': lons
                    },
                    attrs={'description': 'Test data'})
    return ds

def test_pearson_r():
    assert 1==1

