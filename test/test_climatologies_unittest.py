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