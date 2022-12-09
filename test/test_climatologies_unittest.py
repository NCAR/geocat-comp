import sys
import unittest
import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp import anomaly, climatology, month_to_season, calendar_average, climatology_average
else:
    from geocat.comp import anomaly, climatology, month_to_season, calendar_average, climatology_average


class test_climatology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dset_a = xr.tutorial.open_dataset("rasm")
        cls.dset_b = xr.tutorial.open_dataset("air_temperature")
        cls.dset_c = cls.dset_a.copy().rename({"time": "Times"})
        cls.dset_encoded = xr.tutorial.open_dataset("rasm", decode_cf=False)

    def test_climatology_invalid_freq(self):
        with self.assertRaises(ValueError):
            climatology(self.dset_a, 'hourly')

