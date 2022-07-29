from unittest import TestCase
import sys
import numpy as np
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
#if "--cov" in str(sys.argv):
from src.geocat.comp import pearson_r#
#else:
#    from geocat.comp import pearson_r

class Test_pearson_r(TestCase):
    @classmethod
    def setUpClass(cls):
        # Coordinates
        times = xr.cftime_range(start='2022-08-01',
                               end='2022-08-05',
                               freq='D')
        lats = np.linspace(start=-90, stop=90, num=3, dtype='float32')
        lons = np.linspace(start=-180, stop=180, num=4, dtype='float32')

        # Create data variables
        np.random.seed(0)
        cls.a = np.random.random_sample((len(lats), len(lons), len(times)))
        cls.b = np.power(cls.a, 2)
        cls.weights = np.arange(1, 61).reshape(3, 4, 5)
        cls.ds = xr.Dataset(data_vars={'a': (('lat', 'lon', 'time'), cls.a),
                                       'b': (('lat', 'lon', 'time'), cls.b),
                                       'weights': (('lat', 'lon', 'time'), cls.weights)},
                        coords={
                            'lat': lats,
                            'lon': lons,
                            'time': times
                        },
                        attrs={'description': 'Test data'})

        cls.unweighted_r = 0.963472086
        cls.weighted_r = 0.964263374
        cls.weighted_r_time = [[0.995914941, 0.996109426, 0.964059793, 0.996451319],
                               [0.976498563, 0.968421665, 0.955529222, 0.999373275],
                               [0.969600314, 0.979661379, 0.97071373, 0.988741167]]

    # Testing numpy inputs
    def test_pearson_r_np(self):
        a = self.a
        b = self.b
        result = pearson_r(a, b)
        assert np.allclose(self.unweighted_r, result)

    def test_pearson_r_np_weighted(self):
        a = self.a
        b = self.b
        w = self.weights
        result = pearson_r(a, b, weights=w)
        assert np.allclose(self.weighted_r, result)

    def test_pearson_r_np_warn(self):
        a = self.a
        b = self.b
        self.assertWarns(Warning, pearson_r, a, b, dim='lat', axis=0)

    def test_pearson_r_np_time(self):
        a = self.a
        b = self.b
        w = self.weights
        result = pearson_r(a, b, weights=w, axis=2)
        assert np.allclose(self.weighted_r_time, result)

    # Testing xarray inputs
    def test_pearson_r_xr(self):
        a = self.ds.a
        b = self.ds.b
        result = pearson_r(a, b)
        assert np.allclose(self.unweighted_r, result)

    def test_pearson_r_xr_weighted(self):
        a = self.ds.a
        b = self.ds.b
        w = self.ds.weights
        result = pearson_r(a, b, weights=w)
        assert np.allclose(self.weighted_r, result)

    def test_pearson_r_xr_warn(self):
        a = self.ds.a
        b = self.ds.b
        self.assertWarns(Warning, pearson_r, a, b, dim='lat', axis=0)

    def test_pearson_r_xr_time(self):
        a = self.ds.a
        b = self.ds.b
        w = self.ds.weights[0,0,:]  # size of dim and weights must be the same
        result = pearson_r(a, b, weights=w, dim='time')
        assert np.allclose(self.weighted_r_time, result)