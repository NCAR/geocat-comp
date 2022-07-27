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
        time = xr.cftime_range(start='2022-07-25',
                               end='2022-07-26',
                               freq='D')
        lats = np.linspace(start=-90, stop=90, num=2, dtype='float32')
        lons = np.linspace(start=-180, stop=180, num=18, dtype='float32')

        # Create data variables
        np.random.seed(0)
        cls.a = np.random.random_sample((len(lats), len(lons)))
        cls.b = np.power(cls.a, 2)
        cls.weights = np.arange(1, 37).reshape(2, 18)
        cls.ds = xr.Dataset(data_vars={'a': (('lat', 'lon'), cls.a),
                                       'b': (('lat', 'lon'), cls.b),
                                       'weights': (('lat','lon'), cls.weights)},
                        coords={
                            'lat': lats,
                            'lon': lons
                        },
                        attrs={'description': 'Test data'})

        cls.unweighted_r = 0.960868163
        cls.weighted_r = 0.962039034

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

    def test_pearson_r_np_warn(self):
        a = self.a
        b = self.b
        self.assertWarns(Warning, pearson_r, a, b, dim='lat', axis=0)