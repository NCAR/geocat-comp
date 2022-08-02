from unittest import TestCase
import sys
import numpy as np
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
#if "--cov" in str(sys.argv):
from src.geocat.comp import pearson_r
#else:
#    from geocat.comp import pearson_r

class Test_pearson_r(TestCase):
    @classmethod
    def setUpClass(cls):
        # Coordinates
        times = xr.cftime_range(start='2022-08-01',
                               end='2022-08-05',
                               freq='D')
        lats = np.linspace(start=-45, stop=45, num=3, dtype='float32')
        lons = np.linspace(start=-180, stop=180, num=4, dtype='float32')

        # Create data variables
        x, y, z = np.meshgrid(lons, lats, times)
        np.random.seed(0)
        cls.a = np.random.random_sample((len(lats), len(lons), len(times)))
        cls.b = np.power(cls.a, 2)
        cls.weights = np.cos(np.deg2rad(y))
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
        cls.weighted_r = 0.963209755
        cls.weighted_r_lat = [[0.995454445, 0.998450821, 0.99863877, 0.978765291, 0.982350092],
                              [0.99999275, 0.995778831, 0.998994355, 0.991634937, 0.999868279],
                              [0.991344899, 0.998632079, 0.99801552, 0.968517489, 0.985215828],
                              [0.997034735, 0.99834464, 0.987382522, 0.99646236, 0.989222738]]

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

    def test_pearson_r_np_lat(self):
        a = self.a
        b = self.b
        w = self.weights
        result = pearson_r(a, b, weights=w, axis=0)
        assert np.allclose(self.weighted_r_lat, result)

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

    def test_pearson_r_xr_lat(self):
        a = self.ds.a
        b = self.ds.b
        w = self.ds.weights[:,0,0]
        result = pearson_r(a, b, weights=w, dim='lat')
        assert np.allclose(self.weighted_r_lat, result)

    def test_keep_attrs(self):
        a = self.ds.a
        b = self.ds.b
        a.attrs.update({'Description' : 'Test Data'})
        b.attrs.update({'2nd Description' : 'Dummy Data'})
        result = pearson_r(a, b, keep_attrs=True)
        assert result.attrs == a.attrs