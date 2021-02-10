import unittest
import pytest
import numpy as np
import xarray as xr
import dask.array as da
import dask.distributed as dd
from geocat.comp.dewtemp import dewtemp

t_def = [
    29.3, 28.1, 23.5, 20.9, 18.4, 15.9, 13.1, 10.1, 6.7, 3.1, -0.5, -4.5, -9.0,
    -14.8, -21.5, -29.7, -40.0, -52.4
]

rh_def = [
    75.0, 60.0, 61.1, 76.7, 90.5, 89.8, 78.3, 76.5, 46.0, 55.0, 63.8, 53.2,
    42.9, 41.7, 51.0, 70.6, 50.0, 50.0
]

dt_1 = 6.3

dt_2 = [
    24.38342, 19.55563, 15.53281, 16.64218, 16.81433, 14.22482, 9.401337,
    6.149719, -4.1604, -5.096619, -6.528168, -12.61957, -19.38332, -25.00714,
    -28.9841, -33.34853, -46.51273, -58.18289
]


class Test_dewtemp(unittest.TestCase):

    def test_float_input(self):
        tk = 18. + 273.15
        rh = 46.5

        assert dewtemp(tk, rh) - 273.15 == pytest.approx(dt_1, 0.1)

    def test_list_input(self):
        tk = (np.asarray(t_def) + 273.15).tolist()

        assert dewtemp(tk, rh_def) - 273.15 == pytest.approx(dt_2, 0.1)

    def test_numpy_input(self):
        tk = np.asarray(t_def) + 273.15
        rh = np.asarray(rh_def)

        assert dewtemp(tk, rh) - 273.15 == pytest.approx(dt_2, 0.1)

    def test_xarray_input(self):
        tk = xr.DataArray(np.asarray(t_def) + 273.15)
        rh = xr.DataArray(rh_def)

        assert dewtemp(tk, rh) - 273.15 == pytest.approx(dt_2, 0.1)

    def test_dask_unchunked_input(self):
        tk = da.from_array(np.asarray(t_def) + 273.15)
        rh = da.from_array(rh_def)

        # Start dask cluster
        cluster = dd.LocalCluster(n_workers=3, threads_per_worker=2)
        print(cluster.dashboard_link)
        client = dd.Client(cluster)

        assert np.allclose(dewtemp(tk, rh) - 273.15, dt_2, atol=0.1)

        client.close()

    def test_dask_chunked_input(self):
        tk = da.from_array(np.asarray(t_def) + 273.15, chunks="auto")
        rh = da.from_array(rh_def, chunks="auto")

        # Start dask cluster
        cluster = dd.LocalCluster(n_workers=3, threads_per_worker=2)
        print(cluster.dashboard_link)
        client = dd.Client(cluster)

        assert np.allclose(dewtemp(tk, rh) - 273.15, dt_2, atol=0.1)

        client.close()


a = Test_dewtemp()
a.test_dask_unchunked_input()
