from abc import ABCMeta
from unittest import TestCase
import numpy as np
from dask.array.tests.test_xarray import xr

import geocat.comp
from geocat.comp._ncomp import _moc_globe_atl


class BaseTestClass(metaclass=ABCMeta):

    # Dimensions
    _nyaux = 3            # nyaux = lat_aux_grid->shape[0]
    _kdep = 4             # kdep  = a_wvel/a_bolus/a_submeso->shape[0]
    _nlat = 5             # nlat  = a_wvel/a_bolus/a_submeso->shape[1] AND tlat->shape[0] AND rmlak->shape[1]
    _mlon = 6             # mlon  = a_wvel/a_bolus/a_submeso->shape[2] AND tlat->shape[1] AND rmlak->shape[2]

    _kdepnyaux2   = 2 * _kdep * _nyaux
    _nlatmlon     = _nlat * _mlon
    _kdepnlatmlon = _kdep * _nlatmlon

    # Generate arbitrary data
    tmp_a_double = np.linspace(1, _kdepnlatmlon, num = _kdepnlatmlon, dtype = np.double).reshape((_kdep, _nlat, _mlon))

    tmp_a_float64 = np.linspace(1, _kdepnlatmlon, num = _kdepnlatmlon, dtype = np.float64).reshape((_kdep, _nlat, _mlon))

    tmp_a_float32 = np.linspace(1, _kdepnlatmlon, num = _kdepnlatmlon, dtype = np.float32).reshape((_kdep, _nlat, _mlon))

    tmp_a_int64 = np.linspace(1, _kdepnlatmlon, num = _kdepnlatmlon, dtype = np.int64).reshape((_kdep, _nlat, _mlon))

    tmp_a_double_msg_99 = tmp_a_double
    tmp_a_double_msg_99[1,1,1] = -99
    tmp_a_double_msg_99[1,2,3] = -99

    tmp_a_double_msg_nan = tmp_a_double_msg_99
    tmp_a_double_msg_nan[tmp_a_double_msg_nan == -99] = np.nan

    # Generate test data
    # _lat_aux_grid = np.asarray([5, 10, 15], dtype='double').reshape((1,3))
    _lat_aux_grid = np.asarray([5, 10, 15], dtype='double')

    _rmlak = np.ones((2, _nlat, _mlon), dtype = np.int64)

    _t_lat = np.linspace(1, _nlatmlon, num = _nlatmlon, dtype = np.double).reshape((_nlat, _mlon))

    # Initialize _a type data
    _a_wvel = []
    _a_bolus = []
    _a_submeso = []

    _a_wvel.append(tmp_a_double)
    _a_wvel.append(tmp_a_float64)
    _a_wvel.append(tmp_a_float32)
    _a_wvel.append(tmp_a_int64)
    _a_wvel.append(tmp_a_double_msg_99)
    _a_wvel.append(tmp_a_double_msg_nan)

    _a_bolus.append(tmp_a_double)
    _a_bolus.append(tmp_a_float64)
    _a_bolus.append(tmp_a_float32)
    _a_bolus.append(tmp_a_int64)
    _a_bolus.append(tmp_a_double_msg_99)
    _a_bolus.append(tmp_a_double_msg_nan)

    _a_submeso.append(tmp_a_double)
    _a_submeso.append(tmp_a_float64)
    _a_submeso.append(tmp_a_float32)
    _a_submeso.append(tmp_a_int64)
    _a_submeso.append(tmp_a_double_msg_99)
    _a_submeso.append(tmp_a_double_msg_nan)


class Test_Moc_Globe_Atl(TestCase, BaseTestClass):
    def test_moc_globe_atl_01(self):

        out_arr = geocat.comp.moc_globe_atl(self._lat_aux_grid,
                                  self._a_wvel[0],
                                  self._a_bolus[0],
                                  self._a_submeso[0],
                                  self._t_lat,
                                  self._rmlak)

        self.assertEqual((3, 2, self._kdep, self._nyaux), out_arr.shape)

        #for e in np.nditer(result):
        #    self.assertAlmostEqual(0.25, e, 2)
