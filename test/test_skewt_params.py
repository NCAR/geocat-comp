import sys

import metpy.calc as mpcalc
import numpy as np
import numpy.testing as nt
from metpy.units import units

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp import get_skewt_vars, showalter_index
else:
    from geocat.comp import get_skewt_vars, showalter_index

p_upper = np.arange(1000, 200, -50) * units.hPa
p_lower = np.arange(175, 0, -25) * units.hPa
p = np.append(p_upper, p_lower)  # Pressure levels in hPa
tc = np.linspace(30, -30, 23) * units.degC  # Env temp in degC
tdc = np.linspace(10, -30, 23) * units.degC  # DewPt temp in degC
pro = mpcalc.parcel_profile(p, tc[0], tdc[0])  # Parcel Profile


def test_showalter_index():

    result = showalter_index(p, tc, tdc)
    expected = 21.353962321924012 * units.delta_degree_Celsius

    nt.assert_almost_equal(result, expected, 4)


def test_get_skewt_vars():

    result = get_skewt_vars(p, tc, tdc, pro)
    expected = 'Plcl= 747 Tlcl[C]= 6 Shox= 21 Pwat[cm]= 5 Cape[J]= 0'

    nt.assert_almost_equal(result, expected, 4)
