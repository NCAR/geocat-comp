import sys
import metpy.calc as mpcalc
import numpy as np
import xarray as xr
import numpy.testing as nt
from metpy.units import units
import geocat.datafiles as gdf
import pandas as pd

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    from src.geocat.comp import get_skewt_vars, showalter_index
else:
    from geocat.comp import get_skewt_vars, showalter_index

out = xr.open_dataset("skewt_params_output.nc")
ds = pd.read_csv(gdf.get('ascii_files/sounding.testdata'),
                 delimiter='\\s+',
                 header=None)

# Extract the data from ds
p = ds[1].values * units.hPa  # Pressure [mb/hPa]
tc = (ds[5].values + 2) * units.degC  # Temperature [C]
tdc = ds[9].values * units.degC  # Dew pt temp  [C]
pro = mpcalc.parcel_profile(p, tc[0], tdc[0]).to('degC')

# Extract Showalter Index from NCL out file and convert to int
Shox = np.round(out['Shox'])  # Use np.round to avoid rounding issues
NCL_shox = int(Shox[0])  # Convert to int


def test_shox_vals():
    """Testing for Showalter Index only. While MetPy handles all five
    parameters, the Showalter Index was contributed to MetPy by the GeoCAT team
    because of the skewt_params function. Additionally, a discrepency between
    NCL and MetPy calculations of CAPE has been identified. After validating
    the CAPE value by hand using the method outlined in Hobbs 2006, it was
    determined that the MetPy calculation was closer to the CAPE value than the
    NCL calculation. To overcome any issues with validating the dataset, it was
    decided that skewt_params would only test against the Showalter Index for
    validation and not against all five paramters.

    Citation:
    Hobbs, P. V., and J. M. Wallace, 2006:
    Atmospheric Science: An Introductory Survey. 2nd ed. Academic Press,
    pg 345
    """

    # Showalter index
    shox = showalter_index(p, tc, tdc)
    shox = shox[0].magnitude

    # Place calculated values in iterable list
    vals = np.round(shox).astype(int)

    # Compare calculated values with expected
    nt.assert_equal(vals, NCL_shox)


def test_get_skewt_vars():
    """With resepct to the note in test_vars, the MetPy calculated values for
    Plcl, Tlcl, Pwat, and CAPE along with the tested value for Showalter Index
    are pre-defined in this test.

    This test is to ensure that the values of each are being read,
    assigned, and placed correctly in get_skewt_vars.
    """

    expected = 'Plcl= 927 Tlcl[C]= 24 Shox= 3 Pwat[cm]= 5 Cape[J]= 2958'
    result = get_skewt_vars(p, tc, tdc, pro)
    nt.assert_equal(result, expected)


test_shox_vals()
test_get_skewt_vars()
