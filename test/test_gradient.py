import sys
import unittest
import xarray as xr

# Import from directory structure if coverage test, or from installed
# packages otherwise
if "--cov" in str(sys.argv):
    pass
elif "-v" in str(sys.argv):
    pass
else:
    pass


class Test_Gradient(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data = xr.load_dataset('gradient_test_data.nc')
