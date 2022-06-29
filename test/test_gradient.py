import sys
import unittest

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
        return garbage
