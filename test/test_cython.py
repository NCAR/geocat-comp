import numpy
import os
import sys
import pyximport;

PREFIX = os.path.normpath(sys.prefix)
include_dirs = [os.path.join(PREFIX, 'include'), numpy.get_include()]

os.environ['CFLAGS'] = "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION {}".format(os.environ['CFLAGS']) if 'CFLAGS' in os.environ else "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"

pyximport.install(setup_args={"script_args": ["--force", "--verbose"],
                              "include_dirs": include_dirs,
                              "verbose": True},
                  language_level=3)

from cython_tests.test_cython import *
