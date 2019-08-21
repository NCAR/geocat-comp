try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

#import distutils.sysconfig
from Cython.Build import cythonize
import numpy
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    PREFIX = os.path.normpath(os.environ["NCOMP_SRC"])
    NCOMP_INC = os.path.join(PREFIX, 'src')
    NCOMP_LIB = os.path.join(PREFIX, 'src', '.libs')
except KeyError:
    PREFIX = sys.prefix
    NCOMP_INC = os.path.join(PREFIX, 'include')
    NCOMP_LIB = os.path.join(PREFIX, 'lib')

include_dirs = [NCOMP_INC, numpy.get_include()]
library_dirs = [NCOMP_LIB]

print(include_dirs)
print(library_dirs)

extensions = [
    Extension("geocat.comp._ncomp", ["src/geocat/comp/_ncomp/ncomp.pyx"],
        include_dirs=include_dirs,
        libraries=["ncomp"],
),
]
setup(
    name="geocat.comp",
    ext_modules=cythonize(extensions,
                          # help cythonize find my own .pxd files
                          include_path=[os.path.join(SRC_DIR, "src/geocat/comp/_ncomp")]),
    package_dir={'': 'src'},
    namespace_packages=['geocat'],
    packages=["geocat.comp"],
    version='0.1a'
)
