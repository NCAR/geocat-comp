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

with open("src/geocat/comp/version.py") as f:
    exec(f.read())

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.path.normpath(sys.prefix)

include_dirs = [os.path.join(PREFIX, 'include'), numpy.get_include()]

extensions = [
    Extension("geocat.comp._ncomp", ["src/geocat/comp/_ncomp.pyx"],
        include_dirs=include_dirs,
        libraries=["ncomp"],
),
]
setup(
    name="geocat.comp",
    ext_modules=cythonize(extensions,
                          # help cythonize find my own .pxd files
                          include_path=[os.path.join(SRC_DIR, "src/geocat/comp/_ncomp")]),
    package_dir={'geocat': 'src/geocat', 'geocat.comp': 'src/geocat/comp'},
    package_data={'geocat': ['__init__.pxd', 'comp/*.pxd']},
    namespace_packages=['geocat'],
    packages=["geocat", "geocat.comp"],
    version=__version__, 
    install_requires=[
        'numpy', 
        'xarray', 
        'dask[complete]'
    ]
)
