try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

#import distutils.sysconfig

with open("src/geocat/comp/version.py") as f:
    exec(f.read())

setup(name="geocat.comp",
      package_dir={
          '': 'src',
          'geocat': 'src/geocat',
          'geocat.comp': 'src/geocat/comp'
      },
      namespace_packages=['geocat'],
      packages=["geocat", "geocat.comp"],
      version=__version__,
      install_requires=['numpy', 'xarray', 'dask[complete]'])
