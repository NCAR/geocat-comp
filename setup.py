#''' setup.py is needed, but only to make namespaces happen,
version = '2021.01.1'

from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="geocat.comp",
    version=version,
    package_dir={
      '': 'src',
      'geocat': 'src/geocat',
      'geocat.comp': 'src/geocat/comp'
    },
    namespace_packages=['geocat'],
    packages=["geocat", "geocat.comp"],
)
