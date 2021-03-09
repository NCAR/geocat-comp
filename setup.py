#''' setup.py is needed, but only to make namespaces happen
from pathlib import Path

from setuptools import find_packages, setup


#''' moved into function, can now be used other places
def version():
    for line in open('meta.yaml').readlines():
        index = line.find('set version')
        if index > -1:
            return line[index + 15:].replace('\" %}', '').strip()


setup(
    name='geocat.comp',
    version=version(),
    package_dir={
        '': 'src',
        'geocat': 'src/geocat',
        'geocat.comp': 'src/geocat/comp'
    },
    namespace_packages=['geocat'],
    packages=['geocat', 'geocat.comp'],
)
