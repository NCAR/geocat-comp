#''' setup.py is needed, but only to make namespaces happen
from pathlib import Path

from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().strip().split('\n')


#''' moved into function, can now be used other places
def version():
    for line in open('meta.yaml').readlines():
        index = line.find('set version')
        if index > -1:
            return line[index + 15:].replace('\" %}', '').strip()


setup(
    name='geocat.comp',
    version=version(),
    maintainer='GeoCAT',
    maintainer_email='geocat@ucar.edu',
    python_requires='>=3.7',
    install_requires=requirements,
    description=
    """GeoCAT-comp is computational component of the GeoCAT project and provides
    implementations of computational functions for analysis of geosciences data""",
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
    ],
    include_package_data=True,
    package_dir={
        '': 'src',
        'geocat': 'src/geocat',
        'geocat.comp': 'src/geocat/comp'
    },
    namespace_packages=['geocat'],
    packages=['geocat', 'geocat.comp'],
    url='https://github.com/NCAR/geocat-comp',
    project_urls={
        'Documentation': 'https://geocat-comp.readthedocs.io',
        'Source': 'https://github.com/NCAR/geocat-comp',
        'Tracker': 'https://github.com/NCAR/geocat-comp/issues',
    },
    zip_safe=False)
