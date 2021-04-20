| CI           | [![GitHub Workflow Status][github-ci-badge]][github-ci-link] [![GitHub Workflow Status][github-conda-build-badge]][github-conda-build-link] [![Code Coverage Status][codecov-badge]][codecov-link] |
| :----------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **Docs**     |                                                                    [![Documentation Status][rtd-badge]][rtd-link]                                                                    |
| **Package**  |                                                         [![Conda][conda-badge]][conda-link] [![PyPI][pypi-badge]][pypi-link]                                                         |
| **License**  |                                                                        [![License][license-badge]][repo-link]                                                                        |
| **Citing**  |                                                                              [![DOI][doi-badge]][doi-link]                                                                            |



GeoCAT-comp is both the whole computational component of the [GeoCAT](https://geocat.ucar.edu/)
project and a single Github repository as described here. As the computational component of
[GeoCAT](https://geocat.ucar.edu/), GeoCAT-comp provides implementations of computational functions for operating
on geosciences data. Many of these functions originated in NCL and were translated into Python with the help of GeoCAT-comp;
however, developers are welcome to come up with novel computational functions for geosciences data.

Many of the computational functions in GeoCAT are implemented in a pure Python fashion. However,
there are some others that are implemented in Fortran but wrapped up in Python. To facilitate
contribution, the whole GeoCAT-comp structure is split into two repositories with respect to
being pure-Python or Python with compiled codes (i.e. Fortran) implementations. Such implementation
layers are handled within [GeoCAT-comp](https://github.com/NCAR/geocat-comp) and
[GeoCAT-f2py](https://github.com/NCAR/geocat-f2py) repositories, respectively (The
[GeoCAT-f2py](https://github.com/NCAR/geocat-f2py) repo is documented on its own).


# GeoCAT-comp

GeoCAT-comp repo does not explicitly contain or require any compiled code, making it more
accessible to the general Python community at large. However, if
[GeoCAT-f2py](https://github.com/NCAR/geocat-f2py) is installed, then all functions contained in
the "geocat.f2py" package are imported seamlessly into the "geocat.comp" namespace. Thus,
GeoCAT-comp repo serves as a user API to access the entire computational toolkit even though the
repo itself only contains pure Python code from the contributor’s perspective. Whenever prospective
contributors want to add new computational functionality implemented as pure Python, GeoCAT-comp
is the repo to do so. Therefore, there is no onus on contributors of pure python code to
build/compile/test any compiled code (i.e. Fortran) at GeoCAT-comp level.


# Documentation

[GeoCAT Homepage](https://geocat.ucar.edu/)

[GeoCAT Contributor's Guide](https://geocat.ucar.edu/pages/contributing.html)

[GeoCAT-comp documentation on Read the Docs](https://geocat-comp.readthedocs.io)


# Installation and build instructions

Please see our documentation for
[installation and build instructions](https://github.com/NCAR/geocat-comp/blob/main/INSTALLATION.md).


# Xarray interface vs NumPy interface

GeoCAT-comp provides a high-level [Xarray](http://xarray.pydata.org/en/stable/) interface under the
`geocat.comp` namespace. However, a stripped-down NumPy interface is used under the hood to bridge
the gap between NumPy arrays and the compiled language data structures used by
[GeoCAT-f2py](https://github.com/NCAR/geocat-f2py). These functions are accessible under the
`geocat.comp` namespace, but are minimally documented and are
intended primarily for internal use.

# Citing GeoCAT-comp

Cite GeoCAT-comp using the following text:

<> Visualization & Analysis Systems Technologies. (Year).
Geoscience Community Analysis Toolkit (GeoCAT-comp version \<version\>) [Software].
Boulder, CO: UCAR/NCAR - Computational and Informational System Lab. doi:10.5065/A8PP-4358.

Update the GeoCAT-comp version and year as appropriate. For example:

<> Visualization & Analysis Systems Technologies. (2021).
Geoscience Community Analysis Toolkit (GeoCAT-comp version 2021.04.0) [Software].
Boulder, CO: UCAR/NCAR - Computational and Informational System Lab. doi:10.5065/A8PP-4358.

For further information, please refer to
[GeoCAT homepage - Citation](https://geocat.ucar.edu/pages/citation.html).





[github-ci-badge]: https://img.shields.io/github/workflow/status/NCAR/geocat-comp/CI?label=CI&logo=github&style=for-the-badge
[github-conda-build-badge]: https://img.shields.io/github/workflow/status/NCAR/geocat-comp/build_test?label=conda-builds&logo=github&style=for-the-badge
[github-ci-link]: https://github.com/NCAR/geocat-comp/actions?query=workflow%3ACI
[github-conda-build-link]: https://github.com/NCAR/geocat-comp/actions?query=workflow%3Abuild_test
[codecov-badge]: https://img.shields.io/codecov/c/github/NCAR/geocat-comp.svg?logo=codecov&style=for-the-badge
[codecov-link]: https://codecov.io/gh/NCAR/geocat-comp
[rtd-badge]: https://img.shields.io/readthedocs/geocat-comp/latest.svg?style=for-the-badge
[rtd-link]: https://geocat-comp.readthedocs.io/en/latest/?badge=latest
[pypi-badge]: https://img.shields.io/pypi/v/geocat-comp?logo=pypi&style=for-the-badge
[pypi-link]: https://pypi.org/project/geocat-comp
[conda-badge]: https://img.shields.io/conda/vn/ncar/geocat-comp?logo=anaconda&style=for-the-badge
[conda-link]: https://anaconda.org/ncar/geocat-comp
[license-badge]: https://img.shields.io/github/license/NCAR/geocat-comp?style=for-the-badge
[doi-badge]: https://img.shields.io/badge/DOI-10.5065%2Fa8pp--4358-brightgreen?style=for-the-badge
[doi-link]: https://doi.org/10.5065/a8pp-4358
[repo-link]: https://github.com/NCAR/geocat-comp
