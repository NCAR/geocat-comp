geocat-comp
===========

GeoCAT-comp is the computational component of the [GeoCAT](https://ncar.github.io/GeoCAT) project. GeoCAT-comp wraps NCL's non-WRF Fortran routines into Python.

Many of the computational functions under GeoCAT-comp component are implemented in Fortran (or possibly C). However, others can be implemented in a pure Python fashion. To facilitate contribution, the whole GeoCAT-comp structure is split into repositories with respect to being pure-Python, Python with Cython wrappers for compiled codes, and compiled language (C and Fortran) implementations. Such implementation layers are handled within GeoCAT-comp (this repository), [GeoCAT-ncomp](https://github.com/NCAR/geocat-ncomp), and [libncomp](https://github.com/NCAR/libncomp) repositories, respectively (GeoCAT-ncomp and libncomp repos are documented on their own).

This repo, GeoCAT-comp, does not explicitly contain or require any compiled code, making it more accessible to the general Python community at large. However, if [GeoCAT-ncomp](https://github.com/NCAR/geocat-ncomp) is installed, then all functions contained in the “geocat.ncomp” module are imported into the “geocat.comp” namespace. Thus, GeoCAT-comp repo serves as a user API to access the entire computational toolkit even though the repo itself only contains pure Python code from the contributor’s perspective. Whenever prospective contributors want to add new computational functionality implemented as pure Python, GeoCAT-comp is the repo to do so. Therefore, there is no onus on contributors of pure python code to build/compile/test any compiled code (Cython, C, C++, Fortran) at GeoCAT-comp level.


Documentation
=============

[GeoCAT Homepage](https://geocat.ucar.edu/)

[GeoCAT Contributor's Guide](https://geocat.ucar.edu/pages/contributing.html)

[GeoCAT-comp documentation on Read the Docs](https://geocat-comp.readthedocs.io)


Installation and build instructions
===================================

Please see our documentation for [installation and build instructions](https://github.com/NCAR/geocat-comp/INSTALLATION.md).


Xarray interface vs NumPy interface
===================================

GeoCAT-comp provides a high-level Xarray interface under the `geocat.comp` namespace. However, a stripped-down NumPy interface is used under the hood to bridge the gap between NumPy arrays and the C data structures used by `libncomp`. These functions are accessible under the `geocat.comp._ncomp` namespace, but are minimally documented and are intended primarily for internal use.
