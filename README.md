geocat-comp
===========

GeoCAT-comp is the computational component of the [GeoCAT](https://ncar.github.io/GeoCAT) project. GeoCAT-comp wraps NCL's non-WRF Fortran routines into Python.

GeoCAT-comp depends on a separate C/Fortran library called "[ncomp](https://github.com/NCAR/ncomp)", which contains these Fortran routines.


Documentation
=============

[GeoCAT-comp documentation on Read the Docs](https://geocat-comp.readthedocs.io)



Build instructions
==================

GeoCAT-comp requires the following dependencies to be installed:

* Python
* Cython
* Numpy
* Xarray
* Dask
* Any C compiler (GCC and Clang have been tested)
* gfortran
* [ncomp](https://github.com/NCAR/ncomp) 

GeoCAT-comp can be built by running one of the following commands from the root directory of this repository:
```
python setup.py install --prefix=$PREFIX
```
or
```
pip install --prefix $PREFIX .
```

where $PREFIX is the path that `ncomp` is installed.


Xarray interface vs NumPy interface
===================================

GeoCAT-comp provides a high-level Xarray interface under the `geocat.comp` namespace. However, a stripped-down NumPy interface is used under the hood to bridge the gap between NumPy arrays and the C data structures used by `NComp`. These functions are accessible under the `geocat.comp._ncomp` namespace, but are minimally documented and are intended primarily for internal use.
