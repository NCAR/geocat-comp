geocat-comp
===========

GeoCAT-comp is the computational component of the [GeoCAT](https://ncar.github.io/GeoCAT) project. GeoCAT-comp wraps NCL's non-WRF Fortran routines into Python.

GeoCAT-comp depends on a separate C/Fortran library called "[ncomp](https://github.com/NCAR/ncomp)", which contains these Fortran routines.


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
