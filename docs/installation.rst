Installation
============

Installing GeoCAT-comp via Conda
--------------------------------

The easiest way to install GeoCAT-comp is using
`Conda <http://conda.pydata.org/docs/>`_::

    conda install -c conda-forge -c ncar geocat-comp

It is often preferable to create a separate conda "environment" in order to
avoid conflicts with other packages, for example::

    conda create -n geocat -c conda-forge -c ncar geocat-comp

where "geocat" is the name of a new conda environment, which can then be
activated using::

    conda activate geocat

Example code provided with this documentation occasionally makes use of other
software packages, such as Matplotlib, Cartopy, PyNGL, and Jupyter, which you
may wish to install into your geocat environment.  The following `conda create`
command can be used to create a new conda environment that includes some
additional commonly used Python packages pre-installed::

    conda create -n geocat -c conda-forge -c ncar geocat-comp pyngl matplotlib cartopy jupyter

Note that the Conda package manager automatically installs all `required`
dependencies, meaning it is not necessary to explicitly install Python, NumPy,
Xarray, or Dask when creating an envionment.  Although packages like Matplotlib
are often used with GeoCAT-comp, they are considered `optional` dependencies and
must be explicitly installed.


Required dependencies for building GeoCAT-comp
----------------------------------------------

    - Python 3.5+
    - numpy
    - xarray
    - cython
    - dask
    - `ncomp <http://github.com/NCAR/ncomp/>`_
