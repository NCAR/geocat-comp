# Installation

This installation guide includes only the GeoCAT-comp installation and build instructions.
Please refer to [GeoCAT Contributor's Guide](https://geocat.ucar.edu/pages/contributing.html) for installation of
the whole GeoCAT project.


## Installing GeoCAT-comp via Conda

The easiest way to install GeoCAT-comp is using [Conda](http://conda.pydata.org/docs/):

    conda create -n geocat -c conda-forge -c ncar geocat-comp

where "geocat" is the name of a new conda environment, which can then be
activated using:

    conda activate geocat

If you somewhat need to make use of other software packages, such as Matplotlib,
Cartopy, Jupyter, etc. with GeoCAT-comp, you may wish to install into your `geocat`
environment.  The following `conda create` command can be used to create a new
`conda` environment that includes some of these additional commonly used Python
packages pre-installed:

    conda create -n geocat -c conda-forge -c ncar geocat-comp matplotlib cartopy jupyter

Alternatively, if you already created a conda environment using the first
command (without the extra packages), you can activate and install the packages
in an existing environment with the following commands:

    conda activate geocat   # or whatever your environment is called
    conda install -c conda-forge matplotlib cartopy jupyter

Please note that the use of the `conda-forge` channel is essential to guarantee
compatibility between dependency packages.

Also, note that the Conda package manager automatically installs all *required*
dependencies of GeoCAT-comp, meaning it is not necessary to explicitly install
Python, NumPy, Xarray, or Dask when creating an environment and installing GeoCAT-comp.
Although packages like Matplotlib are often used with GeoCAT-comp, they are considered
*optional* dependencies and must be explicitly installed.

If you are interested in learning more about how Conda environments work, please visit
the [managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
page of the Conda documentation.


## Building GeoCAT-comp from source

Building GeoCAT-comp from source code is a fairly straightforward task, but
doing so should not be necessary for most users. If you *are* interested in
building GeoCAT-comp from source, you will need the following packages to be
installed.

### Required dependencies for building and testing GeoCAT-comp

- Python 3.7+
- [GeoCAT-datafiles](https://github.com/NCAR/geocat-datafiles)  (For tests only)
- [GeoCAT-f2py](https://github.com/NCAR/geocat-f2py)
- [cf_xarray](https://cf-xarray.readthedocs.io/en/latest/)
- [cftime](https://unidata.github.io/cftime/)
- [eofs](https://ajdawson.github.io/eofs/latest/index.html)
- [dask](https://dask.org/)
- [distributed](https://distributed.readthedocs.io/en/latest/)
- [netcdf4](https://unidata.github.io/netcdf4-python/)  (For tests only)
- [numpy](https://numpy.org/doc/stable/)
- [pytest](https://docs.pytest.org/en/stable/)  (For tests only)
- [xarray](http://xarray.pydata.org/en/stable/)

Note: [GeoCAT-f2py](https://github.com/NCAR/geocat-f2py) dependency will automatically
install further dependencies for compiled language implementation.

### How to create a Conda environment for building GeoCAT-comp

The GeoCAT-comp source code includes a conda environment definition file in
the `/build_envs` folder under the root directory that can be used to create a
development environment containing all of the packages required to build GeoCAT-comp.
The file `environment.yml` is intended to be used on Linux systems and macOS.
The following commands should work on both Linux and macOS:

    conda env create -f build_envs/environment.yml
    conda activate geocat_comp_build

### Installing GeoCAT-comp

Once the dependencies listed above are installed, you can install GeoCAT-comp
with running the following command from the root-directory:

    pip install .

For compatibility purposes, we strongly recommend using Conda to
configure your build environment as described above.


### Testing a GeoCAT-comp build

A GeoCAT-comp build can be tested from the root directory of the source code
repository using the following command (Explicit installation of the
[pytest](https://docs.pytest.org/en/stable/) package may be required, please
see above):

    pytest test
