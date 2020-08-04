# Installation

## Installing GeoCAT-comp via Conda

The easiest way to install GeoCAT-comp is using
[Conda](http://conda.pydata.org/docs/):

    conda create -n geocat -c conda-forge -c ncar geocat-comp

where "geocat" is the name of a new conda environment, which can then be
activated using:

    conda activate geocat

Example code provided with this documentation frequently makes use of other
software packages, such as Matplotlib, Cartopy, PyNGL, and Jupyter, which you
may wish to install into your geocat environment.  The following `conda create`
command can be used to create a new conda environment that includes some of
these additional commonly used Python packages pre-installed::

    conda create -n geocat -c conda-forge -c ncar geocat-comp pyngl matplotlib cartopy jupyter

Alternatively, if you already created a conda environment using the first
command (without the extra packages), you can activate and install the packages
in an existing environment with the following commands::

    conda activate geocat # or whatever your environment is called
    conda install -c conda-forge pyngl matplotlib cartopy jupyter

Please note that the use of the **conda-forge** channel is essential to guarantee
compatibility between dependency packages.

Also, note that the Conda package manager automatically installs all `required`
dependencies, meaning it is not necessary to explicitly install Python, NumPy,
Xarray, or Dask when creating an envionment.  Although packages like Matplotlib
are often used with GeoCAT-comp, they are considered `optional` dependencies and
must be explicitly installed.

If you are interested in learning more about how Conda environments work, please
visit the [managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) 
page of the Conda documentation.


## Building GeoCAT-comp from source

Building GeoCAT-comp from source code is a fairly straightforward task, but
doing so should not be necessary for most users. If you are interested in
building GeoCAT-comp from source, you will need the following packages to be
installed.

### Required dependencies for building GeoCAT-comp

- Python 3.6+
- numpy
- xarray
- dask
- distributed
- pytest
- [GeoCAT-ncomp](http://github.com/NCAR/geocat-ncomp/)
    
Note: [GeoCAT-ncomp](http://github.com/NCAR/geocat-ncomp/) will handle further dependencies for compiled language implementation.

### How to create a Conda environment for building GeoCAT-comp

The GeoCAT-comp source code includes two Conda environment definition files in
the `/build_envs` directory that can be used to create a development environment
containing all of the packages required to build GeoCAT-comp.  The file
`environment_Linux.yml` is intended to be used on Linux systems, while
`environment_Darwin.yml` should be used on macOS.  It is necessary to have
separate `environment_*.yml` files because Linux and macOS use different C
compilers, although the following commands should work on both Linux and macOS:

    conda env create -f build_envs/environment_$(uname).yml
    conda activate geocat_build


### Installing GeoCAT-comp
 
Once the dependencies listed above are installed, you can install GeoCAT-comp
with running the following command from the root-directory:

    pip install .

If you are using a conda environment as described above, this command should
work as-is. However, if you have chosen to use a different Python binary and
have installed dependencies elsewhere, you may need to set certain environment
variables (CFLAGS, CPPFLAGS, or LDFLAGS) in order for the setup.py script to
find all of the necessary dependency packages.  Due to the potentially
complicated nature of the build process, we strongly recommend using Conda to
configure your build environment.


### Testing a GeoCAT-comp build

A GeoCAT-comp build can be tested from the root directory of the source code
repository using the following command:

    pytest test
