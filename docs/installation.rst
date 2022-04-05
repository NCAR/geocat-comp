Installation
============

This installation guide includes only the GeoCAT-comp installation and build instructions.
Please refer to `GeoCAT Contributor's Guide <https://geocat.ucar.edu/pages/contributing.html>`_ for installation of
the whole GeoCAT project.

Installing GeoCAT-comp via Conda
--------------------------------

The easiest way to install GeoCAT-comp is using
`Conda <http://conda.pydata.org/docs/>`_::

    conda create -n geocat -c conda-forge -c ncar geocat-comp

where "geocat" is the name of a new conda environment, which can then be
activated using::

    conda activate geocat

If you somewhat need to make use of other software packages, such as Matplotlib,
Cartopy, Jupyter, etc. with GeoCAT-comp, you may wish to install into your :code:`geocat`
environment.  The following :code:`conda create` command can be used to create a new
:code:`conda` environment that includes some of these additional commonly used Python
packages pre-installed::

    conda create -n geocat -c conda-forge -c ncar geocat-comp matplotlib cartopy jupyter

Alternatively, if you already created a conda environment using the first
command (without the extra packages), you can activate and install the packages
in an existing environment with the following commands::

    conda activate geocat # or whatever your environment is called
    conda install -c conda-forge matplotlib cartopy jupyter

Please note that the use of the :code:`conda-forge` channel is essential to guarantee
compatibility between dependency packages.

Also, note that the Conda package manager automatically installs all `required`
dependencies of GeoCAT-comp, meaning it is not necessary to explicitly install
Python, NumPy, Xarray, or Dask when creating an envionment and installing GeoCAT-comp.
Although packages like Matplotlib are often used with GeoCAT-comp, they are considered
`optional` dependencies and must be explicitly installed.

If you are interested in learning more about how Conda environments work, please
visit the `managing environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
page of the Conda documentation.

Installing GeoCAT-comp in a Pre-existing Conda Environment
----------------------------------------------------------

If you started a project and later decided to use GeoCAT-comp, you will need to install it in your pre-existing environment.

1.  Make sure your conda is up to date by running this command from the
    terminal::

    conda update conda

2.  Activate the conda environment you want to add GeoCAT to. In this example, the environment is called :code:`foo`::

    conda activate foo

3. Install geocat-comp::

    conda install -c ncar -c conda-forge geocat-comp

Updating GeoCAT-comp via Conda
-------------------------------

It is important to keep your version of :code:`geocat-comp` up to date. This can be done as follows:

1.  Make sure your Conda is up to date by running this command from the terminal::

    conda update conda

2.  Activate the conda environment you want to update. In this example, the environment is called :code:`geocat`::

    conda activate geocat

3. Update :code:`geocat-comp`::

    conda update geocat-comp

Building GeoCAT-comp from source
--------------------------------

Building GeoCAT-comp from source code is a fairly straightforward task, but
doing so should not be necessary for most users. If you `are` interested in
building GeoCAT-comp from source, you will need the following packages to be
installed.

Required dependencies for building and testing GeoCAT-comp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    - Python 3.8+
    - `GeoCAT-datafiles <https://github.com/NCAR/geocat-datafiles>`_  (For tests only)
    - `GeoCAT-f2py <https://github.com/NCAR/geocat-f2py>`_
    - `cf_xarray <https://cf-xarray.readthedocs.io/en/latest/>`_
    - `cftime <https://unidata.github.io/cftime/>`_
    - `eofs <https://ajdawson.github.io/eofs/latest/index.html>`_
    - `dask <https://dask.org/>`_
    - `distributed <https://distributed.readthedocs.io/en/latest/>`_
    - `netcdf4 <https://unidata.github.io/netcdf4-python/>`_  (For tests only)
    - `numpy <https://numpy.org/doc/stable/>`_
    - `pytest <https://docs.pytest.org/en/stable/>`_  (For tests only)
    - `xarray <http://xarray.pydata.org/en/stable/>`_

Note: `GeoCAT-f2py <https://github.com/NCAR/geocat-f2py>`_ dependency will automatically
install further dependencies for compiled language implementation.


How to create a Conda environment for building GeoCAT-comp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GeoCAT-comp source code includes a conda environment definition file in
the :code:`/build_envs` folder under the root directory that can be used to create a
development environment containing all of the packages required to build GeoCAT-comp.
The file :code:`environment.yml` is intended to be used on Linux systems and macOS.
The following commands should work on both Linux and macOS::

    conda env create -f build_envs/environment.yml
    conda activate geocat_comp_build


Installing GeoCAT-comp
^^^^^^^^^^^^^^^^^^^^^^

Once the dependencies listed above are installed, you can install GeoCAT-comp
with running the following command from the root-directory::

    pip install .

For compatibility purposes, we strongly recommend using Conda to
configure your build environment as described above.


Testing a GeoCAT-comp build
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A GeoCAT-comp build can be tested from the root directory of the source code
repository using the following command (Explicit installation of the
`pytest <https://docs.pytest.org/en/stable/>`_ package may be required, please
see above)::

    pytest test
