.. currentmodule:: geocat.comp

.. _installation:

Installation
============

This installation guide includes only the GeoCAT-comp installation and build instructions.

Please refer to the relevant project documentation for how to install other GeoCAT packages.

Installing GeoCAT-comp via Conda in a New Environment
-----------------------------------------------------

The easiest way to install GeoCAT-comp is using
`Conda <https://docs.conda.io/projects/conda/en/latest/>`__::

    conda create -n geocat -c conda-forge geocat-comp

where "geocat" is the name of a new conda environment, which can then be
activated using::

    conda activate geocat

If you somewhat need to make use of other software packages, such as Matplotlib,
Cartopy, Jupyter, etc. with GeoCAT-comp, you may wish to install into your ``geocat``
environment.  The following ``conda create`` command can be used to create a new
``conda`` environment that includes some of these additional commonly used Python
packages pre-installed::

    conda create -n geocat -c conda-forge geocat-comp matplotlib cartopy jupyter

Alternatively, if you already created a conda environment using the first
command (without the extra packages), you can activate and install the packages
in an existing environment with the following commands::

    conda activate geocat # or whatever your environment is called
    conda install -c conda-forge matplotlib cartopy jupyter

Please note that the use of the ``conda-forge`` channel is essential to guarantee
compatibility between dependency packages.

Also, note that the Conda package manager automatically installs all required
dependencies of GeoCAT-comp, meaning it is not necessary to explicitly install
Python, NumPy, Xarray, or Dask when creating an environment and installing GeoCAT-comp.
Although packages like Matplotlib are often used with GeoCAT-comp, they are considered
`optional` dependencies and must be explicitly installed.

If you are interested in learning more about how Conda environments work, please
visit the `managing environments <https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html>`__
page of the Conda documentation.

Installing GeoCAT-comp in a Pre-existing Conda Environment
----------------------------------------------------------

If you started a project and later decided to use GeoCAT-comp, you will need to install it in your pre-existing environment.

1.  Make sure your conda is up to date by running this command from the
    terminal::

        conda update conda

2.  Activate the conda environment you want to add GeoCAT to. In this example, the environment is called ``foo``::

        conda activate foo

3. Install geocat-comp::

    conda install -c conda-forge geocat-comp


Updating GeoCAT-comp via Conda
-------------------------------

It is important to keep your version of ``geocat-comp`` up to date. This can be done as follows:

1.  Make sure your Conda is up to date by running this command from the terminal::

        conda update conda

2.  Activate the conda environment you want to update. In this example, the environment is called ``geocat``::

        conda activate geocat

3. Update ``geocat-comp``::

        conda update geocat-comp


Installing GeoCAT-comp via PyPi
-------------------------------
GeoCAT-comp is distributed also in PyPI; therefore, the above Conda installation instructions should, in theory,
apply to PyPI installation through using ``pip install`` commands instead of ``conda install`` wherever they occur.

Building GeoCAT-comp from source
--------------------------------

Building GeoCAT-comp from source code is a fairly straightforward task, but
doing so should not be necessary for most users. If you `are` interested in
building GeoCAT-comp from source, you will need the following packages to be
installed.

Required dependencies for building and testing GeoCAT-comp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Below are the contents of ``environment.yml``. This file contains all of
the dependencies for building and testing GeoCAT-comp.

.. include:: ../build_envs/environment.yml
    :literal:


How to create a Conda environment for building GeoCAT-comp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GeoCAT-comp source code includes a conda environment definition file in
the ``build_envs`` folder under the root directory that can be used to create a
development environment containing all of the packages required to build GeoCAT-comp.
The file ``environment.yml`` is intended to be used on Linux systems and macOS.
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
`pytest <https://docs.pytest.org/en/stable/>`__ package may be required, please
see above)::

    pytest test
