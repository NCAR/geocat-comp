.. _contributing:

===========================
Contributing to geocat-comp
===========================

************
Introduction
************

Thank you for considering making a contribution to ``geocat-comp``! There are
many ways to contribute to this project, including reporting bugs, requesting
additional or new functionality, improving our documentation, or contributing
your own code and we appreciate all of them.

If you have any questions, please feel free to reach out to us on `GitHub
Discussions <https://github.com/NCAR/geocat-comp/discussions>`__ You can also
reach us by email at geocat@ucar.edu.

**************
Where to start
**************

Look through our open issues and see if there is anything you would like to take
on! We recomment working with core developers to implement new functionality. We
can help you get started and make sure your code is consistent with the rest of
the project.

Also check out any beginner-friendly issues we have tagged with `good first
issue <https://github.com/NCAR/geocat-comp/labels/good%20first%20issue>`__.

We do not officially "assign" issues to contributors, but if you are interested
in working on an issue, please comment on the issue to let us know you are
working on it. This will help us avoid duplicate work.

********************************
Bug reports and feature requests
********************************

We have issue templates for both `bug reports
<https://github.com/NCAR/geocat-comp/issues/new?assignees=&labels=bug%2C+support&projects=&template=bug_report.md&title=>`__
and `feature requests
<https://github.com/NCAR/geocat-comp/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.md&title=>`__.

When reporting a bug, please include as much information as possible. This will
help us reproduce the bug and fix it efficiently. For more information on how to
write a good bug report, see this stackoverflow post on `how to make a good bug
report <https://stackoverflow.com/help/minimal-reproducible-example>`__.

**************
Git and GitHub
**************

The code for ``geocat-comp`` is hosted on GitHub. If you do not have one, you
will need to create a `free GitHub account <https://github.com/signup/free>`__.
The `GitHub Quickstart Guide
<https://docs.github.com/en/get-started/quickstart>`__ is a great place to get
started with git and GitHub.

********************
Development workflow
********************

Overview
--------

This is a brief overview of the development workflow we use for ``geocat-comp``.
A more detailed description of each step is provided in following sections.

**Get set up to develop** ``geocat-comp`` **on your local machine.**

#. Fork the ``geocat-comp`` repository on GitHub.
#. Clone your fork to your local machine.
#. Make your development environment.
#. Create a new branch for your changes.

**Make your changes.**

#. Commit your changes.
#. Write and run tests.
#. Adding to, generating, and checking documentation.

**Contribute your code.**

#. Push your changes to your fork.
#. Open a pull request.
#. Address any feedback.
#. Wait for your pull request to be merged.
#. Delete your branch.


Get set up to develop ``geocat-comp`` on your local machine
-----------------------------------------------------------

Get the code
^^^^^^^^^^^^

Get started by forking the NCAR/geocat-comp repository on GitHub. To do this,
find the "Fork" button near the top of the page and click it. This will create a
copy of the project under your personal github account.

Next, clone your forked copy to your local machine.

.. code-block:: bash

    git clone https://github.com/your-user-name/geocat-comp.git


Enter the project folder and set the upstream remote to the NCAR/geocat-comp
repository. This will allow you to keep your fork up to date with the main
repository.

.. code-block:: bash

    cd geocat-comp git remote add upstream https://github.com/NCAR/xarray.git

For more information, see the `GitHub quickstart section on forking a repository
<https://docs.github.com/en/get-started/quickstart/fork-a-repo>`__.

Create a development environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run and test any changes you make in ``geocat-comp``, you will need to create
a development environment. We recommend installing and using `conda
<https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`__
and/or `mamba
<https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install>`__.

Use the following commands to create a new conda environment to develop
``geocat-comp`` in.

.. code-block:: bash

    # Create a new conda environment
    conda create -c conda-forge -n geocat_comp_build python=3.10

    # Use the environment file to populate the environment with the required
    dependencies conda env update -f build_envs/environment.yml

    # Activate your new environment
    conda activate geocat_comp_build

    # Install your local copy of geocat-comp in interactive mode
    pip install -e .

To test your new install, open a python session and try importing
``geocat.comp``. You can also try printing the version number, which should be
unique to the latest commit on your fork.

.. code-block:: python

    >>> import geocat.comp as gc
    >>> gc.__version__
    '2023.5.1.dev8+g3f0ee48.d20230605'

See the `conda documentation
<https://docs.conda.io/projects/conda/en/latest/>`__ for more information.

Contibuting to documentation
----------------------------

- where
- sphinx
- rst
- how to build


Opening a pull request
----------------------
- pull request template
