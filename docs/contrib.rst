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
#. Install pre-commit hooks.

**How to make your changes.**

#. Understanding the codebase.
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

Creating a branch for your changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We highly recommend creating a new branch on your fork for each new feature or
bug that you work on.

To create and check out a new branch, use the following command:

.. code-block:: bash

    git checkout -b <branch-name>

You can see a list of all branches in your local repository by running:

.. code-block:: bash

    git branch

For more information on branching, check out this `learn git branching
<https://learngitbranching.js.org/>`__ interactive tool.

Pre-commit hooks
^^^^^^^^^^^^^^^^

``geocat-comp`` uses pre-commit hooks to ensure a standardized base-level code
formatting and style.

The ``pre-commit`` package is installed by default when using the
``build_envs/environment.yml`` file. To set up the pre-commit hooks, run the
following command from the root of the repository:

.. code-block:: bash

    pre-commit install

Now, whenever you commit changes, the pre-commit hooks will run and may make
small modifications to your code. If the pre-commit hooks make any changes, you
will need to re-add the files and commit them again in order to sucessfully make
the commit.

To manually run the pre-commit hooks, use the following command:

.. code-block:: bash

    pre-commit run --all-files

You can skip the pre-commit hooks by adding the ``--no-verify`` flag to your
commit command like this:

.. code-block:: bash

    git commit -m "your commit message" --no-verify

For more information on pre-commit hooks, see the `pre-commit documentation <https://pre-commit.com/>`__.


Make your changes
-----------------

After you're all set up to develop ``geocat-comp``, you can start making your
changes. This section describes where, how, and what to change to add your
contributions to the ``geocat-comp`` codebase.


Understanding the codebase
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``geocat-comp`` top-level direcory is organized as follows:

.. code-block:: bash

    geocat-comp
    ├── build_envs
    ├── docs
    ├── geocat
    │   └── comp
    └── test


* The ``build_envs`` directory contains the ``environment.yml`` file used to
  create your development environment. It also contains additional environment
  files used for testing and building the documentation.

* The ``docs`` directory contains the ``sphinx`` documentation for
  ``geocat-comp``.

* The ``geocat/comp`` directory, contains the code for the ``geocat.comp``
  package. This is the place to add new functionality. The ``geocat.comp`` code
  is organized into modules, each of which is contained in its own file. It is
  recommended that you add new functionality to an existing file, though it may
  be appropriate to make a new file.

* The ``test`` directory contains the unit tests for ``geocat-comp``. Each
  module in ``geocat.comp`` has a corresponding test module in the ``test``
  directory.


When adding new functionality, there are multiple auxiliary files that you may
need to modify to encorporate your code into the package. These include:

* ``geocat/comp/__init__.py``: This file imports all of the functions intended
  for the public API.

* ``docs/internal_api/index.rst`` and ``docs/user_api/index.rst``: These files
  are used to generate the API documentation from docstrings.

* ``docs/release-notes.rst``: This file documents changes to the codebase that
  we add to in the same PR as the code changes.

* ``tests/test_<module>.py``: This file contains the unit tests for the module
  you are adding to. It is highly encouraged to add unit tests for any new
  functionality you add to ``geocat-comp``.


Write and run tests
^^^^^^^^^^^^^^^^^^^

``geocat-comp`` uses `pytest <https://pytest.org/>`__ for unit tests. Currently,
we have unit tests written in both ``pytest`` and ``unittest``. We are in the
process of converting all of our tests to ``pytest`` and we encourage you to
write new tests using ``pytest``.

To run the tests locally, use the following command from the root of the
repository:

.. code-block:: bash

    pytest

To run a specific test, use the following command:

.. code-block:: bash

    pytest tests/test_mod.py::test_func

These tests will also run automatically when you open a pull request using
GitHub Actions and the ``.github/workflows/ci.yml`` file.

See the `pytest documentation <https://pytest.org/>`__ for more information.


Documentation
-------------

``geocat-comp`` uses `sphinx <https://www.sphinx-doc.org/en/master/>`__  and
`ReadTheDocs <https://docs.readthedocs.io/en/stable/>`__` to build and host the
documentation.


Docstrings
^^^^^^^^^^

The most common situation in which you will need to add to the documentation is
through docstrings.

``geocat-comp`` uses `numpydoc
<https://numpydoc.readthedocs.io/en/latest/format.html>`__ style docstrings. See
`sphinx's example numpydoc docstring
<https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html#example-numpy>`__.

To include your docstring documentation in the API reference, you will need to
add it to either the ``docs/internal_api/index.rst`` or
``docs/user_api/index.rst`` file, depending on whether the function is intended
for internal or external use.

Editing other documentation files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We welcome changes and improvements to all parts of our documentation (including
this guide)! You can find these files in the ``docs`` directory.

These files are mainly written in `reStructuredText
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__,
but additional file types such as ``.md`` and ``.ipynb`` are also used.

Important documentation files to know about include:

* ``docs/index.rst``: This file is the main page of the documentation. Files
  added to ``toctree`` blocks in this file will be included in the documentation
  as top-level subpages.

* ``docs/contrib.rst``: This file is the source for this guide!

* ``docs/conf.py``: This file contains the configuration for building the documentation.

* ``docs/examples/*.ipynb``, ``docs/examples.rst``, and ``docs/gallery.yml``:
  These files are used to generate the jupyter notebook examples in the
  documentation. Notebooks in the ``docs/examples/`` directory are added to the
  documentation by adding them to the ``toctree`` in ``docs/examples.rst`` and
  linked to their cover picture by addidng them to the ``docs/gallery.yml``
  file.

See the `sphinx documentation <https://www.sphinx-doc.org/en/master/>`__ for
more informatiion about writing sphinx documentation.


Generate the documentation locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To generate the documentation locally, follow the steps below.

#. Create and activate the ``gc-docs`` conda environment using the ``build_envs/docs.yml`` file.
#. Enter the ``docs`` directory.
#. Run ``make html`` or to build the documentation.
#. Open ``docs/_build/html/index.html`` in your browser to view the documentation.


Check the documentation
^^^^^^^^^^^^^^^^^^^^^^^

As well as checking local documentation generation, you should also check the
preview documentation generated as part of a PR. To do this, scroll down to the
"checks" section of the PR and click on the "Details" link next to the
"docs/readthedocs.org:geocat-comp" check. This will take you to the
correspinding build on ReadTheDocs, where you can view the documentation built
from your PR and see any warnings or errors on your build.


Contribute your code
--------------------

Once you have prepared your changes and are ready for them to be reviewed by the
GeoCAT team, you can open a pull request. This section describes how to open a
pull request and what to expect after you open it.

Push your changes to your fork
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have made your changes locally, you will need to push them to your
branch on your fork on GitHub. To do this, use the following command:

.. code-block:: bash

    git push

From here, you can request that your changes be merged into the main repository in the form of a pull request.

Making a pull request
^^^^^^^^^^^^^^^^^^^^^

GitHub has extensive `pull request guides and documentation
<https://docs.github.com/en/pull-requests>`__ that we recommend. This section
describes the basics for our workflow.

From your branch on your fork, open the "Pull requests" tab and click the "New
pull request" button. Make sure the "base repository" is "NCAR/geocat-comp" and
the "base" branch is set to "main", wiht the "head repository" and "compare"
branch set to your fork and prepared branch, respectively.

From this page, you can see a view of the changes you have made in your branch.

We recommend adding a short, descriptive title to your pull request. The body of
the pull request should autofill with our pull request template, which has it's
own set of directions. Please fill out the relevant sections of the template,
including adding a more detailed description of your changes.

Once you have filled out the template, click the "Create pull request" button.
This will open your pull request on the ``geocat-comp`` repository.

If you want to open a pull request but are not ready for it to be reviewed, you
can open the pull request as a draft. This is also a good way to get feedback on
your work that might not be ready to contribute yet.

Addressing feedback
^^^^^^^^^^^^^^^^^^^

After you open your pull request, the GeoCAT team will review it and
may provide feedbac like asking for changes or suggesting improvements. You can
address this feedback by making changes to your branch and pushing them to your
fork. The pull request will automatically update with your changes.

The GeoCAT team appreciates your contributions and the interactive process of
reviewing pull requests, and will do our best to review your pull request in a
timely manner. It is totally normal to have to make several rounds of changes to
your pull request before it is ready to be merged, especially if you are new to
the project.

Once your pull request is approved by a core maintainer and passes the relevant
checks, it will be merged into the main repository!


Delete your branch
^^^^^^^^^^^^^^^^^^

We recommend deleting your branch after your pull request is merged. This will
help keep your fork clean and organized, but is not required.
