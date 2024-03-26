.. currentmodule:: geocat.comp

.. _overview:

Overview
========
GeoCAT-comp is the computational component of the `GeoCAT project <https://geocat.ucar.edu/>`__. It houses Python
implementations of NCL's non-WRF computational routines and additional geoscientific analysis functions that go beyond
what NCL offers. It is a principle component of NSF NCAR's `Pivot to Python Initiative <https://www.ncl.ucar.edu/Document/Pivot_to_Python/>`__.

Why GeoCAT-comp?
----------------
GeoCAT-comp is a geoscience data analysis Python package created as part of NSF NCAR's `Pivot to Python Initiative <https://www.ncl.ucar.edu/Document/Pivot_to_Python/>`__.
`NCL <https://www.ncl.ucar.edu/>`__, the prior language of choice for geoscience work, has been put into maintenance mode
in favor of Python. This is due to Python's easy-to-learn, open-source nature and the benefits that provides. There are
a plethora of scientific analysis packages designed for general use or for niche applications. Numerous tutorials exist
for learning Python basics and for data analysis workflows. Python also enables scalability through parallel computation
which was never possible with NCL, enabling geoscientists to tackle analysis workflows on large volumes of data.

GeoCAT-comp draws from well-established analysis packages like `NumPy <https://numpy.org/>`__, `Xarray <https://docs.xarray.dev/>`__,
`SciPy <https://scipy.org/>`__, and `MetPy <https://unidata.github.io/MetPy/>`__ to recreate and expand upon NCL
functions in pure Python. With so many tools being used under the hood, GeoCAT-comp users have access to geoscience
specific computational tools without needing extensive working knowledge of `Pangeo <https://pangeo.io/>`__ stack.

There are syntactical benefits to using the Pangeo stack, one of which is indexing by label. NCL requires the
dimensions of input data to be in a particular order. This results in functions with rigid data format requirements and
multiple versions of the same function for different dimension orders. By using `Xarray <https://docs.xarray.dev/>`__,
GeoCAT-comp avoids those data format requirements. The xarray data structures allow users to
`refer to dimensions by label <https://docs.xarray.dev/en/stable/getting-started-guide/why-xarray.html#what-labels-enable>`__
rather than index, thereby removing the need for multiple versions of the same function. This makes it easier for the
user to find what they need and adds flexibility so that the user need not spend as much time wrangling their data
into the correct format.

GeoCAT-comp's open-source nature allows for more community engagement than a traditional software development workflow
does. As advances are made in the realm of geoscience, new tools will be needed to analyze new datasets. We are dedicated
to addressing user needs and encourage users to submit feature requests and contributions to our GitHub as well as
participate in discussions. See our :ref:`support` page for info on how to submit bug reports, request new
features, and get involved.

GeoCAT-f2py
-----------
While our goal is to recreate NCL functions in pure Python, translating some NCL routines is challenging and time
consuming. To ensure GeoCAT users have access to those functions while we work on full Python versions, the Fortran code
they are based upon is wrapped in Python in the GeoCAT-f2py package. Information about
GeoCAT-f2py can be found on the `package's homepage <https://geocat-f2py.readthedocs.io/>`__.
