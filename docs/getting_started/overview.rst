Overview
========
GeoCAT-comp is the computational component of the `GeoCAT project <https://geocat.ucar.edu/>`_, and it houses Python
implementations of NCL's non-WRF computational routines and additional geoscientific analysis functions that go beyond
what NCL offers. It is a principle component of NCAR's `Pivot to Python <https://www.ncl.ucar.edu/Document/Pivot_to_Python/>`_.

More Than Just An NCL Clone
---------------------------
`NCL <https://www.ncl.ucar.edu/>`_, the prior language of choice for geoscience work, required the dimensions of data to
be in a particular order. This resulted in functions with rigid data format requirements or multiple versions of the
same function for different dimension orders.

Rather than maintaining those data format requirements, GeoCAT-comp uses `xarray <https://docs.xarray.dev/en/stable/getting-started-guide/why-xarray.html#what-labels-enable>`_
data structures so users can refer to dimensions by name rather than index. This removes the need for different function
versions which makes the finding the right function easier for the user. This also adds flexibility so that the user
need not spend time wrangling their data into the correct format.


GeoCAT-f2py
-----------
Translating some NCL routines to pure Python is challenging and time consuming. To ensure GeoCAT users have access to
those functions, the Fortran they are based upon is wrapped in Python in the GeoCAT-f2py (Fortran 2 Python) package.
GeoCAT-f2py is accessible through GeoCAT-comp, so there is no need for users to install GeoCAT-f2py directly.
Information about GeoCAT-f2py can be found on the `package's homepage <https://geocat-f2py.readthedocs.io/en/latest/#>`_.