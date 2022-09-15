.. currentmodule:: geocat.comp

.. _tutorial:

Usage Examples
==============

Short Tutorial
--------------
Calculate saturation vapor pressure from :code:`Numpy` Data::

    >>> import numpy as np
    >>> from geocat.comp import saturation_vapor_pressure
    >>> temp = np.array([50, 60, 70])
    >>> saturation_vapor_pressure(temp)

    array([1.22796262, 1.76730647, 2.50402976])

Calculate daily climate averages from NetCDF Data::

    >>> import xarray as xr
    >>> from geocat.comp import calendar_average
    >>> import geocat.datafiles as gdf

    >>> # get data from geocat.datafiles repo or use your own data
    >>> file = gdf.get('netcdf_files/atm.20C.hourly6-1990-1995-TS.nc')
    >>> ds = xr.open_dataset(file)  # open NetCDF file using xarray
    >>> temp = ds.TS
    >>> temp

    <xarray.DataArray 'TS' (member_id: 40, time: 8761)>
    [350440 values with dtype=float32]
    Coordinates:
        lat        float64 ...
        lon        float64 ...
      * member_id  (member_id) int64 1 2 3 4 5 6 7 8 ... 34 35 101 102 103 104 105
      * time       (time) object 1990-01-01 00:00:00 ... 1996-01-01 00:00:00
    Attributes:
        long_name:  Surface temperature (radiative)
        units:      K

    >>> calendar_average(temp, 'day')
    <xarray.DataArray 'TS' (member_id: 40, time: 2191)>
    array([[301.88226, 301.9291 , 302.0053 , ..., 302.24304, 302.31415,
            302.33292],
           [300.3797 , 300.3421 , 300.28976, ..., 299.82782, 299.83026,
            299.8336 ],
           [300.31702, 300.29077, 300.28403, ..., 300.08377, 300.02185,
            300.0073 ],
           ...,
           [302.60565, 302.48895, 302.3853 , ..., 300.181  , 300.16293,
            300.15826],
           [299.1427 , 299.12122, 299.06543, ..., 302.36343, 302.35126,
            302.35077],
           [297.99374, 298.0177 , 298.03265, ..., 300.71463, 300.68396,
            300.67422]], dtype=float32)
    Coordinates:
        lat        float64 ...
        lon        float64 ...
      * member_id  (member_id) int64 1 2 3 4 5 6 7 8 ... 34 35 101 102 103 104 105
      * time       (time) object 1990-01-01 12:00:00 ... 1996-01-01 12:00:00
    Attributes:
        long_name:  Surface temperature (radiative)
        units:      K
