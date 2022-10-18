.. currentmodule:: geocat.comp

User API
========

GeoCAT-comp Native Functions
----------------------------
Climatologies
^^^^^^^^^^^^^
.. currentmodule:: geocat.comp.climatologies
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   anomaly
   calendar_average
   climatology
   climatology_average
   month_to_season

Crop
^^^^
.. currentmodule:: geocat.comp.crop
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   actual_saturation_vapor_pressure
   max_daylight
   psychrometric_constant
   saturation_vapor_pressure
   saturation_vapor_pressure_slope

Gradient
^^^^^^^^
.. currentmodule:: geocat.comp.gradient
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   gradient
   arc_lat_wgs84
   arc_lon_wgs84
   rad_lat_wgs84

EOF Functions
^^^^^^^^^^^^^
.. currentmodule:: geocat.comp.eofunc
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   eofunc_eofs
   eofunc_pcs

Fourier Filters
^^^^^^^^^^^^^^^
.. currentmodule:: geocat.comp.fourier_filters
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   fourier_band_block
   fourier_band_pass
   fourier_filter
   fourier_high_pass
   fourier_low_pass


Iterpolation
^^^^^^^^^^^^
.. currentmodule:: geocat.comp.interpolation
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   interp_hybrid_to_pressure
   interp_sigma_to_hybrid
   interp_multidim

Meteorology
^^^^^^^^^^^
.. currentmodule:: geocat.comp.meteorology
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   dewtemp
   heat_index
   relhum
   relhum_ice
   relhum_water

Polynomial
^^^^^^^^^^
.. currentmodule:: geocat.comp.polynomial
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   detrend
   ndpolyfit
   ndpolyval

Skew-T Plot Parameters
^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: geocat.comp.skewt_params
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   get_skewt_vars
   showalter_index

Spherical Harmonics
^^^^^^^^^^^^^^^^^^^
.. currentmodule:: geocat.comp.spherical
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   decomposition
   recomposition
   scale_voronoi

Statistics
^^^^^^^^^^
.. currentmodule:: geocat.comp.stats
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   pearson_r


GeoCAT-comp routines from GeoCAT-f2py
-------------------------------------
.. currentmodule:: geocat.comp
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   dpres_plevel
   grid_to_triple
   linint1
   linint2
   linint2pts
   moc_globe_atl
   rcm2points
   rcm2rgrid
   rgrid2rcm
   triple_to_grid
