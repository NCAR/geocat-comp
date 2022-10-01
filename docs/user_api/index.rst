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
   :toctree: ../_build/user_generated/

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
   :toctree: ../_build/user_generated/

   actual_saturation_vapor_pressure
   max_daylight
   psychrometric_constant
   saturation_vapor_pressure
   saturation_vapor_pressure_slope

EOF Functions
^^^^^^^^^^^^^
.. currentmodule:: geocat.comp.eofunc
.. autosummary::
   :nosignatures:
   :toctree: ../_build/user_generated/

   eofunc_eofs
   eofunc_pcs

Fourier Filters
^^^^^^^^^^^^^^^
.. currentmodule:: geocat.comp.fourier_filters
.. autosummary::
   :nosignatures:
   :toctree: ../_build/user_generated/

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
   :toctree: ../_build/user_generated/

   interp_hybrid_to_pressure
   interp_sigma_to_hybrid
   interp_multidim

Meteorology
^^^^^^^^^^^
.. currentmodule:: geocat.comp.meteorology
.. autosummary::
   :nosignatures:
   :toctree: ../_build/user_generated/

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
   :toctree: ../_build/user_generated/

   detrend
   ndpolyfit
   ndpolyval

Skew-T Plot Parameters
^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: geocat.comp.skewt_params
.. autosummary::
   :nosignatures:
   :toctree: ../_build/user_generated/

   get_skewt_vars
   showalter_index

Spherical Harmonics
^^^^^^^^^^^^^^^^^^^
.. currentmodule:: geocat.comp.spherical
.. autosummary::
   :nosignatures:
   :toctree: ../_build/user_generated/

   decomposition
   recomposition
   scale_voronoi

Statistics
^^^^^^^^^^
.. currentmodule:: geocat.comp.stats
.. autosummary::
   :nosignatures:
   :toctree: ../_build/user_generated/

   pearson_r


GeoCAT-comp routines from GeoCAT-f2py
-------------------------------------
.. currentmodule:: geocat.comp
.. autosummary::
   :nosignatures:
   :toctree: ../_build/user_generated/

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
