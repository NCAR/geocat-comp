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

   calendar_average
   climate_anomaly
   climatology_average
   month_to_season

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

Gradient
^^^^^^^^
.. currentmodule:: geocat.comp.gradient
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   gradient

Interpolation
^^^^^^^^^^^^^
.. currentmodule:: geocat.comp.interpolation
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   delta_pressure_hybrid
   interp_hybrid_to_pressure
   interp_sigma_to_hybrid
   interp_multidim
   pressure_at_hybrid_levels

Meteorology
^^^^^^^^^^^
.. currentmodule:: geocat.comp.meteorology
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   actual_saturation_vapor_pressure
   delta_pressure
   dewtemp
   heat_index
   max_daylight
   psychrometric_constant
   relhum
   relhum_ice
   relhum_water
   saturation_vapor_pressure
   saturation_vapor_pressure_slope
   zonal_meridional_psi

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

   eofunc_eofs
   eofunc_pcs
   nmse
   pearson_r

NCL Function Name Wrappers
--------------------------
.. currentmodule:: geocat.comp
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   meteorology.dpres_plev
   meteorology.zonal_mpsi

Deprecated Functions
---------------------
.. currentmodule:: geocat.comp.deprecated
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   grid_to_triple
   linint1
   linint2
   linint2pts
   moc_globe_atl
   rcm2points
   rcm2rgrid
   rgrid2rcm
   triple_to_grid
