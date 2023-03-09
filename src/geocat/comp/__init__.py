# move functions into geocat.comp namespace
from .climatologies import anomaly, climatology, month_to_season, calendar_average, climatology_average, climate_anomaly
from .fourier_filters import (fourier_band_block, fourier_band_pass,
                              fourier_filter, fourier_high_pass,
                              fourier_low_pass)
from .gradient import gradient, _arc_lon_wgs84, _arc_lat_wgs84, _rad_lat_wgs84
from .interpolation import interp_hybrid_to_pressure, interp_sigma_to_hybrid, interp_multidim
from .meteorology import (dewtemp, heat_index, relhum, relhum_ice, relhum_water,
                          actual_saturation_vapor_pressure, max_daylight,
                          psychrometric_constant, saturation_vapor_pressure,
                          saturation_vapor_pressure_slope, delta_pressure,
                          dpres_plev)
from .spherical import decomposition, recomposition, scale_voronoi
from .stats import eofunc, eofunc_eofs, eofunc_pcs, eofunc_ts, pearson_r
# bring all functions from geocat.f2py into the geocat.comp namespace
try:
    from geocat.f2py import *
except ImportError:
    pass
