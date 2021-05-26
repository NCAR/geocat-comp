# move functions into geocat.comp namespace
from .climatology import anomaly, climatology, month_to_season
from .crop import (actual_saturation_vapor_pressure, max_daylight,
                   psychrometric_constant, saturation_vapor_pressure,
                   saturation_vapor_pressure_slope)
from .dewtemp import dewtemp
from .eofunc import eofunc, eofunc_eofs, eofunc_pcs, eofunc_ts
from .errors import (AttributeError, ChunkError, CoordinateError,
                     DimensionError, Error, MetaError)
from .fourier_filters import (fourier_band_block, fourier_band_pass,
                              fourier_filter, fourier_high_pass,
                              fourier_low_pass)
from .heat_index import heat_index
from .interp_hybrid_to_pressure import interp_hybrid_to_pressure
from .polynomial import detrend, ndpolyfit, ndpolyval
from .relhum import relhum, relhum_ice, relhum_water
from .skewt_params import get_skewt_vars, showalter_index

# bring all functions from geocat.f2py into the geocat.comp namespace
try:
    from geocat.f2py import *
except ImportError:
    pass
