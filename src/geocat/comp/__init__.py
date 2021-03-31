# move functions into geocat.comp namespace
from .climatology import anomaly, climatology, month_to_season
from .dewtemp import dewtemp
from .eofunc import eofunc, eofunc_eofs, eofunc_pcs, eofunc_ts
from .errors import (AttributeError, ChunkError, CoordinateError,
                     DimensionError, Error, MetaError)
from .interp_hybrid_to_pressure import interp_hybrid_to_pressure
from .polynomial import detrend, ndpolyfit, ndpolyval
from .relhum import relhum, relhum_ice, relhum_water
from .skewt_params import showalter_index, get_skewt_vars
# bring all functions from geocat.f2py into the geocat.comp namespace
try:
    from geocat.f2py import *
except ImportError:
    pass
