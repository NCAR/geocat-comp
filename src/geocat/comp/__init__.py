# move functions into geocat.comp namespace
from .climatology import (climatology, anomaly, month_to_season)
from .dewtemp import (dewtemp)
from .eofunc import (eofunc_eofs, eofunc_pcs, eofunc, eofunc_ts)
from .errors import (Error, AttributeError, ChunkError, CoordinateError,
                     DimensionError, MetaError)
from .polynomial import (ndpolyfit, ndpolyval, detrend)
from .relhum import (relhum, relhum_ice, relhum_water)
from .interp_hybrid_to_pressure import interp_hybrid_to_pressure

# bring all functions from geocat.f2py into the geocat.comp namespace
try:
    from geocat.f2py import *
except ImportError:
    pass