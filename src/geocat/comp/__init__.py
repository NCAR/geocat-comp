# move functions into geocat.comp namespace
from .errors import (Error, AttributeError, ChunkError, CoordinateError,
                     DimensionError, MetaError)
from .polynomial import (ndpolyfit, ndpolyval, detrend, isvector)
from .climatology import climatology, anomaly, month_to_season
from .version import __version__

# bring all functions from geocat.ncomp into the geocat.comp namespace
try:
    from geocat.ncomp import *
except ImportError:
    pass
