# move functions into geocat.comp namespace
from .climatology import climatology, anomaly
from .errors import (Error, AttributeError, ChunkError, CoordinateError,
                     DimensionError, MetaError)
from .polynomial import (ndpolyfit, ndpolyval, detrend, isvector)
from .version import __version__

# bring all functions from geocat.ncomp into the geocat.comp namespace
try:
    from geocat.f2py import *
except ImportError:
    pass
