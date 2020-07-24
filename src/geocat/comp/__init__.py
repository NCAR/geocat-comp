# move functions into geocat.comp namespace
from .errors import (Error, AttributeError, ChunkError, CoordinateError,
                     DimensionError, MetaError)
from .polynomial import (ndpolyfit, ndpolyval, detrend, isvector)
from .version import __version__

# bring all functions from geocat.ncomp into the geocat.comp namespace
try:
    from geocat.ncomp import *  # bring all functions from geocat.ncomp into the geocat.comp namespace
except ImportError:
    pass
