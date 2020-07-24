from .errors import (Error, AttributeError, ChunkError, CoordinateError, DimensionError, MetaError)
from .polynomial import (ndpolyfit, ndpolyval, detrend, isvector)
from .version import __version__


try:
    from geocat.ncomp import *
except ImportError:
    pass
