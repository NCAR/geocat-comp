# move functions into geocat.comp namespace
from .eofs_wrapper import (eofunc_eofs, eofunc_pcs, eofunc, eofunc_ts)
from .errors import (Error, AttributeError, ChunkError, CoordinateError,
                     DimensionError, MetaError)
from .polynomial import (ndpolyfit, ndpolyval, detrend, isvector)
# bring all functions from geocat.ncomp into the geocat.comp namespace
try:
    from geocat.f2py import *
except ImportError:
    pass
