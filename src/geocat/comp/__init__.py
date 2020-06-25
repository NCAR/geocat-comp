from typing import Iterable

from .version import __version__
import numpy as np
import xarray as xr
import dask.array as da
from dask.array.core import map_blocks

from .polynomial import ndpolyfit, ndpolyval, detrend

try:
  from geocat.ncomp import *
except ImportError:
  pass


class Error(Exception):
  """Base class for exceptions in this module."""
  pass


class ChunkError(Error):
  """Exception raised when a Dask array is chunked in a way that is
    incompatible with an _ncomp function."""
  pass


class CoordinateError(Error):
  """Exception raised when a GeoCAT-comp function is passed a NumPy array as
    an argument without a required coordinate array being passed separately."""
  pass


class DimensionError(Error):
  """Exception raised when the arguments of GeoCAT-comp functions argument
    has a mismatch of the necessary dimensionality."""
  pass


class AttributeError(Error):
  """Exception raised when the arguments of GeoCAT-comp functions argument
    has a mismatch of attributes with other arguments."""
  pass


class MetaError(Error):
  """Exception raised when the support for the retention of metadata is not
    supported."""
  pass
