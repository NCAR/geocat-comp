from . import _ncomp
import numpy as np
import xarray as xr
from dask.array.core import map_blocks

@xr.register_dataarray_accessor('ncomp')
class Ncomp(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
