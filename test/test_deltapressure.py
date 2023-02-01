
from unittest import TestCase
from geocat.comp.delta_pressure import _calc_deltapressure_1D, calc_deltapressure
import numpy as np
import xarray as xr

class TestDeltaPressure():
    pressure_lev = [1, 2, 50, 100, 200, 500, 700, 1000]
    pressure_lev_da = xr.DataArray(pressure_lev)
    pressure_lev_da.attrs = {"long name": "pressure level", "units": "hPa", "direction": "descending"}
    
    surface_pressure_scalar = 1018
    
    surface_pressure_1D = [1018, 1019]
    coords = {'lon': [5, 6]}
    dims = ["lon"]
    attrs = {"long name": "surface pressure", "units": "hPa"}
    surface_pressure_1D_da = xr.DataArray(surface_pressure_1D, coords = coords, dims = dims, attrs = attrs)
    
    surface_pressure_2D = [[1018, 1019], [1017, 1019.5]]
    coords = {'lat': [3, 4], 'lon': [5, 6]}
    dims = ["lat", "lon"]
    surface_pressure_2D_da = xr.DataArray(surface_pressure_2D, coords = coords, dims = dims, attrs = attrs)

    
    surface_pressure_3D = [[[1018, 1019], [1017, 1019.5]], [[1019, 1020], [1018, 1020.5]]]
    coords = {'time': [1, 2],'lat': [3, 4], 'lon': [5, 6]}
    dims = ["time", "lat", "lon"]
    surface_pressure_3D_da = xr.DataArray(surface_pressure_3D, coords = coords, dims = dims, attrs = attrs)

    
    def test_deltapressure_1D(pressure_lev, surface_pressure_scalar):
        pressure_top = min(pressure_lev)
        delta_pressure = _calc_deltapressure_1D(pressure_lev, surface_pressure_scalar)
        assert sum(delta_pressure) == surface_pressure_scalar - pressure_top
        
    # test that we raise desired warnings
    if surface_pressure != True:
        warnings.warn("'surface_pressure` can't equal a missing value.")
    if pressure_top <= 0:
        warnings.warn("'pressure_lev` values must all be positive.")
    if pressure_top > surface_pressure:
        warnings.warn("`surface_pressure` must be greater than minimum `pressure_lev` value.")
        
        
    # test that we raise desired errors
        try:
        pressure_lev = np.asarray(pressure_lev)
    except AttributeError:
        print("`pressure_lev` must be array-like.")

    try:
        surface_pressure = np.asarray(surface_pressure)
    except AttributeError:
        print("`surface_pressure` must be array-like.")
        
    # test that xarray in means xarray out
    # test that dimensions of delta_pressure xarray out match surface_pressure dimensions plus lev dimension
    # test attributes out
    # assert desired dimension sizes of output
