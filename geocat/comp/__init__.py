# move functions into geocat.comp namespace
from .climatologies import (
    month_to_season,
    calendar_average,
    climatology_average,
    climate_anomaly,
)
from .fourier_filters import (
    fourier_band_block,
    fourier_band_pass,
    fourier_filter,
    fourier_high_pass,
    fourier_low_pass,
)
from .gradient import gradient, _arc_lon_wgs84, _arc_lat_wgs84, _rad_lat_wgs84
from .interpolation import (
    interp_hybrid_to_pressure,
    interp_sigma_to_hybrid,
    interp_multidim,
)
from .meteorology import (
    dewtemp,
    heat_index,
    relhum,
    relhum_ice,
    relhum_water,
    actual_saturation_vapor_pressure,
    max_daylight,
    psychrometric_constant,
    saturation_vapor_pressure,
    saturation_vapor_pressure_slope,
    delta_pressure,
    dpres_plev,
)
from .spherical import decomposition, recomposition, scale_voronoi
from .stats import eofunc, eofunc_eofs, eofunc_pcs, eofunc_ts, pearson_r
from .deprecated import (
    grid_to_triple,
    linint1,
    linint2,
    linint2pts,
    moc_globe_atl,
    rcm2points,
    rcm2rgrid,
    rgrid2rcm,
    triple_to_grid,
)

# get version from pyproject.toml
from importlib.metadata import version as _version

try:
    __version__ = _version("geocat.comp")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"
