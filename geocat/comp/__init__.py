# move functions into geocat.comp namespace
# get version from pyproject.toml
from importlib.metadata import version as _version

from .climatologies import (
    calendar_average,
    climate_anomaly,
    climatology_average,
    month_to_season,
)
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
from .fourier_filters import (
    fourier_band_block,
    fourier_band_pass,
    fourier_filter,
    fourier_high_pass,
    fourier_low_pass,
)
from .gradient import _arc_lat_wgs84, _arc_lon_wgs84, _rad_lat_wgs84, gradient
from .interpolation import (
    delta_pressure_hybrid,
    interp_hybrid_to_pressure,
    interp_multidim,
    interp_sigma_to_hybrid,
    pressure_at_hybrid_levels,
)
from .meteorology import (
    actual_saturation_vapor_pressure,
    delta_pressure,
    dewtemp,
    dpres_plev,
    heat_index,
    max_daylight,
    psychrometric_constant,
    relhum,
    relhum_ice,
    relhum_water,
    saturation_vapor_pressure,
    saturation_vapor_pressure_slope,
    zonal_meridional_psi,
    zonal_mpsi,
)
from .spherical import decomposition, recomposition, scale_voronoi
from .stats import eofunc, eofunc_eofs, eofunc_pcs, eofunc_ts, nmse, pearson_r

try:
    __version__ = _version("geocat.comp")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999"
