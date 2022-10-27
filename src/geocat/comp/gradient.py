from typing import Union

import numpy as np
import xarray as xr

SupportedTypes = Union[np.ndarray, xr.DataArray]
XTypes = Union[xr.DataArray, xr.Dataset]

d2r = 1.74532925199432957692369e-02  # degrees to radians conversion factor


def rad_lat_wgs84(lat: SupportedTypes,):
    r"""The radius calculation for the wgs84 ellipsoid at a latitude uses a
    taylor series from.

    .. math::
        radius = \sqrt{a^2 \cdot cos(lat)^2+b^2 \cdot sin(lat)^2}

    This returns the radius of the ellipsoid for a given latitude
    This is accurate to within floating point error.

    note: This doesn't need to be a taylor series, though the taylor series
    is faster
    and a needed step for the arc_lat_wgs84 function to avoid the elliptic
    integral

    Parameters
    ----------
    lat : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        2-dimensional dataset, of orthographic latitude coordinates

    Returns
    -------
    gradients : list of :class:`numpy.ndarray`, list of
    :class:`xarray.DataArray`
        latitudinal radii calculated using th WGS84 geoid.
    """
    return \
        8.05993093251779959604912e-107 * lat ** 48 - \
        1.26581811418535723456176e-102 * lat ** 46 - \
        9.10951565776720242021392e-98 * lat ** 44 + \
        7.49821836126201522491765e-93 * lat ** 42 - \
        2.27271626922827622448519e-88 * lat ** 40 - \
        2.71379394439952534826763e-84 * lat ** 38 + \
        6.14832871773468827219624e-79 * lat ** 36 - \
        2.84993185787053811467259e-74 * lat ** 34 + \
        3.04677525449067422497153e-70 * lat ** 32 + \
        4.32285294849656618043972e-65 * lat ** 30 - \
        3.09781963775152747156702e-60 * lat ** 28 + \
        8.20657704323096444142572e-56 * lat ** 26 + \
        2.18501284875664232841136e-51 * lat ** 24 - \
        3.14445926697498412770696e-46 * lat ** 22 + \
        1.33970531285992980064454e-41 * lat ** 20 - \
        3.64729257084431972115219e-38 * lat ** 18 - \
        3.19397401776292877914625e-32 * lat ** 16 + \
        2.09755935705627800528960e-27 * lat ** 14 - \
        3.93234660296994989005722e-23 * lat ** 12 - \
        4.67346944715944948938249e-18 * lat ** 10 + \
        5.23053397296821359502194e-13 * lat ** 8 - \
        2.61499198791487361911660e-08 * lat ** 6 + \
        6.57016735969856188450312e-04 * lat ** 4 - \
        6.50322744547926518806010e+00 * lat ** 2 + \
        6378137.0


def arc_lat_wgs84(lat: SupportedTypes,):
    r"""The arc length calculation for the wgs84 ellipsoid at a latitude uses a
    taylor series to obtain the value of the elliptic integral.

    .. math::
      arclat = \int_{0}^{lat}\sqrt{a^2 \cdot cos(lat)^2+b^2 \cdot sin(lat)^2}\ dlat

    The integral of the radius taylor series gives an arc length taylor series
    This returns the distance from the equator to a given latitude
    This is accurate to within floating point error.

    note: This needs to be a taylor series to avoid the elliptic integral

    Parameters
    ----------
    lat : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        2-dimensional dataset, of orthographic latitude coordinates

    Returns
    -------
    gradients : list of :class:`numpy.ndarray`, list of
    :class:`xarray.DataArray`
        latitudinal arc from equator calculated using th WGS84 geoid.
    """
    return \
        2.87086392358719396475614e-110 * lat ** 49 - \
        4.70057315402553703681995e-106 * lat ** 47 - \
        3.53313425533365879608994e-101 * lat ** 45 + \
        3.04345577761664668327703e-96 * lat ** 43 - \
        9.67472728333544072524681e-92 * lat ** 41 - \
        1.21447793719117046293341e-87 * lat ** 39 + \
        2.90023188160517270041180e-82 * lat ** 37 - \
        1.42116269649485608032749e-77 * lat ** 35 + \
        1.61140181088334593373753e-73 * lat ** 33 + \
        2.43380700099386902295507e-68 * lat ** 31 - \
        1.86438456247248912270599e-63 * lat ** 29 + \
        5.30488110085042076590693e-59 * lat ** 27 + \
        1.52542673636737189511924e-54 * lat ** 25 - \
        2.38613771319829866779682e-49 * lat ** 23 + \
        1.11344136742221455419521e-44 * lat ** 21 - \
        3.35038232340852202916191e-41 * lat ** 19 - \
        3.27913899018323295027126e-35 * lat ** 17 + \
        2.44062113577649370926265e-30 * lat ** 15 - \
        5.27941504241845042397655e-26 * lat ** 13 - \
        7.41522084948104995730556e-21 * lat ** 11 + \
        1.01433377184128234589268e-15 * lat ** 9 - \
        6.52003144319804471773524e-11 * lat ** 7 + \
        2.29342105667605006614878e-06 * lat ** 5 - \
        3.78342436432244021373642e-02 * lat ** 3 + \
        111319.490793273572647713 * lat


def arc_lon_wgs84(
    lon: SupportedTypes,
    lat: SupportedTypes,
):
    r"""The arc length calculation for the wgs84 ellipsoid at a longitude uses a
    taylor series from.

    .. math::
        arclon = lon \cdot \cos(lat) \cdot \sqrt{a^2 \cdot \cos(lat)^2+b^2
        \cdot \sin(lat)^2}

    This returns the distance from the Greenwich Meridian to a given latitude
    This is accurate to within floating point error.

    note: This doesn't need to be a taylor series, though the taylor series
    is faster and a needed step for the arc_lat_wgs84 function to avoid the
    elliptic integral

    Parameters
    ----------
    lon : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        2-dimensional dataset, of orthographic longitude coordinates

    lat : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        2-dimensional dataset, of orthographic latitude coordinates

    Returns
    -------
    gradients : list of :class:`numpy.ndarray`, list of
    :class:`xarray.DataArray`
        longitudinal arc from Prime Meridian calculated using th WGS84 geoid.
    """
    return rad_lat_wgs84(lat) * np.cos(lat * d2r) * lon * d2r


def gradient(data: xr.DataArray) -> [xr.DataArray]:
    r"""Extract and return the gradient values of a dataset at each point in the
    dataset. Assuming that the data points are on the surface of the WGS84
    ellipsoid.

    Parameters
    ----------
    data : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        n-dimensional dataset, with orthographic latitude longitude coordinates

    Returns
    -------
    gradients : list of :class:`numpy.ndarray`, list of :class:`xarray.DataArray`
        longitudinal and latitudinal gradients calculated using th WGS84 geoid.
    """

    lon2d, lat2d = np.meshgrid(data.coords['lon'], data.coords['lat'])

    axis0loc = xr.DataArray(
        arc_lat_wgs84(lat2d),
        coords=data.coords,
        dims=data.dims,
    )
    axis1loc = xr.DataArray(
        arc_lon_wgs84(lon2d, lat2d),
        coords=data.coords,
        dims=data.dims,
    )

    axis0dist = xr.DataArray(
        np.gradient(axis0loc, axis=0),
        coords=data.coords,
        dims=data.dims,
    )
    axis1dist = xr.DataArray(
        np.gradient(axis1loc, axis=1),
        coords=data.coords,
        dims=data.dims,
    )

    grad = xr.DataArray(
        np.gradient(data),
        dims=('dir',) + data.dims,
    )
    axis0grad = grad[0] / axis0dist
    axis1grad = grad[1] / axis1dist

    return [axis0grad, axis1grad]
