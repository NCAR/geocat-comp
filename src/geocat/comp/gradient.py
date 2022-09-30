from typing import Union

import numpy as np
import xarray as xr
from scipy.ndimage import convolve

SupportedTypes = Union[np.ndarray, xr.DataArray]
XTypes = Union[xr.DataArray, xr.Dataset]

d2r = 1.74532925199432957692369e-02  # degrees to radians conversion factor


def rad_lat_wgs84(lat: SupportedTypes,):
    """The radius calculation for the wgs84 ellipsoid at a latitude uses a
    taylor series from.

    .. math:
        :nowrap:
        radius = \sqrt{a^2 \cdot cos(lat)^2+b^2 \cdot sin(lat)^2}
    This returns the radius of the elipsoid for a given latitude
    This is accurate to within floating point error.

    note: This doesn't need to be a taylor series, though the taylor series
    is faster
    and a needed step for the arc_lat_wgs84 function to avoid the eliptic
    integral
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
    """The arc length calculation for the wgs84 ellipsoid at a latitude uses a
    taylor series to obtain the value of the elliptic integral.

    .. math:
        :nowrap:
        arclat = \int_{0}^{lat}\sqrt{a^2 \cdot cos(lat)^2+b^2 \cdot sin(
        lat)^2}\ dlat
    The integral of the radius taylor series gives an arc length taylor series
    This returns the distance from the equator to a given latitude
    This is accurate to within floating point error.

    note: This needs to be a taylor series to avoid the eliptic integral
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

    .. math:
        :nowrap:
        arclon = lon \cdot cos(lat) \cdot \sqrt{a^2 \cdot cos(lat)^2+b^2
        \cdot sin(lat)^2}
    This returns the distance from the Greenwich Meridian  to a given latitude
    This is accurate to within floating point error.

    note: This doesn't need to be a taylor series, though the taylor series
    is faster
    and a needed step for the arc_lat_wgs84 function to avoid the eliptic
    integral
    """
    return rad_lat_wgs84(lat) * np.cos(lat * d2r) * lon * d2r


def gradient(data: xr.DataArray) -> [xr.DataArray]:
    """Filter a dataset by frequency. This function allowes for low_pass, high_
    pass, band_pass, or band_block filtering of the data's freqency
    representation.

    Parameters
    ----------
    data : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        n-dimensional dataset, with orthographic latitude longitude coordinates

    Returns
    -------
    gradients : list of :class:`numpy.ndarray`, list of
    :class:`xarray.DataArray`
        longitudinal and latitudinal gradients calculated using th WGS84 geoid.
    """

    gradkernlat = np.array([
        [0, 1, -1],
    ])
    gradkernlon = np.array([
        [0],
        [1],
        [-1],
    ])

    sumkernlat = np.array([
        [0.5, 0.5, 0],
    ])
    sumkernlon = np.array([
        [0.5],
        [0.5],
        [0],
    ])

    lon2d, lat2d = np.meshgrid(data.coords['lon'], data.coords['lat'])

    datagradlat = xr.DataArray(
        convolve(data, gradkernlat, mode='wrap', origin=0),
        dims=('lon', 'lat'),
    )
    datagradlon = xr.DataArray(
        convolve(data, gradkernlon, mode='wrap', origin=0),
        dims=('lon', 'lat'),
    )

    arclon2d = xr.DataArray(
        arc_lon_wgs84(lon2d, lat2d),
        dims=('lon', 'lat'),
    )
    arclat2d = xr.DataArray(
        arc_lat_wgs84(lat2d),
        dims=('lon', 'lat'),
    )

    arclatgrad = xr.DataArray(
        convolve(arclon2d, gradkernlat, mode='mirror', origin=0),
        dims=('lon', 'lat'),
    )
    arclongrad = xr.DataArray(
        convolve(arclat2d, gradkernlon, mode='mirror', origin=0),
        dims=('lon', 'lat'),
    )

    datascalelat = np.divide(datagradlat, arclatgrad)
    datascalelon = np.divide(datagradlon, arclongrad)

    datasumlat = xr.DataArray(
        convolve(datascalelat, sumkernlat, mode='wrap', origin=0),
        dims=('lon', 'lat'),
    )
    datasumlon = xr.DataArray(
        convolve(datascalelon, sumkernlon, mode='wrap', origin=0),
        dims=('lon', 'lat'),
    )

    # do lat and lon differences, can use the gradient kernel

    return [datasumlon[1:-1, 1:-1], datasumlat[1:-1, 1:-1]]


# '''
#
# def grad_wgs84_xr(
#     data: XTypes,
#     wrap_longitude: bool = True,
# ):
#     # remove data and coords
#     # then check data dimensions to slice as needed for mutidimensional arrays
#     # for now I think I run each slice and reassemble again after since it
#     #   cannot be easily done with a kernel convolution
#     # reassemble the xarray and return
#
#     lon2d, lat2d = np.meshgrid(data.coords['lon'], data.coords['lat'])
#     return grad_wgs84(data.values, lon2d, lat2d)
#
#
# def grad_wgs84_np(
#     data: np.array,
#     longitude: np.array,
#     latitude: np.array,
#     wrap_longitude: bool = True,
# ):
#     # todo look into dynamically creating tuples for pad values based on dims
#     if wrap_longitude:
#         datapad = np.pad(data, ((0, 0), (1, 1)), mode='wrap')
#         lonpad = np.pad(longitude, ((0, 0), (1, 1)), mode='wrap')
#         lonpad[:, 0] = lonpad[:, 0] - 360
#         lonpad[:, -1] = lonpad[:, -1] + 360
#         latpad = np.pad(latitude, ((0, 0), (1, 1)), mode='wrap')
#     else:
#         datapad = np.pad(
#             data,
#             ((0, 0), (1, 1)),
#             mode='constant',
#             constant_values=np.nan,
#         )
#         lonpad = np.pad(
#             longitude,
#             ((0, 0), (1, 1)),
#             mode='constant',
#             constant_values=np.nan,
#         )
#         latpad = np.pad(
#             latitude,
#             ((0, 0), (1, 1)),
#             mode='constant',
#             constant_values=np.nan,
#         )
#
#     datapad = np.pad(
#         datapad,
#         ((1, 1), (0, 0)),
#         mode='constant',
#         constant_values=np.nan,
#     )
#     lonpad = np.pad(
#         lonpad,
#         ((1, 1), (0, 0)),
#         mode='constant',
#         constant_values=np.nan,
#     )
#     latpad = np.pad(
#         latpad,
#         ((1, 1), (0, 0)),
#         mode='constant',
#         constant_values=np.nan,
#     )
#
#     arclonpad = arc_lon_wgs84(lonpad, latpad)
#     arclatpad = arc_lat_wgs84(latpad)
#
#     lonresult = np.zeros(data.shape)
#     latresult = np.zeros(data.shape)
#
#     # this can be refactored to use slices for a speed improvement
#     # need specific nan_average function to return appropriate results
#     for latloc in range(1, datapad.shape[0] - 1):
#         for lonloc in range(1, datapad.shape[1] - 1):
#             lonbac = (datapad[latloc, lonloc] - datapad[latloc, lonloc -
#             1]) / \
#                      (arclonpad[latloc, lonloc] - arclonpad[latloc, lonloc
#                      - 1])
#             lonfor = (datapad[latloc, lonloc + 1] - datapad[latloc,
#             lonloc]) / \
#                      (arclonpad[latloc, lonloc + 1] - arclonpad[latloc,
#                      lonloc])
#             if not np.isnan(lonbac) and not np.isnan(lonfor):
#                 longrad = (lonbac + lonfor) / 2
#             elif not np.isnan(lonbac):
#                 longrad = lonbac
#             elif not np.isnan(lonfor):
#                 longrad = lonfor
#             else:
#                 longrad = np.nan
#             lonresult[latloc - 1, lonloc - 1] = longrad
#
#             latbac = (datapad[latloc, lonloc] - datapad[latloc - 1,
#             lonloc]) / \
#                      (arclatpad[latloc, lonloc] - arclatpad[latloc - 1,
#                      lonloc])
#             latfor = (datapad[latloc + 1, lonloc] - datapad[latloc,
#             lonloc]) / \
#                      (arclatpad[latloc + 1, lonloc] - arclatpad[latloc,
#                      lonloc])
#             if not np.isnan(latbac) and not np.isnan(latfor):
#                 latgrad = (latbac + latfor) / 2
#             elif not np.isnan(latbac):
#                 latgrad = latbac
#             elif not np.isnan(latfor):
#                 latgrad = latfor
#             else:
#                 latgrad = np.nan
#             latresult[latloc - 1, lonloc - 1] = latgrad
#
#     return [lonresult, latresult]
##'''
