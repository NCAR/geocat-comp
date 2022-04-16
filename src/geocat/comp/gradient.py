from typing import Union

import numpy as np
import xarray as xr

SupportedTypes = Union[np.ndarray, xr.DataArray]
d2r = 1.745329251994330e-02


def rad_lat_wgs84(lat: SupportedTypes,):
    '''
    The radius calculation for the wgs84 ellipsoid uses a taylor series from
    radius = ((a*cos(lat))**2+(b*sin(lat))**2)**(1/2)
    The taylor series is the radius of the elipsoid for a given latitude
    This is accurate to within floating point error.

    note: This doesn't need to be a taylor series, though the taylor series
    was a step for the arc_lat_wgs84 function to avoid the eliptic integral
    '''
    return \
        8.05993093298849335845093e-107 * lat ** 48 - \
        1.26581811433077485833316e-102 * lat ** 46 - \
        9.10951565786630085206383e-98 * lat ** 44 + \
        7.49821836157842689028464e-93 * lat ** 42 - \
        2.27271626937506094190001e-88 * lat ** 40 - \
        2.71379394421318726884417e-84 * lat ** 38 + \
        6.14832871790710039645791e-79 * lat ** 36 - \
        2.84993185799516722577218e-74 * lat ** 34 + \
        3.04677525482475781021008e-70 * lat ** 32 + \
        4.32285294855311071974471e-65 * lat ** 30 - \
        3.09781963784390194038817e-60 * lat ** 28 + \
        8.20657704361231447958419e-56 * lat ** 26 + \
        2.18501284873018898427381e-51 * lat ** 24 - \
        3.14445926703415013437784e-46 * lat ** 22 + \
        1.33970531289669032782347e-41 * lat ** 20 - \
        3.64729257165139301286738e-38 * lat ** 18 - \
        3.19397401779331194046889e-32 * lat ** 16 + \
        2.09755935708956809813879e-27 * lat ** 14 - \
        3.93234660308697071375021e-23 * lat ** 12 - \
        4.67346944717280103014443e-18 * lat ** 10 + \
        5.23053397300715692655636e-13 * lat ** 8 - \
        2.61499198793628148606271e-08 * lat ** 6 + \
        6.57016735975347744792879e-04 * lat ** 4 - \
        6.50322744553389692456410e+00 * lat ** 2 + \
        6378137.0


def arc_lat_wgs84(lat: SupportedTypes,):
    '''
    The arc length calculation for the wgs84 ellipsoid uses a taylor series from
    radius = ((a*cos(lat))**2+(b*sin(lat))**2)**(1/2)
    The integral of the radius taylor series gives an arc length taylor series
    The taylor series is the distance from the equator to a given latitude
    This is accurate to within floating point error.
    '''
    return \
        2.87086392375485020807270e-110 * lat ** 49 - \
        4.70057315456554051027470e-106 * lat ** 47 - \
        3.53313425537209421651981e-101 * lat ** 45 + \
        3.04345577774507517984383e-96 * lat ** 43 - \
        9.67472728396028867902680e-92 * lat ** 41 - \
        1.21447793710778038581080e-87 * lat ** 39 + \
        2.90023188168650132848542e-82 * lat ** 37 - \
        1.42116269655700431839954e-77 * lat ** 35 + \
        1.61140181106003861674601e-73 * lat ** 33 + \
        2.43380700102570413214524e-68 * lat ** 31 - \
        1.86438456252808355807529e-63 * lat ** 29 + \
        5.30488110109693238696142e-59 * lat ** 27 + \
        1.52542673634890397698471e-54 * lat ** 25 - \
        2.38613771324319612960829e-49 * lat ** 23 + \
        1.11344136745276656534003e-44 * lat ** 21 - \
        3.35038232414989498944290e-41 * lat ** 19 - \
        3.27913899021442625628565e-35 * lat ** 17 + \
        2.44062113581522849116351e-30 * lat ** 15 - \
        5.27941504257555801373336e-26 * lat ** 13 - \
        7.41522084950223435244285e-21 * lat ** 11 + \
        1.01433377184883444985903e-15 * lat ** 9 - \
        6.52003144325142154116932e-11 * lat ** 7 + \
        2.29342105669521921399609e-06 * lat ** 5 - \
        3.78342436435422366967231e-02 * lat ** 3 + \
        111319.490793273572647713 * lat


def arc_lon_wgs84(
    lon: SupportedTypes,
    lat: SupportedTypes,
):
    """this uses."""
    return rad_lat_wgs84(lat) * np.cos(lat * d2r) * lon * d2r


def gradient_sphere(
    data: SupportedTypes,
    longitude: SupportedTypes,
    latitude: SupportedTypes,
    wrap_longitude: bool = True,
):
    datapad = np.pad(data, ((0, 0), (1, 1)), mode='wrap')
    datapad = np.pad(
        datapad,
        ((1, 1), (0, 0)),
        mode='constant',
        constant_values=np.nan,
    )

    if wrap_longitude:
        latpad = np.pad(latitude, ((0, 0), (1, 1)), mode='wrap')
        lonpad = np.pad(longitude, ((0, 0), (1, 1)), mode='wrap')
        lonpad[:, 0] = lonpad[:, 0] - 360
        lonpad[:, -1] = lonpad[:, -1] + 360
    else:
        latpad = np.pad(
            latitude,
            ((0, 0), (1, 1)),
            mode='constant',
            constant_values=np.nan,
        )
        lonpad = np.pad(
            longitude,
            ((0, 0), (1, 1)),
            mode='constant',
            constant_values=np.nan,
        )

    lonpad = np.pad(
        lonpad,
        ((1, 1), (0, 0)),
        mode='constant',
        constant_values=np.nan,
    )
    latpad = np.pad(
        latpad,
        ((1, 1), (0, 0)),
        mode='constant',
        constant_values=np.nan,
    )

    arclonpad = arc_lon_wgs84(lonpad, latpad)
    arclatpad = arc_lat_wgs84(latpad)

    lonresult = np.zeros(data.shape)
    latresult = np.zeros(data.shape)

    for latloc in range(1, datapad.shape[0] - 1):
        for lonloc in range(1, datapad.shape[1] - 1):
            lonbac = (datapad[latloc, lonloc] - datapad[latloc, lonloc - 1]) / \
                     (arclonpad[latloc, lonloc] - arclonpad[latloc, lonloc - 1])
            lonfor = (datapad[latloc, lonloc + 1] - datapad[latloc, lonloc]) / \
                     (arclonpad[latloc, lonloc + 1] - arclonpad[latloc, lonloc])
            if not np.isnan(lonbac) and not np.isnan(lonfor):
                longrad = (lonbac + lonfor) / 2
            elif not np.isnan(lonbac):
                longrad = lonbac
            elif not np.isnan(lonfor):
                longrad = lonfor
            else:
                longrad = np.nan
            lonresult[latloc - 1, lonloc - 1] = longrad

            latbac = (datapad[latloc, lonloc] - datapad[latloc - 1, lonloc]) / \
                     (arclatpad[latloc, lonloc] - arclatpad[latloc - 1, lonloc])
            latfor = (datapad[latloc + 1, lonloc] - datapad[latloc, lonloc]) / \
                     (arclatpad[latloc + 1, lonloc] - arclatpad[latloc, lonloc])
            if not np.isnan(latbac) and not np.isnan(latfor):
                latgrad = (latbac + latfor) / 2
            elif not np.isnan(latbac):
                latgrad = latbac
            elif not np.isnan(latfor):
                latgrad = latfor
            else:
                latgrad = np.nan
            latresult[latloc - 1, lonloc - 1] = latgrad

    return [lonresult, latresult]
