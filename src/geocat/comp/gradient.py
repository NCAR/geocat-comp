from typing import Union

import numpy as np
import xarray as xr

SupportedTypes = Union[np.ndarray, xr.DataArray]
d2r = 1.745329251994330e-02


def rad_lat_wgs84(lat):
    '''
    based on the taylor series expansion of
    radius = sqrt(0.5*(a^2+b^2+(-a^2+b^2)*cos(2*t*pi/180)))
    where a in the minor axis and b is the major axis.
    it has an accuracy from 0 µm at the equator to 66µm at the poles.

    note: this doesn't need to be a taylor series, though the taylor series
    was a step for the arc_lat_wgs84 function to avoid the eliptic integral.
    '''
    return \
        3.046775254824756e-70 * lat ** 32 + \
        4.322852948553110e-65 * lat ** 30 - \
        3.097819637843901e-60 * lat ** 28 + \
        8.206577043612313e-56 * lat ** 26 + \
        2.185012848730189e-51 * lat ** 24 - \
        3.144459267034150e-46 * lat ** 22 + \
        1.339705312896690e-41 * lat ** 20 - \
        3.647292571651390e-38 * lat ** 18 - \
        3.193974017793312e-32 * lat ** 16 + \
        2.097559357089568e-27 * lat ** 14 - \
        3.932346603086970e-23 * lat ** 12 - \
        4.673469447172801e-18 * lat ** 10 + \
        5.230533973007157e-13 * lat ** 8 - \
        2.614991987936281e-08 * lat ** 6 + \
        6.570167359753477e-04 * lat ** 4 - \
        6.503227445533897e+00 * lat ** 2 + \
        6378137.0


def arc_lat_wgs84(lat):
    '''
    based on the taylor series expansion of
    radius = sqrt(0.5*(a^2+b^2+(-a^2+b^2)*cos(2*t*pi/180)))
    out the the 32nd order.
    The taylor series was integrated to give a fast way to calcuate distance
    along a latitude relative to the prime meridian.
    '''
    return \
        1.611401811060038e-73 * lat ** 33 + \
        2.433807001025704e-68 * lat ** 31 - \
        1.864384562528083e-63 * lat ** 29 + \
        5.304881101096931e-59 * lat ** 27 + \
        1.525426736348904e-54 * lat ** 25 - \
        2.386137713243196e-49 * lat ** 23 + \
        1.113441367452766e-44 * lat ** 21 - \
        3.350382324149892e-41 * lat ** 19 - \
        3.279138990214426e-35 * lat ** 17 + \
        2.440621135815228e-30 * lat ** 15 - \
        5.279415042575557e-26 * lat ** 13 - \
        7.415220849502234e-21 * lat ** 11 + \
        1.014333771848834e-15 * lat ** 9 - \
        6.520031443251421e-11 * lat ** 7 + \
        2.293421056695219e-06 * lat ** 5 - \
        3.783424364354224e-02 * lat ** 3 + \
        111319.4907932736 * lat


def arc_lon_wgs84(lon, lat):
    """this uses."""
    return rad_lat_wgs84(lat) * np.cos(lat * d2r) * lon * d2r


def gradient_sphere(
    data,
    longitude,
    latitude,
    wrap_longitude=True,
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
