from typing import Union

import numpy as np
from packaging.version import Version
from scipy.spatial import SphericalVoronoi
from scipy import __version__ as scipy_version
import xarray as xr

# import scipy shp_harm[_y] function depending on scipy version
old_scipy = False
scipy_version = Version(scipy_version)
if scipy_version < Version('1.15.0'):
    from scipy.special import sph_harm

    def sph_harm_y(n, m, theta, phi):
        return sph_harm(m, n, phi, theta)

    old_scipy = True
else:
    from scipy.special import sph_harm_y

SupportedTypes = Union[np.ndarray, xr.DataArray]
default_max_harm = 23  # 300 harmonics from 0,0 to 23,23


def decomposition(
    data: SupportedTypes,
    scale: SupportedTypes,
    theta: SupportedTypes,
    phi: SupportedTypes,
    max_harm: int = default_max_harm,
) -> SupportedTypes:
    """Calculate the spherical harmonics of a dataset.

    Parameters
    ----------
    data : ndarray, :class:`xarray.DataArray`
        2-dimensional dataset

    scale : ndarray, :class:`xarray.DataArray`
        2-dimensional array containing the weighting of each point in the data.
        This is usually the area of the voronoi cell centered on the corresponding datapoint.
        the ``geocat.comp.spherical.scale_voronoi(theta,phi)`` function can provide a scale array.

    theta : ndarray, :class:`xarray.DataArray`
        2-dimensional array containing the theta (longitude in radians) values for each datapoint in data.

    phi : ndarray, :class:`xarray.DataArray`
        2-dimensional array containing the ``theta`` (latitude in radians) values for each datapoint in data.
        ``Phi`` is zero at the top of the sphere and ``pi`` at the bottom, ``phi = (lat_degrees-90)*(-1)*pi/180``

    max_harm: int, optional
        The maximum harmonic value for both m and n.
        The total of harmonics calculated is ``(max_harm+1)*(max_harm+2)/2``
        Defaults to 23, for 300 total harmonics.

    Returns
    -------
    decomposition : ndarray, :class:`xarray.DataArray`
        The spherical harmonic decomposition of the input data
    """

    in_type = type(data)

    # scale_val is the inverse of the total sphere area times the magnitude of
    # the first harmonic. This is used to scale the output so that the output
    # is unaffected by the surface area of the original sphere.
    scale_val = np.array(
        (1 / (np.sum(scale, axis=(0, 1)) * sph_harm_y(0, 0, 0, 0) ** 2))
    )

    mlist = []  # ordered list of the m harmonics
    nlist = []  # ordered list of the n harmonics
    # the real value of the output varies on a factor of two when m is 0,
    # due to m=0 being a symmetric case with no imaginary component,
    # this accounts for that difference in output sum.
    scale_mul = []
    for nvalue in range(max_harm + 1):
        for mvalue in range(nvalue + 1):
            mlist.append(mvalue)
            nlist.append(nvalue)
            if mvalue == 0:
                scale_mul.append(1)
            else:
                scale_mul.append(2)
    m = np.expand_dims(mlist, axis=(1, 2))
    n = np.expand_dims(nlist, axis=(1, 2))
    scale_mul = np.array(scale_mul)
    scale_res = scale_mul * scale_val
    scale_dat = np.multiply(data, scale)

    # set dims for broadcasting
    if in_type is xr.DataArray:
        if not old_scipy:
            scale_res = xr.DataArray(scale_res, dims=['harmonic'])
            scale_dat = xr.DataArray(scale_dat, dims=data.dims)
            theta = xr.DataArray(theta, dims=data.dims)
            phi = xr.DataArray(phi, dims=data.dims)
        else:
            theta = np.expand_dims(theta, axis=0)
            phi = np.expand_dims(phi, axis=0)

        # np.multiply produces (lat, lon, harmonic) dimensions for xarray input
        sum_ax = (0, 1)
    else:
        theta = np.expand_dims(theta, axis=0)
        phi = np.expand_dims(phi, axis=0)

        # np.multiply produces (harmonic, lat, lon) dimensions for numpy input
        sum_ax = (1, 2)

    # scipy 1.15 flips the definitions of theta and phi
    theta, phi = phi, theta

    harmonics = sph_harm_y(n, m, theta, phi)

    # if xarray, make harmonics into xarray and align dims
    if in_type is xr.DataArray:
        harmonics = xr.DataArray(
            harmonics, dims=['harmonic', data.dims[0], data.dims[1]]
        )

    results = np.sum(np.multiply(scale_dat, harmonics), axis=sum_ax) * scale_res

    return results


def recomposition(
    data: SupportedTypes,
    theta: SupportedTypes,
    phi: SupportedTypes,
    max_harm: int = default_max_harm,
) -> SupportedTypes:
    """Calculate a dataset from spherical harmonics.

    Parameters
    ----------
    data : ndarray, :class:`xarray.DataArray`
        1-dimensional array of spherical harmonics.
        These must be in the same order output by ``geocat.comp.spherical.decomposition``.

    theta : ndarray, :class:`xarray.DataArray`
        2-dimensional array containing the theta (longitude in radians) values for each datapoint in data.

    phi : ndarray, :class:`xarray.DataArray`
        2-dimensional array containing the theta (latitude in radians) values for each datapoint in data.
        ``Phi`` is zero at the top of the sphere and ``pi`` at the bottom, ``phi = (lat_degrees-90)*(-1)*pi/180``

    max_harm: int, optional
        The maximum harmonic value for both m and n.
        The total of harmonics calculated is ``(max_harm+1)*(max_harm+2)/2``
        The number of total harmonics must equal the number of harmoncs in the input data.
        Defaults to 23, for 300 total harmonics.

    Returns
    -------
    recomposition : ndarray, :class:`xarray.DataArray`
        The spherical harmonic recomposition of the input data
    """

    in_type = type(data)

    # if xarray, standardize data input dimension names
    if in_type is xr.DataArray:
        data = data.rename({data.dims[0]: 'harmonic'})
        theta_dims = theta.dims

    mlist = []  # ordered list of the m harmonics
    nlist = []  # ordered list of the n harmonics
    for nvalue in range(max_harm + 1):
        for mvalue in range(nvalue + 1):
            mlist.append(mvalue)
            nlist.append(nvalue)

    m = np.expand_dims(mlist, axis=(1, 2))
    n = np.expand_dims(nlist, axis=(1, 2))

    # set dims for broadcasting
    if in_type is xr.DataArray and not old_scipy:
        theta = theta.expand_dims(dim='harmonic', axis=0)
        phi = phi.expand_dims(dim='harmonic', axis=0)
    else:
        data = np.expand_dims(data, axis=(1, 2))
        theta = np.expand_dims(theta, axis=0)
        phi = np.expand_dims(phi, axis=0)

    # scipy 1.15 flips the definitions of theta and phi
    theta, phi = phi, theta

    harmonics = sph_harm_y(n, m, theta, phi)

    # if xarray, make harmonics into xarray and align dims
    if in_type is xr.DataArray:
        harmonics = xr.DataArray(
            harmonics, dims=['harmonic', theta_dims[0], theta_dims[1]]
        )

    results = np.sum(np.multiply(harmonics.real, data.real), axis=0) + np.sum(
        np.multiply(harmonics.imag, data.imag), axis=0
    )

    return results.real


def scale_voronoi(
    theta: SupportedTypes,
    phi: SupportedTypes,
) -> SupportedTypes:
    """Calculate the area weighting for dataset.

    Parameters
    ----------
    theta : ndarray, :class:`xarray.DataArray`
        2-dimensional array containing the theta (longitude in radians) values for each datapoint in data.

    phi : ndarray, :class:`xarray.DataArray`
        2-dimensional array containing the theta (latitude in radians) values for each datapoint in data.
        ``Phi`` is zero at the top of the sphere and ``pi`` at the bottom, ``phi = (lat_degrees-90)*(-1)*pi/180``

    Returns
    -------
    scale : ndarray, :class:`xarray.DataArray`
        2-dimensional array containing the area of the spherical voronoi cell for each ``theta`` and ``phi`` pair.
    """

    if type(theta) is xr.DataArray:
        theta = theta.to_numpy()
        phi = phi.to_numpy()

    theta_1d = theta.reshape((theta.shape[0] * theta.shape[1],))
    phi_1d = phi.reshape((phi.shape[0] * phi.shape[1],))

    data_locs_3d = np.zeros((len(phi_1d), 3))
    data_locs_3d[:, 0] = np.sin(phi_1d) * np.sin(theta_1d)
    data_locs_3d[:, 1] = np.sin(phi_1d) * np.cos(theta_1d)
    data_locs_3d[:, 2] = np.cos(phi_1d)

    scale = np.array(
        SphericalVoronoi(
            data_locs_3d,
            radius=1.0,
            center=np.array([0, 0, 0]),
        ).calculate_areas()
    ).reshape(theta.shape)

    if type(theta) is xr.DataArray:
        scale = xr.DataArray(scale, dims=theta.dims)

    return scale
