from typing import Union

import numpy as np
import scipy.special as sspecial
import scipy.spatial as sspatial
import xarray as xr

SupportedTypes = Union[np.ndarray, xr.DataArray]
default_max_harm = 23  # 300 harmonics from 0,0 to 23,23


def decomposition(
    data: SupportedTypes,
    scale: SupportedTypes,
    theta: SupportedTypes,
    phi: SupportedTypes,
    max_harm: int = default_max_harm,
    chunk_size: int = 'auto',
) -> SupportedTypes:
    """Calculate the spherical harmonics of a dataset.

    Parameters
    ----------
    data : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        2-dimensional dataset

    scale : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        2-dimensional array containing the weighting of each point in the data.
        This is usually the area of the voronoi cell centered on the corresponding datapoint.
        the geocat.comp.spherical.scale_voronoi(theta,phi) function can provide a scale array.

    theta : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        2-dimensional array containing the theta (longitude in radians) values for each datapoint in data.

    phi : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        2-dimensional array containing the theta (latitude in radians) values for each datapoint in data.
        Phi is zero at the top of the sphere and pi at the bottom, phi = (lat_degrees-90)*(-1)*pi/180

    max_harm: :class: `int`, Optional
        The maximum harmonic value for both m and n.
        The total of harmonics calculated is (max_harm+1)*(max_harm+2)/2
        Defaults to 23, for 300 total harmonics.

    chunk_size: :class: `int`, Optional
        The size of the each edge of the dask chunks if using xarray.DataArray inputs.
        Some arrays will be 2d, and others 1d, and the final calculation operates on a 3d array.
        thus the chunks used in the largest calculation scale at chunk_size^3
        A chunk size of 256 is recommended. Defaults to 'auto'

    Returns
    -------
    decomposition : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        the spherical harmonic decomposition of the input data
    """

    # scale_val is the inverse of the total sphere area times the magnitude of
    # the first harmonic. This is used to scale the output so that the output
    # is unaffected by the surface area of the original sphere.
    scale_val = 1 / (np.sum(scale, axis=(0, 1)) *
                     sspecial.sph_harm(0, 0, 0, 0)**2)

    mlist = []  # ordered list of the m harmonics sspecial.sphere(m,n,theta,phi)
    nlist = []  # ordered list of the n harmonics sspecial.sphere(m,n,theta,phi)
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
    m = np.array(mlist)
    n = np.array(nlist)
    scale_mul = np.array(scale_mul)

    # if numpy, change dimensions to allow for broadcast in ss.sph_harm
    if type(data) is np.ndarray:
        m = np.expand_dims(m, axis=(0, 1))
        n = np.expand_dims(n, axis=(0, 1))
        theta = np.expand_dims(theta, axis=(2))
        phi = np.expand_dims(phi, axis=(2))
        scale_res = scale_mul * scale_val
        scale_dat = np.expand_dims(np.multiply(data, scale), axis=(2))

    # if xarray, set dims and chunks for broadcast in ss.sphere_harm
    if type(data) is xr.DataArray:
        m = xr.DataArray(m, dims=['har']).chunk((chunk_size))
        n = xr.DataArray(n, dims=['har']).chunk((chunk_size))
        scale_res = xr.DataArray(
            scale_mul,
            dims=['har'],
        ).chunk((chunk_size)) * scale_val
        scale_dat = xr.DataArray(
            np.multiply(data, scale),
            dims=data.dims,
        ).chunk((chunk_size))
        theta = xr.DataArray(theta, dims=data.dims).chunk((chunk_size))
        phi = xr.DataArray(phi, dims=data.dims).chunk((chunk_size))

    results = np.sum(
        np.multiply(scale_dat, sspecial.sph_harm(m, n, theta, phi)),
        axis=(0, 1),
    ) * scale_res
    return results


def recomposition(
    data: SupportedTypes,
    theta: SupportedTypes,
    phi: SupportedTypes,
    max_harm: int = default_max_harm,
    chunk_size: int = 'auto',
) -> SupportedTypes:
    """Calculate a dataset from spherical harmonics.

    Parameters
    ----------
    data : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        1-dimensional array of spherical harmonics.
        These must by in the same order output by geocat.comp.spherical.decomposition.

    theta : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        2-dimensional array containing the theta (longitude in radians) values for each datapoint in data.

    phi : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        2-dimensional array containing the theta (latitude in radians) values for each datapoint in data.
        Phi is zero at the top of the sphere and pi at the bottom, phi = (lat_degrees-90)*(-1)*pi/180

    max_harm: :class: `int`, Optional
        The maximum harmonic value for both m and n.
        The total of harmonics calculated is (max_harm+1)*(max_harm+2)/2
        The number of total harmonics must equal the number of harmoncs in the input data.
        Defaults to 23, for 300 total harmonics.

    chunk_size: :class: `int`, Optional
        The size of the each edge of the dask chunks if using xarray.DataArray inputs.
        Some arrays will be 2d, and others 1d, and the final calculation operates on a 3d array.
        thus the chunks used in the largest calculation scale at chunk_size^3
        A chunk size of 256 is recommended. Defaults to 'auto'

    Returns
    -------
    recomposition : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        the spherical harmonic recomposition of the input data
    """

    mlist = []  # ordered list of the m harmonics sspecial.sphere(m,n,theta,phi)
    nlist = []  # ordered list of the n harmonics sspecial.sphere(m,n,theta,phi)
    for nvalue in range(max_harm + 1):
        for mvalue in range(nvalue + 1):
            mlist.append(mvalue)
            nlist.append(nvalue)
    m = np.array(mlist)
    n = np.array(nlist)

    # if numpy, change dimensions to allow for broadcast in ss.sph_harm
    if type(data) is np.ndarray:
        m = np.expand_dims(m, axis=(1, 2))
        n = np.expand_dims(n, axis=(1, 2))
        data = np.expand_dims(data, axis=(1, 2))
        theta = np.expand_dims(theta, axis=(0))
        phi = np.expand_dims(phi, axis=(0))

    # if xarray, set dims and chunks for broadcast in ss.sphere_harm
    if type(data) is xr.DataArray:
        m = xr.DataArray(m, dims=['har']).chunk((chunk_size))
        n = xr.DataArray(n, dims=['har']).chunk((chunk_size))
        data = xr.DataArray(data, dims=['har']).chunk((chunk_size))
        theta = xr.DataArray(theta, dims=theta.dims).chunk((chunk_size))
        phi = xr.DataArray(phi, dims=phi.dims).chunk((chunk_size))

    results = np.sum(
        np.multiply(sspecial.sph_harm(m, n, theta, phi).real, data.real),
        axis=(0),
    ) + np.sum(
        np.multiply(sspecial.sph_harm(m, n, theta, phi).imag, data.imag),
        axis=(0),
    )

    return results.real


def scale_voronoi(
    theta: SupportedTypes,
    phi: SupportedTypes,
    chunk_size: int = 'auto',
) -> SupportedTypes:
    """Calculate the area weighting for dataset.

    Parameters
    ----------
    theta : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        2-dimensional array containing the theta (longitude in radians) values for each datapoint in data.

    phi : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        2-dimensional array containing the theta (latitude in radians) values for each datapoint in data.
        Phi is zero at the top of the sphere and pi at the bottom, phi = (lat_degrees-90)*(-1)*pi/180

    chunk_size: :class: `int`, Optional
        The size of the each edge of the dask chunks if using xarray.DataArray inputs.
        Some arrays will be 2d, and others 1d, and the final calculation operates on a 3d array.
        thus the chunks used in the largest calculation scale at chunk_size^3
        A chunk size of 256 is recommended. Defaults to 'auto'

    Returns
    -------
    scale : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        2-dimensional array containing the area of the spherical voronoi cell for each theta and phi pair.
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
        sspatial.SphericalVoronoi(
            data_locs_3d,
            radius=1.0,
            center=np.array([0, 0, 0]),
        ).calculate_areas()).reshape(theta.shape)

    if type(theta) is xr.DataArray:
        scale = xr.DataArray(scale, dims=theta.dims).chunk(chunk_size)

    return scale
