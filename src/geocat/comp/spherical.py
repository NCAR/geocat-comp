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
    chunk_size: dict = {},
) -> SupportedTypes:
    """Calculate the spherical harmonics of a dataset. This function allows for
    the use of any 2d grid.

    Parameters
    ----------
    data : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        2-dimensional dataset

    scale : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        2-dimensional array containing the weighting of each point in the data. This is usually the area of the voronoi cell centered on the corresponding datapoint.



    Returns
    -------
    decomposition : :class:`numpy.ndarray`, :class:`xarray.DataArray`
        the spherical harmonic decomposition of the input data
    """

    scale_val = 1 / (np.sum(scale, axis=(0, 1)) *
                     sspecial.sph_harm(0, 0, 0, 0)**2)
    scale_mul = []
    mlist = []
    nlist = []
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
        scale_dat = np.expand_dims(np.multiply(data, scale), axis=(2))

    # if xarray, set dims and chunks for broadcast in ss.sphere_harm
    if type(data) is xr.DataArray:
        m = xr.DataArray(m, dims=['har']).chunk((chunk_size))
        n = xr.DataArray(n, dims=['har']).chunk((chunk_size))
        scale_mul = xr.DataArray(scale_mul, dims=['har']).chunk((chunk_size))
        scale_dat = xr.DataArray(
            np.multiply(data, scale),
            dims=data.dims,
        ).chunk((chunk_size))
        theta = xr.DataArray(theta, dims=data.dims).chunk((chunk_size))
        phi = xr.DataArray(phi, dims=data.dims).chunk((chunk_size))

    results = np.sum(
        np.multiply(scale_dat, sspecial.sph_harm(m, n, theta, phi)),
        axis=(0, 1),
    ) * scale_mul * scale_val
    return results


def recomposition(
    data: SupportedTypes,
    theta: SupportedTypes,
    phi: SupportedTypes,
    max_harm: int = default_max_harm,
    chunk_size: dict = {},
) -> SupportedTypes:
    mlist = []
    nlist = []
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
        theta = xr.DataArray(theta, dims=theta.dims).chunk((chunk_size))
        phi = xr.DataArray(phi, dims=phi.dims).chunk((chunk_size))
        data = xr.DataArray(data, dims=['har']).chunk((chunk_size))

    results = np.sum(np.multiply(
        sspecial.sph_harm(m, n, theta, phi).real, data.real),
                     axis=(0)) + np.sum(np.multiply(
                         sspecial.sph_harm(m, n, theta, phi).imag, data.imag),
                                        axis=(0))

    return results.real


def scale_voronoi(
    theta: SupportedTypes,
    phi: SupportedTypes,
    chunk_size: dict = {},
) -> SupportedTypes:
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
        )).reshape(theta.shape)
    if type(theta) is xr.DataArray:
        scale = xr.DataArray(scale, dims=theta.dims).chunk(chunk_size)
    return scale
