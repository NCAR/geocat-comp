from typing import Union

import numpy as np
import scipy.special as ss
import xarray as xr

SupportedTypes = Union[np.ndarray, xr.DataArray]
default_max_harm = 23  # 300 harmonics from 0,0 to 23,23


def harmonic_decomposition(
    data: SupportedTypes,
    scale: SupportedTypes,
    theta: SupportedTypes,
    phi: SupportedTypes,
    max_harm: int = default_max_harm,
    chunk_size: dict = {},
) -> SupportedTypes:

    scale_val = 1 / (np.sum(scale, axis=(0, 1)) * ss.sph_harm(0, 0, 0, 0)**2)
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
        scale_dat = xr.DataArray(np.multiply(data, scale),
                                 dims=data.dims).chunk((chunk_size))
        theta = xr.DataArray(theta, dims=data.dims).chunk((chunk_size))
        phi = xr.DataArray(phi, dims=data.dims).chunk((chunk_size))

    results = np.sum(np.multiply(scale_dat, ss.sph_harm(m, n, theta, phi)),
                     axis=(0, 1)) * scale_mul * scale_val
    return results


def harmonic_recomposition(
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

    results = np.sum(np.multiply(data, ss.sph_harm(m, n, theta, phi)), axis=(0))

    return results.real
