from typing import Union

import numpy as np
import scipy.special as ss
import xarray as xr

SupportedTypes = Union[np.array, xr.DataArray]
default_max_harm = 23  # 300 harmonics from 0,0 to 23,23


def harmonic_decomposition(
    data: SupportedTypes,
    scale: SupportedTypes,
    theta: SupportedTypes,
    phi: SupportedTypes,
    max_harm: int = default_max_harm,
    chunk_size={},
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
    data_scaled = np.multiply(data, scale)

    # if numpy, change dimensions to allow for broadcast in ss.sph_harm
    if type(data) is np.ndarray:
        m = np.expand_dims(m, axis=(0, 1))
        n = np.expand_dims(n, axis=(0, 1))
        theta = np.expand_dims(theta, axis=(2))
        phi = np.expand_dims(phi, axis=(2))
        data_scaled = np.expand_dims(data_scaled, axis=(2))

    # if xarray, set dims and chunks for
    if type(data) is xr.DataArray:
        m = xr.DataArray(m, dims=['har']).chunk((chunk_size))
        n = xr.DataArray(n, dims=['har']).chunk((chunk_size))
        scale_mul = xr.DataArray(scale_mul, dims=['har']).chunk((chunk_size))
        data_scaled = \
            xr.DataArray(data_scaled, dims=['lat', 'lon']).chunk((chunk_size))
        theta = xr.DataArray(theta, dims=['lat', 'lon']).chunk((chunk_size))
        phi = xr.DataArray(phi, dims=['lat', 'lon']).chunk((chunk_size))

    results = np.sum(np.multiply(data_scaled, ss.sph_harm(m, n, theta, phi)),
                     axis=(0, 1)) * scale_mul * scale_val
    return results


def harmonic_recomposition(
    input_data: SupportedTypes,
    input_theta: SupportedTypes,
    input_phi: SupportedTypes,
    harms: SupportedTypes = None,
    max_harm: int = default_max_harm,
) -> SupportedTypes:
    # if no harmonic info provided by the user:
    if max_harm is None and harms is None:
        max_harm = default_max_harm

        # in the case of max_harm, provide full set up to max_harm
        harms = []
        for n in range(max_harm + 1):
            for m in range(n + 1):
                harms.append([m, n])

    # return same data type as input
    results = np.zeros(input_theta.shape, dtype=complex)
    if type(input_theta) is xr.DataArray:
        results = xr.DataArray(results,
                               dims=input_theta.dims).chunk(input_theta.chunks)

    for harm, value in zip(harms, input_data):
        sphere = ss.sph_harm(harm[0], harm[1], input_theta, input_phi)
        results += sphere.real * value.real + sphere.imag * value.imag

    return results.real
