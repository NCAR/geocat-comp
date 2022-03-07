from typing import Union

import numpy as np
import scipy.special as ss
import xarray as xr

DataA = Union[np.array, xr.DataArray]
Harms = Union[list[list], DataA]
default_max_harm = 23  # 300 harmonics from 0,0 to 23,23


def harmonic_decomposition(
    input_data: DataA,
    input_scale: DataA,
    input_theta: DataA,
    input_phi: DataA,
    harms: Harms = None,
    max_harm: int = None,
) -> DataA:
    # if no harmonic info provided by the user:
    if max_harm is None and harms is None:
        max_harm = default_max_harm

    # in the case of max_harm, provide full set up to max_harm
    if harms is None and max_harm is not None:
        harms = []
        for n in range(max_harm + 1):
            for m in range(n + 1):
                harms.append([m, n])

    results = []
    scale0 = 1 / (np.sum(input_scale, axis=(0, 1)) * ss.sph_harm(0, 0, 0, 0)**2)
    if input_scale is xr.DataArray:
        scale0 = scale0.compute().persist()
    scale1 = scale0 * 2
    input_data_scaled = np.multiply(input_data, input_scale)
    if input_data is xr.DataArray:
        input_data_scaled = input_data_scaled.persist()

    for harm in harms:
        results.append(
            np.sum(np.multiply(
                input_data_scaled,
                ss.sph_harm(harm[0], harm[1], input_theta, input_phi)),
                   axis=(0, 1)))
        if harm[0] == 0:
            results[-1] = results[-1] * scale0
        else:
            results[-1] = results[-1] * scale1

    # return same data type as input
    results = np.asarray(results)
    if type(input_data) is xr.DataArray:
        results = xr.DataArray(results)
    return results


def harmonic_recomposition(
    input_data: DataA,
    input_theta: DataA,
    input_phi: DataA,
    harms: Harms = None,
    max_harm: int = None,
) -> DataA:
    # if no harmonic info provided by the user:
    if max_harm is None and harms is None:
        max_harm = default_max_harm

    # in the case of max_harm, provide full set up to max_harm
    if harms is None and max_harm is not None:
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

    return results
