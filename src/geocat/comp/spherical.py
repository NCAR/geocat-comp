from typing import Union

import numpy as np
import scipy.special as ss
import xarray as xr

Harms = Union[list[list]]  # , list[tuple], tuple[list], tuple[tuple]]
InputData = Union[np.array, xr.DataArray]


def harmonic_decomposition(
    input_data: InputData,
    input_theta: InputData,
    input_phi: InputData,
    harms: Harms = None,
    max_harm: int = None,
):
    results = []

    # if no harmonic info provided by the user:
    if max_harm is None and harms is None:
        max_harm = 24  # 300 total harmonics

    # in the case of max_harm, provide full set up to max_harm
    if harms is None and max_harm is not None:
        harms = []
        for n in range(max_n):
            for m in range(n + 1):
                harms.append([m, n])

    for m, n in harms:
        results.append(
            np.sum(np.multiply(input_data, ss.sph_harm(m, n, theta, phi)),
                   axis=(0, 1)))


# ''' todo
# def harmonic_recomposition(
#         input_data: InputData,
#         input_theta: InputData,
#         input_phi: InputData,
#         harms: Harms,
# ):
