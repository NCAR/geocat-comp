from typing import Union

import numpy as np
import scipy.special as ss
import xarray as xr

Harms = Union[list[list]]  # , list[tuple], tuple[list], tuple[tuple]]
ValidData = Union[np.array, xr.DataArray]


def harmonic_decomposition(
    input_data: ValidData,
    input_scale: ValidData,
    input_theta: ValidData,
    input_phi: ValidData,
    harms: Harms = None,
    max_harm: int = None,
) -> list[ValidData]:
    # if no harmonic info provided by the user:
    if max_harm is None and harms is None:
        max_harm = 24  # 300 total harmonics

    # in the case of max_harm, provide full set up to max_harm
    if harms is None and max_harm is not None:
        harms = []
        for n in range(max_harm):
            for m in range(n + 1):
                harms.append([m, n])

    results = []
    input_data_scaled = np.multiply(demo_data, scale_phi).persist()
    for harm in harms:
        results.append(
            np.sum(np.multiply(
                input_data_scaled,
                ss.sph_harm(harm[0], harm[1], input_theta, input_phi)),
                   axis=(0, 1)))

    return results


# ''' todo
# def harmonic_recomposition(
#         input_data: ValidData,
#         input_theta: ValidData,
#         input_phi: ValidData,
#         harms: Harms,
# ):
