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
        for n in range(max_harm + 1):
            for m in range(n + 1):
                harms.append([m, n])

    results = []
    scale0 = 1 / (np.sum(scale_phi, axis=(0, 1)) * ss.sph_harm(0, 0, 1, 1)**2)
    scale0 = scale0.compute().persist()
    scale1 = scale0 * 2
    # print(scale1.compute())
    input_data_scaled = np.multiply(demo_data, input_scale).persist()
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

    return results


# ''' todo
# def harmonic_recomposition(
#         input_data: ValidData,
#         input_theta: ValidData,
#         input_phi: ValidData,
#         harms: Harms,
# ):
