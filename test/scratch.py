import numpy as np
import xarray as xr
from src.geocat.comp.interpolation import interp_hybrid_to_pressure
import matplotlib.pyplot as plt
ds_in = xr.open_dataset("ccsm35.h0.0021-01.demo.nc", decode_times=False)

_hyam = ds_in.hyam
_hybm = ds_in.hybm
_p0 = 1000. * 100  # Pa

data = ds_in.T[:, :, 0:80, 0:80]
ps = ds_in.PS[:, 0:80, 0:80]
new_levels = np.asarray([100, 200, 300, 400, 500, 600, 700, 750, 850, 925, 950, 1000])

ts = ds_in.TS[:, 0:80, 0:80]
phis = ds_in.PHIS[:, 0:80, 0:80]

output = interp_hybrid_to_pressure(data,
                          ps,
                          _hyam,
                          _hybm,
                          p0=_p0,
                          new_levels=new_levels,
                          extrapolate=True,
                          var='temperature',
                          t_sfc=ts,
                          phi_sfc=phis)
print(output)