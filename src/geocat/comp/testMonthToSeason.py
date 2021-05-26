import xarray as xr
import pandas as pd
import numpy as np
import time
from climatology import month_to_season, month_to_season_optimized

dates = pd.date_range(start="2000/01/01", freq="M", periods=60)
million_6 = xr.DataArray(np.random.rand(60, 10, 100, 100), dims=["time", "lat", "lon", "lev"], coords={"time": dates})
million_60 = xr.DataArray(np.random.rand(60, 100, 100, 100), dims=["time", "lat", "lon", "lev"], coords={"time": dates})
million_600 = xr.DataArray(np.random.rand(60, 100, 100, 1000), dims=["time", "lat", "lon", "lev"], coords={"time": dates})

print('---Test 6 million data points---')
print('Old Method')
start = time.perf_counter()
data = month_to_season(million_6, "DJF")
end = time.perf_counter()
old_time = end - start
print(old_time, "seconds")

print('\nNew Method')
start = time.perf_counter()
data = month_to_season_optimized(million_6, "DJF")
end = time.perf_counter()
new_time = end-start
print(new_time, "seconds")

print('\nChange: ', (old_time-new_time)/old_time * 100, '%')

print('\n\n---Test 60 million data points---')
print('Old Method')
start = time.perf_counter()
data = month_to_season(million_60, "DJF")
end = time.perf_counter()
old_time = end - start
print(old_time, "seconds")

print('New Method')
start = time.perf_counter()
data = month_to_season_optimized(million_60, "DJF")
end = time.perf_counter()
new_time = end - start
print(new_time, "seconds")
print('\nChange: ', (old_time-new_time)/old_time * 100, '%')

print('\n\n---Test 600 million data points---')
print('Old Method')
start = time.perf_counter()
data = month_to_season(million_600, "DJF")
end = time.perf_counter()
old_time = end - start
print(old_time, "seconds")

print('New Method')
start = time.perf_counter()
data = month_to_season_optimized(million_600, "DJF")
end = time.perf_counter()
new_time = end - start
print(new_time, "seconds")
print('\nChange: ', (old_time-new_time)/old_time * 100, '%')