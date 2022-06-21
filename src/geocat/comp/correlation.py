import numpy as np
import xarray as xr

x = xr.DataArray(np.arange(0, 34))
y = xr.DataArray([5,5,3,4,8,6,4,5,6,66,4,6,16,5,3,3,4,9,2,8,2,24,15,62,25,16,39,38,35,30,43,50,56,51])
w = xr.DataArray([1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4])

# TODO: make compatible with dask
# TODO: documentation
# TODO: make test file
def covar(x, y, weights=None):
    if weights is None:
        numer = ((x - x.mean()) * (y - y.mean())).sum()
        denom = len(x) - 1
    else:
        # Frequency weights https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Frequency_weights_2
        numer = (weights * (x - x.weighted(weights).mean()) * (y - y.weighted(weights).mean())).sum()
        denom = weights.sum() - 1
    return numer / denom


def variance(x, weights=None):
    # Unweighted
    if weights is None:
        numer = (np.power(x - x.mean(), 2)).sum()
        denom = x.size - 1
    else:
        # Frequency Weights https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Frequency_weights
        numer = (weights * np.power(x - x.weighted(weights).mean(), 2)).sum()
        denom = weights.sum() - 1
    return numer / denom


def correlation_coeff(x, y, weights=None):
    return covar(x, y, weights) / np.sqrt(covar(x, x, weights) * covar(y, y, weights))


print("Covar x,x: " + str(covar(x,x,w)))
print("Var x: " + str(variance(x, w)))

print()
print("Covar y,y: " + str(covar(y,y,w)))
print("Var y: " + str(variance(y,w)))

print()
print("Covar x,y: " + str(covar(x,y,w)))
print("Covar x,y: " + str(covar(x,y)))

print()
print("r weighted: " + str(correlation_coeff(x,y,weights=w)))
print("r unweighted: " + str(correlation_coeff(x, y)))