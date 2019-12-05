from typing import Iterable

import numpy as np
import xarray as xr
import dask.array as da


def ndpolyfit(x: Iterable, y: Iterable, **kwargs) -> (xr.DataArray, da.Array):
    axis = kwargs.get("axis", 0)
    deg = kwargs.get("deg", 1)
    rcond = kwargs.get("rcond", None)
    full = kwargs.get("full", False)
    w = kwargs.get("w", None)
    cov = kwargs.get("cov", False)
    missing_value = kwargs.get("missing_value", np.nan)
    meta = kwargs.get("meta", True)

    # converting x to numpy.ndarray
    if isinstance(x, xr.DataArray):
        x = x.data
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x)

    if isinstance(y, np.ndarray):
        return _ndpolyfit(y, x, axis, deg, rcond, full, w, cov, missing_value)
    if isinstance(y, xr.DataArray):
        output = _ndpolyfit(y.data, x, axis, deg, rcond, full, w, cov, missing_value)
        if meta:
            for attr, v in y.attrs.items():
                output.attrs[attr] = v

            output.dims = [("poly_coef" if i == max(axis, axis+y.ndim) else output.dims[i]) for i in range(y.ndim)]
            output.coords = {k: v for (k, v) in y.coords.items() if k != y.dims[max(axis, axis+y.ndim)]}
            output.coords[axis] = list(range(int(deg)))
        return output
    if isinstance(y, da.Array):
        shape = y.shape
        chunksize = y.chunksize
        axis = _check_axis(axis, y.ndim)
        if shape[axis] != chunksize[axis]:
            y = y.rechunk(
                (*chunksize[0:axis], shape[axis], *chunksize[axis + 1:len(chunksize)])
            )

        return y.map_blocks(
            _ndpolyfit,
            x=x,
            axis=axis,
            deg=deg,
            rcond=rcond,
            full=full,
            w=w,
            cov=cov,
            missing_value=missing_value,
            xarray_ouput=False
        )
    else:
        return _ndpolyfit(np.asarray(y), x, axis, deg, rcond, full, w, cov, missing_value)


def _ndpolyfit(
        x: np.ndarray,
        y: np.ndarray,
        axis: int = 0,
        deg: int = 1,
        rcond=None,
        full=False,
        w=None,
        cov=False,
        missing_value=np.nan,
        xarray_output=True) -> (np.ndarray, xr.DataArray):

    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a numpy.ndarray")

    if not isinstance(y, np.ndarray):
        raise TypeError("This function only accepts np.ndarray as its input for y")

    axis = _check_axis(axis, y.ndim)

    if x.size != y.shape[axis]:
        raise ValueError("X must have the same number of elements as the y-dimension defined by axis")

    if x.shape not in ((y.shape[axis], ), (y.shape[axis], 1), (1, y.shape[axis])):
        raise ValueError("x must be of size (M,), (M, 1), or (1, M); where M = y.shape[axis]")


    x = x.reshape(y.shape[axis])

    if deg < 0:
        raise ValueError("deg must be zero or a positive integer.")
    elif int(deg) != deg:
        raise TypeError("deg must be an integral type.")

    deg = int(deg)

    # TODO: check x

    y_rearranged, trailing_shape = _rearrange_axis(y, axis)

    mask = y_rearranged == missing_value
    has_missing = mask.any()
    if (not np.isnan(missing_value)) and has_missing:
        y_rearranged[mask] = np.nan

    if has_missing:
        raise NotImplemented()
    else:
        polyfit_output = np.polyfit(
            x, y_rearranged,
            deg=deg,
            rcond=rcond,
            full=full,
            w=w,
            cov=cov
        )

    p = _reverse_rearrange_axis(
        (polyfit_output if isinstance(polyfit_output, np.ndarray) else polyfit_output[0]),
        axis,
        trailing_shape
    )

    if bool(xarray_output):
        attrs = {
            "deg": deg,
            "provided_rcond": rcond,
            "full": full,
            "weights": w,
            "covariance": cov
        }

        if full:  # full == True
            attrs["residuals"] = polyfit_output[1]
            attrs["rank"] = polyfit_output[2]
            attrs["singular_values"] = polyfit_output[3]
            attrs["rcond"] = polyfit_output[4]
        elif cov:  # (full == False) and (cov == True)
            attrs["V"] = polyfit_output[1]

        return xr.DataArray(
            p,
            attrs=attrs
        )
    else:
        return p


def _check_axis(axis, ndim) -> int:
    if (axis > (ndim - 1)) or (axis < -ndim):
        raise ValueError(f"axis must be an integer between {-ndim} and {ndim - 1}")
    if int(axis) != axis:
        raise TypeError("axis must be an integral type.")
    elif axis < 0:
        axis += ndim

    return int(axis)


def _rearrange_axis(data: np.ndarray, axis: int = 0) -> tuple:
    """
    rearranges the `numpy.ndarray` as a two-dimensional array of size (n, -1), where n is the number of elements of
    the dimension defined by `axis`.
    Args:
        data: a `numpy.ndarray` to be rearranged
        axis: the axis that all other dimensions are rearranged around it. default is 0.

    Returns: a tuple, where the first element contains the reshaped data, and the second is a tuple with all dimensions
             except the one specified by the axis

    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray.")

    axis = _check_axis(axis, data.ndim)

    if axis != 0:
        data = np.moveaxis(data, axis, 0)

    trailing_shape = data.shape[1:]

    data = data.reshape((data.shape[0], -1))

    return data, trailing_shape


def _reverse_rearrange_axis(data: np.ndarray, axis, trailing_shape: tuple) -> np.ndarray:
    return np.moveaxis(data.reshape((data.shape[0], * trailing_shape)), 0, axis)






















































