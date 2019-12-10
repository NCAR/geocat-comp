from typing import Iterable

import numpy as np
import xarray as xr
import dask.array as da


def ndpolyfit(x: Iterable, y: Iterable, deg: int, axis: int = 0, **kwargs) -> (xr.DataArray, da.Array):
    """
    An extension to `numpy.polyfit` function to work with multi-dimensional arrays. If `y` is of shape, let's say
    `(s0, s1, s2, s3)`, the `axis=1`, and `deg=1`, then the output would be `(s0, 2, s2, s3)`. So, the function fits
    a first degree polynomial (because `deg=1`) along the second dimension (because `axis=1`) for every other dimension.
    The other change from `numpy.polyfit` is that this method also handles the missing values.

    Args:

        x (:class:`array_like`):
            x-coordinate, an Iterable object of shape `(M,)`, `(M, 1)`, or `(1, M)` where `M = y.shape(axis)`. It cannot
            have `nan` or missing values.

        y (:class:`array_like`)
            y-coordinate, an Iterable containing the data. It could be list, `numpy.ndarray`, `xr.DataArray`, Dask array.
            or any Iterable convertible to `numpy.ndarray`. In case of Dask Array, The data could be chunked. It is
            recommended no to chunk along the `axis` provided.

        axis (:class:`int`):
            the axis to fit the polynomial to. Default is 0.

        deg (:class:`int`):
            degree of the fitting polynomial

        kwargs (:class:`dict`, optional):
            Extra parameter controlling the method behavior:

            rcond (:class:`float`, optional):
                Relative condition number of the fit. Refer to `numpy.polyfit` for further details.

            full (:class:`bool`, optional):
                Switch determining nature of return value. Refer to `numpy.polyfit` for further details.

            w (:class:`array_like`, optional):
                Weights applied to the y-coordinates of the sample points. Refer to `numpy.polyfit` for further details.

            cov (:class:`bool`, optional):
                Determines whether to return the covariance matrix. Refer to `numpy.polyfit` for further details.

            missing_value (:class:`number` or :class:`np.nan`, optional):
                The value to be treated as missing. Default is `np.nan`

            meta (:class:`bool`, optional):
                If set to `True` and the input, i.e. `y`, is of type `xr.DataArray`, the attributes associated to the
                input are transferred to the output.

    Returns:
        an `xarray.DataArray` or `numpy.ndarray` containing the coefficients of the fitted polynomial.
    """

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
        return _ndpolyfit(x, y, axis, deg, rcond, full, w, cov, missing_value)
    if isinstance(y, xr.DataArray):
        output = _ndpolyfit(x, y.data, axis, deg, rcond, full, w, cov, missing_value)
        attrs = output.attrs

        if meta:
            for attr, v in y.attrs.items():
                attrs[attr] = v
            axis = axis if axis >= 0 else axis+y.ndim
            dims = [("poly_coef" if i == axis else output.dims[i]) for i in range(y.ndim)]
            coords = {k: v for k, v in y.coords.items() if k != y.dims[axis]}
            coords["poly_coef"] = list(range(int(deg) +1))
            output = xr.DataArray(
                output.data,
                attrs=attrs,
                dims=dims,
                coords=coords
            )
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

    """
    An extension to `numpy.polyfit` function to work with multi-dimensional numpy input. If `y` is of shape, let's say
    `(s0, s1, s2, s3)`, the `axis=1`, and `deg=1`, then the output would be `(s0, 2, s2, s3)`. So, the function fits
    a first degree polynomial (because `deg=1`) along the second dimension (because `axis=1`) for every other dimension.
    The other change from `numpy.polyfit` is that this method also handles the missing values.

    Args:

        x (:class:`np.ndarray`):
            x-coordinate, `numpy.ndarray` of shape `(M,)`, `(M, 1)`, or `(1, M)` where `M = y.shape(axis)`. It cannot
            have `nan`.

        y (:class:`np.ndarray`):
            y-coordinate, 'numpy.ndarray`.

        axis (:class:`int`, optional):
            the axis to fit the polynomial to. Default is 0.

        deg (:class:`int`, optional):
            degree of the fitting polynomial

        rcond (:class:`float`, optional):
            Relative condition number of the fit. Refer to `numpy.polyfit` for further details.

        full (:class:`bool`, optional):
            Switch determining nature of return value. Refer to `numpy.polyfit` for further details.

        w (:class:`array_like`, optional):
            Weights applied to the y-coordinates of the sample points. Refer to `numpy.polyfit` for further details.

        cov (:class:`bool`, optional):
            Determines whether to return the covariance matrix. Refer to `numpy.polyfit` for further details.

        missing_value (:class:`number` or :class:`np.nan`, optional):
            The value to be treated as missing. Default is `np.nan`

        xarray_output (:class:`bool`, optional):
            Determines the type of the output. If set to `True` the output would be of type `xr.DataArray`
            and the some extra information are attached to the output as attributes. Otherwise, the output
            would be of type `np.ndarray` containing only the coefficients of the fitted polynomial.

    Returns:
        an `xarray.DataArray` or `numpy.ndarray` containing the coefficients of the fitted polynomial.

    """

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

    y_rearranged, trailing_shape = _rearrange_axis(y, axis)

    if not np.isnan(missing_value):
        y_rearranged[y_rearranged == missing_value] = np.nan

    mask = np.logical_not(np.isfinite(y_rearranged))

    if mask.any():
        all_row_missing = np.all(mask, axis=1)
        row_with_missing = np.any(mask, axis=1)
        if np.all(all_row_missing == row_with_missing):
            rows_to_keep = np.logical_not(all_row_missing)
            y_rearranged = y_rearranged[rows_to_keep, :]
            x = x[rows_to_keep]
            has_missing = False
            # mask = np.full(y_rearranged.shape, False, dtype=bool)  # This is not actually needed any longer
        else:
            has_missing = True
    else:
        has_missing = False

    if np.isnan(x).any():
        raise ValueError("x cannot have missing values")

    if has_missing:
        tmp_results = []

        for c in range(y_rearranged.shape[1]):
            idx = np.logical_not(mask[:, c])
            tmp_results.append(
                np.polyfit(
                    x[idx], y_rearranged[idx, c],
                    deg=deg,
                    rcond=rcond,
                    full=full,
                    w=w,
                    cov=cov
                )
            )

        polyfit_output = tmp_results[0].reshape((-1, 1)) \
            if isinstance(tmp_results[0], np.ndarray) \
            else tmp_results[0][0].reshape((-1, 1))
        for i in range(1, y_rearranged.shape[1]):
            polyfit_output = np.concatenate(
                (
                    polyfit_output,
                    (tmp_results[c].reshape((-1, 1))
                        if isinstance(tmp_results[c], np.ndarray)
                        else tmp_results[c][0].reshape((-1, 1)))
                ),
                axis=1
            )

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

        if not has_missing:
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






















































