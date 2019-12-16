from typing import Iterable, Any

import numpy as np
import xarray as xr
import dask.array as da


def _get_missing_value(data: xr.DataArray, args: dict) -> Any:
    """
     Attempts to extract missing_vallue. It first checks in the arguments, for `missing_value` key. If that's not
     provided, and the data is of type `xr.DataArray`, the `missing_value` and `_FillValue` are looked up in the
     attributes, in that order. If all fails, `np.nan` as default is returned.
    Args:
        data (:class: `xr.DataArray` or `Any`):
            a data with possible attrs property.
        args (:class: `Dict`):
            a dictionary that may contain `missing_value` key.

    Returns:
        of course, a value for missing_value. What else did you expect?!

    """
    if "missing_value" in args:
        missing_value = args["missing_value"]
    elif isinstance(data, xr.DataArray):
        if "missing_value" in data.attrs:
            missing_value = data.attrs["missing_value"]
        elif "_FillValue" in data.attrs:
            missing_value = data.attrs["_FillValue"]
        else:
            missing_value = np.nan
    else:
        missing_value = np.nan

    return missing_value


def _unchunk_ifneeded(data: da.Array, axis: int) -> da.Array:
    """
    Make sures that the `Dask.Array` is not chunked along the specified axis. If it is chunked, it returns a new
    rechunked `Dask.Array` array, with the same chunking except along the specified axis.

    Args:

        data (:class: `Dask.Array`(:
            The data

        axis (:class: `int`):
            The axis that must not be chunked

    Returns (:class: `Dask.Array`):
        a `Dask.Array` which is not chunked along the specified axis.

    """
    if isinstance(data, da.Array):
        shape = data.shape
        chunksize = data.chunksize
        axis = _check_axis(axis, data.ndim)
        if shape[axis] != chunksize[axis]:
            data = data.rechunk({axis: -1})
        return data
    else:
        raise TypeError("data must be a dask array.")



def ndpolyfit(x: Iterable, y: Iterable, deg: int, axis: int = 0, **kwargs) -> (xr.DataArray, da.Array):
    """
    An extension to `numpy.polyfit` function to work with multi-dimensional arrays. If `y` is of shape, let's say
    `(s0, s1, s2, s3)`, the `axis=1`, and `deg=1`, then the output would be `(s0, 2, s2, s3)`. So, the function fits
    a first degree polynomial (because `deg=1`) along the second dimension (because `axis=1`) for every other dimension.
    The other change from `numpy.polyfit` is that this method also handles the missing values. Also, this version, has
    support for Dask array and chunked Dask arrays.

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
    missing_value = _get_missing_value(y, kwargs)
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
        y = _unchunk_ifneeded(y, axis)

        return y.map_blocks(
            lambda b: _ndpolyfit(x, b, axis, deg, rcond, full, w, cov, missing_value, False),
            dtype=np.float64
        ).compute()
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


def isvector(input: Iterable) -> bool:
    if isinstance(input, np.ndarray):
        return (input.size != 1) and \
               (
                   (input.ndim == 1) or
                   ((input.ndim == 2) and (np.any(input.shape == 1)))
               )
    else:
        return isvector(np.asarray(input))


def _to_numpy_ndarray(data: Iterable) -> np.ndarray:
    if isinstance(data, xr.DataArray):
        data = data.data
    elif not isinstance(data, np.ndarray):
        data = np.asarray(data)

    return data


def ndpolyval(p: Iterable, x: Iterable = None, axis: int = 0, **kwargs):
    """
    Extended version of `numpy.polyval` to support multi-dimensional outputs provided by `geocat.comp.ndpolyfit`.

    As the name suggest, this version supports a multi-dimensional `p` array. Let's say `p` is of dimension `(s0,s1,s2)`
    and `axis=1`, then the output would be of dimension `(s0, M, s2)` where M depends on `x`.
    The same way, `x` could be a multi dimensional array or a single array. In another word, `x` is either of
    dimension `(M, )`, `(M, 1)`, `(1, M)` or, in this example, of dimension `(s0, M, s2)`. When `x` is not the vector,
    it must have the same dimension as of `p` except for the dimension that is defined by `axis`.

    Args:

        p (:class:`Iterable`):
            the polynomial coeficients

        x (:class:`Iterable`):
            the coordinates where polynomial must be evaluated

        axis (:class:`int`):
        The axis where the polynomials are there.

        **kwargs:
            Currently not used.

    Returns (:class: `xr.DataArray`:
        polynomial evaluated with the provided coordinates.

    Example:
        * Fitting a first degree polynomial and calculating the residual manually:

        >>> from geocat.comp.polynomial import ndpolyfit, ndpolyval
        >>> import numpy as np
        >>> n = 10
        >>> x = np.linspace(0, 1, n)
        >>> print(x.shape)
        (10,)
        >>> y = np.random.random((3, 4, n, 2, 5))
        >>> print(y.shape)
        (3, 4, 10, 2, 5)
        >>> # First fitting a first degree polynomial to our 5-Dimensional array
        ... p = ndpolyfit(x, y, deg=1, axis=2)
        >>> print("p.shape: ", p.shape)
        p.shape:  (3, 4, 2, 2, 5)
        >>> # Now re-evaluating the polynomial at the same points:
        ... y_fitted = ndpolyval(p, x, axis=2)
        >>> We are ready to manually calculate the residual now:
        ... signed_residual = y_fitted - y
        >>> y_fitted.shape
        (3, 4, 10, 2, 5)

        * when evaluating x can be another multi-dimensional array

        >>> # Let's work on the p that was calculated in previous example
        ... # using the command:
        ... #           p = ndpolyfit(x, y, deg=1, axis=2)
        ... print("p.shape: ", p.shape)
        p.shape:  (3, 4, 2, 2, 5)

        >>> # Let's pass x which has 20 members this time:
        ... x = np.random.random(20)
        >>> print("x.shape: ", x.shape)
        x.shape:  (20,)
        >>> new_y = ndpolyval(p, x, axis=2)
        >>> print("new_y.shape: ", new_y.shape) # Note new_y.shape[2] = 20
        new_y.shape:  (3, 4, 20, 2, 5)

        >>> # Now let's make a multi-dimensional x
        ... # NOTE: all dimension, except axis=2 is the same as the one in p
        ... #       because during fitting step axis was set to 2
        ... x = np.random.random((3, 4, 42, 2, 5))
        >>> new_y = ndpolyval(p, x, axis=2)
        >>> print("new_y.shape: ", new_y.shape) # Note new_y.shape[2] = 42
        new_y.shape:  (3, 4, 42, 2, 5)

        >>> # now if a wrongly sized array is passed, you would get an error:
        ... x = np.random.random((5, 2, 42, 4, 3))
        >>> new_y = ndpolyval(p, x, axis=2)
        Traceback (most recent call last):
          ...
        ValueError: x has invalid shape.

    """
    p_ndarr = _to_numpy_ndarray(p)
    axis = _check_axis(axis, p_ndarr.ndim)
    if isinstance(x, da.Array):
        x = _unchunk_ifneeded(x, axis)
        x_chunks = list(x.chunks)
        x_chunks[axis] = p_ndarr.shape[axis]
        p_dask = da.from_array(p_ndarr, chunks=x_chunks)
        y = da.map_blocks(
            lambda p_blocks, x_blocks: _ndpolyval(p_blocks, x_blocks, axis),
            p_dask, x,
            dtype=np.float64
        ).compute()
    else:
        x_ndarr = _to_numpy_ndarray(x)
        y = _ndpolyval(p_ndarr, x_ndarr, axis)

    attrs = {"p": p} if kwargs.get("return_info", False) else {}

    output = xr.DataArray(
        y,
        attrs=attrs
    )
    return output


def _ndpolyval(p: np.ndarray, x: np.ndarray, axis: int = 0, **kwargs) -> np.ndarray:
    if not isinstance(p, np.ndarray):
        raise TypeError("This function accepts only numpy.ndarray as p.")

    if not isinstance(x, np.ndarray):
        raise TypeError("This function accepts only numpy.ndarray as x.")

    axis = _check_axis(axis, p.ndim)

    if (x.ndim != 1) and (x.ndim != p.ndim):
        raise ValueError("x has invalid number of dimension. x must be either 1 dimensionn.")

    if isvector(x):
        x_original_size = x.size
        other_dims = np.asarray(p.shape)[np.arange(p.ndim) != axis]
        x = np.moveaxis(
            np.tile(
                x.reshape((-1, )),
                np.prod(other_dims)
            ).reshape((*other_dims, x_original_size)),
            -1,
            axis
        )

    else:
        if not ( \
                np.all(x.shape[:axis] == p.shape[:axis]) and \
                np.all(x.shape[(axis+1):x.ndim] == p.shape[(axis+1):p.ndim])
        ):
            raise ValueError("x has invalid shape.")

    y = np.zeros(x.shape)

    for i in range(p.shape[axis]):
        y += p.take([i], axis=axis) * np.power(x, p.shape[axis] - 1 - i)

    return y




















































