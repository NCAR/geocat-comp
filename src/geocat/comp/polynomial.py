import dask.array as da
import numbers
import numpy as np
import typing
import xarray as xr


def _get_missing_value(data: xr.DataArray, args: dict) -> typing.Any:
    """Attempts to extract `missing_value` or `_FillValue` from either `data`
    or `dict`. If not found, returns `numpy.nan`

    Parameters
    ----------
    data : :class:`xarray.DataArray`
        Data which may contain `missing_value` or `_FillValue` attributes.
    args : :class:`dict`
        Dictionary which may contain `missing_value` key.

    Returns
    -------
    missing_value : :class:`Any`
        The `missing_value` representation.
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
    """Returns `data` unchunked along `axis`.

    Parameters
    ----------
    data : :class:`dask.array.Array`
        Data which may be chunked along `axis`.
    axis : :class:`int`
        Axis number which specifies the axis to unchunk.

    Returns
    -------
        data : :class:`dask.array.Array`
            A dask array which is not chunked along the specified axis.
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


def ndpolyfit(x: typing.Iterable,
              y: typing.Iterable,
              deg: int,
              axis: int = 0,
              **kwargs) -> (xr.DataArray, da.Array):
    """An extension to `numpy.polyfit` function to support multi-dimensional
    arrays, Dask arrays, and missing values.

    Parameters
    ----------

    x : :class:`array_like`
        X-coordinate, an iterable object of shape `(M,)`, `(M, 1)`, or `(1, M)` where `M = y.shape(axis)`. It cannot
        have `nan` or missing values.

    y : :class:`array_like`
        Y-coordinate, an iterable containing the data. It could be list, `numpy.ndarray`, `xarray.DataArray`, Dask array.
        or any Iterable convertible to `numpy.ndarray`. In case of Dask Array, The data could be chunked. It is
        recommended not to chunk along the `axis` provided.

    deg : :class:`int`
        Degree of the fitting polynomial

    axis : :class:`int`, Optional
        Axis to fit the polynomial to. Default is 0.

    kwargs : :class:`dict`, Optional
        See below

    Keyword Args
    ------------
    rcond : :class:`float`, Optional
        Relative condition number of the fit. Refer to `numpy.polyfit` for further details.

    full : :class:`bool`, Optional
        Switch determining nature of return value. Refer to `numpy.polyfit` for further details.

    w : :class:`array_like`, Optional
        Weights applied to the y-coordinates of the sample points. Refer to `numpy.polyfit` for further details.

    cov : :class:`bool`, Optional
        Determines whether to return the covariance matrix. Refer to `numpy.polyfit` for further details.

    missing_value : :class:`number` or :class:`numpy.nan`, Optional
        The value to be treated as missing. Default is `numpy.nan`

    meta : :class:`bool`, Optional
        If set to `True` and the input, i.e. `y`, is of type `xarray.DataArray`, the attributes associated to the
        input are transferred to the output.

    Returns
    -------
    coefficients : :class:`xarray.DataArray` or :class:`numpy.ndarray`
        An array containing the coefficients of the fitted polynomial.

    Examples
    --------
        * Fitting a line to a one dimensional array:

        >>> import numpy as np
        >>> from geocat.comp.polynomial import ndpolyfit
        >>> x = np.arange(10, dtype=np.float)
        >>> y = 2*x + 3
        >>> p = ndpolyfit(x, y, deg=1)
        >>> print(p)
        <xarray.DataArray (dim_0: 2)>
        array([2., 3.])
        Dimensions without coordinates: dim_0
        Attributes:
            deg:             1
            provided_rcond:  None
            full:            False
            weights:         None
            covariance:      False

        * Fitting a second degree polynomial to a one dimensional array:

        >>> y = 4*x*x + 3*x + 2
        >>> p = ndpolyfit(x, y, deg=2)
        >>> print(p)
        <xarray.DataArray (dim_0: 3)>
        array([4., 3., 2.])
        Dimensions without coordinates: dim_0
        Attributes:
            deg:             2
            provided_rcond:  None
            full:            False
            weights:         None
            covariance:      False

        * Fitting polynomial with missing values: Ordinarily NaN's are treated as missing values.
          In this example let's introduce a different value to indicate missing data.

        >>> # Let's introduce some missing values:
        >>> y[7:] = 999
        >>> p = ndpolyfit(x, y, deg=2)
        >>> print(p)
        <xarray.DataArray (dim_0: 3)>
        array([ 21.15909091, -62.14090909,  20.4       ])
        Dimensions without coordinates: dim_0
        Attributes:
            deg:             2
            provided_rcond:  None
            full:            False
            weights:         None
            covariance:      False
        >>> # As you can see, we got a different coefficients
        >>> # Now let's define 999 as missing value
        >>> p = ndpolyfit(x, y, deg=2, missing_value=999)
        >>> print(p)
        <xarray.DataArray (dim_0: 3)>
        array([4., 3., 2.])
        Dimensions without coordinates: dim_0
        Attributes:
            deg:             2
            provided_rcond:  None
            full:            False
            weights:         None
            covariance:      False
        >>> # Now we got the coefficient we were looking for

        * Fitting polynomial with NaN as missing values: NaN is by default considered a missing value all the time

        >>> import numpy as np
        >>> from geocat.comp.polynomial import ndpolyfit
        >>> x = np.arange(10, dtype=np.float)
        >>> y = 4*x*x + 3*x + 2
        >>> y[7:] = np.nan
        >>> print(y)
        [  2.   9.  24.  47.  78. 117. 164.  nan  nan  nan]
        >>> p = ndpolyfit(x, y, deg=2)
        >>> print(p)
        <xarray.DataArray (dim_0: 3)>
        array([4., 3., 2.])
        Dimensions without coordinates: dim_0
        Attributes:
            deg:             2
            provided_rcond:  None
            full:            False
            weights:         None
            covariance:      False
        >>> # as you can see, despite not specifying NaN as missing value, the coefficients are properly calculated

        * Fitting a line to a multi-dimensional array

        >>> y_md = np.tile(y.reshape(1, 10, 1, 1), [2, 1, 3, 4])
        >>> y_md.shape
        (2, 10, 3, 4)
        >>> print(y)
        [  2.   9.  24.  47.  78. 117. 164. 219. 282. 353.]
        >>> print(y_md[1, :, 1, 1])
        [  2.   9.  24.  47.  78. 117. 164. 219. 282. 353.]
        >>> p = ndpolyfit(x, y_md, deg=2, axis=1)
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
        output = _ndpolyfit(x, y.data, axis, deg, rcond, full, w, cov,
                            missing_value)
        attrs = output.attrs

        if meta:
            for attr, v in y.attrs.items():
                attrs[attr] = v
            axis = axis if axis >= 0 else axis + y.ndim
            dims = [("poly_coef" if i == axis else output.dims[i])
                    for i in range(y.ndim)]
            coords = {k: v for k, v in y.coords.items() if k != y.dims[axis]}
            coords["poly_coef"] = list(range(int(deg) + 1))
            output = xr.DataArray(output.data,
                                  attrs=attrs,
                                  dims=dims,
                                  coords=coords)
        return output
    if isinstance(y, da.Array):
        y = _unchunk_ifneeded(y, axis)

        return y.map_blocks(lambda b: _ndpolyfit(x, b, axis, deg, rcond, full,
                                                 w, cov, missing_value, False),
                            dtype=np.float64).compute()
    else:
        return _ndpolyfit(np.asarray(y), x, axis, deg, rcond, full, w, cov,
                          missing_value)


def _ndpolyfit(x: typing.Iterable,
               y: typing.Iterable,
               axis: int = 0,
               deg: int = 1,
               rcond: float = None,
               full: bool = False,
               w: typing.Iterable = None,
               cov: bool = False,
               missing_value: typing.Union[numbers.Number] = np.nan,
               xarray_output: bool = True) -> (np.ndarray, xr.DataArray):
    """An extension to `numpy.polyfit` function to support multi-dimensional
    arrays, Dask arrays, and missing values.

    Parameters
    ----------

    x : :class:`array_like`
        X-coordinate, an iterable object of shape `(M,)`, `(M, 1)`, or `(1, M)` where `M = y.shape(axis)`.
        It cannot have `nan` or missing values.

    y : :class:`array_like`
        Y-coordinate, an iterable containing the data. It could be list, `numpy.ndarray`, `xarray.DataArray`, Dask array.
        or any Iterable convertible to `numpy.ndarray`. In case of Dask Array, The data could be chunked. It is
        recommended not to chunk along the `axis` provided.

    axis : :class:`int`, Optional
        Axis to fit the polynomial to. Defaults to 0.

    deg : :class:`int`, Optional
        Degree of the fitting polynomial. Defaults to 1.

    rcond : :class:`float`, Optional
        Relative condition number of the fit. Defaults to None. Refer to `numpy.polyfit` for further details.

    full : :class:`bool`, Optional
        Switch determining nature of return value. Defaults to False. Refer to `numpy.polyfit` for further details.

    w : :class:`array_like`, Optional
        Weights applied to the y-coordinates of the sample points. Defaults to None. Refer to `numpy.polyfit` for further details.

    cov : :class:`bool`, Optional
        Determines whether to return the covariance matrix. Defaults to False. Refer to `numpy.polyfit` for further details.

    missing_value : :class:`number` or :class:`numpy.nan`, Optional
        The value to be treated as missing. Default is `numpy.nan`

    xarray_output : :class:`bool`, Optional
        Determines the type of the output. If set to `True` the output would be of type `xarray.DataArray`
        and the some extra information are attached to the output as attributes. Otherwise, the output
        would be of type `numpy.ndarray` containing only the coefficients of the fitted polynomial. Defaults to True.

    Returns
    -------
    coefficients : :class:`xarray.DataArray` or :class:`numpy.ndarray`
        An array containing the coefficients of the fitted polynomial.
    """

    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a numpy.ndarray")

    if not isinstance(y, np.ndarray):
        raise TypeError(
            "This function only accepts np.ndarray as its input for y")

    axis = _check_axis(axis, y.ndim)

    if x.size != y.shape[axis]:
        raise ValueError(
            "X must have the same number of elements as the y-dimension defined by axis"
        )

    if x.shape not in ((y.shape[axis],), (y.shape[axis], 1), (1,
                                                              y.shape[axis])):
        raise ValueError(
            "x must be of size (M,), (M, 1), or (1, M); where M = y.shape[axis]"
        )

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
            # mask = np.full(y_rearranged.shape, False, dtype=bool)  # This is
            # not actually needed any longer
            has_missing = False
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
                np.polyfit(x[idx],
                           y_rearranged[idx, c],
                           deg=deg,
                           rcond=rcond,
                           full=full,
                           w=w,
                           cov=cov))

        polyfit_output = tmp_results[0].reshape((-1, 1)) if isinstance(
            tmp_results[0], np.ndarray) else tmp_results[0][0].reshape((-1, 1))
        for i in range(1, y_rearranged.shape[1]):
            polyfit_output = np.concatenate(
                (polyfit_output, (tmp_results[c].reshape((-1, 1)) if isinstance(
                    tmp_results[c], np.ndarray) else tmp_results[c][0].reshape(
                        (-1, 1)))),
                axis=1)

    else:
        polyfit_output = np.polyfit(x,
                                    y_rearranged,
                                    deg=deg,
                                    rcond=rcond,
                                    full=full,
                                    w=w,
                                    cov=cov)

    p = _reverse_rearrange_axis((polyfit_output if isinstance(
        polyfit_output, np.ndarray) else polyfit_output[0]), axis,
                                trailing_shape)

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

        return xr.DataArray(p, attrs=attrs)
    else:
        return p


def _check_axis(axis, ndim) -> int:
    if (axis > (ndim - 1)) or (axis < -ndim):
        raise ValueError(
            f"axis must be an integer between {-ndim} and {ndim - 1}")
    if int(axis) != axis:
        raise TypeError("axis must be an integral type.")
    elif axis < 0:
        axis += ndim

    return int(axis)


def _rearrange_axis(data: np.ndarray,
                    axis: int = 0) -> tuple([np.ndarray, tuple]):
    """rearranges the `numpy.ndarray` as a two-dimensional array of size (n,

    -1), where n is the number of elements of the dimension defined by `axis`.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        An array to be rearranged
    axis : :class:`int`, Optional
        The axis that all other dimensions are rearranged around it. Defaults to 0.

    Returns
    -------
    tuple (data :class:`numpy.ndarray`, shape :class:`tuple`
        A tuple, where the first element contains the reshaped data, and the second is a tuple with all dimensions except the one specified by the axis.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray.")

    axis = _check_axis(axis, data.ndim)

    if axis != 0:
        data = np.moveaxis(data, axis, 0)

    trailing_shape = data.shape[1:]

    data = data.reshape((data.shape[0], -1))

    return data, trailing_shape


def _reverse_rearrange_axis(data: np.ndarray, axis,
                            trailing_shape: tuple) -> np.ndarray:
    return np.moveaxis(data.reshape((data.shape[0], *trailing_shape)), 0, axis)


def _isvector(input: typing.Iterable) -> bool:
    if isinstance(input, np.ndarray):
        return (input.size != 1) and ((input.ndim == 1) or
                                      ((input.ndim == 2) and
                                       (np.any(input.shape == 1))))
    else:
        return _isvector(np.asarray(input))


def _to_numpy_ndarray(data: typing.Iterable) -> np.ndarray:
    if isinstance(data, xr.DataArray):
        data = data.data
    elif not isinstance(data, np.ndarray):
        data = np.asarray(data)

    return data


def ndpolyval(p: typing.Iterable,
              x: typing.Iterable,
              axis: int = 0,
              **kwargs) -> xr.DataArray:
    """Extended version of `numpy.polyval` to support multi-dimensional outputs
    provided by `geocat.comp.ndpolyfit`.

    As the name suggest, this version supports a multi-dimensional `p` array. Let's say `p` is of dimension `(s0,s1,s2)`
    and `axis=1`, then the output would be of dimension `(s0, M, s2)` where M depends on `x`.
    The same way, `x` could be a multi dimensional array or a single array. In another word, `x` is either of
    dimension `(M, )`, `(M, 1)`, `(1, M)` or, in this example, of dimension `(s0, M, s2)`. When `x` is not the vector,
    it must have the same dimension as of `p` except for the dimension that is defined by `axis`.

    Parameters
    ----------

    p : :class:`Iterable`
        Polynomial coeficients.

    x : :class:`Iterable`
        Coordinates where polynomial must be evaluated.

    axis : :class:`int`, Optional
        The axis along which to evaluate the polynomial. Defaults to 0.

    **kwargs:
        Currently not used.

    Returns
    -------
    output : :class:`xarray.DataArray`
        Polynomial evaluated with the provided coordinates.

    Examples
    --------

        * Evaluating a polynomial:

        >>> p = [2, 3] # representing y = 2*x + 3
        >>> x = np.arange(-5, 6, dtype="float")
        >>> x
        array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.])
        >>> from geocat.comp.polynomial import ndpolyval
        >>> y = ndpolyval(p, x)
        >>> y.shape
        (11,)
        >>> print(y)
        <xarray.DataArray (dim_0: 11)>
        array([-7., -5., -3., -1.,  1.,  3.,  5.,  7.,  9., 11., 13.])
        Dimensions without coordinates: dim_0
        >>> np.testing.assert_almost_equal(y, 2*x+3)

        * evaluating a multi-dimensional fitted polynomial:

        >>> p = np.tile(np.asarray(p).reshape(1, 2, 1, 1), [3, 1, 4, 5])
        >>> p.shape
        (3, 2, 4, 5)
        >>> p[1, :, 1, 1]
        array([2, 3])
        >>> p[1, :, 1, 2]
        array([2, 3])
        >>> y = ndpolyval(p, x, axis=1)
        >>> y.shape
        (3, 11, 4, 5)

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
            p_dask,
            x,
            dtype=np.float64).compute()
    else:
        x_ndarr = _to_numpy_ndarray(x)
        y = _ndpolyval(p_ndarr, x_ndarr, axis)

    attrs = {"p": p} if kwargs.get("return_info", False) else {}

    output = xr.DataArray(y, attrs=attrs)
    return output


def _ndpolyval(p: np.ndarray,
               x: np.ndarray,
               axis: int = 0,
               **kwargs) -> np.ndarray:
    if not isinstance(p, np.ndarray):
        raise TypeError("This function accepts only numpy.ndarray as p.")

    if not isinstance(x, np.ndarray):
        raise TypeError("This function accepts only numpy.ndarray as x.")

    axis = _check_axis(axis, p.ndim)

    if (x.ndim != 1) and (x.ndim != p.ndim):
        raise ValueError(
            "x has invalid number of dimension. x must be either 1 dimensionn.")

    if _isvector(x):
        x_original_size = x.size
        other_dims = np.asarray(p.shape)[np.arange(p.ndim) != axis]
        x = np.moveaxis(
            np.tile(x.reshape((-1,)), np.prod(other_dims)).reshape(
                (*other_dims, x_original_size)), -1, axis)

    else:
        if not (np.all(x.shape[:axis] == p.shape[:axis]) and np.all(
                x.shape[(axis + 1):x.ndim] == p.shape[(axis + 1):p.ndim])):
            raise ValueError("x has invalid shape.")

    y = np.zeros(x.shape)

    for i in range(p.shape[axis]):
        y += p.take([i], axis=axis) * np.power(x, p.shape[axis] - 1 - i)

    return y


def detrend(data: typing.Iterable, deg=1, axis=0, **kwargs) -> xr.DataArray:
    """Estimates and removes the trend of the leftmost dimension from all grid
    points. This method, at the minimum, provides all the functionality that is
    provided by NCL's 'dtrend', 'dtrend_quadratic', 'dtrend_quadratic_msg_n',
    'dtrend_msg_n', 'dtrend_msg', 'dtrend_n'. However, this function is not
    limited to quadratic detrending and you could use higher polynomial degree
    as well.

    Parameters
    ----------
    data : :class:`array_like`
        a multi-dimensional numeric array

    deg : :class:`int`, Optional
        a non-negative integer determining the degree of the polynomial to use for detrending. Default value is 1.

    axis : :class:`int`, Optional
        the axis along which the data is detrended. Default value is 0.

    kwargs : :class:`dict`, Optional
        See below

    Keyword Args
    ------------
    return_info : :class:`bool`
        If set to true, the fitted polynomial is returned as part of the attributes. Default value is `True`.

    missing_value : :class:`numeric`
        A value that must be ignored. Default is NaN.

    Returns
    -------
    detrended_data : :class:`xarray.DataArray`
        Array containing the detrended data.

    Examples
    --------

        * Detrending a data:

        >>> from geocat.comp.polynomial import ndpolyfit
        >>> from geocat.comp.polynomial import detrend
        >>> # Creating synthetic data
        >>> x = np.linspace(-8*np.pi, 8 * np.pi, 33, dtype=np.float64)
        >>> y0 = 1.0 * x
        >>> y1 = np.sin(x)
        >>> y = y0 + y1
        >>> p = ndpolyfit(np.arange(x.size), y, deg=1)
        >>> y_trend = ndpolyval(p, np.arange(x.size))
        >>> y_detrended = detrend(y)
        >>> np.testing.assert_almost_equal(y_detrended + y_trend, y)


        * Detrending a multi-dimensional data:

        >>> # Creating synthetic data
        >>> x = np.linspace(-8*np.pi, 8 * np.pi, 33, dtype=np.float64)
        >>> y0 = 1.0 * x
        >>> y1 = np.sin(x)
        >>> y = np.tile((y0 + y1).reshape((1, -1, 1, 1)), (2, 1, 3, 4))
        >>> p = ndpolyfit(x, y, deg=1, axis=1)
        >>> y_trend = ndpolyval(p, x, axis=1)
        >>> y_detrended = detrend(y, x=x, axis=1)
        >>> np.testing.assert_almost_equal(y_detrended + y_trend, y)

    See Also
    --------
    Related NCL Functions:
    `dtrend <https://www.ncl.ucar.edu/Document/Functions/Built-in/dtrend.shtml>`_,
    `dtrend_n <https://www.ncl.ucar.edu/Document/Functions/Built-in/dtrend_n.shtml>`_,
    `dtrend_msg <https://www.ncl.ucar.edu/Document/Functions/Built-in/dtrend_msg.shtml>`_,
    `dtrend_msg_n <https://www.ncl.ucar.edu/Document/Functions/Built-in/dtrend_msg_n.shtml>`_,
    `dtrend_quadratic <https://www.ncl.ucar.edu/Document/Functions/Built-in/dtrend_quadratic.shtml>`_,
    `dtrend_quadratic_msg_n <https://www.ncl.ucar.edu/Document/Functions/Built-in/dtrend_quadratic_msg_n.shtml>`_
    """
    if (int(deg) != deg) or (deg < 0):
        raise ValueError("deg must be non-negative integer value.")

    return_info = bool(kwargs.get("return_info", True))

    missing_value = _get_missing_value(data, kwargs)

    try:
        data_shape = data.shape if isinstance(
            data,
            (np.ndarray, xr.DataArray, da.Array)) else np.asarray(data).shape
    except BaseException:
        raise TypeError("Could not extract the shape of data")

    x = kwargs.get("x", None)
    if x is None:
        x = np.arange(data_shape[axis], dtype=np.float)
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x).astype(np.float)

    x = x.reshape((-1,))

    p = ndpolyfit(x, data, deg, axis, missing_value=missing_value)

    data_fitted = ndpolyval(p, x, axis)

    data_detrended = xr.DataArray(data) - data_fitted

    if return_info:
        data_detrended.attrs["p"] = p

    return data_detrended
