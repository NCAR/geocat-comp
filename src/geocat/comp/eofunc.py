import warnings
from typing import Iterable

import numpy as np
import xarray as xr
from eofs.xarray import Eof


def _generate_eofs_solver(data, time_dim=0, weights=None, center=True, ddof=1):
    """Convenience function to be used in both `eofunc_eofs` and `eofunc_pcs`
    functions."""

    # ''' Start of boilerplate
    if not isinstance(data, xr.DataArray):

        data = np.asarray(data)

        if (time_dim >= data.ndim) or (time_dim < -data.ndim):
            raise ValueError("ERROR eofunc_efs: `time_dim` out of bound.")

        # Transpose data if time_dim is not 0 (i.e. the first/left-most dimension)
        dims_to_transpose = np.arange(data.ndim).tolist()
        dims_to_transpose.insert(
            0, dims_to_transpose.pop(dims_to_transpose.index(time_dim)))
        data = np.transpose(data, axes=dims_to_transpose)

        dims = [f"dim_{i}" for i in range(data.ndim)]
        dims[0] = 'time'

        data = xr.DataArray(
            data,
            dims=dims,
        )

    solver = Eof(data, weights=weights, center=center, ddof=ddof)

    return data, solver


def eofunc_eofs(data,
                neofs=1,
                time_dim=0,
                eofscaling=0,
                weights=None,
                center=True,
                ddof=1,
                vfscaled=False,
                meta=False):
    """Computes empirical orthogonal functions (EOFs, aka: Principal Component
    Analysis).

    Note: `eofunc_eofs` allows to perform the EOF analysis that was previously done via the NCL function `eofunc`.
    However, there are a few changes to the NCL flow such as : (1) Only `np.nan` is supported as missing value,
    (2) EOFs are computed only from covariance matrix and there is no support for computation from correlation matrix,
    (3) percentage of non-missing points that must exist at any single point is no longer an input.

    This implementation uses `eofs` package (https://anaconda.org/conda-forge/eofs), which is built upon the
    following study: Dawson, Andrew, "eofs: A library for EOF analysis of meteorological, oceanographic, and
    climate data," Journal of Open Research Software, vol. 4, no. 1, 2016. Further information about this
    package can be found at: https://ajdawson.github.io/eofs/latest/index.html#

    This implementation provides a few conveniences to the user on top of `eofs` package that are described below
    in the Parameters section.

    Parameters
    ----------
    data : (:class:`xarray.DataArray` or :class:`numpy.ndarray` or :class:`list`)
        Should contain numbers or `np.nan` for missing value representation. It must be at least a 2-dimensional array.

        When input data is of type `xarray.DataArray`, `eofs.xarray` interface assumes the left-most dimension
        (i.e. `dim_0`) is the `time` dimension. In this case, that dimension should have the name "time".

        When input data is of type `numpy.ndarray` or `list`, this function still assumes the leftmost dimension
        to be the number of observations or `time` dimension: however, in this case, user is allowed to input otherwise.
        If the input do not have its leftmost dimension as the `time` or number of observations, then the user should
        specify with `time_dim=x` to define which dimension must be treated as time or number of observations

    neofs : (:class:`int`, Optional)
        A scalar integer that specifies the number of empirical orthogonal functions (i.e. eigenvalues and
        eigenvectors) to be returned. This is usually less than or equal to the minimum number of observations or
        number of variables. Defaults to 1.

    time_dim : (:class:`int`, Optional)
        An integer defining the time dimension if it is not the leftmost dimension. When input data is of type
        `xarray.DataArray`, this is ignored (assuming `xarray.DataArray` has its leftmost dimension with the exact
        name 'time'). It must be between ``0`` and ``data.ndim - 1`` or it could be ``-1`` indicating the last
        dimension. Defaults to 0.

        Note: The `time_dim` argument allows to perform the EOF analysis that was previously done via the NCL
        function `eofunc_n`.

    eofscaling : (:class:`int`, Optional)
        (From `eofs` package): Sets the scaling of the EOFs. The following values are accepted:

            - 0 : Un-scaled EOFs (default).
            - 1 : EOFs are divided by the square-root of their eigenvalues.
            - 2 : EOFs are multiplied by the square-root of their eigenvalues.

    weights : (:class:`array_like`, Optional)
        (From `eofs` package): An array of weights whose shape is compatible with those of the input array dataset.
        The weights can have the same shape as dataset or a shape compatible with an array broadcast (i.e., the shape
        of the weights can can match the rightmost parts of the shape of the input array dataset). If the input array
        dataset does not require weighting then the value None may be used. Defaults to None (no weighting).

    center : (:class:`bool`, Optional)
        (From `eofs` package): If True, the mean along the first axis of dataset (the time-mean) will be removed prior
        to analysis. If False, the mean along the first axis will not be removed. Defaults to True (mean is removed).

        The covariance interpretation relies on the input data being anomaly data with a time-mean of 0. Therefore this
        option should usually be set to True. Setting this option to True has the useful side effect of propagating
        missing values along the time dimension, ensuring that a solution can be found even if missing values occur
        in different locations at different times.

    ddof : (:class:`int`, Optional)
        (From `eofs` package): ‘Delta degrees of freedom’. The divisor used to normalize the covariance matrix is
        N - ddof where N is the number of samples. Defaults to 1.

    vfscaled : (:class:`bool`, Optional)
        (From `eofs` package): If True, scale the errors by the sum of the eigenvalues. This yields typical errors
        with the same scale as the values returned by Eof.varianceFraction. If False then no scaling is done.
        Defaults to False.

    meta : (:class:`bool`, Optional)
        If set to True and the input array is an Xarray, the metadata from the input array will be copied to the
        output array. Defaults to False.

    Returns
    -------
        A multi-dimensional array containing EOFs. The returned array will be of the same size as data with the
        leftmost dimension removed and an additional dimension of the size `neofs` added.

        The return variable will have associated with it the following attributes:

        eigenvalues:
            A one-dimensional array of size `neofs` that contains the eigenvalues associated with each EOF.

        northTest:
            (From `eofs` package): Typical errors for eigenvalues.

            The method of North et al. (1982) is used to compute the typical error for each eigenvalue. It is
            assumed that the number of times in the input data set is the same as the number of independent
            realizations. If this assumption is not valid then the result may be inappropriate.

            Note: The `northTest` attribute allows to perform the error analysis that was previously done via the NCL
            function `eofunc_north`.

        totalAnomalyVariance:
            (From `eofs` package): Total variance associated with the field of anomalies (the sum of the eigenvalues).

        varianceFraction:
            (From `eofs` package): Fractional EOF mode variances.

            The fraction of the total variance explained by each EOF mode, values between 0 and 1 inclusive..

    See Also
    --------
    Related NCL Functions:
    `eofunc <https://www.ncl.ucar.edu/Document/Functions/Built-in/eofunc.shtml>`_,
    `eofunc_Wrap <https://www.ncl.ucar.edu/Document/Functions/Contributed/eofunc_Wrap.shtml>`_,
    `eofunc_north <https://www.ncl.ucar.edu/Document/Functions/Contributed/eofunc_north.shtml>`_,
    `eofunc_n <https://www.ncl.ucar.edu/Document/Functions/Built-in/eofunc_n.shtml>`_,
    `eofunc_n_Wrap <https://www.ncl.ucar.edu/Document/Functions/Contributed/eofunc_n_Wrap.shtml>`_
    """

    data, solver = _generate_eofs_solver(data,
                                         time_dim=time_dim,
                                         weights=weights,
                                         center=center,
                                         ddof=ddof)

    # Checking number of EOFs
    if neofs <= 0:
        raise ValueError(
            "ERROR eofunc_eofs: num_eofs must be a positive non-zero integer value."
        )

    eofs = solver.eofs(neofs=neofs, eofscaling=eofscaling)

    # Populate attributes for output
    attrs = {}

    if meta:
        attrs = data.attrs

    attrs['eigenvalues'] = solver.eigenvalues(neigs=neofs)
    attrs['northTest'] = solver.northTest(neigs=neofs, vfscaled=vfscaled)
    attrs['totalAnomalyVariance'] = solver.totalAnomalyVariance()
    attrs['varianceFraction'] = solver.varianceFraction(neigs=neofs)

    if meta:
        dims = ["eof"
               ] + [data.dims[i] for i in range(data.ndim) if i != time_dim]
        coords = {
            k: v for (k, v) in data.coords.items() if k != data.dims[time_dim]
        }
    else:
        dims = ["eof"] + [f"dim_{i}" for i in range(data.ndim) if i != time_dim]
        coords = {}

    return xr.DataArray(eofs, attrs=attrs, dims=dims, coords=coords)


def eofunc_pcs(data,
               npcs=1,
               time_dim=0,
               pcscaling=0,
               weights=None,
               center=True,
               ddof=1,
               meta=False):
    """Computes the principal components (time projection) in the empirical
    orthogonal function analysis.

    Note: `eofunc_pcs` allows to perform the analysis that was previously done via the NCL function `eofunc_ts`.
    However, there are a few changes to the NCL flow such as : (1) Only `np.nan` is supported as missing value,
    (2) EOFs are computed only from covariance matrix and there is no support for computation from correlation matrix,
    (3) percentage of non-missing points that must exist at any single point is no longer an input.

    This implementation uses `eofs` package (https://anaconda.org/conda-forge/eofs), which is built upon the
    following study: Dawson, Andrew, "eofs: A library for EOF analysis of meteorological, oceanographic, and
    climate data," Journal of Open Research Software, vol. 4, no. 1, 2016. Further information about this
    package can be found at: https://ajdawson.github.io/eofs/latest/index.html#

    This implementation provides a few conveniences to the user on top of `eofs` package that are described below
    in the Parameters section.

    Parameters
    ----------
    data : :class:`xarray.DataArray` or :class:`numpy.ndarray` or :class:`list`
        Should contain numbers or `np.nan` for missing value representation. It must be at least a 2-dimensional array.

        When input data is of type `xarray.DataArray`, `eofs.xarray` interface assumes the left-most dimension
        (i.e. `dim_0`) is the `time` dimension. In this case, that dimension should have the name "time".

        When input data is of type `numpy.ndarray` or `list`, this function still assumes the leftmost dimension
        to be the number of observations or `time` dimension: however, in this case, user is allowed to input otherwise.
        If the input do not have its leftmost dimension as the `time` or number of observations, then the user should
        specify with `time_dim=x` to define which dimension must be treated as time or number of observations

    npcs : (:class:`int`, Optional)
        A scalar integer that specifies the number of principal components (i.e. eigenvalues and eigenvectors) to be
        returned. This is usually less than or equal to the minimum number of observations or number of variables.
        Defaults to 1.

    time_dim : (:class:`int`, Optional)
        An integer defining the time dimension if it is not the leftmost dimension. When input data is of type
        `xarray.DataArray`, this is ignored (assuming `xarray.DataArray` has its leftmost dimension with the exact
        name 'time'). It must be between ``0`` and ``data.ndim - 1`` or it could be ``-1`` indicating the last
        dimension. Defaults to 0.

        Note: The `time_dim` argument allows to perform the EOF analysis that was previously done via the NCL
        function `eofunc_ts_n`.

    pcscaling : (:class:`int`, Optional)
        (From `eofs` package): Sets the scaling of the retrieved PCs. The following values are accepted:
            - 0 : Un-scaled PCs (default).
            - 1 : PCs are divided by the square-root of their eigenvalues.
            - 2 : PCs are multiplied by the square-root of their eigenvalues.

    weights : (:class:`array_like`, Optional)
        (From `eofs` package): An array of weights whose shape is compatible with those of the input array dataset.
        The weights can have the same shape as dataset or a shape compatible with an array broadcast (i.e., the shape
        of the weights can can match the rightmost parts of the shape of the input array dataset). If the input array
        dataset does not require weighting then the value None may be used. Defaults to None (no weighting).

    center : (:class:`bool`, Optional)
        (From `eofs` package): If True, the mean along the first axis of dataset (the time-mean) will be removed prior
        to analysis. If False, the mean along the first axis will not be removed. Defaults to True (mean is removed).

        The covariance interpretation relies on the input data being anomaly data with a time-mean of 0. Therefore this
        option should usually be set to True. Setting this option to True has the useful side effect of propagating
        missing values along the time dimension, ensuring that a solution can be found even if missing values occur
        in different locations at different times.

    ddof : (:class:`int`, Optional)
        (From `eofs` package): ‘Delta degrees of freedom’. The divisor used to normalize the covariance matrix is
        N - ddof where N is the number of samples. Defaults to 1.

    meta : (:class:`bool`, Optional)
        If set to True and the input array is an Xarray, the metadata from the input array will be copied to the
        output array. Defaults to False.

    Returns
    -------

    See Also
    --------
    Related NCL Functions:
    `eofunc_ts <https://www.ncl.ucar.edu/Document/Functions/Built-in/eofunc_ts.shtml>`_,
    `eofunc_ts_Wrap <https://www.ncl.ucar.edu/Document/Functions/Contributed/eofunc_ts_Wrap.shtml>`_,
    `eofunc_ts_n <https://www.ncl.ucar.edu/Document/Functions/Built-in/eofunc_ts_n.shtml>`_,
    `eofunc_ts_n_Wrap <https://www.ncl.ucar.edu/Document/Functions/Contributed/eofunc_ts_n_Wrap.shtml>`_
    """

    data, solver = _generate_eofs_solver(data,
                                         time_dim=time_dim,
                                         weights=weights,
                                         center=center,
                                         ddof=ddof)

    # Checking number of EOFs
    if npcs <= 0:
        raise ValueError(
            "ERROR eofunc_pcs: num_pcs must be a positive non-zero integer value."
        )

    solver = Eof(data, weights=weights, center=center, ddof=ddof)

    pcs = solver.pcs(npcs=npcs, pcscaling=pcscaling)
    pcs = pcs.transpose()

    # Populate attributes for output
    attrs = {}

    if meta:
        attrs = data.attrs

    dims = ["pc", "time"]
    if meta:
        coords = {"time": data.coords[data.dims[time_dim]]}
    else:
        coords = {}

    return xr.DataArray(pcs, attrs=attrs, dims=dims, coords=coords)


# Transparent wrappers for geocat.comp backwards compatibility


def eofunc(data: Iterable, neval, **kwargs) -> xr.DataArray:
    warnings.warn(
        "eofunc will be deprecated soon in a future version and may not currently generate proper results for some of "
        "its arguments including `pcrit`, `jopt="
        "correlation"
        "`, and 'missing_value' other than np.nan. The output "
        " and its attributes may thus not be as expected, too. Use `eofunc_eofs` instead.",
        PendingDeprecationWarning)

    if not isinstance(data, xr.DataArray) or not isinstance(data, np.ndarray):
        data = np.asarray(data)

    time_dim = int(kwargs.get("time_dim", data.ndim - 1))
    meta = bool(kwargs.get("meta"))

    return eofunc_eofs(data, neofs=neval, time_dim=time_dim, meta=meta)


def eofunc_ts(data: Iterable, evec, **kwargs) -> xr.DataArray:
    warnings.warn(
        "eofunc_ts will be deprecated soon in a future version and may not currently generate proper results for "
        "some of its arguments including `evec`, `jopt="
        "correlation"
        "`, and 'missing_value' other than np.nan. The output "
        " and its attributes may thus not be as expected, too. Use `eofunc_pcs` instead.",
        PendingDeprecationWarning)

    if not isinstance(data, xr.DataArray) or not isinstance(data, np.ndarray):
        data = np.asarray(data)

    time_dim = int(kwargs.get("time_dim", data.ndim - 1))
    meta = bool(kwargs.get("meta"))

    return eofunc_pcs(data, npcs=evec.shape[0], time_dim=time_dim, meta=meta)
