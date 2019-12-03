from typing import Iterable

from . import _ncomp
from .version import __version__
import numpy as np
import xarray as xr
import dask.array as da
from dask.array.core import map_blocks


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class ChunkError(Error):
    """Exception raised when a Dask array is chunked in a way that is
    incompatible with an _ncomp function."""
    pass


class CoordinateError(Error):
    """Exception raised when a GeoCAT-comp function is passed a NumPy array as
    an argument without a required coordinate array being passed separately."""
    pass


def linint2(fi, xo, yo, icycx, msg=None, meta=True, xi=None, yi=None):
    """Interpolates a regular grid to a rectilinear one using bi-linear
    interpolation.

    linint2 uses bilinear interpolation to interpolate from one
    rectilinear grid to another. The input grid may be cyclic in the x
    direction. The interpolation is first performed in the x direction,
    and then in the y direction.

    Args:

        fi (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            An array of two or more dimensions. If xi is passed in as an
            argument, then the size of the rightmost dimension of fi
            must match the rightmost dimension of xi. Similarly, if yi
            is passed in as an argument, then the size of the second-
            rightmost dimension of fi must match the rightmost dimension
            of yi.

            If missing values are present, then linint2 will perform the
            bilinear interpolation at all points possible, but will
            return missing values at coordinates which could not be
            used.

            Note:

                This variable must be
                supplied as a :class:`xarray.DataArray` in order to copy
                the dimension names to the output. Otherwise, default
                names will be used.

        xo (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            A one-dimensional array that specifies the X coordinates of
            the return array. It must be strictly monotonically
            increasing, but may be unequally spaced.

            For geo-referenced data, xo is generally the longitude
            array.

            If the output coordinates (xo) are outside those of the
            input coordinates (xi), then the fo values at those
            coordinates will be set to missing (i.e. no extrapolation is
            performed).

        yo (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            A one-dimensional array that specifies the Y coordinates of
            the return array. It must be strictly monotonically
            increasing, but may be unequally spaced.

            For geo-referenced data, yo is generally the latitude array.

            If the output coordinates (yo) are outside those of the
            input coordinates (yi), then the fo values at those
            coordinates will be set to missing (i.e. no extrapolation is
            performed).

        icycx (:obj:`bool`):
            An option to indicate whether the rightmost dimension of fi
            is cyclic. This should be set to True only if you have
            global data, but your longitude values don't quite wrap all
            the way around the globe. For example, if your longitude
            values go from, say, -179.75 to 179.75, or 0.5 to 359.5,
            then you would set this to True.

        msg (:obj:`numpy.number`):
            A numpy scalar value that represent a missing value in fi.
            This argument allows a user to use a missing value scheme
            other than NaN or masked arrays, similar to what NCL allows.

        meta (:obj:`bool`):
            Set to False to disable metadata; default is True.

        xi (:class:`numpy.ndarray`):
            An array that specifies the X coordinates of the fi array.
            Most frequently, this is a 1D strictly monotonically
            increasing array that may be unequally spaced. In some
            cases, xi can be a multi-dimensional array (see next
            paragraph). The rightmost dimension (call it nxi) must have
            at least two elements, and is the last (fastest varying)
            dimension of fi.

            If xi is a multi-dimensional array, then each nxi subsection
            of xi must be strictly monotonically increasing, but may be
            unequally spaced. All but its rightmost dimension must be
            the same size as all but fi's rightmost two dimensions.

            For geo-referenced data, xi is generally the longitude
            array.

            Note:
                If fi is of type :class:`xarray.DataArray` and xi is
                left unspecified, then the rightmost coordinate
                dimension of fi will be used. If fi is not of type
                :class:`xarray.DataArray`, then xi becomes a mandatory
                parameter. This parameter must be specified as a keyword
                argument.

        yi (:class:`numpy.ndarray`):
            An array that specifies the Y coordinates of the fi array.
            Most frequently, this is a 1D strictly monotonically
            increasing array that may be unequally spaced. In some
            cases, yi can be a multi-dimensional array (see next
            paragraph). The rightmost dimension (call it nyi) must have
            at least two elements, and is the second-to-last dimension
            of fi.

            If yi is a multi-dimensional array, then each nyi subsection
            of yi must be strictly monotonically increasing, but may be
            unequally spaced. All but its rightmost dimension must be
            the same size as all but fi's rightmost two dimensions.

            For geo-referenced data, yi is generally the latitude array.

            Note:
                If fi is of type :class:`xarray.DataArray` and xi is
                left unspecified, then the second-to-rightmost
                coordinate dimension of fi will be used. If fi is not of
                type :class:`xarray.DataArray`, then xi becomes a
                mandatory parameter. This parameter must be specified as
                a keyword argument.

    Returns:
        :class:`xarray.DataArray`: The interpolated grid. If the *meta*
        parameter is True, then the result will include named dimensions
        matching the input array. The returned value will have the same
        dimensions as fi, except for the rightmost two dimensions which
        will have the same dimension sizes as the lengths of yo and xo.
        The return type will be double if fi is double, and float
        otherwise.

    Examples:

        Example 1: Using linint2 with :class:`xarray.DataArray` input

        .. code-block:: python

            import numpy as np
            import xarray as xr
            import geocat.comp

            fi_np = np.random.rand(30, 80)  # random 30x80 array

            # xi and yi do not have to be equally spaced, but they are
            # in this example
            xi = np.arange(80)
            yi = np.arange(30)

            # create target coordinate arrays, in this case use the same
            # min/max values as xi and yi, but with different spacing
            xo = np.linspace(xi.min(), xi.max(), 100)
            yo = np.linspace(yi.min(), yi.max(), 50)

            # create :class:`xarray.DataArray` and chunk it using the
            # full shape of the original array.
            # note that xi and yi are attached as coordinate arrays
            fi = xr.DataArray(fi_np,
                              dims=['lat', 'lon'],
                              coords={'lat': yi, 'lon': xi}
                             ).chunk(fi_np.shape)

            fo = geocat.comp.linint2(fi, xo, yo, 0)

    """

    if not isinstance(fi, xr.DataArray):
        fi = xr.DataArray(fi)
        if xi is None or yi is None:
            raise CoordinateError("linint2: arguments xi and yi must be passed"
                                  " explicitly if fi is not an xarray.DataArray.")

    if xi is None:
        xi = fi.coords[fi.dims[-1]].values
    elif isinstance(xi, xr.DataArray):
        xi = xi.values

    if yi is None:
        yi = fi.coords[fi.dims[-2]].values
    elif isinstance(yi, xr.DataArray):
        yi = yi.values

    # ensure xo and yo are numpy.ndarrays
    if isinstance(xo, xr.DataArray):
        xo = xo.values
    if isinstance(yo, xr.DataArray):
        yo = yo.values

    fi_data = fi.data

    if isinstance(fi_data, da.Array):
        chunks = list(fi.chunks)

        # ensure rightmost dimensions of input are not chunked
        if chunks[-2:] != [yi.shape, xi.shape]:
            raise ChunkError("linint2: the two rightmost dimensions of fi must"
                             " not be chunked.")

        # ensure rightmost dimensions of output are not chunked
        chunks[-2:] = (yo.shape, xo.shape)

        # map_blocks maps each chunk of fi_data to a separate invocation of
        # _ncomp._linint2. The "chunks" keyword argument should be the chunked
        # dimensionality of the expected output; the number of chunks should
        # match that of fi_data. Additionally, "drop_axis" and "new_axis" in
        # this case indicate that the two rightmost dimensions of the input
        # will be dropped from the output array, and that two new axes will be
        # added instead.
        fo = map_blocks(_ncomp._linint2, xi, yi, fi_data, xo, yo, icycx, msg,
                        chunks=chunks, dtype=fi.dtype,
                        drop_axis=[fi.ndim-2, fi.ndim-1],
                        new_axis=[fi.ndim-2, fi.ndim-1])
    elif isinstance(fi_data, np.ndarray):
        fo = _ncomp._linint2(xi, yi, fi_data, xo, yo, icycx, msg)
    else:
        raise TypeError

    if meta:
        coords = {k:v if k not in fi.dims[-2:]
                  else (xo if k == fi.dims[-1] else yo)
                  for (k, v) in fi.coords.items()}

        fo = xr.DataArray(fo, attrs=fi.attrs, dims=fi.dims,
                              coords=coords)
    else:
        fo = xr.DataArray(fo)

    return fo


def eofunc(data: Iterable, neval, **kwargs) -> xr.DataArray:
    """
    Computes empirical orthogonal functions (EOFs, aka: Principal Component Analysis).

    Args:
        data:
            an iterable object containing numbers. It must be at least a 2-dimensional array. The right-most dimension
            is assumed to be the number of observations. Generally this is the time time dimension. If your right-most
            dimension is not time, you could pass ``time_dim=x`` as an argument to define which dimension must be
            treated as time and/or number of observations. Data must be convertible to numpy.array
        neval:
            A scalar integer that specifies the number of eigenvalues and eigenvectors to be returned. This is usually
            less than or equal to the minimum number of observations or number of variables.
        **kwargs:
            extra options controlling the behavior of the function. Currently the following are supported:
            - ``jopt``: a string that indicates whether to use the covariance matrix or the correlation
                        matrix. The default is to use the covariance matrix.
            - ``pcrit``: a float value between ``0`` and ``100`` that indicates the percentage of non-missing points
                         that must exist at any single point in order to be calculated. The default is 50%. Points that
                         contain all missing values will automatically be set to missing.
            - ''time_dim``: an integer defining the time dimension. it must be between ``0`` and ``data.ndim - 1`` or it
                            could be ``-1`` indicating the last dimension. The default value is -1.
            - ``missing_value``: a value defining the missing value. The default is ``np.nan``.
            - ``meta``: if set to ``True`` (or a value that evaluates to ``True``) the properties or attributes
                        associated to the input data are also transferred to the output data. This is equivalent
                        to the ``_Wrap`` version of the functions in ``NCL``. This only works if the input data is
                        of type ``xarray.DataArray``.

    """
    # Parsing Options
    options = {}
    if "jopt" in kwargs:
        if not isinstance(kwargs["jopt"], str):
            raise TypeError('jopt must be a string set to either "correlation" or "covariance".')
        if str.lower(kwargs["jopt"]) not in {"covariance", "correlation"}:
            raise ValueError("jopt must be set to either covariance or correlation.")

        options[b'jopt'] = np.asarray(1) if str.lower(kwargs["jopt"]) == "correlation" else np.asarray(0)

    if "pcrit" in kwargs:
        provided_pcrit = np.asarray(kwargs["pcrit"]).astype(np.float64)
        if provided_pcrit.size != 1:
            raise ValueError("Only a single number must be provided for pcrit.")

        if (provided_pcrit >= 0.0) and (provided_pcrit <= 100.0):
            options[b'pcrit'] = provided_pcrit
        else:
            raise ValueError("pcrit must be between 0 and 100")

    missing_value = kwargs.get("missing_value", np.nan)

    # the input data must be convertible to numpy array
    np_data = None
    if isinstance(data, np.ndarray):
        np_data = data
    elif isinstance(data, xr.DataArray):
        np_data = data.data
    else:
        np_data = np.asarray(data)

    time_dim = int(kwargs.get("time_dim", -1))

    if (time_dim >= np_data.ndim) or (time_dim < -np_data.ndim):
        raise ValueError(f"dimension out of bound. The input data has {np_data.ndim} dimension."
                         f" hence, time_dim must be between {-np_data.ndim} and {np_data.ndim - 1 }")

    if time_dim < 0:
        time_dim = np_data.ndim + time_dim

    # checking neval
    accepted_neval = int(neval)
    if accepted_neval <= 0:
        raise ValueError("neval must be a positive non-zero integer value.")

    if (time_dim == (np_data.ndim - 1)):
        response = _ncomp._eofunc(np_data, accepted_neval, options, missing_value=missing_value)
    else:
        response = _ncomp._eofunc_n(np_data, accepted_neval, time_dim, options, missing_value=missing_value)

    attrs = data.attrs if isinstance(data, xr.DataArray) and bool(kwargs.get("meta", False)) else {}
    attrs["_FillValue"] = np.nan
    attrs["missing_value"] = np.nan

    # converting the keys to string instead of bytes also fixing matrix and method
    # TODO: once Kevin's work on char * is merged, we could remove this part or change it properly.
    for k, v in response[1].items():
        if k in {b'matrix', b'method'}:
            attrs[k.decode('utf-8')] = v.tostring().decode('utf-8')[:-1]
        else:
            attrs[k.decode('utf-8')] = v

    if isinstance(data, xr.DataArray) and bool(kwargs.get("meta", False)):
        dims = ["evn"] + [data.dims[i] for i in range(data.ndim) if i != time_dim]
        coords = {k: v for (k, v) in data.coords.items() if k != data.dims[time_dim]}
    else:
        dims = ["evn"] + [f"dim_{i}" for i in range(np_data.ndim) if i != time_dim]
        coords = {}

    return xr.DataArray(
        response[0],
        attrs=attrs,
        dims=dims,
        coords=coords
    )


def eofunc_ts(data: Iterable, evec, **kwargs) -> xr.DataArray:
    """
    Calculates the time series of the amplitudes associated with each eigenvalue in an EOF.
    Args:
        data: An Iterable convertible to `numpy.ndarray` in which the rightmost dimension is the number of
              observations. Generally, this is the time dimension. If your rightmost dimension is not time, then pass
              `time_dim` as an extra options.
        evec: An Iterable convertible to `numpy.ndarray` containing the EOFs calculated using `eofunc`.
        **kwargs:
            extra options controlling the behavior of the function. Currently the following are supported:
            - ``jopt``: a string that indicates whether to use the covariance matrix or the correlation
                        matrix. The default is to use the covariance matrix.
            - ''time_dim``: an integer defining the time dimension. it must be between ``0`` and ``data.ndim - 1`` or it
                            could be ``-1`` indicating the last dimension. The default value is -1.
            - ``missing_value``: defines the missing_value. The default is ``np.nan``.
            - ``meta``: if set to ``True`` (or a value that evaluates to ``True``) the properties or attributes
                        associated to the input data are also transferred to the output data. This is equivalent
                        to the ``_Wrap`` version of the functions in ``NCL``. This only works if the input data is
                        of type ``xarray.DataArray``.

    Returns: A two-dimensional array dimensioned by the number of eigenvalues selected in `eofunc` by the size of the
             time dimension of data. Will contain the following attribute:
             - `ts_mean`: an array of the same size and type as `evec` containing the means removed from data as part
                          of the calculation.

    Examples:
        * Passing a xarray:

        >>> # Openning a data set:
        ... ds = xr.open_dataset("dataset.nc")
        >>> # Extracting SST (Sea Surface temperature)
        ... sst = ds.sst
        >>> evec = eofunc(sst, 5)
        >>> ts = eofunc(sst, evec)

        * Passing a numpy array:

        >>> # Openning a data set:
        ... ds = xr.open_dataset("dataset.nc")
        >>> # Extracting SST (Sea Surface temperature) as Numpy Array
        ... sst = ds.sst.data
        >>> evec = eofunc(sst, 5)
        >>> ts = eofunc(sst, evec.data)

        * Transferring the attributes from input to the output:

        >>> # Openning a data set:
        ... ds = xr.open_dataset("dataset.nc")
        >>> # Extracting SST (Sea Surface temperature)
        ... sst = ds.sst
        >>> evec = eofunc(sst, 5)
        >>> ts = eofunc(sst, evec, meta=True)

        * Defining the time dimension:

        >>> # Openning a data set:
        ... ds = xr.open_dataset("dataset.nc")
        >>> # Extracting SST (Sea Surface temperature)
        ... sst = ds.sst
        >>> evec = eofunc(sst, 5, time_dim=0)
        >>> ts = eofunc(sst, evec, time_dim=0)


    """
    # Parsing Options
    options = {}
    if "jopt" in kwargs:
        if not isinstance(kwargs["jopt"], str):
            raise TypeError('jopt must be a string set to either "correlation" or "covariance".')
        if str.lower(kwargs["jopt"]) not in {"covariance", "correlation"}:
            raise ValueError("jopt must be set to either covariance or correlation.")

        options[b'jopt'] = np.asarray(1) if str.lower(kwargs["jopt"]) == "correlation" else np.asarray(0)

    missing_value = kwargs.get("missing_value", np.nan)

    # the input data must be convertible to numpy array
    if isinstance(data, np.ndarray):
        np_data = data
    elif isinstance(data, xr.DataArray):
        np_data = data.data
    else:
        np_data = np.asarray(data)

    # the input data must be convertible to numpy array
    if isinstance(evec, np.ndarray):
        np_evec = evec
    elif isinstance(evec, xr.DataArray):
        np_evec = evec.data
    else:
        np_evec = np.asarray(evec)

    time_dim = int(kwargs.get("time_dim", -1))

    if (time_dim >= np_data.ndim) or (time_dim < -np_data.ndim):
        raise ValueError(f"dimension out of bound. The input data has {np_data.ndim} dimension."
                             f" hence, time_dim must be between {-np_data.ndim} and {np_data.ndim - 1 }")
    if time_dim < 0:
        time_dim = np_data.ndim + time_dim

    if (time_dim == (np_data.ndim - 1)):
        response = _ncomp._eofunc_ts(np_data, np_evec, options, missing_value=missing_value)
    else:
        response = _ncomp._eofunc_ts_n(np_data, np_evec, time_dim, options, missing_value=missing_value)

    attrs = data.attrs if isinstance(data, xr.DataArray) and bool(kwargs.get("meta", False)) else {}
    attrs["_FillValue"] = np.nan
    attrs["missing_value"] = np.nan

    # converting the keys to string instead of bytes also fixing matrix and method
    # TODO: once Kevin's work on char * is merged, we could remove this part or change it properly.
    for k, v in response[1].items():
        if k in {b'matrix'}:
            attrs[k.decode('utf-8')] = v.tostring().decode('utf-8')[:-1]
        else:
            attrs[k.decode('utf-8')] = v

    dims = ["neval", "time"]
    if isinstance(data, xr.DataArray) and bool(kwargs.get("meta", False)):
        coords = {"time": data.coords[data.dims[time_dim]]}
    else:
        coords = {}

    return xr.DataArray(
        response[0],
        attrs=attrs,
        dims=dims,
        coords=coords
    )


def moc_globe_atl(lat_aux_grid, a_wvel, a_bolus, a_submeso, tlat, rmlak,
                  msg=None, meta=False):
    """Facilitates calculating the meridional overturning circulation for the
    globe and Atlantic.

    Args:

        lat_aux_grid (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            Latitude grid for transport diagnostics.

        a_wvel (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            Area weighted Eulerian-mean vertical velocity [TAREA*WVEL].

        a_bolus (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            Area weighted Eddy-induced (bolus) vertical velocity [TAREA*WISOP].

        a_submeso (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            Area weighted submeso vertical velocity [TAREA*WSUBM].

        tlat (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            Array of t-grid latitudes.

        rmlak (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            Basin index number: [0]=Globe, [1]=Atlantic

        msg (:obj:`numpy.number`):
          A numpy scalar value that represent a missing value.
          This argument allows a user to use a missing value scheme
          other than NaN or masked arrays, similar to what NCL allows.

        meta (:obj:`bool`):
          Set to False to disable metadata; default is True.

        Returns:
            :class:`xarray.DataArray`: A multi-dimensional array of size [moc_comp] x
            [n_transport_reg] x [kdepth] x [nyaux] where:

            - moc_comp refers to the three components returned
            - n_transport_reg refers to the Globe and Atlantic
            - kdepth is the the number of vertical levels of the work arrays
            - nyaux is the size of the lat_aux_grid

            The type of the output data will be double only if a_wvel or a_bolus or
            a_submesa is of type double. Otherwise, the return type will be float.


    Examples:

        # TODO: To be included

    """

    # Ensure input arrays are numpy.ndarrays
    if isinstance(lat_aux_grid, xr.DataArray):
        lat_aux_grid = lat_aux_grid.values

    if isinstance(a_wvel, xr.DataArray):
        a_wvel = a_wvel.values

    if isinstance(a_bolus, xr.DataArray):
        a_bolus = a_bolus.values

    if isinstance(a_submeso, xr.DataArray):
        a_submeso = a_submeso.values

    if isinstance(tlat, xr.DataArray):
        tlat = tlat.values

    if isinstance(rmlak, xr.DataArray):
        rmlak = rmlak.values

    # Make sure msg has the correct dtype even if given wrong type or a scalar instead of np.num
    if a_wvel.dtype == np.float64:
        msg = np.float64(msg)
    else:
        msg = np.float32(msg)


    # Call ncomp function
    out_arr = _ncomp._moc_globe_atl(lat_aux_grid, a_wvel, a_bolus, a_submeso,
                                    tlat, rmlak, msg)

    if meta and isinstance(input, xr.DataArray):
        pass
        # TODO: Retaining possible metadata might be revised in the future
    else:
        out_arr = xr.DataArray(out_arr)

    return out_arr
