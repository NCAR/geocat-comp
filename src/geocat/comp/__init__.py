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


def eofunc(data, neval, **kwargs):
    """
    Computes empirical orthogonal functions (EOFs, aka: Principal Component Analysis).

    Args:
        data:
            an iterable object containing numbers. It must be at least a 2-dimensional array. The right-most dimension
            is assumed to be the number of observations. Generally this is the time time dimension. If your right-most
            dimension is not time, you could pass ``time_dim=x`` as an argument to define which dimension must be treated
            as time and/or number of observations.
            Data must be convertible to numpy.array
        neval:
            A scalar integer that specifies the number of eigenvalues and eigenvectors to be returned. This is usually
            less than or equal to the minimum number of observations or number of variables.
        **kwargs:
            extra options controlling the behavior of the function. Currently the following are supported:
            - ``jopt``: an integer that indicates whether to use the covariance matrix or the correlation
                        matrix. The default is to use the covariance matrix.
            - ``pcrit``: a float value between ``0`` and ``100`` that indicates the percentage of non-missing points
                         that must exist at any single point in order to be calculated. The default is 50%. Points that
                         contain all missing values will automatically be set to missing.
            - ''time_dim``: an integer defining the time dimension. it must be between ``0`` and ``data.ndim - 1`` or it
                            could be ``-1`` indicating the last dimension. The default balue is -1.
            - ``missing_value``: a value defining the missing value. The default is ``np.nan``.

    """
    # Parsing Options
    options = {}
    if "jopt" in kwargs:
        if not isinstance(kwargs["jopt"], str):
            raise TypeError('jopt must be a string set to eirther "correlation" or "covariance".')
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

    missing_value = kwargs["missing_value"] if "missing_value" in kwargs else np.nan


    np_data = np.asarray(data)
    time_dim = -1
    if "time_dim" in kwargs:
        time_dim = int(kwargs["time_dim"])
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

    eof = response[0]
    attrs = {}
    # converting the keys to string instead of bytes also fixing matrix and method
    # TODO: once Kevin's work on char * is merged, we could remove this part or change it properly.
    for k, v in response[1].items():
        if k in {b'matrix', b'method'}:
            attrs[k.decode('utf-8')] = v.tostring().decode('utf-8')[:-1]
        else:
            attrs[k.decode('utf-8')] = v

    return eof, attrs



