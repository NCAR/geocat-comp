from typing import Iterable

from . import _ncomp
from .version import __version__
import numpy as np
import xarray as xr
import dask.array as da
from dask.array.core import map_blocks

from .polynomial import ndpolyfit, ndpolyval


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

class DimensionError(Error):
     """Exception raised when the arguments of GeoCAT-comp functions argument
     has a mismatch of the necessary dimensionality."""
     pass

class AttributeError(Error):
     """Exception raised when the arguments of GeoCAT-comp functions argument
     has a mismatch of attributes with other arguments."""
     pass

class MetaError(Error):
     """Exception raised when the support for the retention of metadata is not
     supported."""
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

def rcm2rgrid(lat2d, lon2d, fi, lat1d, lon1d, msg=None, meta=False):
    """Interpolates data on a curvilinear grid (i.e. RCM, WRF, NARR) to a rectilinear grid.

    Args:

        lat2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the latitudes locations
	    of fi. Because this array is two-dimensional it is not an associated
	    coordinate variable of `fi`. The latitude order must be south-to-north.

        lon2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the longitude locations
	    of fi. Because this array is two-dimensional it is not an associated
	    coordinate variable of `fi`. The latitude order must be west-to-east.

        fi (:class:`numpy.ndarray`):
	    A multi-dimensional array to be interpolated. The rightmost two
	    dimensions (latitude, longitude) are the dimensions to be interpolated.

        lat1d (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the latitude coordinates of
	    the regular grid. Must be monotonically increasing.

        lon1d (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the longitude coordinates of
	    the regular grid. Must be monotonically increasing.

        msg (:obj:`numpy.number`):
            A numpy scalar value that represent a missing value in fi.
            This argument allows a user to use a missing value scheme
            other than NaN or masked arrays, similar to what NCL allows.

        meta (:obj:`bool`):
            Set to False to disable metadata; default is True.

    Returns:
        :class:`numpy.ndarray`: The interpolated grid. A multi-dimensional array
	of the same size as fi except that the rightmost dimension sizes have been
	replaced by the sizes of lat1d and lon1d respectively.
	Double if fi is double, otherwise float.

    Description:
        Interpolates RCM (Regional Climate Model), WRF (Weather Research and Forecasting) and
        NARR (North American Regional Reanalysis) grids to a rectilinear grid. Actually, this
	function will interpolate most grids that use curvilinear latitude/longitude grids.
	No extrapolation is performed beyond the range of the input coordinates. Missing values
	are allowed but ignored.

	The weighting method used is simple inverse distance squared. Missing values are allowed
	but ignored.

	The code searches the input curvilinear grid latitudes and longitudes for the four
	grid points that surround a specified output grid coordinate. Because one or more of
	these input points could contain missing values, fewer than four points
	could be used in the interpolation.

	Curvilinear grids which have two-dimensional latitude and longitude coordinate axes present
	some issues because the coordinates are not necessarily monotonically increasing. The simple
	search algorithm used by rcm2rgrid is not capable of handling all cases. The result is that,
	sometimes, there are small gaps in the interpolated grids. Any interior points not
	interpolated in the initial interpolation pass will be filled using linear interpolation.
        In some cases, edge points may not be filled.

    Examples:

        Example 1: Using rcm2rgrid with :class:`xarray.DataArray` input

        .. code-block:: python

            import numpy as np
            import xarray as xr
            import geocat.comp

            # Open a netCDF data file using xarray default engine and load the data stream
            ds = xr.open_dataset("./ruc.nc")

            # [INPUT] Grid & data info on the source curvilinear
            ht_curv=ds.DIST_236_CBL[:]
            lat2D_curv=ds.gridlat_236[:]
            lon2D_curv=ds.gridlon_236[:]

            # [OUTPUT] Grid on destination rectilinear grid (or read the 1D lat and lon from
            #          an other .nc file.
            newlat1D_rect=np.linspace(lat2D_curv.min(), lat2D_curv.max(), 100)
            newlon1D_rect=np.linspace(lon2D_curv.min(), lon2D_curv.max(), 100)

            ht_rect = geocat.comp.rcm2rgrid(lat2D_curv, lon2D_curv, ht_curv, newlat1D_rect, newlon1D_rect)


    """

    # Basic sanity checks
    if lat2d.shape[0] != lon2d.shape[0] or lat2d.shape[1] != lon2d.shape[1]:
        raise DimensionError("ERROR rcm2rgrid: The input lat/lon grids must be the same size !")

    if lat2d.shape[0] < 2 or lon2d.shape[0] < 2 or lat2d.shape[1] < 2 or lon2d.shape[1] < 2:
        raise DimensionError("ERROR rcm2rgrid: The input/output lat/lon grids must have at least 2 elements !")

    if fi.ndim < 2:
        raise DimensionError("ERROR rcm2rgrid: fi must be at least two dimensions !\n")

    if fi.shape[fi.ndim - 2] != lat2d.shape[0] or fi.shape[fi.ndim - 1] != lon2d.shape[1]:
        raise DimensionError("ERROR rcm2rgrid: The rightmost dimensions of fi must be (nlat2d x nlon2d),"
                             "where nlat2d and nlon2d are the size of the lat2d/lon2d arrays !")

    if isinstance(lat2d, xr.DataArray):
        lat2d = lat2d.values

    if isinstance(lon2d, xr.DataArray):
        lon2d = lon2d.values

    if not isinstance(fi, xr.DataArray):
        fi = xr.DataArray(fi)

    # ensure lat1d and lon1d are numpy.ndarrays
    if isinstance(lat1d, xr.DataArray):
        lat1d = lat1d.values
    if isinstance(lon1d, xr.DataArray):
        lon1d = lon1d.values

    fi_data = fi.data

    if isinstance(fi_data, da.Array):
        chunks = list(fi.chunks)

        # ensure rightmost dimensions of input are not chunked
        if chunks[-2:] != [lon2d.shape, lat2d.shape]:
            raise ChunkError("rcm2rgrid: the two rightmost dimensions of fi must"
                             " not be chunked.")

        # ensure rightmost dimensions of output are not chunked
        chunks[-2:] = (lon1d.shape, lat1d.shape)

        fo = map_blocks(_ncomp._rcm2rgrid, lat2d, lon2d, fi_data, lat1d, lon1d, msg,
                        chunks=chunks, dtype=fi.dtype,
                        drop_axis=[fi.ndim-2, fi.ndim-1],
                        new_axis=[fi.ndim-2, fi.ndim-1])
    elif isinstance(fi_data, np.ndarray):
        fo = _ncomp._rcm2rgrid(lat2d, lon2d, fi_data, lat1d, lon1d, msg)
    else:
        raise TypeError

    if meta and isinstance(input, xr.DataArray):
        raise MetaError("ERROR rcm2rgrid: retention of metadata is not yet supported !")
    else:
        fo = xr.DataArray(fo)

    return fo

def rgrid2rcm(lat1d, lon1d, fi, lat2d, lon2d, msg=None, meta=False):
    """Interpolates data on a rectilinear lat/lon grid to a curvilinear grid like
       those used by the RCM, WRF and NARR models/datasets.

    Args:

        lat1d (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the latitude coordinates of
	    the regular grid. Must be monotonically increasing.

        lon1d (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the longitude coordinates of
	    the regular grid. Must be monotonically increasing.

        fi (:class:`numpy.ndarray`):
	    A multi-dimensional array to be interpolated. The rightmost two
	    dimensions (latitude, longitude) are the dimensions to be interpolated.

        lat2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the latitude locations
	    of fi. Because this array is two-dimensional it is not an associated
	    coordinate variable of `fi`.

        lon2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the longitude locations
	    of fi. Because this array is two-dimensional it is not an associated
	    coordinate variable of `fi`.

        msg (:obj:`numpy.number`):
            A numpy scalar value that represent a missing value in fi.
            This argument allows a user to use a missing value scheme
            other than NaN or masked arrays, similar to what NCL allows.

        meta (:obj:`bool`):
            Set to False to disable metadata; default is True.

    Returns:
        :class:`numpy.ndarray`: The interpolated grid. A multi-dimensional array of the
	same size as `fi` except that the rightmost dimension sizes have been replaced
	by the sizes of `lat2d` and `lon2d` respectively. Double if `fi` is double,
	otherwise float.

    Description:
        Interpolates data on a rectilinear lat/lon grid to a curvilinear grid, such as those
	used by the RCM (Regional Climate Model), WRF (Weather Research and Forecasting) and
	NARR (North American Regional Reanalysis) models/datasets. No extrapolation is
	performed beyond the range of the input coordinates. The method used is simple inverse
	distance weighting. Missing values are allowed but ignored.

    Examples:

        Example 1: Using rgrid2rcm with :class:`xarray.DataArray` input

        .. code-block:: python

            import numpy as np
            import xarray as xr
            import geocat.comp

            # Open a netCDF data file using xarray default engine and load the data stream
            # input grid and data
            ds_rect = xr.open_dataset("./DATAFILE_RECT.nc")

            # [INPUT] Grid & data info on the source rectilinear
            ht_rect   =ds_rect.SOME_FIELD[:]
            lat1D_rect=ds_rect.gridlat_[:]
            lon1D_rect=ds_rect.gridlon_[:]

            # Open a netCDF data file using xarray default engine and load the data stream
            # for output grid
            ds_curv = xr.open_dataset("./DATAFILE_CURV.nc")

            # [OUTPUT] Grid on destination curvilinear grid (or read the 2D lat and lon from
            #          an other .nc file
            newlat2D_rect=ds_curv.gridlat2D_[:]
            newlon2D_rect=ds_curv.gridlat2D_[:]

            ht_curv = geocat.comp.rgrid2rcm(lat1D_rect, lon1D_rect, ht_rect, newlat2D_curv, newlon2D_curv)


    """

    # Basic sanity checks
    if lat2d.shape[0] != lon2d.shape[0] or lat2d.shape[1] != lon2d.shape[1]:
        raise DimensionError("ERROR rgrid2rcm: The output lat2D/lon2D grids must be the same size !")

    if lat2d.shape[0] < 2 or lon2d.shape[0] < 2 or lat2d.shape[1] < 2 or lon2d.shape[1] < 2:
        raise DimensionError("ERROR rgrid2rcm: The input/output lat/lon grids must have at least 2 elements !")

    if fi.ndim < 2:
        raise DimensionError("ERROR rgrid2rcm: fi must be at least two dimensions !\n")

    if fi.shape[fi.ndim - 2] != lat1d.shape[0] or fi.shape[fi.ndim - 1] != lon1d.shape[0]:
        raise DimensionError("ERROR rgrid2rcm: The rightmost dimensions of fi must be (nlat1d x nlon1d),"
                             "where nlat1d and nlon1d are the size of the lat1d/lon1d arrays !")

    if isinstance(lat1d, xr.DataArray):
        lat1d = lat1d.values

    if isinstance(lon1d, xr.DataArray):
        lon1d = lon1d.values

    if not isinstance(fi, xr.DataArray):
        fi = xr.DataArray(fi)

    # ensure lat2d and lon2d are numpy.ndarrays
    if isinstance(lat2d, xr.DataArray):
        lat2d = lat2d.values
    if isinstance(lon2d, xr.DataArray):
        lon2d = lon2d.values

    fi_data = fi.data

    if isinstance(fi_data, da.Array):
        chunks = list(fi.chunks)

        # ensure rightmost dimensions of input are not chunked
        if chunks[-2:] != [lon1d.shape, lat1d.shape]:
            raise ChunkError("rgrid2rcm: the two rightmost dimensions of fi must"
                             " not be chunked.")

        # ensure rightmost dimensions of output are not chunked
        chunks[-2:] = (lon2d.shape, lat2d.shape)

        fo = map_blocks(_ncomp._rgrid2rcm, lat1d, lon1d, fi_data, lat2d, lon2d, msg,
                        chunks=chunks, dtype=fi.dtype,
                        drop_axis=[fi.ndim-2, fi.ndim-1],
                        new_axis=[fi.ndim-2, fi.ndim-1])
    elif isinstance(fi_data, np.ndarray):
        fo = _ncomp._rgrid2rcm(lat1d, lon1d, fi_data, lat2d, lon2d, msg)
    else:
        raise TypeError

    if meta and isinstance(input, xr.DataArray):
        raise MetaError("ERROR rgrid2rcm: retention of metadata is not yet supported !")
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
        raise MetaError("ERROR moc_globe_atl: retention of metadata is not yet supported !")
    else:
        out_arr = xr.DataArray(out_arr)

    return out_arr

def dpres_plevel(plev, psfc, ptop=None, msg=None, meta=False):
    """Calculates the pressure layer thicknesses of a constant pressure level coordinate system.

    Args:

        plev (:class:`numpy.ndarray`):
            A one dimensional array containing the constant pressure levels. May be
            in ascending or descending order. Must have the same units as `psfc`.

        psfc (:class:`numpy.ndarray`):
            A scalar or an array of up to three dimensions containing the surface
            pressure data in Pa or hPa (mb). The rightmost dimensions must be latitude
            and longitude. Must have the same units as `plev`.

        ptop (:class:`numpy.number`):
            A scalar specifying the top of the column. ptop should be <= min(plev).
            Must have the same units as `plev`.

        msg (:obj:`numpy.number`):
            A numpy scalar value that represent a missing value in fi.
            This argument allows a user to use a missing value scheme
            other than NaN or masked arrays, similar to what NCL allows.

        meta (:obj:`bool`):
            Set to False to disable metadata; default is False.

    Returns:
        :class:`numpy.ndarray`: If psfc is a scalar the return variable will be a
        one-dimensional array the same size as `plev`; if `psfc` is two-dimensional
        [e.g. (lat,lon)] or three-dimensional [e.g. (time,lat,lon)] then the return
        array will have an additional level dimension: (lev,lat,lon) or (time,lev,lat,lon).
        The returned type will be double if `psfc` is double, float otherwise.

    Description:
        Calculates the layer pressure thickness of a constant pressure level system. It
        is analogous to `dpres_hybrid_ccm` for hybrid coordinates. At each grid point the
        sum of the pressure thicknesses equates to [psfc-ptop]. At each grid point, the
        returned values above `ptop` and below `psfc` will be set to the missing value of `psfc`.
        If there is no missing value for `psfc` then the missing value will be set to the default
        for float or double appropriately. If `ptop` or `psfc` is between plev levels
        then the layer thickness is modifed accordingly. If `psfc` is set to a missing value, all
        layer thicknesses are set to the appropriate missing value.

        The primary purpose of this function is to return layer thicknesses to be used to
        weight observations for integrations.

    Examples:

        Example 1: Using dpres_plevel with :class:`xarray.DataArray` input

        .. code-block:: python

            import numpy as np
            import xarray as xr
            import geocat.comp

            # Open a netCDF data file using xarray default engine and load the data stream
            ds = xr.open_dataset("./SOME_NETCDF_FILE.nc")

            # [INPUT] Grid & data info on the source
            psfc = ds.PS
            plev = ds.LEV
            ptop = 0.0

            # Call the function
            result_dp = geocat.comp.dpres_plevel(plev, psfc, ptop)
    """

    if isinstance(psfc, np.ndarray):
        if psfc.ndim > 3:
            raise DimensionError("ERROR dpres_plevel: The 'psfc' array must be a scalar or be a 2 or 3 dimensional array with right most dimensions lat x lon !")
    if plev.ndim != 1:
        raise DimensionError("ERROR dpres_plevel: The 'plev' array must be 1 dimensional array !")
    if isinstance(ptop, np.ndarray):
        raise DimensionError("ERROR dpres_plevel: The 'ptop' value must be a scalar !")
    if isinstance(plev, xr.DataArray) and isinstance(psfc, xr.DataArray):
        if plev.attrs["units"] != psfc.attrs["units"]:
            raise AttributeError("ERROR dpres_plevel: Units of 'plev' and 'psfc' needs to match !")

    if isinstance(plev, xr.DataArray):
        plev = plev.values

    if isinstance(psfc, xr.DataArray):
        psfc = psfc.values
    elif np.size(psfc)==1: # if it is a scalar, then construct a ndarray
        psfc = np.asarray(psfc)
        psfc = np.ndarray([1], buffer=psfc, dtype=psfc.dtype)

    if ptop is None:
        ptop = min(plev)
    else:
        if ptop > min(plev):
            raise ValueError("ERROR dpres_plevel: The 'ptop' value must be <= min(plev) !")

    # call the ncomp 'dpres_plevel' function
    result_dp = _ncomp._dpres_plevel(plev, psfc, ptop, msg)

    if meta and isinstance(input, xr.DataArray):
        raise MetaError("ERROR dpres_plevel: retention of metadata is not yet supported !")

        pass     # TODO: Retaining possible metadata might be revised in the future
    else:
        result_dp = xr.DataArray(result_dp)

    return result_dp

def rcm2points(lat2d, lon2d, fi, lat1dPoints, lon1dPoints, opt=0, msg=None, meta=False):
    """Interpolates data on a curvilinear grid (i.e. RCM, WRF, NARR) to an unstructured grid.

    Args:

	lat2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the latitudes locations
	    of fi. The latitude order must be south-to-north.

	lon2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the longitude locations
	    of fi. The latitude order must be west-to-east.

	fi (:class:`numpy.ndarray`):
	    A multi-dimensional array to be interpolated. The rightmost two
	    dimensions (latitude, longitude) are the dimensions to be interpolated.

	lat1dPoints (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the latitude coordinates of
	    the output locations.

	lon1dPoints (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the longitude coordinates of
	    the output locations.

	opt (:obj:`numpy.number`):
	    opt=0 or 1 means use an inverse distance weight interpolation.
	    opt=2 means use a bilinear interpolation.

	msg (:obj:`numpy.number`):
	    A numpy scalar value that represent a missing value in fi.
	    This argument allows a user to use a missing value scheme
	    other than NaN or masked arrays, similar to what NCL allows.

	meta (:obj:`bool`):
	    Set to False to disable metadata; default is False.

    Returns:
	:class:`numpy.ndarray`: The interpolated grid. A multi-dimensional array
	of the same size as fi except that the rightmost dimension sizes have been
	replaced by the number of coordinate pairs (lat1dPoints, lon1dPoints).
	Double if fi is double, otherwise float.

    Description:
	Interpolates data on a curvilinear grid, such as those used by the RCM (Regional Climate Model),
	WRF (Weather Research and Forecasting) and NARR (North American Regional Reanalysis)
	models/datasets to an unstructured grid. All of these have latitudes that are oriented south-to-north.

	A inverse distance squared algorithm is used to perform the interpolation.

	Missing values are allowed and no extrapolation is performed.

    Examples:

	Example 1: Using rcm2points with :class:`xarray.DataArray` input

	.. code-block:: python

	    import numpy as np
	    import xarray as xr
	    import geocat.comp

	    # Open a netCDF data file using xarray default engine and load the data stream
	    ds = xr.open_dataset("./ruc.nc")

	    # [INPUT] Grid & data info on the source curvilinear
	    ht_curv=ds.DIST_236_CBL[:]
	    lat2D_curv=ds.gridlat_236[:]
	    lon2D_curv=ds.gridlon_236[:]

	    # [OUTPUT] Grid on destination points grid (or read the 1D lat and lon from
	    #	       an other .nc file.
	    newlat1D_points=np.linspace(lat2D_curv.min(), lat2D_curv.max(), 100)
	    newlon1D_points=np.linspace(lon2D_curv.min(), lon2D_curv.max(), 100)

	    ht_points = geocat.comp.rcm2points(lat2D_curv, lon2D_curv, ht_curv, newlat1D_points, newlon1D_points)
    """

    # Basic sanity checks
    if lat2d.shape[0] != lon2d.shape[0] or lat2d.shape[1] != lon2d.shape[1]:
        raise DimensionError("ERROR rcm2points: The input lat/lon grids must be the same size !")

    if lat1dPoints.shape[0] != lon1dPoints.shape[0]:
        raise DimensionError("ERROR rcm2points: The output lat/lon grids must be same size !")

    if lat2d.shape[0] < 2 or lon2d.shape[0] < 2 or lat2d.shape[1] < 2 or lon2d.shape[1] < 2:
        raise DimensionError("ERROR rcm2points: The input/output lat/lon grids must have at least 2 elements !")

    if fi.ndim < 2:
        raise DimensionError("ERROR rcm2points: fi must be at least two dimensions !\n")

    if fi.shape[fi.ndim - 2] != lat2d.shape[0] or fi.shape[fi.ndim - 1] != lon2d.shape[1]:
        raise DimensionError("ERROR rcm2points: The rightmost dimensions of fi must be (nlat2d x nlon2d),"
			     "where nlat2d and nlon2d are the size of the lat2d/lon2d arrays !")

    if isinstance(lat2d, xr.DataArray):
        lat2d = lat2d.values
    if isinstance(lon2d, xr.DataArray):
        lon2d = lon2d.values
    if not isinstance(fi, xr.DataArray):
        fi = xr.DataArray(fi)

    # ensure lat1d and lon1d are numpy.ndarrays
    if isinstance(lat1dPoints, xr.DataArray):
        lat1dPoints = lat1dPoints.values
    if isinstance(lon1dPoints, xr.DataArray):
        lon1dPoints = lon1dPoints.values

    fi_data = fi.values

    if isinstance(fi_data, np.ndarray):
        fo = _ncomp._rcm2points(lat2d, lon2d, fi_data, lat1dPoints, lon1dPoints, opt, msg)
    else:
        raise TypeError

    if meta and isinstance(input, xr.DataArray):
        raise MetaError("ERROR rcm2points: retention of metadata is not yet supported !")
    else:
        fo = xr.DataArray(fo)

    return fo


def linint2_points(fi, xo, yo, icycx, msg=None, meta=False, xi=None, yi=None):
    """Interpolates from a rectilinear grid to an unstructured grid or locations using bilinear interpolation.

    Args:

        fi (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            An array of two or more dimensions. The two rightmost
            dimensions (nyi x nxi) are the dimensions to be used in
            the interpolation. If missing values are present, the
            value of `msg` must be set appropriately.

        xo (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            A One-dimensional array that specifies the X (longitude)
            coordinates of the unstructured grid.

        yo (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            A One-dimensional array that specifies the Y (latitude)
            coordinates of the unstructured grid. It must be the same
            length as `xo`.

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
            Set to True for metadata; default is False.

        xi (:class:`numpy.ndarray`):
            A strictly monotonically increasing array that specifies
            the X [longitude] coordinates of the `fi` array.

        yi (:class:`numpy.ndarray`):
            A strictly monotonically increasing array that specifies
            the Y [latitude] coordinates of the `fi` array.

    Returns:
	:class:`numpy.ndarray`: The returned value will have the same
        dimensions as `fi`, except for the rightmost dimension which will
        have the same dimension size as the length of `yo` and `xo`. The
        return type will be double if fi is double, and float otherwise.

    Description:
        The inint2_points uses bilinear interpolation to interpolate from
        a rectilinear grid to an unstructured grid.

        If missing values are present, then linint2_points will perform the
        piecewise linear interpolation at all points possible, but will return
        missing values at coordinates which could not be used. If one or more
        of the four closest grid points to a particular (xo,yo) coordinate
        pair are missing, then the return value for this coordinate pair will
        be missing.

        If the user inadvertently specifies output coordinates (xo,yo) that
        are outside those of the input coordinates (xi,yi), the output value
        at this coordinate pair will be set to missing as no extrapolation
        is performed.

        linint2_points is different from linint2 in that `xo` and `yo` are
        coordinate pairs, and need not be monotonically increasing. It is
        also different in the dimensioning of the return array.

        This function could be used if the user wanted to interpolate gridded
        data to, say, the location of rawinsonde sites or buoy/xbt locations.

        Warning: if xi contains longitudes, then the xo values must be in the
        same range. In addition, if the xi values span 0 to 360, then the xo
        values must also be specified in this range (i.e. -180 to 180 will not work).

    Examples:

        Example 1: Using linint2_points with :class:`xarray.DataArray` input

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

            fo = geocat.comp.linint2_points(fi, xo, yo, 0)

    """

    # Basic sanity checks
    if not isinstance(fi, xr.DataArray):
        fi = xr.DataArray(fi)
        if xi is None or yi is None:
            raise CoordinateError("linint2_points: arguments xi and yi must be passed"
                                  " explicitly if fi is not an xarray.DataArray !")
    if xo.shape[0] != yo.shape[0]:
        raise DimensionError("ERROR linint2_points: The xo and yo must be the same size !")

    if fi.ndim < 2:
        raise DimensionError("ERROR linint2_points: fi must be at least two dimensions !\n")

    if xi is None:
        xi = fi.coords[fi.dims[-1]].values
    elif isinstance(xi, xr.DataArray):
        xi = xi.values

    if yi is None:
        yi = fi.coords[fi.dims[-2]].values
    elif isinstance(yi, xr.DataArray):
        yi = yi.values

    if isinstance(xo, xr.DataArray):
        xo = xo.values
    if isinstance(yo, xr.DataArray):
        yo = yo.values

    fi_data = fi.data

    if isinstance(fi_data, np.ndarray):
        fo = _ncomp._linint2_points(xi, yi, fi_data, xo, yo, icycx, msg)
    else:
        raise TypeError

    if meta and isinstance(input, xr.DataArray):
        raise MetaError("ERROR linint2_points: retention of metadata is not yet supported !")
    else:
        fo = xr.DataArray(fo)

    # OERO: Above two if-blocks should be changed with two if-blocks similar to the following (would require corrections
    # though) when parallelization for differently-shaped input (fi) and output (fo) arrays in this case is resolved:

    # if isinstance(fi_data, da.Array):
    #     chunks = list(fi.chunks)
    #
    #     # ensure rightmost dimensions of input are not chunked
    #     if chunks[-2:] != [yi.shape, xi.shape]:
    #         raise ChunkError("linint2_points: the two rightmost dimensions of fi must not be chunked.")
    #
    #     # Ensure rightmost dimensions of output are not chunked
    #     # chunks[-2:] = [yo.shape, xo.shape]
    #
    #     # map_blocks maps each chunk of fi_data to a separate invocation of _ncomp._linint2_points. The "chunks"
    #     # keyword argument should be the chunked dimensionality of the expected output; the number of chunks should
    #     # match that of fi_data. Additionally, "drop_axis" and "new_axis" in this case indicate that the two rightmost
    #     # dimensions of the input will be dropped from the output array, and that two new axes will be added instead.
    #     fo = map_blocks(_ncomp._linint2_points, xi, yi, fi_data, xo, yo, icycx, msg,
    #                     chunks=chunks, dtype=fi.dtype,
    #                     drop_axis=[fi.ndim-2, fi.ndim-1],
    #                     new_axis=[fi.ndim-2, fi.ndim-1])
    #
    # elif isinstance(fi_data, np.ndarray):
    #     fo = _ncomp._linint2_points(xi, yi, fi_data, xo, yo, icycx, msg)
    #
    # else:
    #     raise TypeError
    #
    # if meta:
    #     coords = {k:v if k not in fi.dims[-2:]
    #               else (xo if k == fi.dims[-1] else yo)
    #               for (k, v) in fi.coords.items()}
    #
    #     fo = xr.DataArray(fo, attrs=fi.attrs, dims=fi.dims,
    #                           coords=coords)
    # else:
    #     fo = xr.DataArray(fo)

    return fo
