class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class AttributeError(Error):
    """Exception raised when the arguments of GeoCAT-comp functions argument
    has a mismatch of attributes with other arguments."""
    pass


class ChunkError(Error):
    """Exception raised when a Dask array is chunked in a way that is
    incompatible with an f2py function."""
    pass


class CoordinateError(Error):
    """Exception raised when a GeoCAT-comp function is passed a NumPy array as
    an argument without a required coordinate array being passed separately."""
    pass


class DimensionError(Error):
    """Exception raised when the arguments of GeoCAT-comp functions argument
    has a mismatch of the necessary dimensionality."""
    pass


class MetaError(Error):
    """Exception raised when the support for the retention of metadata is not
    supported."""
    pass
