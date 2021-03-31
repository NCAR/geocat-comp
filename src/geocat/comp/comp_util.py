import numpy as np


def _is_duck_array(value):
    """Returns True when ``value`` is array-like."""
    if isinstance(value, np.ndarray):
        return True
    return (hasattr(value, "ndim") and hasattr(value, "shape") and
            hasattr(value, "dtype") and hasattr(value, "__array_function__") and
            hasattr(value, "__array_ufunc__"))
