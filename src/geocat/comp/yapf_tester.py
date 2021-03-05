import numpy as np


def foo_with_very_long_function_name(arg1,
                                     opt1=1,
                                     opt2=2,
                                     opt3=3,
                                     opt4=None,
                                     opt5=None,
                                     opt6=None,
                                     opt7=None,
                                     opt8=None,
                                     opt9=None):
    pass


def foo(arg1, arg2=1):
    """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

    Parameters
    ----------
    arg1 : :class:`xarray.DataArray` or :class:`numpy.ndarray` or :class:`list`
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

    arg2:
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

    Returns
    -------
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

    """

    result1 = foo_with_very_long_function_name(arg1,
                                               opt1=1,
                                               opt2=2,
                                               opt3=3,
                                               opt4=None,
                                               opt5=None,
                                               opt6=None,
                                               opt7=None,
                                               opt8=None,
                                               opt9=None)

    # Checking arg2
    if arg2 <= 0:
        raise ValueError(
            "ERROR Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
        )

    if 1:
        dims = ["dim_0_too_long_dim_name_here"
               ] + [arg1.dims[i] for i in range(arg1.ndim) if i != 0]
        coords = {k: v for (k, v) in arg1.coords.items() if k != arg1.dims[0]}

    return foo_with_very_long_function_name(arg1,
                                            opt1=1,
                                            opt2=2,
                                            opt3=3,
                                            opt4=None,
                                            opt5=None,
                                            opt6=None,
                                            opt7=None,
                                            opt8=None,
                                            opt9=None)
