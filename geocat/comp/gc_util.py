import typing


def _generate_wrapper_docstring(
    wrapper_fcn: typing.Callable, base_fcn: typing.Callable
) -> None:
    """Generate the docstring for a wrapper function in the form of: 'This
    method is a wrapper for base_fcn', with a generated link to the base
    function.

    Parameters
    ----------
    wrapper_fcn : function
        The wrapper function to generate and assign a docstring

    base_fcn : function
        The wrapped function that the wrapper function's docstring is based
        on and links to
    """

    # create wrapper docstring
    wrapper_docstring = (
        f".. attention:: This method is a wrapper for "
        f":func:`{base_fcn.__name__} <{base_fcn.__module__}.{base_fcn.__name__}>`"
        f"\n\n    {base_fcn.__doc__}"
    )

    # assign docstring to wrapper function
    setattr(wrapper_fcn, '__doc__', wrapper_docstring)


def _find_coord(ds, possible_names, description="coordinate"):
    """
    Find a coordinate/variable in uxds by checking multiple possible names.

    Parameters
    ----------
    ds : Xr.Dataset
        The dataset to search
    possible_names : list of str
        List of possible names for the coordinate, in priority order
    description : str, optional
        Description of the coordinate for error messages

    Returns
    -------
    str
        The name of the found coordinate

    Raises
    ------
    KeyError
        If none of the possible names are found
    """
    for name in possible_names:
        if name in ds:
            return name

    raise KeyError(
        f"Could not find {description}. Tried: {possible_names}. "
        f"Available variables: {list(ds.data_vars)}"
    )


def _find_optional_coord(ds, possible_names):
    """
    Find a coordinate that may or may not exist.

    Parameters
    ----------
    ds : Xr.Dataset
        The dataset to search
    possible_names : list of str
        List of possible names for the coordinate

    Returns
    -------
    str or None
        The name of the found coordinate, or None if not found
    """
    for name in possible_names:
        if name in ds:
            return name
    return None
