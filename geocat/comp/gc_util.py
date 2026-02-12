import typing
import warnings


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


def _find_var(
    ds,
    standard_name=None,
    long_name=None,
    possible_names=None,
    units=None,
    description='variable',
):
    """
    Find a variable using CF-compliant checks.

    Searches in priority order:
    1. CF standard_name attribute match (if standard_names provided)
    2. Name attribute match (if long_names provided)
    3. Direct variable name match (if possible_names provided)
    4. Units match (if units provided)

    Parameters
    ----------
    ds : xr.Dataset or ux.UxDataset
        The dataset to search
    standard_name : str, optional
        CF standard_name to check in attrs
    long_name : str, optional
        long_name to check in attrs
    possible_names : list of str, optional
        List of possible variable names to check first
    units : str, optional
        Possible units to check in attrs
    description : str, optional
        String

    Returns
    -------
    str
        The name of the found variable

    Raises
    ------
    KeyError
        If no matching variable is found
    """
    error_parts = []

    # First try CF standard_name attribute match
    if standard_name:
        for var_name in ds.data_vars:
            var_attrs = ds[var_name].attrs
            if var_name == standard_name or standard_name in var_attrs.get(
                'standard_name', ''
            ):
                return var_name
        error_parts.append(f"Tried standard_name: {standard_name}. ")

    # Then try long_name attribute match
    if long_name:
        for var_name in ds.data_vars:
            var_attrs = ds[var_name].attrs
            if long_name in var_attrs.get('long_name', ''):
                return var_name
        error_parts.append(f"Tried long_name: {long_name}. ")

    # Then try direct name match
    if possible_names:
        for name in possible_names:
            if name in ds:
                return name
        error_parts.append(f"Tried names: {possible_names}. ")

    # Finally try units match (less reliable)
    if units:
        for var_name in ds.data_vars:
            var_attrs = ds[var_name].attrs
            if 'units' in var_attrs:
                if units in var_attrs.get('units', ''):
                    warnings.warn(
                        f"Found {description} '{var_name}' using units attribute only. "
                        f"This is unreliable - multiple variables may share the same units. "
                        f"Please verify this is correct and add CF standard_name. {error_parts}",
                        UserWarning,
                        stacklevel=3,
                    )
                    return var_name
        error_parts.append(f"Tried units: {units}. ")

    raise KeyError(f"Could not find {description} in dataset. {' '.join(error_parts)}")


def _find_optional_var(
    ds, standard_name=None, long_name=None, possible_names=None, units=None
):
    """
    Find an optional variable using CF-compliant checks.

    Parameters
    ----------
    ds : xr.Dataset or ux.UxDataset
        The dataset to search
    standard_name : str, optional
        CF standard_name to check in attrs
    long_name : str, optional
        Long_name to check in attrs
    possible_names : list of str, optional
        List of possible variable names
    units : str, optional
        Possible units to check in attrs

    Returns
    -------
    str or None
        The name of the found coordinate, or None if not found
    """
    try:
        return _find_var(ds, standard_name, long_name, possible_names, units)
    except KeyError:
        return None
