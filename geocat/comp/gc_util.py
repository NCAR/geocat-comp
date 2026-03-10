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
    description='variable',
):
    """
    Find a variable using CF-compliant checks.

    Searches in priority order:
    1. CF standard_name attribute match (if standard_names provided)
    2. Name attribute match (if long_names provided)
    3. Direct variable name match (if possible_names provided)

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
    description : str, optional
        String for descriptive Error message

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
        standard_name_variations = {
            standard_name,
            standard_name.lower(),
            standard_name.upper(),
            standard_name.capitalize(),
            '_'.join(word.capitalize() for word in standard_name.split('_')),
        }

        for var_name in ds.data_vars:
            var_attrs = ds[var_name].attrs
            attr_standard_name = var_attrs.get('standard_name', '')
            if (
                var_name in standard_name_variations
                or attr_standard_name in standard_name_variations
            ):
                return var_name

        error_parts.append(f"Tried standard_name: {standard_name}.")

    # Then try long_name attribute match
    if long_name:
        long_name_variations = {
            long_name,
            long_name.lower(),
            long_name.upper(),
            long_name.capitalize(),
            ' '.join(word.capitalize() for word in long_name.split(' ')),
            '_'.join(word.capitalize() for word in long_name.split('_')),
            long_name.replace(' ', '_'),
            long_name.replace('_', ' '),
        }

        for var_name in ds.data_vars:
            var_attrs = ds[var_name].attrs
            attr_long_name = var_attrs.get('long_name', '')
            if (
                var_name in long_name_variations
                or attr_long_name in long_name_variations
            ):
                return var_name

        error_parts.append(f"Tried long_name: {long_name}.")

    # Then try direct name match
    if possible_names:
        possible_names_expanded = set()
        for name in possible_names:
            possible_names_expanded.update(
                {
                    name,
                    name.lower(),
                    name.upper(),
                    name.capitalize(),
                    '_'.join(word.capitalize() for word in name.split('_')),
                }
            )

        for name in possible_names_expanded:
            if name in ds:
                return name
        error_parts.append(f"Tried names: {possible_names}.")

    raise KeyError(f"Could not find {description} in dataset. {' '.join(error_parts)}")


def _find_optional_var(
    ds,
    standard_name=None,
    long_name=None,
    possible_names=None,
    description=None,
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
    description: str
        String for descriptive Error message


    Returns
    -------
    str or None
        The name of the found coordinate, or None if not found
    """
    try:
        return _find_var(ds, standard_name, long_name, possible_names, description)
    except KeyError:
        return None
