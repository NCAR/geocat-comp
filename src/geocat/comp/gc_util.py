import typing


def _generate_wrapper_docstring(wrapper_fcn: typing.Callable,
                                base_fcn: typing.Callable) -> None:
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
    wrapper_docstring = f".. attention:: This method is a wrapper for " \
                        f":func:`{base_fcn.__name__} <{base_fcn.__module__}.{base_fcn.__name__}>`" \
                        f"\n\n    {base_fcn.__doc__}"

    # assign docstring to wrapper function
    setattr(wrapper_fcn, '__doc__', wrapper_docstring)
