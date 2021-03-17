Please first refer to [GeoCAT Contributor's Guide](https://geocat.ucar.edu/pages/contributing.html) for overall
contribution guidelines (such as detailed description of GeoCAT structure, forking, repository cloning,
branching, etc.). Once you determine that a function should be contributed under this repo, please refer to the
following contribution guidelines:

# Adding new functions to the Geocat-comp repo

1. For a new function or family of functions that handle similar computations, create a new Python file in
`$GEOCATCOMP/src/geocat/comp/`.

2. For implementation guidelines (such as Xarray and Dask usage), please refer to:
   - Previously implemented functionality as examples,
    e.g. [polynomial.py](https://github.com/NCAR/geocat-comp/blob/main/src/geocat/comp/polynomial.py) or others.
   - [GeoCAT Contributor's Guide](https://geocat.ucar.edu/pages/contributing.html) for further information.

3. In any Python script under `$GEOCATCOMP/src/geocat/comp/`, there may be user API functions, which are
supposed to be included in the `geocat.comp` namespace, and internal API functions, which are used by the
user API functions as helpers, preferably starts with an underscore ("_") in their names, as well as are
not included in the `geocat.comp` namespace.

4. The user API functions should be imported in `$GEOCATCOMP/src/geocat/comp/__init__.py` to be included in
the namespace.

5. For appropriate documentation, each user API and internal API function should be listed in the
`$GEOCATCOMP/docs/user_api/index.rst` and `$GEOCATCOMP/docs/internal_api/index.rst`, respectively.

# Adding unit tests

All new computational functions need to include unit testing. For that purpose, please refer to the following
guideline:

1. Unit tests of each function (or function family of similar purposes) should be implemented as a separate
test file under the `$GEOCATCOMP/test` folder.

2. The [pytest](https://docs.pytest.org/en/stable/contents.html) testing framework is used as a “runner” for the tests.
For further information about `pytest`, see: [pytest documentation](https://docs.pytest.org/en/stable/contents.html).
    - Test scripts themselves are not intended to use `pytest` through implementation. Instead, `pytest` should be used
    only for running test scripts as follows:

        `pytest <test_script_name>.py`

    - Not using `pytest` for implementation allows the unit tests to be also run by using:

        `python -m unittest <test_script_name>.py`

3. Python’s unit testing framework [unittest](https://docs.python.org/3/library/unittest.html) is used for
implementation of the test scripts. For further information about `unittest`,
see: [unittest documentation](https://docs.python.org/3/library/unittest.html).

4. Recommended but not mandatory implementation approach is as follows:

    - Common data structures, variables and functions,  as well as
    expected outputs, which could be used by multiple test methods throughout
    the test script, are defined either under a base test class or in the very
    beginning of the test script for being used by multiple unit test cases.

    - Only applies to functions that are replicated from NCL: For the sake
    of having reference results (i.e. expected output or ground truth for not
    all but the most cases), an NCL test script can be written under
    `\test\ncl_tests` folder and its output can be used for each testing
    scenario.

    - Any group of testing functions dedicated to testing a particular
    phenomenon (e.g. a specific edge case, data structure, etc.) is
    implemented by a class, which inherits `TestCase` from Python's
    `unittest` and likely the base test class implemented for the purpose
    mentioned above.

    - Assertions are used for testing various cases such as array comparison.

    - Please see previously implemented test cases for reference of the
    recommended testing approach,
    e.g. [test_polynomial.py](https://github.com/NCAR/geocat-comp/blob/main/test/test_polynomial.py)
