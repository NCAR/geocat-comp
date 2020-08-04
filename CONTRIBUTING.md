Please first refer to [GeoCAT Contributor's Guide](https://geocat.ucar.edu/pages/contributing.html) for overall contribution guidelines (such as detailed description of GeoCAT structure, repository cloning, branching, etc.). Once you determine that a function should be contributed under this repo, please refer to the following contribution guidelines:

# Adding new functions to the repo

1. Create a new function as a Python file in `/src/geocat/comp/`.
2. For implementation guideline (such as XArray and Dask usage), please refer to:
- Previously implemented functionality as examples, e.g. [polynomial.py](https://github.com/NCAR/geocat-comp/blob/develop/src/geocat/comp/polynomial.py) or others.
- [GeoCAT Contributor's Guide](https://geocat.ucar.edu/pages/contributing.html) for further information.

# Adding unit tests

All new computational functionality needs to include unit testing. For that purpose, please refer to the following guideline:

1. Unit tests of the function should be implemented as a separate test file under the `/test` folder in the root directory.
2. The [pytest](https://docs.pytest.org/en/stable/contents.html) testing framework is used as a “runner” for the tests. For further information about pytest, see: [pytest documentation](https://docs.pytest.org/en/stable/contents.html).
    - Test scripts themselves are not intended to use pytest through implementation. Instead, pytest should be used only for running test scripts as follows:
    
        `pytest <test_script_name>.py` 

    - Not using pytest for implementation allows the unit tests to be also run by using: 

        `python -m unittest <test_script_name>.py`
        
3. Python’s unit testing framework [unittest](https://docs.python.org/3/library/unittest.html) is used for implementation of the test scripts. For further information about unittest, see: [unittest documentation](https://docs.python.org/3/library/unittest.html).
4. Recommended but not mandatory implementation approach is as follows:
    - Common data structures as well as variables and functions, which could be used by multiple test methods throughout the test script, are defined under a base test class.
    - Any group of testing functions dedicated to testing a particular phenomenon (e.g. a specific edge case, data structure, etc.) is implemented by a class, which inherits TestCase from Python’s unittest and likely the base test class implemented for the purpose mentioned above.
    - Assertions are used for testing various cases such as array comparison.
    - Please see previously implemented test cases for reference of the recommended testing approach, e.g. [test_polynomial.py](https://github.com/NCAR/geocat-comp/blob/develop/test/test_polynomial.py) 