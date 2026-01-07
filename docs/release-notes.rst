.. currentmodule:: geocat.comp

.. _release:

..
    SEE TEMPLATE AT BOTTOM OF FILE WHEN STARTING A NEW RELEASE SECTION

Release Notes
=============

vYYYY.MM.## (unreleased)
------------------------
This release...

New Features
^^^^^^^^^^^^
* Update ``calendar_average`` and ``climate_anomaly`` to allow for monthly data with non-uniform spacing by `Katelyn FitzGerald`_ in (:pr:`805`)

v2025.12.1 (December 17, 2025)
------------------------------
This release fixes a bug in ``nmse`` and adds a new function, ``delta_pressure_hybrid``.

New Features
^^^^^^^^^^^^
* Adds ``delta_pressure_hybrid`` function by `Anissa Zacharias`_ in (:pr:`797`)
* Add a ``pressure_top`` argument to ``delta_pressure`` by Anissa Zacharias`_ in (:pr:`798`)

Bug Fixes
^^^^^^^^^
* Fix ``nmse`` coordinate matching issue `Anissa Zacharias`_ in (:pr:`796`)

Documentation
^^^^^^^^^^^^^
* Add ``delta_pressure`` calculation reference by `Anissa Zacharias`_ in (:pr:`800`)

v2025.12.0 (December 2, 2025)
-----------------------------
This release adds a normalized mean squared error metric function.

New Features
^^^^^^^^^^^^
* Add normalized mean square error (NMSE) metric function, ``nmse`` by `Anissa Zacharias`_ in (:pr:`787`)

v2025.11.0 (November 24, 2025)
------------------------------
This release adds an initial benchmarking suite, exposes a new ``pressure_at_hybrid_levels`` function,
and addresses several attribute related issues following upstream changes in Xarray.

Bug Fixes
^^^^^^^^^
* Update ``climate_anomaly`` to ensure consistent behavior for the ``keep_attrs`` option by `Katelyn FitzGerald`_ in (:pr:`783`)
* Revert back to passing only `pressure.data` rather than the full DataArray to avoid unit handling issues in ``interp_hybrid_to_pressure`` by `Katelyn FitzGerald`_ in (:pr:`781`)

New Features
^^^^^^^^^^^^
* Move ``_pressure_from_hybrid`` to public API as ``pressure_at_hybrid_levels`` by `Anissa Zacharias`_ in (:pr:`776`)


Developer Features
^^^^^^^^^^^^^^^^^^
* Create initial benchmarking suite by `Anissa Zacharias`_ in (:pr:`772`)

Documentation
^^^^^^^^^^^^^
* Add latitude/longitude axis labels in ``vimfc`` example plot by `Cora Schneck`_ in (:pr:`784`)


v2025.10.01 (October 7, 2025)
-----------------------------
This release updates to dependabot and optional PR ASV runs

Internal Changes
^^^^^^^^^^^^^^^^
* Group Dependabot updates by `Katelyn FitzGerald`_ in (:pr:`764`)

v2025.09.0 (September 9, 2025)
------------------------------
This release removes dask as a mandatory dependency and reworks some internal dask usage.

Breaking Changes
^^^^^^^^^^^^^^^^
* ``scale_voronoi`` no longer accepts ``chunk_size`` as an optional argument and will not automatically chunk xarray inputs from `Anissa Zacharias`_ in (:pr:`749`)

Internal Changes
^^^^^^^^^^^^^^^^
* Update ``interp_hybrid_to_pressure`` to remove forced dask usage by `Anissa Zacharias`_ in (:pr:`748`)
* Remove dask as an explicit dependency and forced dask internal usage by `Anissa Zacharias`_ in (:pr:`749`)

Developer Features
^^^^^^^^^^^^^^^^^^
* Add option to enable asv comparison runs on PR branches by `Anissa Zacharias`_ in (:pr:`753`)

Documentation
^^^^^^^^^^^^^
* Update links to cartopy docs by `Katelyn FitzGerald`_ in (:pr:`757`)

v2025.07.0 (July 15, 2025)
---------------------------
This release updates packaging, replaces deprecated xarray functionality, and
addresses a bug in ``interp_hybrid_to_pressure``.

Bug Fixes
^^^^^^^^^
* Change ``interp_hybrid_to_pressure`` to use ``t_bot`` directly for temperature extrapolation rather than the temperature from the (sometimes incorrectly) presumed lowest level by `Katelyn FitzGerald`_ in (:pr:`737`)

Internal Changes
^^^^^^^^^^^^^^^^
* Replace deprecated ``xarray.cftime_range`` with ``xarray.date_range`` by `Katelyn FitzGerald`_ in (:pr:`739`)

Maintenance
^^^^^^^^^^^
* Streamline packaging and dependencies by `Katelyn FitzGerald`_ in (:pr:`740`)


v2025.05.0 (May 20, 2025)
------------------------
This release includes documentation improvements, fixes a delta_pressure
thickness bug, and adds improved CFTime and Datetime object compatibility to
climatology functionality.

Documentation
^^^^^^^^^^^^^
* Remove reference to NCAR/geocat repo from support page by `Katelyn FitzGerald`_ in (:pr:`709`)
* Add additional relative humidity documentation by `Cora Schneck`_ in (:pr:`710`)
* Clarify definition of ``delta_pressure`` by `Katelyn FitzGerald`_ in (:pr:`718`)
* Correct ``psychrometric_constant`` equation references in documentation by `Cora Schneck`_ in (:pr:`723`)
* Add note about alternate hybrid-sigma pressure formulation to ``interp_hybrid_to_pressure`` docs by `Katelyn FitzGerald`_ in (:pr:`727`)

Bug Fixes
^^^^^^^^^
* Update near surface pressure thickness calculations in ``delta_pressure`` for consistency with NCL by `Katelyn FitzGerald`_ in (:pr:`726`)
* Updates internal climatology function to ensure compatibility with both CFTime and Datetime indices by `Katelyn FitzGerald`_ in (:pr:`717`)

Internal Changes
^^^^^^^^^^^^^^^^
* Updates GitHub Actions workflows per new guidance by `Cora Schneck`_ in (:pr:`716`)
* Adds `blackdoc` to linter for rst and markdown by `Cora Schneck`_ in (:pr:`720`)
* Unpins ``asv`` and adds Python 3.13 to benchmarking matrix by `Anissa Zacharias`_ in (:pr:`722`)
* Updates to GitHub Actions workflows by `Katelyn FitzGerald`_ in (:pr:`733`)

v2025.03.0 (March 25, 2025)
---------------------------
This release unpins scipy, establishes minimum version testing, and switches
from ``yapf`` to ``ruff`` formatting.

Enhancements
^^^^^^^^^^^^
* Add minimum dependency version testing and address minor compatibility issues with Pandas and Xarray by `Katelyn FitzGerald`_ in (:pr:`699`)

Internal Changes
^^^^^^^^^^^^^^^^
* Reconfigure analytics by `Katelyn FitzGerald`_ in (:pr:`698`)
* Remove ``docformatter`` and setup ``ruff`` by `Cora Schneck`_ in (:pr:`700`)

Bug Fixes
^^^^^^^^^
* Unpin scipy and update ``sph_harm`` to ``sph_harm_y`` by `Anissa Zacharias`_ in (:pr:`695`)

Breaking Changes
^^^^^^^^^^^^^^^^
* The ``chunk_size`` parameter was removed from :func:`.decomposition` and :func:`.recomposition` in (:pr:`695`)

v2025.02.0 (February 25, 2025)
------------------------------
This release adds support and testing for Python 3.13, unpins numpy dependency, increases support for
`non-nanosecond datetime objects <https://github.com/NCAR/geocat-comp/issues/682>`__,
and updates our `code of conduct <https://github.com/NCAR/geocat-comp/commit/48b16cc0143ce63ebc4ce2735d0d1afc5f7bee5f>`__.

Enhancements
^^^^^^^^^^^^
* Add tests and support for non-nanosecond datetime objects by `Katelyn FitzGerald`_ in (:pr:`691`)

Maintenance
^^^^^^^^^^^
* Add support and testing for Python 3.13 by `Katelyn FitzGerald`_ in (:pr:`688`)
* Remove NumPy version pin by `Katelyn FitzGerald`_ in (:pr:`686`)

Documentation
^^^^^^^^^^^^^
* Updates Code of Conduct by `Orhan Eroglu`_ in `48b16cc <https://github.com/NCAR/geocat-comp/commit/48b16cc0143ce63ebc4ce2735d0d1afc5f7bee5f>`__

v2025.01.0 (January 28, 2025)
-----------------------------
v2025.01.0 releases a collection of small changes and pins ``scipy`` to <1.15
while we finalize removing the deprecating ``sph_harm`` in favor of
``sph_harm_y``.

Documentation
^^^^^^^^^^^^^
* Fix to documentation theme configuration by `Katelyn FitzGerald`_ in (:pr:`657`)
* Update links to eofs documentation by `Katelyn FitzGerald`_ in (:pr:`661`)
* Update citation documentation removing reference to old GeoCAT website by `Katelyn FitzGerald`_ in (:pr:`666`)
* Fix link to ``satvpr_slope_fao56`` in ``saturation_vapor_pressure_slope`` documentation by `Cora Schneck`_ in (:pr:`671`)

Internal Changes
^^^^^^^^^^^^^^^^
* Pin SciPy version <1.15 by `Katelyn FitzGerald`_ in (:pr:`674`)
* Temporarily remove docformatter from pre-commit by `Katelyn FitzGerald`_ in (:pr:`653`)
* Pre-commit hook to check for valid yml by `Cora Schneck`_ in (:pr:`625`)
* CI to run on macos-latest instead of macos-14 by `Cora Schneck`_ in (:pr:`617`)
* Update and setup autoupdate for pre-commit hooks by `Cora Schneck`_ in (:pr:`604`)
* Removed ``pre-commit.yaml`` action in favor of pre-commit.ci by `Cora Schneck`_ in (:pr:`608`)
* Remove ASV version pin and pin Conda version for benchmarking workflow by `Katelyn FitzGerald`_ in (:pr:`610`)
* Updates to issue and PR templates by `Anissa Zacharias`_ in (:pr:`612`)
* Re-pin ASV and list env info by `Katelyn FitzGerald`_ in (:pr:`613`)
* Refactor ``pre-commit.ci`` by `Anissa Zacharias`_ in (:pr:`628`)
* Temporarily pin micromamba for CI by `Anissa Zacharias`_ in (:pr:`645`)
* Remove micromamba pin for CI by `Katelyn FitzGerald`_ in (:pr:`650`)

v2024.04.0 (April 23, 2024)
---------------------------
This release drops compatibility for Python 3.9 and improves performance for ``interp_hybrid_to_pressure``.

Breaking Changes
^^^^^^^^^^^^^^^^
* Drop Python 3.9 Support by `Cora Schneck`_ in (:pr:`599`)

Bug Fixes
^^^^^^^^^
* Fix to address slow execution times for ``interp_hybrid_to_pressure`` with extrapolation by `Katelyn FitzGerald`_ in (:pr:`592`)
* Pin ``numpy<2.0`` for occasional PyPI install issues by `Anissa Zacharias`_ in (:pr:`600`)

Internal Changes
^^^^^^^^^^^^^^^^
* Update build in pypi workflow and documentation links to reduce warnings in testing and docs by `Cora Schneck`_ in (:pr:`582`)


v2024.03.0 (March 29, 2024)
---------------------------
This release includes a bug fix for ``delta_pressure``.

Internal Changes
^^^^^^^^^^^^^^^^
* Additional pre-commit hook for codespell by `Cora Schneck`_ in (:pr:`579`)
* Remove unused imports, unused variables, code cleanup `Cora Schneck`_ in (:pr:`584`)
* Add M1 runners to CI by `Katelyn FitzGerald`_ in (:pr:`581`)
* Reorganize dask compatibility tests by `Anissa Zacharias`_ in (:pr:`568`)

Bug Fixes
^^^^^^^^^
* Fix ``delta_pressure`` to handle the case where pressure level(s) are greater than surface pressure by `Katelyn FitzGerald`_ in (:pr:`571`)


v2024.02.0 (February 28, 2024)
------------------------------
This release switches the package to use an implicit namespace and adds support
for Python 3.12.

Documentation
^^^^^^^^^^^^^
* Override branding for light/dark theme by `Cora Schneck`_ in (:pr:`545`)
* Updated outdated documentation `Cora Schneck`_ in (:pr:`561`)

Internal Changes
^^^^^^^^^^^^^^^^
* Update hourly frequency aliases by `Katelyn FitzGerald`_ in (:pr:`550`)
* Python 3.12 Support by `Cora Schneck`_ in (:pr:`548`)
* Added ``linkcheck_ignore`` to ``docs/conf.py`` to address erroneous failures `Anissa Zacharias`_ in (:pr:`559`)
* Updated Codecov upload to use token by `Anissa Zacharias`_ in (:pr:`566`)
* Convert to implicit namespace packaging set up by `Anissa Zacharias`_ in (:pr:`563`)
* Temporarily pin ASV to ``<0.6.2`` by `Anissa Zacharias`_ in (:pr:`556` and :pr:`569`)


v2024.01.0 (January 30, 2023)
-----------------------------
This release primarily updates our internal CI.

Internal Changes
^^^^^^^^^^^^^^^^
* Upstream CI improvements by `Anissa Zacharias`_ in (:pr:`527`)
* CI improvements by `Anissa Zacharias`_ in (:pr:`528`)
* Switch to PyPI Trusted Publishing by `Anissa Zacharias`_ in (:pr:`534`)
* Add retry actions to CI by `Anissa Zacharias`_ in (:pr:`532`)
* Improves/standardizes release tests and adds PyPI release testing by `Anissa Zacharias`_ in (:pr:`531`)
* Updates upstream dev workflow to use `scientific-python-nightly-wheels <https://pypi.anaconda.org/scientific-python-nightly-wheels/simple/>`__
  by `Anissa Zacharias`_ and `Katelyn FitzGerald`_ in (:pr:`537`)

Documentation
^^^^^^^^^^^^^
* Update remaining assets to comply with NSF branding theme by `Cora Schneck`_ in (:pr:`541`)


v2023.12.0 (December 5, 2023)
-----------------------------
This release adds official windows support and unpins xarray and numpy.

Internal Changes
^^^^^^^^^^^^^^^^
* Remove unnecessary tag publish trigger for ASV benchmarking CI  by `Anissa Zacharias`_ in (:pr:`509`)
* Add windows to testing strategy by `AnshRoshan`_ in (:pr:`460`)

Bug Fixes
^^^^^^^^^
* Unpin xarray in environment builds with changes to interpolation.py (specify
  dims in xr.DataArray) and ``climatologies.py`` (replace ``loffset`` with
  ``to_offset``) by `Cora Schneck`_ in (:pr:`492`)
* Fixes for Windows tests when EOF throws different signs by `Cora Schneck`_ in
  (:pr:`516`)
* Fix ``extlinks`` for Sphinx 6 compatibility by `Anissa Zacharias`_ in
  (:pr:`520`)

Maintenance
^^^^^^^^^^^
* Remove no longer needed numpy version pin by `Katelyn FitzGerald`_ in (:pr:`515`)

Documentation
^^^^^^^^^^^^^
* Transferred fourier filter example from Geocat-examples by `Julia Kent`_ in (:pr:`511`)
* Updated documentation links by `Anissa Zacharias`_ in (:pr:`518`)
* Augment documentation for ``interp_multidim`` by `Katelyn FitzGerald`_ in (:pr:`504`)


v2023.10.1 (October 31, 2023)
-----------------------------
This release includes minor changes to documentation, a full conversion to
pytest from unittest, and is the first release to include automated
benchmarking.

Maintenance
^^^^^^^^^^^
* Convert Unittest to Pytest by `Cora Schneck`_ in (:pr:`478`)

Documentation
^^^^^^^^^^^^^
* Updated office hours link by `Anissa Zacharias`_ in (:pr:`495`)
* Added benchmark badge to README by `Anissa Zacharias`_ in (:pr:`497`)

Bug Fixes
^^^^^^^^^
* Fix Python version in upstream CI by `Philip Chmielowiec`_ in (:pr:`436`)

Internal Changes
^^^^^^^^^^^^^^^^
* Add benchmarking to commits to main and tagged releases by `Anissa Zacharias`_ in (:pr:`496`)
* Fix benchmarking workflow failures by `Anissa Zacharias`_ in (:pr:`499`)


v2023.10.0 (Oct 3, 2023)
------------------------
This release adds a code of conduct, minor edits to our contributor's guide, and
sets up some structure for future ASV benchmarking

Internal Changes
^^^^^^^^^^^^^^^^
* Sets up ASV for benchmarking by `Anissa Zacharias`_ in (:pr:`474`)

Documentation
^^^^^^^^^^^^^
* New Code of Conduct by `Cora Schneck`_ in (:pr:`461`)
* Updated Pull Request Template by `Cora Schneck`_ in (:pr:`468`)
* Fixes for Contributing Geocat-Comp Contributing by `Cora Schneck`_ in (:pr:`476`)

v2023.09.0 (Sept 8, 2023)
-------------------------
This release adds ``custom_seasons`` to ``climatology_average`` and adds a new
Contributor's Guide to the documentation.

New Features
^^^^^^^^^^^^
* User-defined seasonal boundaries, ``custom_seasons``, enabled for
  ``climatology_average`` by `Julia Kent`_ in (:pr:`411`)

Bug Fixes
^^^^^^^^^
* Fix codecov coverage reporting issue by `Anissa Zacharias`_ in (:pr:`446`)
* Fix xarray inconsistent pinning issue by `Anissa Zacharias`_ in (:pr:`458`)

Documentation
^^^^^^^^^^^^^
* New Contributor's Guide by `Anissa Zacharias`_ in (:pr:`450`)


v2023.06.1 (June 23, 2023)
--------------------------
This releases fixes the unintentional limitation of the 2023.06.0 release to python 3.11.0

Bug Fixes
^^^^^^^^^
* Fix python version limit of 3.11.0 by `Anissa Zacharias`_ in (:pr:`431`)


v2023.06.0 (June 23, 2023)
--------------------------
This release removes the geocat-f2py dependency. To use these functions, users
will need to install the geocat-f2py package directly. Additionally, this
release also drops support for python 3.8 and adds support for 3.11.

Documentation
^^^^^^^^^^^^^
* New *Vertically Integrated Moisture Flux Convergence* (VIMFC) example by `Julia Kent`_ in (:pr:`388`)

Internal Changes
^^^^^^^^^^^^^^^^
* Updates deprecated pre-commit YAPF repository from https://github.com/pre-commit/mirrors-yapf to https://github.com/google/yapf by `Anissa Zacharias`_ in (:pr:`417`)
* Reconfigures package structure to remove top level ``src/`` directory by `Anissa Zacharias`_ in (:pr:`419`)

Breaking Changes
^^^^^^^^^^^^^^^^
* Removed deprecated functions ``climatology`` and ``anomaly`` by `Anissa Zacharias`_ in (:pr:`416`)
* Removed internal functions ``_find_time_invariant_vars`` and ``_setup_clim_anom_input`` by `Anissa Zacharias`_ in (:pr:`416`)
* Dropped support for python 3.8 (and added support for python 3.11) by `Anissa Zacharias`_ in (:pr:`426`)
* Removed ``geocat-f2py`` dependency by `Anissa Zacharias`_ in (:pr:`421`)

Bug Fixes
^^^^^^^^^
* Fix bug in ``_temp_extrapolate`` used by ``interp_hybrid_to_pressure`` by `Katelyn FitzGerald`_ in (:pr:`424`)


v2023.05.0 (4 May 2023)
-----------------------
In this release, we've added support for numpy input and other improvements to the gradient function

Bug Fixes
^^^^^^^^^
* Support for numpy input types and lat/lon kwargs in gradient by `Julia Kent`_ and `Alea Kootz`_ in (:pr:`385`)

Documentation
^^^^^^^^^^^^^
* Update PR template to include manual addition to release notes by `Anissa Zacharias`_ in (:pr:`397`)


v2023.03.2 (Mar 29, 2023)
-------------------------

Bug Fixes
^^^^^^^^^
* type check patch in delta_pressure by `Julia Kent`_ in (:pr:`363`)

Maintenance
^^^^^^^^^^^
* Update internal links to use sphinx internal referencing by `Heather Craker`_ in (:pr:`376`)
* Switch pypi release action to be triggered manually by `Anissa Zacharias`_ in (:pr:`390`)
* Package setup refactor (namespace and versioning fixes) (x2) by `Anissa Zacharias`_ (:pr:`389`)


**Full Changelog**: https://github.com/NCAR/geocat-comp/compare/v2023.03.1...v2023.03.2

v2023.03.1 (Mar 23, 2023)
-------------------------

Bug Fixes
^^^^^^^^^
* Add dpres_plev init file by `Heather Craker`_ in (:pr:`368`)
* Fix argument checks for interp_hybrid_to_pressure by `Heather Craker`_ in (:pr:`372`)
* Extrap args fix by `Heather Craker`_ in (:pr:`369`)
* Revert "Extrap args fix" by `Heather Craker`_ in (:pr:`371`)

Documentation
^^^^^^^^^^^^^
* Create a utility function to generate docstrings on wrapper functions by `Anissa Zacharias`_ in (:pr:`362`)
* adjust calendar example to follow similar template by `Julia Kent`_ in (:pr:`339`)
* Update release notes for v2023.03.0 by `Heather Craker`_ in (:pr:`365`)

Misc
^^^^
* version bump 2023 03 1 by `Alea Kootz`_ in (:pr:`379`)

**Full Changelog**: https://github.com/NCAR/geocat-comp/compare/v2023.03.0...v2023.03.1

v2023.03.0 (Mar 2, 2023)
------------------------

New Features
^^^^^^^^^^^^
* Adding method to calculate delta pressure by `Julia Kent`_ in (:pr:`338`)

Deprecations
^^^^^^^^^^^^
* Remove deprecated functions from v2022.10.0 by `Heather Craker`_ in (:pr:`357`)
* Remove links to deleted function docs by `Heather Craker`_ in (:pr:`359`)

Bug Fixes
^^^^^^^^^
* Partial fix for _vertical_remap_extrap bug by `Heather Craker`_ in (:pr:`360`)

Documentation
^^^^^^^^^^^^^
* Fix typos in PR template by `Heather Craker`_ in (:pr:`353`)
* add climatology_average example by `Julia Kent`_ in (:pr:`341`)
* Fix some delta_pressure docs formatting by `Heather Craker`_ in (:pr:`361`)

Maintenance
^^^^^^^^^^^
* Add PR template to repository by `Heather Craker`_ in (:pr:`344`)
* Fix typos by `Heather Craker`_ in (:pr:`347`)
* Change conda badge to conda-forge channel by `Heather Craker`_ in (:pr:`349`)

New Contributors
^^^^^^^^^^^^^^^^
* `Julia Kent`_ made their first contribution in (:pr:`341`)

**Full Changelog**: https://github.com/NCAR/geocat-comp/compare/v2023.02.0...v2023.03.0

v2023.02.0 (Feb 2, 2023)
------------------------

New Features
^^^^^^^^^^^^
* Add extrapolation feature to interp_hybrid_to_pressure by `Heather Craker`_ in (:pr:`237`)
* Add ``climate_anomaly`` by `Heather Craker`_ in (:pr:`332`)

Enhancements
^^^^^^^^^^^^
* Add check to avoid calculating monthly averages on monthly data by `Heather Craker`_ in (:pr:`304`)
* Add ``keep_attrs`` to climatology functions by `Heather Craker`_ in (:pr:`315`)

Deprecations
^^^^^^^^^^^^
* Fix pint bug in showalter_index and deprecate by `Heather Craker`_ in (:pr:`319`)
* Replace and deprecate ``anomaly`` and ``climatology`` by `Heather Craker`_ in (:pr:`332`)

Documentation
^^^^^^^^^^^^^
* Add rendered equations to docstrings by `Anissa Zacharias`_ in (:pr:`294`)
* Fix badge links by `Anissa Zacharias`_ in (:pr:`321`)

Maintenance
^^^^^^^^^^^
* Adding import nightly CI test by `Anissa Zacharias`_ in (:pr:`300`)
* Expand upstream CI by `Anissa Zacharias`_ in (:pr:`301`)
* Patch for import test by `Anissa Zacharias`_ in (:pr:`302`)
* Fix the import package CI test by `Anissa Zacharias`_ in (:pr:`303`)
* CI failures quick patch by `Anissa Zacharias`_ in (:pr:`312`)
* Resolve CI link-check issue by `Anissa Zacharias`_ in (:pr:`316`)
* Pin numpy for numba compatibility by `Anissa Zacharias`_ in (:pr:`325`)
* Improve speed of spherical harmonics tests by `Alea Kootz`_ in (:pr:`327`)
* Converting pytest to unittest for ``climatology.py`` by `Heather Craker`_ in (:pr:`331`)
* Allow upstream-dev to run on forks by `Anissa Zacharias`_ in (:pr:`335`)


**Full Changelog**: https://github.com/NCAR/geocat-comp/compare/v2022.11.0...v2023.02.0


v2022.11.0 (Nov 17, 2022)
-------------------------

Documentation
^^^^^^^^^^^^^
* Docstring improvements by `Heather Craker`_ in (:pr:`284`)
* Update dependency list by `Heather Craker`_ in (:pr:`293`)


Maintenance
^^^^^^^^^^^
* Module reorganization by `Heather Craker`_ in (:pr:`266`)
* Update to conda-build badge link by `Mario Rodriguez`_ in (:pr:`288`)

**Full Changelog**: https://github.com/NCAR/geocat-comp/compare/v2022.10.1...v2022.11.0


v2022.10.1 (Oct. 28, 2022)
--------------------------

Bug Fixes
^^^^^^^^^
* Pinned Pint version by `Mario Rodriguez`_ in (:pr:`281`)


**Full Changelog**: https://github.com/NCAR/geocat-comp/compare/v2022.10.0...v2022.10.1


v2022.10.0 (Oct. 21, 2022)
--------------------------
New Features
^^^^^^^^^^^^
* Gradient, arc_lon_wgs84, arc_lat_wgs84, rad_lat_wgs84 by `Alea Kootz`_ in (:pr:`256`)

Documentation
^^^^^^^^^^^^^
* Creating quick start guide as part of overall docs improvements by `Heather Craker`_ in (:pr:`249`)
* Documentation style overhaul by `Anissa Zacharias`_ in (:pr:`260`)

Maintenance
^^^^^^^^^^^
* Make local install for docs better by `Anissa Zacharias`_ in (:pr:`273`)
* Documentation and Link Checker by `Anissa Zacharias`_ in (:pr:`274`)



**Full Changelog**: https://github.com/NCAR/geocat-comp/compare/v2022.08.0...v2022.10.0

..
    Add new names and GitHub links as needed

.. _`Heather Craker`: https://github.com/hCraker
.. _`Anissa Zacharias`: https://github.com/anissa111
.. _`Alea Kootz`: https://github.com/pilotchute
.. _`Mario Rodriguez`: https://github.com/marodrig
.. _`Julia Kent`: https://github.com/jukent
.. _`Katelyn FitzGerald`: https://github.com/kafitzgerald
.. _`Cora Schneck`: https://github.com/cyschneck
.. _`Philip Chmielowiec`: https://github.com/philipc2
.. _`AnshRoshan`: https://github.com/AnshRoshan
.. _`Orhan Eroglu`: https://github.com/erogluorhan

..
    TEMPLATE
    vYYYY.MM.## (unreleased)
    ------------------------
    This release...

    New Features
    ^^^^^^^^^^^^

    Breaking Changes
    ^^^^^^^^^^^^^^^^

    Bug Fixes
    ^^^^^^^^^

    Internal Changes
    ^^^^^^^^^^^^^^^^

    Documentation
    ^^^^^^^^^^^^^
