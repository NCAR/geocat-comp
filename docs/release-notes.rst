.. currentmodule:: geocat.comp

.. _release:

Release Notes
=============

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
-----------------------
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
* Remove unnecessary tag publish trigger for ASV benchmarking CI  by `Anissa
  Zacharias`_ in (:pr:`509`)
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
  ``climatology_average`` by `Julia Kent`_ in (:pr:`441`)

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
* Fix bug in ``_temp_extrapolate`` used by ``interp_hybrid_to_pressure`` by `Katelyn FitzGerald`_ in (:pr:`422`)


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
