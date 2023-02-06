## PR Summary
<!-- Summary goes here. Replace XXX with the number of the issue this PR will resolve. -->
Closes XXX

## PR Checklist
<!-- Please mark any checkboxes that do not apply to this PR as [N/A]. -->
**Functionality**
- [ ] Function is in appropriate module file
- [ ] New function(s) intended for public API added to `src/geocat/comp/__init__.py` file

**Testing**
- [ ] Tests for function exists in associated module test file
- [ ] Tests cover all possible logical paths in your function

**Documentation**
- [ ] Passes `precommit`. To set up on your local, run `pre-commit install` from the top level of the repository. To manually run pre-commits, use `pre-commit run --all-files` and re-add any changed files before committing again and pushing.
- [ ] Docstrings have been added to all new functions ([Documentation Standards](https://geocat.ucar.edu/pages/contributing.html#422-documentation))
- [ ] Docstrings have updated with any function changes
- [ ] Internal functions have a preceeding underscore (`_`) and have been added to `docs/internal_api/index.rst`
- [ ] User facing functions have been added to `docs/user_api/index.rst` under their module
- [ ] Appropriate labels have been added to this PR

**Examples**
- [ ] Any new notebook examples added to `docs/examples/` folder
- [ ] Pre-run all notebook cells
- [ ] New notebook files added to `docs/examples.rst` toctree
- [ ] New notebook files added to new entry in `docs/gallery.yml` with appropriate thumbnail photo in `docs/_static/thumbnails/`

**PR Etiquette Reminders**
- This PR should be listed as a draft PR until you are ready to request reviewers
- After making changes in accordance with the reviews, re-request your reviewers
- Do *not* mark conversations as resolved if you didn't start them
- Do mark conversations as resolved *if you opened them* and are satified with the changes/discussion.

<!--
Thank you so much for your PR!  To help us review your contribution, please
consider the following points:

- A development guide is available at https://geocat.ucar.edu/pages/contributing.html.

- Fork this repository and open the PR from your fork. Do not directly work on
  the NCAR/geocat-comp repository.

- The PR title should summarize the changes, for example "Create weighted pearson-r
  correlation coefficient function". Avoid non-descriptive titles such as "Addresses
  issue #229".

- The summary should provide at least 1-2 sentences describing the pull request
  in detail (Why is this change required?  What problem does it solve?) and
  link to any relevant issues.

If you need assistance with your PR, please let the GeoCAT team know by
tagging us with @geocat. We can help if reviews are unclear, the recommended changes
seem overly demanding, you would like help in addressing a reviewer's comments,
or if you have been waiting more than a week to hear back on your PR.
-->
