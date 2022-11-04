Please first refer to [GeoCAT Contributor's Guide](https://geocat.ucar.edu/pages/contributing.html) for overall
contribution guidelines (such as detailed description of GeoCAT structure, forking, repository cloning,
branching, etc.). If your contribution is related to GeoCat-Comp functionality, please see the main [GeoCat-Comp Contributor's Guide](../CONTRIBUTING.md).

Once you determine that your documentation should be contributed under this repo, please refer to the
following contribution guidelines:

# Adding new examples to the Geocat-comp repo

Examples are housed in `docs/examples` as `.ipynb` Jupyter Notebooks. You can clone the template notebooks, `docs/examples/template.ipynb`, as a blueprint for your example notebook.

In order for your notebook to appear in the GeoCat-Comp gallery, you must add it as an entry in the `docs/gallery.yml` file AND in the `docs/examples.rst`.

Additionally, please add a thumbnail `.png` image that represents your new notebook to `docs/_static/thumbnails/`. This will be used for rendering the new gallery card.