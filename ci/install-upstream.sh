#!/usr/bin/env bash
# adapted from https://github.com/pydata/xarray/blob/main/ci/install-upstream-wheels.sh

# forcibly remove packages to avoid artifacts
conda remove -y --force \
    cf_xarray \
    dask \
    distributed \
    metpy \
    numpy \
    pandas \
    pint \
    scipy \
    xarray \
    uxarray \
    xskillscore

# conda list
conda list

# if available install from nightly wheels
python -m pip install \
    -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple/ \
    --no-deps \
    --pre \
    --upgrade \
    numpy \
    pandas \
    scipy \
    xarray

# install rest from source
python -m pip install \
    --no-deps \
    --upgrade \
    git+https://github.com/xarray-contrib/cf-xarray.git \
    git+https://github.com/dask/dask.git \
    git+https://github.com/dask/distributed.git \
    git+https://github.com/Unidata/MetPy.git \
    git+https://github.com/hgrecco/pint.git \
    git+https://github.com/xarray-contrib/xskillscore.git \
    git+https://github.com/Uxarray/uxarray.git
