#!/bin/bash

set -e
set -eo pipefail

conda config --set always_yes true --set changeps1 false --set quiet true
conda config --add channels conda-forge
conda create --clone ${NCOMP_ENV_NAME} --name ${ENV_NAME}
conda env update -f .circleci/environment-dev-$(uname)-${PYTHON}.yml --name ${ENV_NAME} --quiet
conda env list
source activate ${ENV_NAME}
#pip install pip --upgrade
pip install --no-deps --quiet .
conda list -n ${ENV_NAME}
