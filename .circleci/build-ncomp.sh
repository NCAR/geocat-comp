#!/bin/bash

set -e
set -eo pipefail

conda config --set always_yes true --set changeps1 false --set quiet true
conda config --add channels conda-forge
conda install git
git clone ${NCOMP_GIT_REPO}
cd ncomp
git checkout ${CIRCLE_BRANCH} || echo "No ${CIRCLE_BRANCH} on ncomp"
conda env create -f .circleci/environment-dev-$(uname).yml --name ${NCOMP_ENV_NAME} --quiet
conda env list
source activate ${NCOMP_ENV_NAME}
autoreconf --install
./configure --prefix=${CONDA_PREFIX}
make install
