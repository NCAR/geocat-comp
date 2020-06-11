#!/bin/bash

set -e
set -eo pipefail

conda config --set always_yes true --set changeps1 false --set quiet true
conda config --add channels conda-forge
conda list -f python -e >> /usr/local/conda-meta/pinned
conda install git
git clone ${LIBNCOMP_GIT_REPO}
cd libncomp
git checkout ${CIRCLE_BRANCH} || echo "No ${CIRCLE_BRANCH} on libncomp"
conda env create -f .circleci/environment-dev-$(uname).yml --name ${LIBNCOMP_ENV_NAME} --quiet
conda env list
source activate ${LIBNCOMP_ENV_NAME}
autoreconf --install
./configure --prefix=${CONDA_PREFIX}
make install
