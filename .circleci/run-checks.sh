#!/bin/bash

set -e
set -eo pipefail

source activate ${ENV_NAME}
pip install netcdf4
pip instll scipy
pytest --verbose test
