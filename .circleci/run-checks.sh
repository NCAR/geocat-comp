#!/bin/bash

set -e
set -eo pipefail

source activate ${ENV_NAME}

pytest --verbose test
