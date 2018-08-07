#!/bin/bash

# exit on first error
set -e

# directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DEFENSE_DIR="${SCRIPT_DIR}/$1"

python "${SCRIPT_DIR}/run_defense_docker.py" \
  --defense_dir="${DEFENSE_DIR}" \
  --input_dir="${SCRIPT_DIR}/input" \
  --output_dir="${SCRIPT_DIR}/output" \
  --gpu
