#!/bin/bash

# exit on first error
set -e

# directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

TARGETED_ATTACKS_DIR="${SCRIPT_DIR}/sample_targeted_attacks"
DATASET_DIR="${SCRIPT_DIR}/dataset/images"
MAX_EPSILON=32

# Prepare working directory and copy all necessary files.
# In particular copy attacks defenses and dataset, so originals won't
# be overwritten.
if [[ "${OSTYPE}" == "darwin"* ]]; then
    WORKING_DIR="/private"$(mktemp -d -p ${SCRIPT_DIR})
else
    WORKING_DIR=$(mktemp -d -p ${SCRIPT_DIR})
fi
echo "Preparing working directory: ${WORKING_DIR}"
mkdir "${WORKING_DIR}/targeted_attacks"
mkdir "${WORKING_DIR}/dataset"
mkdir "${WORKING_DIR}/intermediate_results"

cp -R "${TARGETED_ATTACKS_DIR}"/* "${WORKING_DIR}/targeted_attacks"
cp -R "${DATASET_DIR}"/* "${WORKING_DIR}/dataset"
cp "${SCRIPT_DIR}/configTargetedAttackChosenList.py" "${WORKING_DIR}/targeted_attacks/configTargetedAttackChosenList.py"

echo "Running attacks and defenses"
python -W ignore "${SCRIPT_DIR}/targetAttack.py" \
  --targeted_attacks_dir="${WORKING_DIR}/targeted_attacks" \
  --dataset_dir="${WORKING_DIR}/dataset" \
  --intermediate_results_dir="${WORKING_DIR}/intermediate_results" \
  --epsilon="${MAX_EPSILON}" \
  --save_all_classification

echo "adv images saved in directory '${WORKING_DIR}/adv_images'"
