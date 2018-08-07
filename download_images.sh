#!/bin/bash
# Ramdonly download 100 images according to dev_dataset.csv

OUTPUT=$1
NUMBERS_IMAGES=$2

cd "$( dirname "${BASH_SOURCE[0]}" )"

rm -rf ${OUTPUT}
mkdir -p ${OUTPUT}
python ./dataset/download_images.py \
  --input_file=dataset/dev_dataset.csv \
  --numbers_images=${NUMBERS_IMAGES} \
  --output_dir=${OUTPUT}
