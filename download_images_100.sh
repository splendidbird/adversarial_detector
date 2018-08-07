#!/bin/bash
# Ramdonly download 100 images according to dev_dataset.csv

cd "$( dirname "${BASH_SOURCE[0]}" )"

rm -rf dataset/images
mkdir -p dataset/images
python ./dataset/download_images.py \
  --input_file=dataset/dev_dataset.csv \
  --numbers_images=100 \
  --output_dir=dataset/images/
