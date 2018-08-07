#!/bin/bash
#
# run_attack.sh is a script which executes the attack
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_attack.sh INPUT_DIR OUTPUT_DIR MAX_EPSILON
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#

INPUT_DIR=$1
OUTPUT_DIR=$2
MAX_EPSILON=$3

# For how many iterations run this attack
NUM_ITERATIONS=40
# EOT in this code is to tune iterations for random noise generations
EOT=15
# alpha factor
ALPHA_FACTOR=10

python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --num_iter="${NUM_ITERATIONS}" \
  --ensemble_size="${EOT}" \
  --alpha_factor="${ALPHA_FACTOR}" \
  --checkpoint_path1=inception_v3.ckpt \
  --checkpoint_path2=adv_inception_v3.ckpt \
  --checkpoint_path3=ens_adv_inception_resnet_v2.ckpt
