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
NUM_ITERATIONS=100
# alpha factor
ALPHA_FACTOR=1.0
#EOT ensemble size
EOT=300

# two kinds of attck here, one using only InceptionResnet_v2, one use both Inception and InceptionResnet_v2
python target_attack_crop.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --num_iter="${NUM_ITERATIONS}" \
  --alpha_factor="${ALPHA_FACTOR}" \
  --ensemble_size="${EOT}" \
  --checkpoint_path1=inception_v3.ckpt \
  --checkpoint_path2=adv_inception_v3.ckpt \
  --checkpoint_path3=ens_adv_inception_resnet_v2.ckpt
