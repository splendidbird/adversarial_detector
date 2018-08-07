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

#for how many iterations to run the attack
ITER_NUM=20
#step size factor
ALPHA_FACTOR=1

python target_attack.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --num_iter="${ITER_NUM}" \
  --alpha="${ALPHA_FACTOR}" \
  --checkpoint_path_inception_v3=inception_v3.ckpt \
  --checkpoint_path_adv_inception_v3=adv_inception_v3_rename.ckpt \
  --checkpoint_path_ens3_adv_inception_v3=ens3_adv_inception_v3_rename.ckpt \
  --checkpoint_path_ens4_adv_inception_v3=ens4_adv_inception_v3_rename.ckpt \
  --checkpoint_path_ens_adv_inception_resnet_v2=ens_adv_inception_resnet_v2_rename.ckpt
