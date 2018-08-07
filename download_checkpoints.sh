#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Download adversarially trained inception v3 checkpoint
cd "${SCRIPT_DIR}/defense/"
wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
tar -xvzf adv_inception_v3_2017_08_18.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz

# Download checkpoints for Guided_denoise defense
# Original file is from https://www.dropbox.com/sh/q9ssnbhpx8l515t/AACvjiMmGRCteaApmj1zTrLTa?dl=0
# or https://pan.baidu.com/s/1hs7ti5Y#list/path=%2F
# I convert it to tar.gz
cd "${SCRIPT_DIR}/Guided_Denoise_14/"
python ../google_drive_downloader.py 1p1zhtUeBA8MJa0p3X2WHxSoanIsEjH38 checkpoints.tar.gz
tar -xvzf checkpoints.tar.gz

# Download checkpoints for Random_padding_IresV2
cd "${SCRIPT_DIR}/Random_padding_IresV2/"
wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# Download checkpoints for Random_Denoise_14
cd "${SCRIPT_DIR}/Random_Denoise_14/"
mv ../Guided_Denoise_14/checkpoints.tar.gz .
tar -xvzf checkpoints.tar.gz

# Download checkpoints for Diff_Random_Denoise_14
cd "${SCRIPT_DIR}/Diff_Random_Denoise_14/"
mv ../Random_Denoise_14/checkpoints.tar.gz .
tar -xvzf checkpoints.tar.gz
mv ../Random_padding_IresV2/ens_adv_inception_resnet_v2_2017_08_18.tar.gz .
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# Download checkpoints for Diff_cv2_Random_Denoise_14_pytorch
cd "${SCRIPT_DIR}/Diff_cv2_Random_Denoise_14_pytorch/"
mv ../Diff_Random_Denoise_14/checkpoints.tar.gz .
tar -xvzf checkpoints.tar.gz

# Download checkpoints for Diff_Random_Denoise_14_pytorch
cd "${SCRIPT_DIR}/Diff_Random_Denoise_14_pytorch/"
mv ../Diff_cv2_Random_Denoise_14_pytorch/checkpoints.tar.gz .
tar -xvzf checkpoints.tar.gz
rm checkpoints.tar.gz
