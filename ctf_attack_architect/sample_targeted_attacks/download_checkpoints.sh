#!/bin/bash
#
# Scripts which download checkpoints for provided models.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Download for attackGD
cd "${SCRIPT_DIR}/attackGD/"
python ../../google_drive_downloader.py 1p1zhtUeBA8MJa0p3X2WHxSoanIsEjH38 checkpoints.tar.gz
tar -xvzf checkpoints.tar.gz

# Download momentum
cd "${SCRIPT_DIR}/momentum/"
python ../../google_drive_downloader.py 1TnT-nHf_a375ilDJKWp5jAu3Bjjwpe8y checkpoints.tar.gz
tar -xvzf checkpoints.tar.gz
rm checkpoints.tar.gz

# Download ToshiK
cd "${SCRIPT_DIR}/ToshiK/"
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
tar -xvzf adv_inception_v3_2017_08_18.tar.gz
wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# Download checkpoints for target_attack_EOT_toshi_on_randomPadding
cd "${SCRIPT_DIR}/RandomPaddingEOT/"
cp ../ToshiK/*.tar.gz .
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz
tar -xvzf adv_inception_v3_2017_08_18.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# Download checkpoints for target_attack_EOT_crop
cd "${SCRIPT_DIR}/CropEOT/"
cp ../ToshiK/*.tar.gz .
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm inception_v3_2016_08_28.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# Download checkpoints for target_attack_EOT_crop
cd "${SCRIPT_DIR}/JpegEOT/"
cp ../ToshiK/*.tar.gz .
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm inception_v3_2016_08_28.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# Download checkpoints for target_class_toshi_k_Sangxia
cd "${SCRIPT_DIR}/ToshiKSangxia/"
cp ../ToshiK/*.tar.gz .
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz
tar -xvzf adv_inception_v3_2017_08_18.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz

# Download checkpoints for toshiksangxia eot
# Download checkpoints for target_class_toshi_k_Sangxia
cd "${SCRIPT_DIR}/ToshiKSangxiaEOT/"
cp ../ToshiK/*.tar.gz .
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz
tar -xvzf adv_inception_v3_2017_08_18.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz
