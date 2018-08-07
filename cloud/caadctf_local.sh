#!/bin/bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y python-pip
sudo pip install --upgrade pip

# fundamental tools
sudo apt-get install -y python-numpy python-scipy python-nose
#sudo apt-get install -y python-matplotlib ipython ipython-notebook python-pandas python-sympy
sudo pip install pandas
sudo pip install pillow

# gpu installation
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-0; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
  sudo apt-get update
  sudo apt-get install -y cuda-9-0
fi
# Enable persistence mode
nvidia-smi -pm 1

# cuDNN installation
wget https://storage.googleapis.com/caadctf/libcudnn7_7.1.4.18-1%2Bcuda9.0_amd64.deb
wget https://storage.googleapis.com/caadctf/libcudnn7-dev_7.1.4.18-1%2Bcuda9.0_amd64.deb
sudo apt install ./libcudnn7_7.1.4.18-1+cuda9.0_amd64.deb
sudo apt install ./libcudnn7-dev_7.1.4.18-1+cuda9.0_amd64.deb

sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

# below two lines may need manual input
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

sudo pip install opencv-python
sudo pip install -U tensorflow-gpu
sudo pip install torch torchvision
sudo pip install cleverhans
sudo pip install pathtools
