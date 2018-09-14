# Adversarial Image Attack Sensor

A tool to evalute whether an input image has been attacked (as an adversarial example) to fool conventional classifiers.
A score is output from the binary classifier, based on which the input image can be categorized into 'Safe', 'Suspicious' and 'Danger'.

Two deployments - stand-alone and cloud:

Stand-alone version is based on a tool kit for the
[Competition on Adversarial Attacks and Defenses 2018](http://caad.geekpwn.org/) CTF @ LV

The cloud version is deployed on AWS for real-time classification, with API available in the future.

## Algorithm
![AdvDistribution]
(https://github.com/splendidbird/adversarial_detector/raw/master/images/adv_disc_dist.png)

## Installation (Stand-alone)

### Prerequisites

Following software required to use this package:

* Python 2.7 with installed [Numpy](http://www.numpy.org/)
  and [Pillow](https://python-pillow.org/) packages.

Following libs are required by some attacks/defenses, but not all of them.
* [Docker](https://www.docker.com/) (optional)
* [TensorFlow](https://www.tensorflow.org/)
* [PyTorch](https://pytorch.org/)
* [cleverhans](https://github.com/tensorflow/cleverhans)

### Defenses
* adv_inception_v3(/defense/)
* Guided_Denoise_14
  * Dataset2 class has been created to only load one image from input folder
    * Redefine functions __getitem__ and __len__
  * Move inference code into function of event handler, which is triggered when a new file is created.
* Random_Denoise_14
  * Add random padding on to Guided_Denoise
  * Classify 1 picture needs **4 seconds** when iteration = 30.
* Random_padding_IresV2
  * **Slow!!**. Classify 1 picture needs **5-6 seconds** when iteration = 30.
* Diff_Random_Denoise_14_pytorch
  * Difference filter of two defenses: Random padding and Denoise. it is pytorch version. The random padding is implemented with inception_resnet_v2 instead of ens_adv_inception_resnet_v2 in the original version above, becuase I only found the pre-trained weights for the former.
* Diff_cv2_Random_Denoise_14_pytorch
  * Difference filter of two defenses: (cv2_Random padding) and (Denoise). it is pytorch version. The random padding is implemented with inception_resnet_v2 instead of ens_adv_inception_resnet_v2 in the original version above, becuase I only found the pre-trained weights for the former.
* [Not work]Diff_Random_Denoise
  * defense.py: Seperate two classifier into two watchdog functions to handler two events: file creation and file deletion, sperately. But it still doesn't work.
  * defense2.py: Try to combine tensorflow and pytorch into the same function, but it doesn't work. Tenserflow fails to get CUDNN hanle after pytorch code.
  pytorch will be out of memeory after tensorflow code.
  Don't know how to fix it.

### Download check points and images

To be able to run the examples you need to download checkpoints for provided models
as well as dataset.

To all checkpoints run following:

```bash
./download_checkpoints.sh
```
To download randomly 100 images run following:

```bash
./download_images_100.sh
```
Or you can define number of images:

```bash
./download_images.sh <output_folder> <number of images>
```

## Cloud Service

## Docker

## API

## Dataset

This toolkit includes 1000 labelled images.
Details about dataset are [here](./dataset/README.md).

## category helper
```bash
# check the label by label id
python ./category/category.py <label_id>
```

## How to use this tool kit
### Only defense


### Commands accepted by server
