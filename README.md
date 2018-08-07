# CTF Toolkit

This is modified from the toolkit for the
[Competition on Adversarial Attacks and Defenses 2018](http://caad.geekpwn.org/) CTF @ LV

## Installation

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
```bash
./run_defense.sh ../input ../output
```
starts a classifier and keeps it monitoring the input folder. If there’s a new image, the classifier will inference the result into a txt file with the same name as image, and move both files into output folder.

You can also run defense in docker
```bash
# container and entry_point are defined in matadata.json under defense folder
# This shell file must be in the same folder as 'input' and 'output' 
./run_defense_docker.sh <defense_dir>
```

Or
```bash
#  Firstly start the container
docker run --rm -ti -v /…/defense:/code -v /…/input:/input -v /…/output:/output -w/code tensorflow/tensorflow:1.1.0 /bin/bash
```
```bash
# Go inside docker container, and run the defense
./run_defense2.sh /input /output
```

### Simulate the CTF
1. Start the classifier/defense
2. Start the server by running the following script
```bash
# It starts a server running at 127.0.0.1:8888.
Python defensewrapper.py
```


3. Run our codes to get info from and post attack to the server
* ctf.py: a class having all functions required for this CTF
  * getmydefense
  * getdefenses
  * getpng
  * postpng
* post-attack.py: a example of posting an adversarial image to server


### Commands accepted by server

#### *getmydefense* 
get your team's defense id
```
127.0.0.1:8888/?cmd=getmydefense&attackid=12345
```

```json
Response sample in json forma
{
 "msg": "getmydefense succeed",
 "classlabel": 123,
 "defenseid": 1,
 "result": 0
}
```
#### *getdefenses* will get all defenses you can attack
```
127.0.0.1:8888/?cmd=getdefenses&attackid=12345
```
```json
Response sample in json format
{
 [
 {
 "classlabel": 260,
 "defenseid": 2
 },
 {
 "classlabel": 851,
 "defenseid": 3
 },
 {
 "classlabel": 74,
 "defenseid": 4
 },
 {
 "classlabel": 309,
 "defenseid": 5
 }
]
}
```

#### *getdefenses* 
get all defenses you can attack
```
127.0.0.1:8888/?cmd=getdefenses&attackid=12345
```
```json
Response sample in json format
{
 [
 {
 "classlabel": 260,
 "defenseid": 2
 },
 {
 "classlabel": 851,
 "defenseid": 3
 },
 {
 "classlabel": 74,
 "defenseid": 4
 },
 {
 "classlabel": 309,
 "defenseid": 5
 }
]
}
```
#### *getteamlogo*
This command will respond the image which the team need to use as the source
image. Create adversarial examples based on this image to attack other teams.
```bash
127.0.0.1:8888/?cmd=getteamlogo&attackid=12345
Response: binary data content of .png file.
```

#### *postattack*
This command will be used to attack other defenses. The defenseid parameter in the
request is the target defense. The epsilon parameter is the perturbation. The epsilon value can be 1,2,4,8,16,32
```bash
127.0.0.1:8888/?cmd=postattack&attackid=abcde&defenseid=2&epsilon=32
Post data body: binary data content of adversarial example .png file.
Response sample: json format
{
 "msg": "attack performed",
 "classlabel": 34,
 "attackresult": "negative",
 "result": 0,
 "description": "loggerhead, loggerhead turtle, Caretta caretta\n"
}
```
If attack is successful, the “attackresult”field will be positive. Otherwise, it is negative. The classlabel is the class the defense put the image in. Result is 0 if attack performed, otherwise there are problems when attacking.





