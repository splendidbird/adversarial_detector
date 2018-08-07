from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import requests
from ctf import ctf
import os
import glob

url = 'http://35.230.92.142:8080'
attackid = 'F98Z16'

#inialize ctf object
getpost = ctf(url, attackid)

# get opponents' defenseid
defense_ids = getpost.getdefense()
print(defense_ids)

# get original image
png_file = 'original.png'
original = getpost.getpng(png_file)
# print(original)

# create adversarial images

# post attacks to all opponents
# folder storing the images
attack_folder = '.'
path_attack_folder = os.path.join(attack_folder, '**/*.png')
adv_images = glob.glob(path_attack_folder, recursive=True)
print(adv_images)
epsilon = 32
#input = 
# list of lists storing result
#         defense1   defense2   defense3 ...
#image1  [resutl1,   result2,   result3, ...] 
results = []
for i in enumerate(adv_images):
    results_per_image = []
    for id in defense_ids:
        label, result = getpost.postpng('original.png', id, epsilon)
        if (result == 'negative'):
            results_per_image.append(label)
        else:
            results_per_image.append(0)
    results.append(results_per_image)

print(results)