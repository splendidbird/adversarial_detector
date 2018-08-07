from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from NWS_Attack import *

with open("url_attackid.txt",'r') as f:
    [url, attackid] = f.read().splitlines()

nwsec = NWS_Attack(url, attackid)

# get our defense id and original classlabel
nwsec.getmyid()

# get attack mission list
nwsec.getattacklist()

# get orig image and load in attack list
# prepared and saved properly

parser = argparse.ArgumentParser()
parser.add_argument('--img_path_file', help='full-directory filename of input png')
args = parser.parse_args()
img_path_file = args.img_path_file

imgname = img_path_file.split('/')[-1]
imgpath = '/'.join(img_path_file.split('/')[0:-1])

nwsec.postpng(imgname, imgpath)
