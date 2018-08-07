"""Implementation of sample defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import argparse
import math
import numpy as np
import shutil

from scipy.misc import imread

from watchdog.observers import Observer
from watchdog.events import *
from category.category import CategoryHelper

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision
import torchvision.datasets.folder
import torchvision.transforms as transforms

from dataset import Dataset2
from res152_wide import get_model as get_model1
from inres import get_model as  get_model2
from v3 import get_model as get_model3
from resnext101 import get_model as get_model4

parser = argparse.ArgumentParser(description='Defense')
parser.add_argument('--input_dir', metavar='DIR', default='', required=True,
                    help='Input directory with images.')
parser.add_argument('--output_dir', metavar='DIR', default='', required=True,
                    help='Output directory to save images and labels.')
parser.add_argument('--checkpoint_path', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--checkpoint_path2', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--img-size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='Batch size (default: 16)')
parser.add_argument('--no-gpu', action='store_true', default=False,
                    help='disables GPU training')

class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """

    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


class FileEventHandler(FileSystemEventHandler):
    def __init__(self, batch_size, input_dir, net1, net4, output_dir, no_gpu):
        FileSystemEventHandler.__init__(self)
        self._batch_size = batch_size
        self._input_dir = input_dir
        self._net1 = net1
        self._net4 = net4
        self._output_dir = output_dir
        self._no_gpu = no_gpu
        self._category_helper = CategoryHelper("category/categories.csv")

    def on_moved(self, event):
        if event.is_directory:
            print("directory moved from {0} to {1}".format(event.src_path, event.dest_path))
        else:
            print("file moved from {0} to {1}".format(event.src_path, event.dest_path))

    def _defense_for_img_created(self, imgfile):
        """ defense one image: xxx.png,
            write res to xxx.txt with two line(lable human_string),
            copy the src image file to output dir then delete it
        :param img_file:
        :return None:
        """
        start_time = time.time()
        
        tf = transforms.Compose([
            transforms.Resize([299,299]),
            transforms.ToTensor()
        ])
        with torch.no_grad():
            mean_torch = autograd.Variable(torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1,3,1,1]).astype('float32')).cuda())
            std_torch = autograd.Variable(torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1,3,1,1]).astype('float32')).cuda())
            mean_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).cuda())
            std_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).cuda())
            dataset = Dataset2(imgfile, transform=tf)
            loader = data.DataLoader(dataset, batch_size=self._batch_size, shuffle=False)
            net1 = self._net1
            net4 = self._net4

        outputs = []
        for batch_idx, input in enumerate(loader):
            if not self._no_gpu:
                input = input.cuda()
            with torch.no_grad():
                input_var = autograd.Variable(input)
                input_tf = (input_var-mean_tf)/std_tf
                input_torch = (input_var - mean_torch)/std_torch
            
                labels1 = net1(input_torch,True)[-1]
                labels4 = net4(input_torch,True)[-1]
                labels = (labels1+labels4).max(1)[1] + 1  # argmax + offset to match Google's Tensorflow + Inception 1001 class ids
            outputs.append(labels.data.cpu().numpy())
        outputs = np.concatenate(outputs, axis=0)   
    
        
        filenames = [imgfile]
        output_file_name = ""
        for filename, label in zip(filenames, outputs):
            res_file_name = os.path.basename(filename)[:-4] + '.txt'
            output_file_name = os.path.join(self._output_dir, os.path.basename(filename))
            print(output_file_name)
            print("res_file_name: " + res_file_name)
            with open(os.path.join(self._output_dir, res_file_name), 'w+') as res_file:
                res_file.write('{0}\n{1}\n'.format(label,
                                                   self._category_helper.get_category_name(label)))
                res_file.flush()
            if os.path.exists(output_file_name):
                os.remove(output_file_name)
            shutil.copy(filename, output_file_name)
            os.remove(filename)
        elapsed_time = time.time() - start_time
        print('elapsed time: {0:.0f} [s]'.format(elapsed_time))

    def on_created(self, event):
        if event.is_directory:
            print("directory created:{0}".format(event.src_path))
        else:
            print("file created:{0}".format(event.src_path))
            self._defense_for_img_created(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            print("directory deleted:{0}".format(event.src_path))
        else:
            print("file deleted:{0}".format(event.src_path))

    def on_modified(self, event):
        if event.is_directory:
            print("directory modified:{0}".format(event.src_path))
        else:
            print("file modified:{0}".format(event.src_path))

def main():
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print("Error: Invalid input folder %s" % args.input_dir)
        exit(-1)
    if not os.path.exists(args.output_dir):
        print("Error: Invalid output folder %s" % args.output_dir)
        exit(-1)
        
    with torch.no_grad():
        config, resmodel = get_model1()
        config, inresmodel = get_model2()
        config, incepv3model = get_model3()
        config, rexmodel = get_model4()
        net1 = resmodel.net    
        net2 = inresmodel.net
        net3 = incepv3model.net
        net4 = rexmodel.net

    checkpoint = torch.load('denoise_res_015.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        resmodel.load_state_dict(checkpoint['state_dict'])
    else:
        resmodel.load_state_dict(checkpoint)

    checkpoint = torch.load('denoise_inres_014.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        inresmodel.load_state_dict(checkpoint['state_dict'])
    else:
        inresmodel.load_state_dict(checkpoint)

    checkpoint = torch.load('denoise_incepv3_012.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        incepv3model.load_state_dict(checkpoint['state_dict'])
    else:
        incepv3model.load_state_dict(checkpoint)
    
    checkpoint = torch.load('denoise_rex_001.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        rexmodel.load_state_dict(checkpoint['state_dict'])
    else:
        rexmodel.load_state_dict(checkpoint)

    if not args.no_gpu:
        inresmodel = inresmodel.cuda()
        resmodel = resmodel.cuda()
        incepv3model = incepv3model.cuda()
        rexmodel = rexmodel.cuda()
    inresmodel.eval()
    resmodel.eval()
    incepv3model.eval()
    rexmodel.eval()

    ''' watch the input dir for defense '''
    observer = Observer()
    event_handler = FileEventHandler(batch_size=args.batch_size,
                                     input_dir=args.input_dir,
                                     net1=net1,
                                     net4=net4,
                                     output_dir=args.output_dir,
                                     no_gpu=args.no_gpu)

    observer.schedule(event_handler, args.input_dir, recursive=True)
    observer.start()

    print("watchdog start...")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    print("\nwatchdog stoped!")

if __name__ == '__main__':
    main()
