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
import random
import numpy as np
import shutil
import cv2
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

# Denoise
from dataset import Dataset2
from res152_wide import get_model as get_model1
from inres import get_model as  get_model2
from v3 import get_model as get_model3
from resnext101 import get_model as get_model4

#Random padding
from pretrainedmodels.models import inceptionresnetv2   

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR', default='',
                    help='Input directory with images.')
parser.add_argument('--output_dir', metavar='DIR', default='',
                    help='Output directory for adv images.')
parser.add_argument('--img-size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='Batch size (default: 16)')
parser.add_argument('--no-gpu', action='store_true', default=False,
                    help='disables GPU training')
parser.add_argument('--itr', type=int, default=30, metavar='N',
                    help='Time of iteration (default: 30)')

def batch_transform(inputs, transform, size):
    input_shape = list(inputs.size())
    res = torch.zeros(input_shape[0], input_shape[1], size, size)
    for i in range(input_shape[0]):
        res[i,:,:,:] = transform(inputs[i,:,:,:])
    return res

# codes for random padding
def padding_layer_iyswim(inputs, shape, transform):
    h_start = shape[0]
    w_start = shape[1]
    output_short = shape[2]
    # print(output_short)
    input_shape = list(inputs.size())
    #print(input_shape)
    # input shape (16, 3, 299, 299)
    input_short = min(input_shape[2:4])
    input_long = max(input_shape[2:4])
    #print(input_long, input_short)
    output_long = int(math.ceil( 1. * float(output_short) * float(input_long) / float(input_short)))
    output_height = output_long if input_shape[1] >= input_shape[2] else output_short
    output_width = output_short if input_shape[1] >= input_shape[2] else output_long  
    # print(output_height, output_width, output_long)
    padding = torch.nn.ConstantPad3d((w_start, output_width - w_start - input_shape[3], h_start, output_height - h_start - input_shape[2], 0,0), 0)
    outputs = padding(inputs)
    # print(type(outputs))
    return batch_transform(outputs, transform, 299)

class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """

    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


class FileEventHandler(FileSystemEventHandler):
    def __init__(self, batch_size, input_dir, net1, net4, model, itr, output_dir, no_gpu):
        FileSystemEventHandler.__init__(self)
        self._batch_size = batch_size
        self._input_dir = input_dir
        self._net1 = net1
        self._net4 = net4
        self._model = model
        self._itr = itr
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

        tf_shrink = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([299,299]),
            transforms.ToTensor()
        ]) 
        tf_flip = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
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

        labels_denoise = {}
        labels_random = {}
        random_outputs = []
        denoise_outputs = []
        for batch_idx, input in enumerate(loader):
	        #cv2
            temp_numpy = input.data.numpy()
            temp_numpy = np.reshape(temp_numpy, (3, 299, 299))
            temp_numpy = np.moveaxis(temp_numpy, -1, 0)
            temp_numpy = np.moveaxis(temp_numpy, -1, 0)
            temp_numpy = cv2.bilateralFilter(temp_numpy, 5, 50, 50)
            temp_numpy = np.moveaxis(temp_numpy, -1, 0) 
            temp_numpy = np.reshape(temp_numpy, (1, 3, 299, 299))
            input2 = torch.from_numpy(temp_numpy)

	        # Random padding
            length_input, _, _, _ = input.size()
            iter_labels = np.zeros([length_input, 1001, self._itr])
            for j in range(self._itr):
                # random fliping
                input0 = batch_transform(input2, tf_flip, 299)
                # random resizing
                resize_shape_ = random.randint(310, 331)
                image_resize = 331
                tf_rand_resize = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize([resize_shape_, resize_shape_]),
                    transforms.ToTensor()
                ]) 
                input1 = batch_transform(input0, tf_rand_resize, resize_shape_)

                # ramdom padding
                shape = [random.randint(0, image_resize - resize_shape_), random.randint(0, image_resize - resize_shape_), image_resize]
                # print(shape)
       
                new_input = padding_layer_iyswim(input1, shape, tf_shrink)
                #print(type(new_input))
                if not self._no_gpu:
                    new_input = new_input.cuda()
                with torch.no_grad():
                    input_var = autograd.Variable(new_input)
                    logits = self._model(input_var)
                    labels = logits.max(1)[1]
                    labels_index = labels.data.tolist() 
                    #print(len(labels_index))
                    iter_labels[range(len(iter_labels)), labels_index, j] = 1
            final_labels = np.sum(iter_labels, axis=-1)
            labels = np.argmax(final_labels, 1)
            print(labels)
            random_outputs.append(labels)   

            # Denoise
            if not self._no_gpu:
                input = input.cuda()
            with torch.no_grad():
                input_var = autograd.Variable(input)
                input_tf = (input_var-mean_tf)/std_tf
                input_torch = (input_var - mean_torch)/std_torch
        
                labels1 = net1(input_torch,True)[-1]
                # labels2 = net2(input_tf,True)[-1]
                # labels3 = net3(input_tf,True)[-1]
                labels4 = net4(input_torch,True)[-1]

                labels = (labels1+labels4).max(1)[1] + 1  # argmax + offset to match Google's Tensorflow + Inception 1001 class ids
            denoise_outputs.append(labels.data.cpu().numpy())
        
        denoise_outputs = np.concatenate(denoise_outputs, axis=0)
        random_outputs = np.concatenate(random_outputs, axis=0)

        filenames = [imgfile]
        # filenames = [ os.path.basename(ii) for ii in filenames ]
        labels_denoise.update(dict(zip(filenames, denoise_outputs)))
        labels_random.update(dict(zip(filenames, random_outputs)))
        
        # diff filtering
        print('diff filtering...')
        if (len(labels_denoise) == len(labels_random)):
            # initializing 
            final_labels = labels_denoise
            # Compare
            diff_index = [ii for ii in labels_denoise if labels_random[ii] != labels_denoise[ii]]
            if (len(diff_index) != 0):
                # print(diff_index)
                for index in diff_index:
                    final_labels[index] = 0
        else:
            print("Error: Number of labels returned by two defenses doesn't match")
            exit(-1)   

        output_file_name = ""
        for filename, label in final_labels.items():
            res_file_name = os.path.basename(filename)[:-4] + '.txt'
            output_file_name = os.path.join(self._output_dir, os.path.basename(filename))
            #print(output_file_name)
            #print("res_file_name: " + res_file_name)
            with open(os.path.join(self._output_dir, res_file_name), 'w+') as res_file:
                res_file.write('{0}\n{1}\n'.format(label,
                                                   self._category_helper.get_category_name(label)))
                res_file.flush()
            if os.path.exists(output_file_name):
                os.remove(output_file_name)
            shutil.copy(filename, output_file_name)
            os.remove(filename)     
        
        elapsed_time = time.time() - start_time
        print('elapsed time: {0:.1f} [s]'.format(elapsed_time))

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
        #config, inresmodel = get_model2()
        #config, incepv3model = get_model3()
        config, rexmodel = get_model4()
        net1 = resmodel.net    
        #net2 = inresmodel.net
        #net3 = incepv3model.net
        net4 = rexmodel.net

    checkpoint = torch.load('denoise_res_015.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        resmodel.load_state_dict(checkpoint['state_dict'])
    else:
        resmodel.load_state_dict(checkpoint)

    #checkpoint = torch.load('denoise_inres_014.ckpt')
    #if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        #inresmodel.load_state_dict(checkpoint['state_dict'])
    #else:
        #inresmodel.load_state_dict(checkpoint)

    #checkpoint = torch.load('denoise_incepv3_012.ckpt')
    #if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        #incepv3model.load_state_dict(checkpoint['state_dict'])
    #else:
        #incepv3model.load_state_dict(checkpoint)
    
    checkpoint = torch.load('denoise_rex_001.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        rexmodel.load_state_dict(checkpoint['state_dict'])
    else:
        rexmodel.load_state_dict(checkpoint)

    if not args.no_gpu:
        #inresmodel = inresmodel.cuda()
        resmodel = resmodel.cuda()
        #incepv3model = incepv3model.cuda()
        rexmodel = rexmodel.cuda()
    #inresmodel.eval()
    resmodel.eval()
    #incepv3model.eval()
    rexmodel.eval()

    #inceptionresnetv2 for ramdon padding
    model = inceptionresnetv2(num_classes=1001, pretrained='imagenet+background')
    model = model.cuda()
    model.eval()

    ''' watch the input dir for defense '''
    observer = Observer()
    event_handler = FileEventHandler(batch_size=args.batch_size,
                                     input_dir=args.input_dir,
                                     net1=net1,
                                     net4=net4,
                                     model=model,
                                     itr=args.itr,
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
