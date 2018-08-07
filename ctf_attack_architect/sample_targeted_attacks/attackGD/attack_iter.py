"""Sample Pytorch defense.

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
import csv
from scipy.misc import imsave
from shutil import copy

import torch
from torch import nn
import torch.autograd as autograd
import torch.utils.data as data
import torchvision
import torchvision.datasets.folder
import torchvision.transforms as transforms

import tensorflow as tf
from torch.autograd.gradcheck import zero_gradients
from dataset import Dataset
from res152_wide import get_model as get_model1
from inres import get_model as  get_model2
from v3 import get_model as get_model3
from resnext101 import get_model as get_model4

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR', default='',
                    help='Input directory with images.')
parser.add_argument('--output_dir', metavar='DIR', default='',
                    help='Output directory with adv images.')
parser.add_argument('--output_attack_dir', metavar='DIR', default=r'../../../adv_images',
                    help='Output attack directory with adv images.')
parser.add_argument('--checkpoint_path', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--checkpoint_path2', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--img_size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='Batch size (default: 1)')
parser.add_argument('--max_epsilon', type=int, default=32, metavar='N',
                    help='Batch size (default: 32)')
parser.add_argument('--num_iter', type=int, default=30, metavar='N',
                    help='number of iterations (default: 50)')
parser.add_argument('--no-gpu', action='store_true', default=False,
                    help='disables GPU training')
parser.add_argument('--alpha_factor', type=float, default=1.0, metavar='N',
                    help='alpha_factor (alpha = alpha_factor * 0.01)')

args = parser.parse_args()

ITER_NUM = args.num_iter
ALPHA_FACTOR = args.alpha_factor
EOT=0

def get_file_config():
    fileConfig = '_'
    fileConfig = fileConfig + 'EOT-' + str(EOT) + '_' + 'iterNum-'+ str(ITER_NUM) + '_' + 'alphaFactor-' + str(ALPHA_FACTOR) + '_' + str(int(args.max_epsilon)) + '_'
    return fileConfig

def load_target_class(input_dir):
  """Loads target classes."""
  with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:
    return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}

def convert_tf_dimensions(image, shape_tf):
    image_converted = np.zeros(shape_tf)
    for i in range(3):
        image_converted[:, :, i] = image[i, :, :]
    return image_converted

def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  batch_shape = images.shape

  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    output_path = filename.split(r"/")[-1].split(r'_')[-1]
    output_path = 'AttackGD'+ get_file_config() + output_path

    with tf.gfile.Open(os.path.join(output_dir, output_path), 'w') as f:
      tf_image = convert_tf_dimensions(images[i, :, :, :], (batch_shape[2], batch_shape[3], batch_shape[1]))
      #imsave(f, (tf_image + 1.0) * 0.5, format='png')
      imsave(f, tf_image * 255)
    copy(os.path.join(output_dir, output_path), os.path.join(args.output_attack_dir, output_path))

class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """

    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor

def get_filenames_batch(filenames, batch_idx, batch_size):

    num_batches = len(filenames) / batch_size

    if batch_idx > num_batches - 1:
        return filenames[batch_idx * batch_size :]
    else:
        return filenames[batch_idx * batch_size : (batch_idx + 1) * batch_size]

def load_target_class(input_dir):
  """Loads target classes."""
  with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:
    return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}

def _get_diff_img(adv_images, orig_images):
    print('adv.shape = {0}'.format(adv_images.shape))
    print('orig_images.shape = {0}'.format(orig_images.shape))

    diff_img = np.absolute(adv_images - orig_images)
    max_diff = 0
    for i in range(diff_img.shape[0]):
        max_diff = max(max_diff, np.amax(diff_img[i]))
    print('max_diff = {0}'.format(max_diff))

def main():
    start_time = time.time()

    if not os.path.exists(args.input_dir):
        print("Error: Invalid input folder %s" % args.input_dir)
        exit(-1)
    if not args.output_dir:
        print("Error: Please specify an output directory")
        exit(-1)

    trans_forms = transforms.Compose([
           transforms.Resize([299,299]),
            transforms.ToTensor()
    ])

    with torch.no_grad():
        mean_torch = autograd.Variable(torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1,3,1,1]).astype('float32')).cuda())
        std_torch = autograd.Variable(torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1,3,1,1]).astype('float32')).cuda())
        mean_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).cuda())
        std_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).cuda())

        dataset = Dataset(args.input_dir, transform=trans_forms)
        #loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        loader = data.DataLoader(dataset, batch_size=1, shuffle=False)

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

    outputs = []
    filenames = dataset.filenames()
    all_images_taget_class = load_target_class(args.input_dir)
    #print('filenames = {0}'.format(filenames))
    for batch_idx, (input, _) in enumerate(loader):
        #print('input = {0}'.format(input.data.numpy().shape))
        #print('batch_idx = {0}'.format(batch_idx))
        filenames_batch = get_filenames_batch(filenames, batch_idx, args.batch_size)
        filenames_batch = [n.split(r"/")[-1] for n in filenames_batch]
        print('filenames = {0}'.format(filenames_batch))

        target_class_for_batch = (
            [all_images_taget_class[n] - 1 for n in filenames_batch]
            + [0] * (args.batch_size - len(filenames_batch))) # all_images_taget_class[n] - 1 to match imagenet label 1001 classes
        print('target_class_for_batch = {0}'.format(target_class_for_batch))

        #labels1 = net1(input_torch,True)[-1]
        #labels2 = net2(input_tf,True)[-1]
        #labels3 = net3(input_tf,True)[-1]
        #labels4 = net4(input_torch,True)[-1]
        #labels = (labels1+labels2+labels3+labels4).max(1)[1] + 1  # argmax + offset to match Google's Tensorflow + Inception 1001 class ids
        #print('labels1.shape = ', labels1.data.cpu().numpy().shape) # looks like labels1.data.cpu().numpy can be used as logits
        #print('labels1', labels1.data.cpu().numpy())

        loss = nn.CrossEntropyLoss()
        step_alpha = 0.01 * ALPHA_FACTOR
        eps = args.max_epsilon / 255.0 # input in now in [0, 1]
        target_label = torch.Tensor(target_class_for_batch).long().cuda()
        #print('input.cpu().numpy().amax = {0}'.format(np.amax(input.cpu().numpy()))) #1.0
        #print('input.cpu().numpy().amin = {0}'.format(np.amin(input.cpu().numpy()))) #0.0
        #raise ValueError('hold')
        if not args.no_gpu:
            input = input.cuda()
        input_var = autograd.Variable(input, requires_grad=True)
        orig_images = input.cpu().numpy()
        y = autograd.Variable(target_label)
        for step in range(args.num_iter):

            input_tf = (input_var-mean_tf)/std_tf
            input_torch = (input_var - mean_torch)/std_torch
            #input_tf = autograd.Variable(input_tf, requires_grad=True)
            #input_torch = autograd.Variable(input_torch, requires_grad=True)

            zero_gradients(input_tf)
            zero_gradients(input_torch)

            out = net1(input_torch,True)[-1]
            out += net2(input_tf,True)[-1]
            out += net3(input_tf,True)[-1]
            out += net4(input_torch,True)[-1]
            pred = out.max(1)[1] + 1
            if step % 10 == 0:
                print('pred = {0}'.format(pred))
            _loss = loss(out, y)
            #_loss = autograd.Variable(_loss)
            _loss.backward()
            #print('type of input = ', type(input_torch))
            #print('type of input.grad = ', type(input_torch.grad))
            normed_grad = step_alpha * torch.sign(input_var.grad.data)
            step_adv = input_var.data - normed_grad
            adv = step_adv - input.data
            adv = torch.clamp(adv, -eps, eps)
            result = input.data + adv
            result = torch.clamp(result, 0, 1.0)
            input_var.data = result

        adv_image = result.cpu().numpy()
        #_ = _get_diff_img(adv_image, orig_images) # check max diff
        save_images(adv_image, get_filenames_batch(filenames, batch_idx, args.batch_size), args.output_dir)

    elapsed_time = time.time() - start_time
    print('elapsed time: {0:.0f} [s]'.format(elapsed_time))


if __name__ == '__main__':
    main()
