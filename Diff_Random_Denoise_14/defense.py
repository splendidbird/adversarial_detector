"""Implementation of sample defense.

This defense loads inception resnet v2 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import time
import random

import numpy as np
from scipy.misc import imread

# packages for random padding
import tensorflow as tf
import inception_resnet_v2
slim = tf.contrib.slim

# packages for denoise
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

from watchdog.observers import Observer
from watchdog.events import *
from category.category import CategoryHelper


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory to save labels.')

tf.flags.DEFINE_integer(
    'itr_time', 30, 'Time of iteration.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'image_resize', 331, 'Resize of image size.')

FLAGS = tf.flags.FLAGS



def padding_layer_iyswim(inputs, shape, name=None):
    h_start = shape[0]
    w_start = shape[1]
    output_short = shape[2]
    input_shape = tf.shape(inputs)
    input_short = tf.reduce_min(input_shape[1:3])
    input_long = tf.reduce_max(input_shape[1:3])
    output_long = tf.to_int32(tf.ceil(
        1. * tf.to_float(output_short) * tf.to_float(input_long) / tf.to_float(input_short)))
    output_height = tf.to_int32(input_shape[1] >= input_shape[2]) * output_long +\
        tf.to_int32(input_shape[1] < input_shape[2]) * output_short
    output_width = tf.to_int32(input_shape[1] >= input_shape[2]) * output_short +\
        tf.to_int32(input_shape[1] < input_shape[2]) * output_long
    return tf.pad(inputs, tf.to_int32(tf.stack([[0, 0], [h_start, output_height - h_start - input_shape[1]], [w_start, output_width - w_start - input_shape[2]], [0, 0]])), name=name)


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

def load_one_image(src_file, batch_shape):
    """Read one png image from input directory in batches.

    Args:
      src_file: new image file
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    with tf.gfile.Open(src_file) as f:
        image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(src_file))
    idx += 1
    if idx == batch_size:
        yield filenames, images
        filenames = []
        images = np.zeros(batch_shape)
        idx = 0
    if idx > 0:
        yield filenames, images


class FileEventHandler(FileSystemEventHandler):
    def __init__(self, batch_shape, sess, end_points, x_input, img_resize_tensor, shape_tensor, output_dir, itr, img_resize, net1, net4):
        FileSystemEventHandler.__init__(self)
        self._batch_shape = batch_shape
        self._sess = sess
        self._end_points = end_points
        self._x_input = x_input
        self._img_resize_tensor = img_resize_tensor
        self._shape_tensor = shape_tensor
        self._output_dir = output_dir
        self._itr = itr
        self._img_resize = img_resize        
        self._net1 = net1   
        self._net4 = net4
        self._category_helper = CategoryHelper("category/categories.csv")

    def on_moved(self, event):
        if event.is_directory:
            print("directory moved from {0} to {1}".format(event.src_path, event.dest_path))
        else:
            print("file moved from {0} to {1}".format(event.src_path, event.dest_path))

    def _defense_for_img_created(self, img_file):
        """ defense one image: xxx.png,
            write res to xxx.txt with two line(lable human_string),
            copy the src image file to output dir then delete it
        :param img_file:
        :return None:
        """
        if img_file.endswith('.png'):
            start_time = time.time() 
            output_file_name = ""
            
            #random padding
            print('Classifying with random padding...')
            labels_random = {}
            for filenames, images in load_one_image(img_file, self._batch_shape):
                final_preds = np.zeros([self._batch_shape[0], 1001, self._itr])
                for j in range(self._itr):
                    if np.random.randint(0, 2, size=1) == 1:
                        images = images[:, :, ::-1, :]
                    resize_shape_ = np.random.randint(310, 331)
                    pred, aux_pred = self._sess.run([self._end_points['Predictions'], self._end_points['AuxPredictions']],
                                                    feed_dict={self._x_input: images, self._img_resize_tensor: [resize_shape_]*2,
                                                               self._shape_tensor: np.array([random.randint(0, self._img_resize - resize_shape_), random.randint(0, self._img_resize - resize_shape_), self._img_resize])})
                    final_preds[..., j] = pred + 0.4 * aux_pred
                final_probs = np.sum(final_preds, axis=-1)
                random_labels = np.argmax(final_probs, 1)
                labels_random.update(dict(zip(filenames, random_labels)))
                print(labels_random)
            
            for filename, label in zip(filenames, random_labels):
                res_file_name = os.path.basename(filename)[:-4] + '.txt'
                output_file_name = os.path.join(self._output_dir, os.path.basename(filename))
                print("res_file_name: " + res_file_name)
                with open(os.path.join(self._output_dir, res_file_name), 'w+') as res_file:
                    res_file.write('{0}\n{1}\n'.format(label,
                                                       self._category_helper.get_category_name(label)))
                    res_file.flush()

            elapsed_time = time.time() - start_time
            print('elapsed time: {0:.00f} [s]'.format(elapsed_time))
            if os.path.exists(output_file_name):
                os.remove(output_file_name)
            shutil.copy(img_file, output_file_name)
            os.remove(img_file)
    
    def _defense_for_img_deleted(self, img_file):
        """ double check one image deleted with another classifier: xxx.png,
            if agree doing nothing
            if disagree, re-write the xxx.txt with label updated to 0 or random number
        :param img_file:
        :return None:
        """

        if img_file.endswith('.png'):
            start_time = time.time() 
            # since the file has been moved to output folder
            # file name needs update
            filename = os.path.basename(img_file)
            img_file = os.path.join(self._output_dir, filename)
            print(img_file)
            start_time = time.time() 
            output_file_name = ""

            
            # denoise
            print('Classifying with denoise...')
            transf = transforms.Compose([
                transforms.Resize([299,299]),
                transforms.ToTensor()
            ])
            with torch.no_grad():
                mean_torch = autograd.Variable(torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1,3,1,1]).astype('float32')).cuda())
                std_torch = autograd.Variable(torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1,3,1,1]).astype('float32')).cuda())
                mean_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).cuda())
                std_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).cuda())
                dataset = Dataset2(img_file, transform=transf)
                loader = data.DataLoader(dataset, batch_size=self._batch_shape[0], shuffle=False)
                net1 = self._net1
                net4 = self._net4
            labels_denoise = {}
            outputs = []
            for batch_idx, input in enumerate(loader):
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
            filenames = [os.path.basename(img_file)]
            #filenames = dataset.filenames()
            print(filenames)
            #print(zip(filenames, outputs))
            labels_denoise.update(dict(zip(filenames, outputs))) 
            print(labels_denoise)


            elapsed_time = time.time() - start_time
            print('elapsed time: {0:.00f} [s]'.format(elapsed_time))
            if False:
                res_file_name = os.path.basename(img_file)[:-4] + '.txt'
                if os.path.exists(res_file_name):
                    os.remove(res_file_name)
                    update_label = 0
                    with open(os.path.join(self._output_dir, res_file_name), 'w+') as res_file:
                        res_file.write('{0}\n{1}\n'.format(update_label,
                                                           self._category_helper.get_category_name(update_label)))
                        res_file.flush()


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
            self._defense_for_img_deleted(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            print("directory modified:{0}".format(event.src_path))
        else:
            print("file modified:{0}".format(event.src_path))


def main(_):
    print('Loading denoise...')
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

    inresmodel = inresmodel.cuda()
    resmodel = resmodel.cuda()
    incepv3model = incepv3model.cuda()
    rexmodel = rexmodel.cuda()
    inresmodel.eval()
    resmodel.eval()
    incepv3model.eval()
    rexmodel.eval()

    # random padding
    print('Loading random padding...')
    print('Iteration: %d' % FLAGS.itr_time)
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        img_resize_tensor = tf.placeholder(tf.int32, [2])
        x_input_resize = tf.image.resize_images(x_input, img_resize_tensor, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        shape_tensor = tf.placeholder(tf.int32, [3])
        padded_input = padding_layer_iyswim(x_input_resize, shape_tensor)
        # 330 is the last value to keep 8*8 output, 362 is the last value to keep 9*9 output, stride = 32
        padded_input.set_shape(
            (FLAGS.batch_size, FLAGS.image_resize, FLAGS.image_resize, 3))

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2.inception_resnet_v2(
                padded_input, num_classes=num_classes, is_training=False, create_aux_logits=True)

        predicted_labels = tf.argmax(end_points['Predictions'], 1)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=FLAGS.checkpoint_path,
            master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            ''' watch the input dir for defense '''
            observer = Observer()
            event_handler = FileEventHandler(batch_shape=batch_shape,
                                             sess=sess,
                                             end_points=end_points,
                                             x_input=x_input,
                                             img_resize_tensor=img_resize_tensor,
                                             shape_tensor=shape_tensor,
                                             output_dir=FLAGS.output_dir,
                                             itr=FLAGS.itr_time,
                                             img_resize=FLAGS.image_resize,
                                             net1=net1,
                                             net4=net4)

            observer.schedule(event_handler, FLAGS.input_dir, recursive=True)
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
    tf.app.run()
