"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import time
import random
import numpy as np

from scipy.misc import imread
from shutil import copy

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import inception_resnet_v2
from util import gkern
from PIL import Image

import sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)
from configTargetedAttackChosenList import _targetAttackChosenDiction

slim = tf.contrib.slim

tf.flags.DEFINE_string(
     'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path1', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path2', '', 'Path to checkpoint for adversarial trained inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path3', '', 'Path to checkpoint for adversarial trained inception-resnet network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
     'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_string(
     'output_attack_dir', r'../../../adv_images', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 8.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_float(
     'iter_alpha', 1.0, 'Step size for one iteration.')

tf.flags.DEFINE_float(
     'alpha_factor', 1.0, 'LR = LR * alpha_factor.')

tf.flags.DEFINE_integer(
    'num_iter', 200, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
     'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'debug', 1, 'debug flag')

tf.flags.DEFINE_integer(
    'ensemble_size', 500, 'how many times to tile the input tensor')

FLAGS = tf.flags.FLAGS

DEBUG = FLAGS.debug == 1
EOT = FLAGS.ensemble_size  # larger ensemble size -> better
ALPHA_FACTOR=FLAGS.alpha_factor
ITER_NUM = FLAGS.num_iter

def get_file_config():
    fileConfig = '_'
    fileConfig = fileConfig + 'EOT-' + str(EOT) + '_' + 'iterNum-'+ str(ITER_NUM) + '_' + 'alphaFactor-' + str(ALPHA_FACTOR) + '_' + str(int(FLAGS.max_epsilon)) + '_'
    return fileConfig

def load_target_class(input_dir):
    """Loads target classes."""
    with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:
        return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}


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
    print('_targetAttackChosenDiction = {0}'.format(_targetAttackChosenDiction))
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        # if filepath not in dict['filename']:
              # pass
        defenseID = filepath.split('/')[-1][:-4].split('_')[-1]        
        print('!!!!!!! defenseID = {0}'.format(defenseID))
        if int(defenseID) not in _targetAttackChosenDiction['CropEOT']:
            continue
        with tf.gfile.Open(filepath, 'rb') as f:
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


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].

        filename = filename[:-4].split(r'_')[-1]
        filename = 'ToshiKCrop'+ get_file_config() + filename + r'.png'
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            # imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')
            img = np.round(255.0 * (images[i, :, :, :] + 1.0) * 0.5).astype(np.uint8)
            #img = np.round(255.0 * (images[i, :, :, :] )).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')
        copy(os.path.join(output_dir, filename), os.path.join(FLAGS.output_attack_dir, filename))

class InceptionModel(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes, scope=''):
        self.num_classes = num_classes
        self.built = False
        self.scope = scope

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(
                x_input, num_classes=self.num_classes, is_training=False, reuse=tf.AUTO_REUSE, scope=self.scope)

        self.built = True
        return logits, end_points


class IrNetModel(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes, scope=''):
        self.num_classes = num_classes
        self.built = False
        self.scope = scope

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=self.num_classes, reuse=tf.AUTO_REUSE, is_training=False, scope=self.scope, create_aux_logits = False)
        #create_aux_logits=True, reuse=tf.AUTO_REUSE
        self.built = True
        return logits, end_points

# input_tensor should be of shape [1, 299, 299, 3]
# output is of shape [1, 331, 331, 3]

def defend_crop(x, crop_size=90, ensemble_size=30):
    x_size = tf.to_float(x.shape[1])
    frac = crop_size/x_size
    start_fraction_max = (x_size - crop_size)/x_size
    def randomizing_crop(x):
        start_x = tf.random_uniform((), 0, start_fraction_max)
        start_y = tf.random_uniform((), 0, start_fraction_max)
        return tf.image.crop_and_resize(x, boxes=[[start_y, start_x, start_y+frac, start_x+frac]],
                                 box_ind=[0], crop_size=[crop_size, crop_size])

    return tf.concat([randomizing_crop(x) for _ in range(ensemble_size)], axis=0)

def main(_):

    start_time = time.time()
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    LR = 0.2 * ALPHA_FACTOR

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    if FLAGS.max_epsilon <= 4:
        sig = 1000
    elif FLAGS.max_epsilon <= 8:
        sig = 12
    elif FLAGS.max_epsilon <= 12:
        sig = 8
    else:
        sig = 4

    print('MAX_EPSILON: {0:f} sig = {1:d}'.format(FLAGS.max_epsilon, sig))

    tf.logging.set_verbosity(tf.logging.ERROR)

    all_images_taget_class = load_target_class(FLAGS.input_dir)

    with tf.Graph().as_default():

        # ---------------------------------
        # define graph

        x_expanded = tf.placeholder(tf.float32, shape=batch_shape)

        cropped_xs = defend_crop(x_expanded, ensemble_size = EOT)

        l2_x = tf.placeholder(tf.float32, shape=batch_shape)
        l2_orig = tf.placeholder(tf.float32, shape=batch_shape)
        normalized_l2_loss = tf.nn.l2_loss(l2_orig - l2_x) / tf.nn.l2_loss(l2_orig)
        # Prepare graph
        # build computational graph
        #model1 = InceptionModel(num_classes, scope='sc1')
        #model2 = InceptionModel(num_classes, scope='sc2')
        model3 = IrNetModel(num_classes, scope='sc3')

        target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        one_hot_target_class = tf.one_hot(target_class_input, num_classes)

        #ensemble_logits_cs1, ensemble_end_points_cs1 = model1(ensemble_xs)
        #ensemble_logits_cs2, ensemble_end_points_cs2 = model2(ensemble_xs)
        cropped_logits, crop_end_points = model3(cropped_xs)

        if DEBUG:
            cropped_probs = tf.reduce_mean(tf.nn.softmax(cropped_logits), axis=0, keep_dims=True)
            cropped_preds = tf.argmax(cropped_probs, 1)

        cropped_labels = tf.tile(one_hot_target_class, (cropped_logits.shape[0], 1))
        #tf.losses.softmax_cross_entropy(ensemble_labels, ensemble_logits_cs3, label_smoothing=0.1, weights=1.0)
        cropped_xent = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=cropped_logits, onehot_labels=cropped_labels))

        lam = tf.placeholder(tf.float32, ())
        cropped_loss = cropped_xent + lam * normalized_l2_loss

        cropped_grad, = tf.gradients(cropped_loss, x_expanded)
        """
        # apply gradient smoothing
        kernel = gkern(7, sig).astype(np.float32)
        stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
        stack_kernel = np.expand_dims(stack_kernel, 3)
        ensemble_grad = tf.nn.depthwise_conv2d(ensemble_grad, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
        """
        # ---------------------------------
        # set input
        all_vars = tf.global_variables()
        #model1_vars = [k for k in all_vars if k.name.startswith('sc1')]
        #model2_vars = [k for k in all_vars if k.name.startswith('sc2')]
        model3_vars = [k for k in all_vars if k.name.startswith('sc3')]

        # name of variable `my_var:0` corresponds `my_var` for loader
        #model1_keys = [s.name.replace('sc1', 'InceptionV3')[:-2] for s in model1_vars]
        #model2_keys = [s.name.replace('sc2', 'InceptionV3')[:-2] for s in model2_vars]
        model3_keys = [s.name.replace('sc3', 'InceptionResnetV2')[:-2] for s in model3_vars]

        #saver1 = tf.train.Saver(dict(zip(model1_keys, model1_vars)))
        #saver2 = tf.train.Saver(dict(zip(model2_keys, model2_vars)))
        saver3 = tf.train.Saver(dict(zip(model3_keys, model3_vars)))

        session_creator = tf.train.ChiefSessionCreator(master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:

            #saver1.restore(sess, FLAGS.checkpoint_path1)
            #saver2.restore(sess, FLAGS.checkpoint_path2)
            saver3.restore(sess, FLAGS.checkpoint_path3)

            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                print(filenames)
                target_class_for_batch = (
                        [all_images_taget_class[n] for n in filenames] + [0] * (FLAGS.batch_size - len(filenames)))

                adv = np.copy(images)
                lower = np.clip(images - eps, -1, 1) # doubt here
                upper = np.clip(images + eps, -1, 1) # doubt here
                #print('number of iterations = {0:d}'.format(num_iter))
                for i in range(ITER_NUM):
                    if DEBUG:
                        g, l2, p = sess.run([cropped_grad, normalized_l2_loss, cropped_preds], {x_expanded: adv, target_class_input: target_class_for_batch, lam: 1.0, l2_x: adv, l2_orig: images})
                        #g, p = sess.run([ensemble_grad, ensemble_preds], {x_input: adv, target_class_input: target_class_for_batch, lam: 1.0, l2_x: adv, l2_orig:images})
                        if i % 20 == 0:
                            print('step %d, l2=%f, preds=%s' % (i, l2, p))
                    else:
                        g = sess.run([cropped_grad], {x_expanded: adv, target_class_input: target_class_for_batch, lam: 1.0, l2_x: adv, l2_orig: images})
                        #g = sess.run([ensemble_grad], {x_input: adv, target_class_input: target_class_for_batch, lam: 1.0, l2_x: adv_images, l2_orig:images})
                    # debug-----
                    #print('type g[0] =  {0}'.format(type(g)))
                    #print('g.shape =  {0}'.format(len(g)))
                    #print('adv.shape =  {0}'.format(adv.shape))
                    # raise ValueError('hold')
                    # step
                    adv -= LR * g
                    # project
                    adv = np.clip(adv, lower, upper)
                    adv = np.clip(adv, -1, 1)
                save_images(adv, filenames, FLAGS.output_dir)

    elapsed_time = time.time() - start_time
    print('elapsed time: {0:.0f} [s]'.format(elapsed_time))

if __name__ == '__main__':
    tf.app.run()
