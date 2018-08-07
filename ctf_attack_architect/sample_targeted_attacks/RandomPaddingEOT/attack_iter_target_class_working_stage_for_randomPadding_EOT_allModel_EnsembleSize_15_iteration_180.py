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

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import inception_resnet_v2
from util import gkern
from shutil import copy

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
    'output_attack_dir', r'../../../adv_images/', 'Output directory with images')

tf.flags.DEFINE_float(
    'max_epsilon', 32.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_float(
     'iter_alpha', 1.0, 'Step size for one iteration.')

tf.flags.DEFINE_integer(
    'num_iter', 120, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
     'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
     'resize_size', 331, 'resize size of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'debug', 1, 'debug flag')

tf.flags.DEFINE_integer(
    'EOT', 15, 'how many times to tile the input tensor')

tf.flags.DEFINE_float(
     'alpha_factor', 1.0, 'LR = 0.1 * alpha_factor.')

FLAGS = tf.flags.FLAGS

DEBUG = FLAGS.debug == 1

ITER_NUM = FLAGS.num_iter
EOT = FLAGS.EOT  # larger ensemble size -> better
ALPHA_FACTOR = FLAGS.alpha_factor

num_iter = ITER_NUM

def load_target_class(input_dir):
    """Loads target classes."""
    with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:
        return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}
def get_file_config():
    fileConfig = '_'
    fileConfig = fileConfig + 'EOT-' + str(EOT) + '_' + 'iterNum-'+ str(ITER_NUM) + '_' + 'alphaFactor-' + str(ALPHA_FACTOR) + '_' + str(int(FLAGS.max_epsilon)) + '_'
    return fileConfig

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
        defenseID = filepath.split('/')[-1][:-4].split('_')[-1]        
        print('!!!!!!! defenseID = {0}'.format(defenseID))
        if int(defenseID) not in _targetAttackChosenDiction['RandomPaddingEOT']:
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
        filename = 'ToshikRP'+ get_file_config() + filename + r'.png'
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            # imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')
            img = np.round(255.0 * (images[i, :, :, :] + 1.0) * 0.5).astype(np.uint8)
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
                x_input, num_classes=self.num_classes, reuse=tf.AUTO_REUSE, is_training=False, scope=self.scope, create_aux_logits = True)
        #create_aux_logits=True, reuse=tf.AUTO_REUSE
        self.built = True
        return logits, end_points

PAD_VALUE = 0.5

# input_tensor should be of shape [1, 299, 299, 3]
# output is of shape [1, 331, 331, 3]
def defend(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.resize_size, dtype=tf.int32)
    rescaled = tf.image.crop_and_resize(input_tensor, [[0, 0, 1, 1]], [0], [rnd, rnd])
    h_rem = FLAGS.resize_size - rnd
    w_rem = FLAGS.resize_size - rnd
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=PAD_VALUE)
    padded.set_shape((FLAGS.batch_size, FLAGS.resize_size, FLAGS.resize_size, 3))
    return padded


def padding_layer_iyswim(inputs, shape, name=None):
    """

    :param inputs: input tf tensor
    :param shape:
    :param name:
    :return: padded tf tensor
    """
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


def main(_):

    start_time = time.time()
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0

    LR = 0.1 * ALPHA_FACTOR

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

        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_adv = x_input
        # Prepare graph
        img_resize_tensor = tf.placeholder(tf.int32, [2])
        x_input_resize = tf.image.resize_images(x_adv, img_resize_tensor, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        shape_tensor = tf.placeholder(tf.int32, [3])
        x_adv_padded = padding_layer_iyswim(x_input_resize, shape_tensor)
        # 330 is the last value to keep 8*8 output, 362 is the last value to keep 9*9 output, stride = 32
        x_adv_padded.set_shape(
            (FLAGS.batch_size, FLAGS.resize_size, FLAGS.resize_size, 3))

        # build computational graph
        model1 = InceptionModel(num_classes, scope='sc1')
        model2 = InceptionModel(num_classes, scope='sc2')
        model3 = IrNetModel(num_classes, scope='sc3')

        target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        one_hot_target_class = tf.one_hot(target_class_input, num_classes)
        ensemble_xs = tf.concat([x_adv_padded for _ in range(EOT)], axis=0)

        ensemble_logits_cs1, ensemble_end_points_cs1 = model1(ensemble_xs)
        ensemble_logits_cs2, ensemble_end_points_cs2 = model2(ensemble_xs)
        ensemble_logits_cs3, ensemble_end_points_cs3 = model3(ensemble_xs)

        ensemble_logits = ensemble_logits_cs1
        ensemble_logits += 0.4 * ensemble_end_points_cs1['AuxLogits']
        ensemble_logits += ensemble_logits_cs2
        ensemble_logits += 0.4 * ensemble_end_points_cs2['AuxLogits']
        ensemble_logits += ensemble_logits_cs3
        ensemble_logits += 0.4 * ensemble_end_points_cs3['AuxLogits']

        if DEBUG:
            ensemble_preds = tf.argmax(ensemble_logits, 1)

        ensemble_labels = tf.tile(one_hot_target_class, (ensemble_logits.shape[0], 1))

        ensemble_cross_entropy = tf.losses.softmax_cross_entropy(ensemble_labels, ensemble_logits_cs1,
                                                                 label_smoothing=0.1, weights=1.0)
        ensemble_cross_entropy += tf.losses.softmax_cross_entropy(ensemble_labels, ensemble_end_points_cs1['AuxLogits'],
                                                                  label_smoothing=0.1, weights=0.4)

        ensemble_cross_entropy += tf.losses.softmax_cross_entropy(ensemble_labels, ensemble_logits_cs2,
                                                                  label_smoothing=0.1, weights=1.0)
        ensemble_cross_entropy += tf.losses.softmax_cross_entropy(ensemble_labels, ensemble_end_points_cs2['AuxLogits'],
                                                                  label_smoothing=0.1, weights=0.4)

        ensemble_cross_entropy += tf.losses.softmax_cross_entropy(ensemble_labels, ensemble_logits_cs3,
                                                                  label_smoothing=0.1, weights=1.0)
        ensemble_cross_entropy += tf.losses.softmax_cross_entropy(ensemble_labels, ensemble_end_points_cs3['AuxLogits'],
                                                                  label_smoothing=0.1, weights=0.4)

        ensemble_loss = tf.reduce_mean(ensemble_cross_entropy)
        ensemble_grad, = tf.gradients(ensemble_loss, x_adv)
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

        model1_vars = [k for k in all_vars if k.name.startswith('sc1')]
        model2_vars = [k for k in all_vars if k.name.startswith('sc2')]
        model3_vars = [k for k in all_vars if k.name.startswith('sc3')]

        # name of variable `my_var:0` corresponds `my_var` for loader
        model1_keys = [s.name.replace('sc1', 'InceptionV3')[:-2] for s in model1_vars]
        model2_keys = [s.name.replace('sc2', 'InceptionV3')[:-2] for s in model2_vars]
        model3_keys = [s.name.replace('sc3', 'InceptionResnetV2')[:-2] for s in model3_vars]

        saver1 = tf.train.Saver(dict(zip(model1_keys, model1_vars)))
        saver2 = tf.train.Saver(dict(zip(model2_keys, model2_vars)))
        saver3 = tf.train.Saver(dict(zip(model3_keys, model3_vars)))

        session_creator = tf.train.ChiefSessionCreator(master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:

            saver1.restore(sess, FLAGS.checkpoint_path1)
            saver2.restore(sess, FLAGS.checkpoint_path2)
            saver3.restore(sess, FLAGS.checkpoint_path3)

            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                target_class_for_batch = (
                        [all_images_taget_class[n] for n in filenames] + [0] * (FLAGS.batch_size - len(filenames)))

                adv = np.copy(images)
                lower = np.clip(images - eps, -1, 1) # doubt here
                upper = np.clip(images + eps, -1, 1) # doubt here
                for i in range(num_iter):
                    resize_shape_ = np.random.randint(310, 331)

                    if DEBUG:
                        g, p = sess.run([ensemble_grad, ensemble_preds], {x_input: adv, target_class_input: target_class_for_batch, img_resize_tensor: [resize_shape_] * 2,
                                                          shape_tensor: np.array(
                                                              [random.randint(0, FLAGS.resize_size - resize_shape_),
                                                               random.randint(0, FLAGS.resize_size - resize_shape_),
                                                               FLAGS.resize_size])})
                        if i % 10 == 0:
                            print('step %d, preds=%s' % (i, p))
                    else:
                        g = sess.run([ensemble_grad], {x_input: adv, target_class_input: target_class_for_batch})
                    # debug-----
                    # print('g.shape =  {0}'.format(g.shape))
                    # print('adv.shape =  {0}'.format(adv.shape))
                    # raise ValueError('hold')
                    # step
                    adv -= LR * g
                    # project
                    adv = np.clip(adv, lower, upper)
                save_images(adv, filenames, FLAGS.output_dir)

    elapsed_time = time.time() - start_time
    print('elapsed time: {0:.0f} [s]'.format(elapsed_time))

if __name__ == '__main__':
    tf.app.run()
