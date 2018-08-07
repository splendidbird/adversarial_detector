"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import time

import numpy as np

from scipy.misc import imread

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import inception_resnet_v2
from io import BytesIO
from shutil import copy

import PIL
import PIL.Image

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
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_float(
     'iter_alpha', 1.0, 'Step size for one iteration.')

tf.flags.DEFINE_float(
     'alpha_factor', 1.0, 'Step size for one iteration.')

tf.flags.DEFINE_integer(
    'ensemble_size', 20, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'num_iter', 120, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
     'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS

ITER_NUM = FLAGS.num_iter
EOT = FLAGS.ensemble_size  # larger ensemble size -> better
ALPHA_FACTOR=FLAGS.alpha_factor

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

    filepaths = tf.gfile.Glob(os.path.join(input_dir, '*.png'))

    for count, filepath in enumerate(filepaths):
        defenseID = filepath.split('/')[-1][:-4].split('_')[-1]        
        print('!!!!!!! defenseID = {0}'.format(defenseID))
        if int(defenseID) not in _targetAttackChosenDiction['JpegEOT']:
            continue
        with tf.gfile.Open(filepath, 'rb') as f:
            image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
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
        filename = 'JpegBGPD'+ get_file_config() + filename + r'.png'
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            # img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
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
                x_input, num_classes=self.num_classes, is_training=False, reuse=reuse, scope=self.scope)

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
                x_input, num_classes=self.num_classes, reuse=reuse, is_training=False, scope=self.scope)

        self.built = True
        return logits, end_points

def convert_jpeg(image):

    pil_image = PIL.Image.fromarray((image * 255.0).astype('uint8'), mode='RGB')
    f = BytesIO()
    pil_image.save(f, format='jpeg', quality=75)  # quality level specified in paper
    jpeg_image = (np.asarray(PIL.Image.open(f)).astype(np.float32) / 255.0) * 2 - 1

    return jpeg_image

def defend_jpeg(input_array):
    """
    jpeg denoise
    :param input_array: numpy array [batch_size, heigth, width, 3]
    :return: denoised numpyt array [batch_size, heigth, width, 3]
    """
    input_array = (input_array + 1) / 2
    converted_images_list = []
    for image in input_array:
        image = [convert_jpeg(image)]
        converted_images_list.append(image)
    return np.concatenate(converted_images_list)

def main(_):

    start_time = time.time()

    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0

    #alpha = 0.1 * alpha_factor
    alpha = 0.1 * ALPHA_FACTOR 

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    # gradient smoothing
    # FLAGS.max_epsilon  4: sig = 1000 (no blur)
    # FLAGS.max_epsilon  8: sig = 12
    # FLAGS.max_epsilon 12: sig = 8
    # FLAGS.max_epsilon 16: sig = 4

    if FLAGS.max_epsilon <= 4:
        sig = 1000
    elif FLAGS.max_epsilon <= 8:
        sig = 12
    elif FLAGS.max_epsilon <= 12:
        sig = 8
    else:
        sig = 4

    print('MAX_EPSILON: {0:f} sig = {1:d}'.format(FLAGS.max_epsilon, sig))

    tf.logging.set_verbosity(tf.logging.INFO)

    all_images_taget_class = load_target_class(FLAGS.input_dir)

    with tf.Graph().as_default():

        # ---------------------------------
        # define graph

        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        l2_x = tf.placeholder(tf.float32, shape=batch_shape)
        l2_orig = tf.placeholder(tf.float32, shape=batch_shape)
        normalized_l2_loss = tf.nn.l2_loss(l2_orig - l2_x) / tf.nn.l2_loss(l2_orig)

        model3 = IrNetModel(num_classes, scope='sc3')

        x_max = tf.clip_by_value(l2_orig + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(l2_orig - eps, -1.0, 1.0)

        x_adv = x_input

        target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        one_hot_target_class = tf.one_hot(target_class_input, num_classes)

        logits_cs3, end_points_cs3 = model3(x_adv)

        cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class, logits_cs3,
                                                         label_smoothing=0.1, weights=1.0)
        cross_entropy += tf.losses.softmax_cross_entropy(one_hot_target_class, end_points_cs3['AuxLogits'], label_smoothing=0.1, weights=0.4)


        logits = logits_cs3 + 0.4 * end_points_cs3['AuxLogits']
        preds = tf.argmax(logits, axis = 1)
        lam = tf.placeholder(tf.float32, ())
        loss = cross_entropy + lam * normalized_l2_loss

        # get gradient
        grad = tf.gradients(loss, x_adv)[0]

        #gradient smooth

        # update
        x_next = x_adv - alpha * tf.sign(grad)

        # clip
        x_next = tf.clip_by_value(x_next, x_min, x_max)

        x_adv = x_next

        # ---------------------------------
        # set input

        all_vars = tf.global_variables()

        model3_vars = [k for k in all_vars if k.name.startswith('sc3')]

        # name of variable `my_var:0` corresponds `my_var` for loader
        model3_keys = [s.name.replace('sc3', 'InceptionResnetV2')[:-2] for s in model3_vars]

        saver3 = tf.train.Saver(dict(zip(model3_keys, model3_vars)))

        session_creator = tf.train.ChiefSessionCreator(master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            saver3.restore(sess, FLAGS.checkpoint_path3)
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                target_class_for_batch = (
                        [all_images_taget_class[n] for n in filenames] + [0] * (FLAGS.batch_size - len(filenames)))
                adv_images = images
                for i in range(ITER_NUM):
                    adv_def_images = defend_jpeg(adv_images)  ###### change here for other defend, if the defend is map from numpy array to numpy array
                    [adv_images,  prediction]= sess.run([x_adv, preds], feed_dict={x_input: adv_def_images, target_class_input: target_class_for_batch, lam: 1.0, l2_x: adv_images, l2_orig: images})
                                            
                save_images(adv_images, filenames, FLAGS.output_dir)

    elapsed_time = time.time() - start_time
    print('elapsed time: {0:.0f} [s]'.format(elapsed_time))

if __name__ == '__main__':
    tf.app.run()
