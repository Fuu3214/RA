from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import glob as glob

import models
import pylib
import tensorflow as tf
import tflib as tl


def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        # dataset
        pylib.mkdir('./data/mnist')
        Dataset = partial(tl.Mnist, data_dir='./data/mnist', repeat=1)

        # shape
        img_shape = [28, 28, 1]

        # index func
        def get_imgs(batch):
            return batch['img']
        return Dataset, img_shape, get_imgs

    elif dataset_name == 'celeba':
        # dataset
        def _map_func(img):
            crop_size = 108
            re_size = 64
            img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
            img = tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            return img

        paths = glob.glob('./data/celeba/img_align_celeba/*.jpg')
        Dataset = partial(tl.DiskImageData, img_paths=paths, repeat=1, map_func=_map_func)

        # shape
        img_shape = [64, 64, 3]

        # index func
        def get_imgs(batch):
            return batch

        return Dataset, img_shape, get_imgs


def get_models(model_name):
    return getattr(models, model_name)()





def gaussian_kernel_matrix(x, y, sigmas):
  r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
  We create a sum of multiple gaussian kernels each having a width sigma_i.
  Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
      gaussians in the kernel.
  Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
  """
  beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

  dist = compute_pairwise_distances(x, y)

  s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

  return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

def compute_pairwise_distances(x, y):
  """Computes the squared pairwise Euclidean distances between x and y.
  Args:
    x: a tensor of shape [num_x_samples, num_features]
    y: a tensor of shape [num_y_samples, num_features]
  Returns:
    a distance matrix of dimensions [num_x_samples, num_y_samples].
  Raises:
    ValueError: if the inputs do no matched the specified dimensions.
  """

  if not len(x.get_shape()) == len(y.get_shape()) == 2:
    raise ValueError('Both inputs should be matrices.')

  if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
    raise ValueError('The number of features should be the same.')

  norm = lambda x: tf.reduce_sum(tf.square(x), 1)

  # By making the `inner' dimensions of the two matrices equal to 1 using
  # broadcasting then we are essentially substracting every pair of rows
  # of x and y.
  # x will be num_samples x num_features x 1,
  # and y will be 1 x num_features x num_samples (after broadcasting).
  # After the substraction we will get a
  # num_x_samples x num_features x num_y_samples matrix.
  # The resulting dist will be of shape num_y_samples x num_x_samples.
  # and thus we need to transpose it again.
  return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def permute(z_samples):
  real_samples = z_samples
  permuted_rows = []
  for i in range(real_samples.get_shape()[1]):
    permuted_rows.append(tf.random_shuffle(real_samples[:, i]))
    permuted_samples = tf.stack(permuted_rows, axis=1)
  return permuted_samples