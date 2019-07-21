from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflib as tl

conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(tl.flatten_fully_connected, activation_fn=None)
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)

def vae_mnist():
    def Enc(img, z_dim, dim=512, is_training=True):
        fc_relu = partial(fc, activation_fn=relu)

        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            y = fc_relu(img, dim)
            y = fc_relu(y, dim * 2)
            z_mu = fc(y, z_dim)
            z_log_sigma_sq = fc(y, z_dim)
            return z_mu, z_log_sigma_sq

    def Dec(z, dim=512, channels=1, is_training=True):
        fc_relu = partial(fc, activation_fn=relu)

        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):
            y = fc_relu(z, dim * 2)
            y = fc_relu(y, dim)
            y = tf.tanh(fc(y, 28 * 28 * channels))
            img = tf.reshape(y, [-1, 28, 28, channels])
            return img

    return Enc, Dec


def vae_enc_dec(img, enc, dec, is_training=True):
    # encode
    z_mu, z_log_sigma_sq = enc(img, is_training=is_training)

    # sample
    epsilon = tf.random_normal(tf.shape(z_mu))
    if is_training:
        z = z_mu + tf.exp(0.5 * z_log_sigma_sq) * epsilon
    else:
        z = z_mu

    # decode
    img_rec = dec(z, is_training=is_training)

    return z_mu, z_log_sigma_sq, img_rec 