from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils




# DIFFERENCE LOSS
def difference_loss(z_d, z_e, weight=1.0, name=''):
    z_d -= tf.reduce_mean(z_d, 0)
    z_e -= tf.reduce_mean(z_e, 0)
    z_d = tf.nn.l2_normalize(z_d, 1)
    z_e = tf.nn.l2_normalize(z_e, 1)
    correlation_matrix = tf.matmul( z_d, z_e, transpose_a=True)
    cost = tf.reduce_mean(tf.square(correlation_matrix)) * weight
    cost = tf.where(cost > 0, cost, 0, name='value')
    #tf.summary.scalar('losses/Difference Loss {}'.format(name),cost)
    assert_op = tf.Assert(tf.is_finite(cost), [cost])
    with tf.control_dependencies([assert_op]):
        tf.losses.add_loss(cost)
    return cost

def kl_loss(mu, log_sigma_sq):
    return tf.reduce_mean(0.5 * (1 + log_sigma_sq - mu**2 - tf.exp(log_sigma_sq)))



def mmd_loss(source_samples, target_samples, weight = 1, scope=None):
  """Adds a similarity loss term, the MMD between two representations.
  This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
  different Gaussian kernels.
  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the MMD loss.
    scope: optional name scope for summary tags.
  Returns:
    a scalar tensor representing the MMD loss value.
  """
  sigmas = [
      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
      1e3, 1e4, 1e5, 1e6
  ]
  gaussian_kernel = partial(
      utils.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

  loss_value = maximum_mean_discrepancy(
      source_samples, target_samples, kernel=gaussian_kernel)
  loss_value = tf.maximum(1e-4, loss_value) * weight
  # loss_value = tf.where(tf.equal(source_labels, target_labels), loss_value, 0)

  assert_op = tf.Assert(tf.is_finite(loss_value), [loss_value])
  with tf.control_dependencies([assert_op]):
    tag = 'MMD Loss'
    if scope:
      tag = scope + tag
    tf.losses.add_loss(loss_value)

  return loss_value

def maximum_mean_discrepancy(x, y, kernel=utils.gaussian_kernel_matrix):
  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.
  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.
  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost
