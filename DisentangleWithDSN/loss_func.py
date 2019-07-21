import tensorflow as tf
import tensorflow.contrib.slim as slim



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
