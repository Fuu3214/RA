from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
from functools import partial
import json
import traceback

import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl
import utils
import loss_func


# ==============================================================================
# =                                    param                                   =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', dest='epoch', type=int, default=50)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--z_dim_d', dest='z_dim_d', type=int, default=16, help='dimension of latent')
parser.add_argument('--z_dim_e', dest='z_dim_e', type=int, default=32, help='dimension of latent')
parser.add_argument('--beta', dest='beta', type=float, default=1)
parser.add_argument('--dataset', dest='dataset_name', default='mnist', choices=['mnist', 'celeba'])
parser.add_argument('--model', dest='model_name', default='mlp_mnist', choices=['mlp_mnist'])
parser.add_argument('--experiment_name', dest='experiment_name', default="test")

args = parser.parse_args()

epoch = args.epoch
batch_size = args.batch_size
lr = args.lr
z_dim_d = args.z_dim_d
z_dim_e = args.z_dim_e
z_dim = z_dim_d + z_dim_e
beta = args.beta

dataset_name = args.dataset_name
model_name = args.model_name
experiment_name = args.experiment_name

pylib.mkdir('./output/%s' % experiment_name)
with open('./output/%s/setting.txt' % experiment_name, 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))


# dataset and models
Dataset, img_shape, get_imgs = utils.get_dataset(dataset_name)
dataset = Dataset(batch_size=batch_size)
dataset_val = Dataset(batch_size=100)
Enc, Dec = utils.get_models(model_name)
DSN = utils.get_models("dsn")

Enc_D = partial(Enc, z_dim=z_dim_d, name="Enc_D")
Enc_E = partial(Enc, z_dim=z_dim_e, name="Enc_E")
Dec = partial(Dec, channels=img_shape[2])
DSN = partial(DSN, enc_d=Enc_D, enc_e=Enc_E, dec_shared=Dec)

# ==============================================================================
# =                                    graph                                   =
# ==============================================================================

# input
img = tf.placeholder(tf.float32, [None] + img_shape)
z_sample = tf.placeholder(tf.float32, [None, z_dim])

# encode & decode
z_d_mu, z_d_log_sigma_sq, z_e_mu, z_e_log_sigma_sq, z_d, z_e, img_rec = DSN(img)

# loss
rec_loss = tf.losses.mean_squared_error(img, img_rec)
kld_loss_d = -loss_func.kl_loss(z_d_mu, z_d_log_sigma_sq)
kld_loss_e = -loss_func.kl_loss(z_e_mu, z_e_log_sigma_sq)
l_diff = loss_func.difference_loss(z_d, z_e)

# loss = rec_loss + kld_loss_d * beta1 + kld_loss_e * beta2 + l_diff

loss = 2 * rec_loss + kld_loss_d * 4 + kld_loss_e * 1 + l_diff * 1

# otpim
step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(loss)

# summary
summary = tl.summary({rec_loss: 'rec_loss', kld_loss_d: 'kld_loss_d', kld_loss_e: 'kld_loss_e', l_diff: 'l_diff' })

# sample
_, _, _, _, _, _, img_rec_sample = DSN(img, is_training=False)
img_sample = Dec(z_sample, is_training=False)


# ==============================================================================
# =                                    train                                   =
# ==============================================================================

# session
sess = tl.session()

# saver
saver = tf.train.Saver(max_to_keep=1)

# summary writer
summary_writer = tf.summary.FileWriter('./output/%s/summaries' % experiment_name, sess.graph)

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
pylib.mkdir(ckpt_dir)
try:
    tl.load_checkpoint(ckpt_dir, sess)
except:
    sess.run(tf.global_variables_initializer())

# train
try:
    img_ipt_sample = get_imgs(dataset_val.get_next())
    z_ipt_sample = np.random.normal(size=[100, z_dim])

    it = -1
    for ep in range(epoch):
        dataset.reset()
        it_per_epoch = it_in_epoch if it != -1 else -1
        it_in_epoch = 0
        for batch in dataset:
            it += 1
            it_in_epoch += 1

            # batch data
            img_ipt = get_imgs(batch)

            # train D
            summary_opt, _ = sess.run([summary, step], feed_dict={img: img_ipt})
            summary_writer.add_summary(summary_opt, it)

            # display
            if (it + 1) % 1 == 0:
                print("\r Epoch: (%3d) (%5d/%5d)" % (ep, it_in_epoch, it_per_epoch), end="")

            # sample
            if (it + 1) % 1000 == 0:
                save_dir = './output/%s/sample_training' % experiment_name
                pylib.mkdir(save_dir)

                img_rec_opt_sample = sess.run(img_rec_sample, feed_dict={img: img_ipt_sample})
                ipt_rec = np.concatenate((img_ipt_sample, img_rec_opt_sample), axis=2).squeeze()
                img_opt_sample = sess.run(img_sample, feed_dict={z_sample: z_ipt_sample}).squeeze()

                im.imwrite(im.immerge(ipt_rec, padding=img_shape[0] // 8), '%s/Epoch_(%d)_(%dof%d)_img_rec.jpg' % (save_dir, ep, it_in_epoch, it_per_epoch))
                im.imwrite(im.immerge(img_opt_sample), '%s/Epoch_(%d)_(%dof%d)_img_sample.jpg' % (save_dir, ep, it_in_epoch, it_per_epoch))

        save_path = saver.save(sess, '%s/Epoch_%d.ckpt' % (ckpt_dir, ep))
        print('\nModel is saved in file: %s' % save_path)
except:
    traceback.print_exc()
finally:
    sess.close()
