from __future__ import division, absolute_import, print_function
from six.moves import range, zip

import os
import time
import numpy as np
import scipy.misc as sm
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.streams import DataStream
import tensorflow as tf

from utils import preproc

FLAGS = tf.app.flags.FLAGS


def reset_tmp(path_tmp):
    """
    If path_tmp doesn't exist, create it.
    Otherwise, delete all non-.h5 files in path_tmp.
    """
    # If path_tmp doesn't exist, create it
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)
    
    # Delete files in path_tmp
    for f in os.listdir(path_tmp):
        if f.split('.')[-1] == 'h5':  # skip data file
            continue
        fp = os.path.join(path_tmp, f)
        try:
            if os.path.isfile(fp):
                os.unlink(fp)
        except Exception, e:
            print(e)


def prepare_data(conf):
    """
    Extract strided crops from a set of images and assemble into a 2D matrix.
    Save into an HDF5 file.

    Args:
      conf: dictionary containing data parameters
    Returns:
      tr_stream: DataStream for training set
      te_stream: DataStream for testing set
    """
    preproc.store_hdf5(conf)#, compression='lzf')
    
    path_h5 = conf['path_h5']

    tr_set = H5PYDataset(path_h5, ('train',), sources=('LR', 'HR'),
                         load_in_memory=conf['load_in_memory'])
    tr_scheme = ShuffledScheme(examples=tr_set.num_examples,
                               batch_size=FLAGS.num_gpus * conf['mb_size'])
    tr_stream = DataStream(dataset=tr_set, iteration_scheme=tr_scheme)

    te_set = H5PYDataset(path_h5, ('test',), sources=('LR', 'HR'),
                         load_in_memory=conf['load_in_memory'])
    te_scheme = SequentialScheme(examples=te_set.num_examples,
                                 batch_size=FLAGS.num_gpus * conf['mb_size'])
    te_stream = DataStream(dataset=te_set, iteration_scheme=te_scheme)
    
    if conf['load_in_memory']:
        print("training set: %d mb" % ((tr_set.data_sources[0].nbytes + \
            tr_set.data_sources[1].nbytes) / 1e6))
        print("testing set: %d mb" % ((te_set.data_sources[0].nbytes + \
            te_set.data_sources[1].nbytes) / 1e6))
        time.sleep(2)
    
    return tr_stream, te_stream


def exp_decay_lr(global_step, n_tr, conf, name='lr'):
    """
    Exponential decay learning rate.

    Args:
      global_step: global step tensor
      n_tr: number of training examples
      conf: configuration dictionary
      name: name of exponential decay tensor
    Returns:
      lr: learning rate tensor
    """
    mb_size = conf['mb_size']
    lr0 = conf['lr0']
    n_epochs_per_decay = conf['n_epochs_per_decay']
    lr_decay_factor = conf['lr_decay_factor']

    n_mb_per_epoch = n_tr / (mb_size * FLAGS.num_gpus)
    decay_steps = int(n_mb_per_epoch * n_epochs_per_decay)
    lr = tf.train.exponential_decay(lr0,
                                    global_step,
                                    decay_steps,
                                    lr_decay_factor,
                                    staircase=True,
                                    name=name)

    return lr


def clip_by_norm(gvs, grad_norm_thresh, scope="grad_clip"):
    """
    Clip gradients by norm, and scope.

    Args:
      gvs: list of gradient variable tuples
      grad_norm_thresh: norm threshold to clip
      scope: scope for the clip operation
    """
    if scope:
        with tf.name_scope(scope):
            gvs = [(tf.clip_by_norm(gv[0], grad_norm_thresh), gv[1]) \
                   for gv in gvs if gv[0]]
            return gvs
    else:
        gvs = [(tf.clip_by_norm(gv[0], grad_norm_thresh), gv[1]) \
               for gv in gvs if gv[0]]
        return gvs


def average_gradients(tower_grads, name='avg_grads_sync'):
    """
    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a sync-point across all towers.

    Args:
      tower_grads: list of list of (gradient, variable) tuples. The outer list is
        over individual gradients. The inner list is over the gradient calculation
        for each tower.
      name: name scope for operations in averaging gradients
    Returns:
      list of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    with tf.name_scope(name):
        avg_grads = []
        for grad_vars in zip(*tower_grads):
            # Note that each grad_vars looks like the following:
            # ((grad0_gpu0, var0_gpu), ..., (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
        
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
        
            # Average over the 'tower' dimension
            grad = tf.concat(0, grads)
            grad = tf.reduce_mean(grad, 0)
        
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So, we will just return the first tower's pointer to
            # the Variable.
            v = grad_vars[0][1]
            grad_var = (grad, v)
            avg_grads.append(grad_var)
    
        return avg_grads


def track_err():
    """
    Track error for TensorBoard.
    
    Returns:
      err_sum_op: summary operation
      psnr_tr_t: training error placeholder
      psnr_te_t: testing error placeholder
    """
    psnr_tr_t = tf.placeholder('float32', name='psnr_tr')
    psnr_te_t = tf.placeholder('float32', name='psnr_te')
    tf.scalar_summary('psnr_tr', psnr_tr_t, collections=['error'])
    tf.scalar_summary('psnr_te', psnr_te_t, collections=['error'])
    err_sum_op = tf.merge_all_summaries('error')
    return err_sum_op, psnr_tr_t, psnr_te_t


def tf_boilerplate(summs, conf, ckpt=None):
    """
    TensorFlow boilerplate code
    
    Args:
      summs: summaries
      conf: configuration dictionary
      ckpt (None): if not None, restore from this path
    Returns:
      sess: session
      saver: saver
      summ_writer: summary writer
      summ_op: summary operation
    """
    path_tmp = conf['path_tmp']

    # Create a saver
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

    # Build summary operation based on the collection of summaries
    if summs is None:
        summ_op = tf.merge_all_summaries()
    else:
        summ_op = tf.merge_summary(summs)

    # Initialization
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    summ_writer = tf.train.SummaryWriter(path_tmp, graph_def=sess.graph_def)
    init = tf.initialize_all_variables()
    sess.run(init)
    if ckpt:
        saver.restore(sess, ckpt)

    return sess, saver, summ_writer, summ_op


def eval_psnr(Y, y):
    """Evaluate PSNR between ground-truth `Y` and img `y`"""
    diff = Y.astype(np.float32) - y.astype(np.float32)
    rmse = np.mean(diff ** 2) ** 0.5
    base = 1.0 if Y.dtype != np.uint8 else 255
    psnr = 20 * np.log10(base / rmse)
    return psnr


def baseline_psnr(te_stream):
    """ Calculate the baseline error of test set. """
    baseline_se = 0.
    for X_mb, y_mb in te_stream.get_epoch_iterator():
        baseline_se += np.sum((y_mb - X_mb) ** 2)
    N = te_stream.dataset.num_examples * y_mb.shape[1] * y_mb.shape[2]
    baseline_rmse = np.sqrt(baseline_se / N)
    baseline_psnr = 20 * np.log10(1.0 / baseline_rmse)
    return baseline_psnr
