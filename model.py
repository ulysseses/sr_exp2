from __future__ import division, absolute_import, print_function
from six.moves import range, zip

import re
import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops

FLAGS = tf.app.flags.FLAGS


def _init_shift(n_f, pw):
    """
    Initialize shift filter.

    Args:
      n_f: input dimension
      pw: patch width
    Returns:
      shift: shift convolutional filter
    """
    shift = np.zeros((pw, pw, n_f, pw*pw), dtype='float32')
    for f in range(n_f):
        ind = 0
        for i in range(pw):
            for j in range(pw):
                shift[i, j, f, ind] = 1.
                ind += 1
    return shift


def _init_stitch(pw):
    """
    Initialize stitch filter.

    Args:
      pw: patch width
    Returns:
      stitch: stitch convolutional filter
    """
    stitch = np.zeros((pw, pw, 1, pw*pw), dtype='float32')
    ind = 0
    for i in range(0, pw):
        for j in range(0, pw):
            stitch[pw - i - 1, pw - j - 1, 0, ind] = 1. / (pw*pw)
            ind += 1
    return stitch


def _relu_std(fw, n_chans):
    """
    ReLU initialization based on "Delving Deep into Rectifiers: Surpassing
    Human-Level Performance on ImageNet Classification" by Kaiming He et al.

    Args:
      fw: filter width
      n_chans: filter depth
    Returns:
      see below
    """
    return np.sqrt(2.0 / (fw*fw*n_chans))
    
    
def _arr_initializer(arr):
    ''' https://github.com/tensorflow/tensorflow/issues/434 '''
    def _initializer(_, dtype=tf.float32):
        return tf.constant(arr, dtype=dtype)
    return _initializer


def _st(x, thresh, name='st'):
    """
    L1 Soft threshold operator.

    Args:
      x: input
      thresh: threshold variable
      name: name assigned to this operation
    Returns:
      soft threshold of `x`
    """
    with tf.name_scope('st'):
        return tf.mul(tf.sign(x), tf.nn.relu(tf.nn.bias_add(tf.abs(x), -thresh)))


def _lcod(x, w_e, w_s, thresh, conf):
    """
    Learned Coordinate Descent (LCoD). LCoD is an approximately sparse encoder. It
    approximates (in an L2 sense) a sparse code of `x` according to dictionary `w_e`.
    Note that during backpropagation, `w_e` isn't strictly a dictionary (i.e.
    dictionary atoms are not strictly normalized).

    LCoD is a differentiable version of greedy coordinate descent.

    Args:
      x: [n, n_f] tensor
      w_e: [n_f, n_c] encoder tensor
      w_s: [n_c, n_f] mutual inhibition tensor
      thresh: soft thresold
      conf: configuration dictionary
    Returns:
      z: LCoD output
    """
    def f1():
        '''k_i == 0'''
        forget_z_i = tf.concat(1, [z_ik, tf.zeros([1, n_c - 1],
                                                  dtype=tf.float32)],
                               name='forget_z_%d' % i)
        update_z_i = tf.concat(1, [z_bar_ik, tf.zeros([1, n_c - 1],
                                                      dtype=tf.float32)],
                               name='update_z_%d' % i)
        return forget_z_i, update_z_i

    def f2():
        '''k_i == n_c - 1'''
        forget_z_i = tf.concat(1, [tf.zeros([1, n_c - 1], dtype=tf.float32),
                                   z_ik],
                               name='forget_z_%d' % i)
        update_z_i = tf.concat(1, [tf.zeros([1, n_c - 1], dtype=tf.float32),
                                   z_bar_ik],
                               name='update_z_%d' % i)
        return forget_z_i, update_z_i

    def f3():
        '''k_i > 0 and k_i < n_c - 1'''
        forget_z_i = tf.concat(1, [tf.zeros(tf.pack([1, k_i]), dtype=tf.float32),
                                   z_ik,
                                   tf.zeros(tf.pack([1, n_c - (k_i + 1)]),
                                            dtype=tf.float32)],
                               name='forget_z_%d' % i)
        update_z_i = tf.concat(1, [tf.zeros(tf.pack([1, k_i]), dtype=tf.float32),
                                   z_bar_ik,
                                   tf.zeros(tf.pack([1, n_c - (k_i + 1)]),
                                            dtype=tf.float32)],
                               name='update_z_%d' % i)
        return forget_z_i, update_z_i

    n = conf['mb_size'] * (conf['iw'] // conf['ps'])**2
    n_c = conf['n_c']
    T = conf['T']
    n_f = conf['n_chans'] * conf['pw']**2

    prevs1 = [None for _ in range(n)]
    prevs2 = [None for _ in range(n)]
    with tf.name_scope('slice_x'):
        for i in range(n):
            x_i = tf.slice(x, [i, 0], [1, n_f], name='x_%d' % i)
            prevs1[i] = x_i

    with tf.name_scope('itr_00'):
        for i in range(n):
            x_i = prevs1[i]
            b_i = tf.matmul(x_i, w_e, name='b_%d' % i)
            z_i = tf.zeros_like(b_i, dtype=tf.float32, name='z_%d' % i)
            prevs1[i] = b_i
            prevs2[i] = z_i

    for t in range(1, T):
        with tf.name_scope('itr_%02d' % t):
            for i in range(n):
                b_i = prevs1[i]
                z_i = prevs2[i]
                z_bar_i = _st(b_i, thresh, name='z_bar_%d' % i)
                with tf.name_scope('greedy_heuristic'):
                    if t > 1:
                        tmp_i = z_bar_i - z_i
                    else:
                        tmp_i = z_bar_i
                    tmp2_i = tf.abs(tmp_i)
                    k_i = tf.to_int32(tf.argmax(tmp2_i, 1, name='k_%d' % i))
                    e_i = tf.reshape(tf.slice(tmp_i, tf.pack([0, k_i]), tf.pack([1, 1]),
                                              name='e_%d' % i),
                                     [])
                with tf.name_scope('update_b'):
                    s_slice_i = tf.slice(w_s, tf.pack([k_i, 0]), tf.pack([1, n_c]),
                                         name='s_slice_%d' % i)
                    b_i = tf.add(b_i, e_i * s_slice_i, name='b_%d' % i)
                with tf.name_scope('update_z'):
                    z_bar_ik = tf.slice(z_bar_i, tf.pack([0, k_i]), tf.pack([1, 1]),
                                        name='z_bar_%dk' % i)
                    z_ik = tf.slice(z_i, tf.pack([0, k_i]), tf.pack([1, 1]),
                                    name='z_%dk' % i)
                    tup_i = control_flow_ops.case({tf.equal(k_i, 0): f1,
                                                   tf.equal(k_i, n_c - 1): f2},
                                                  default=f3,
                                                  exclusive=False)
                    forget_z_i, update_z_i = tup_i
                    z_i = tf.add_n([z_i, -forget_z_i, update_z_i], name='z_%d' % i)
                prevs1[i] = b_i
                prevs2[i] = z_i

    with tf.name_scope('itr_%02d' % T):
        for i in range(n):
            b_i = prevs1[i]
            z_i = _st(b_i, thresh, name='z_%d' % i)
            prevs2[i] = z_i

    # Concatenate to full tensor
    z = tf.concat(0, prevs2, name='z')

    return z


def _low_rank_approx(a, low_rank):
    """
    Use SVD to approximate `a` as two matrices, `b` and `c`.
    
    Args:
      a: input matrix
      low_rank: low rank
    Returns:
      b, c: 2 matrices of rank `low_rank` that optimally approximate `a` when
        multiplied together.
    """
    u, s, vh = np.linalg.svd(a)
    b = u[:, :low_rank] * np.sqrt(s[:low_rank])
    c = (vh[:low_rank, :].T * np.sqrt(s[:low_rank])).T
    return b.astype(np.float32), c.astype(np.float32)


def _low_rank_helper(mat, low_rank, suffix):
    """
    Helper function to return a tf.Variable with low rank.
    
    Args:
      mat: input matrix to approximate
      low_rank: intermediate rank
      suffix: suffix name of the low rank matrices
    Returns:
      w_mat: output matrix
    """
    in_dim, out_dim = mat.shape
    if low_rank > 0:
        mat1, mat2 = _low_rank_approx(mat, low_rank)
        w_mat1 = tf.get_variable('w_%s1' % suffix,
            [in_dim, low_rank],
            dtype=tf.float32,
            initializer=_arr_initializer(mat1))
        w_mat2 = tf.get_variable('w_%s2' % suffix,
            [low_rank, out_dim],
            dtype=tf.float32,
            initializer=_arr_initializer(mat2))
        w_mat = tf.matmul(w_mat1, w_mat2, name='w_%s' % suffix)
    else:
        w_mat = tf.get_variable('w_%s' % suffix,
            [in_dim, out_dim],
            dtype=tf.float32,
            initializer=_arr_initializer(mat))
    return w_mat


def inference(x, conf):
    """
    Sparse Coding Based Network for Super-Resolution. This is a convolutional
    neural network formulation of Coupled Dictionary Learning. Features are
    extracted via convolutional filters. Overlapping patches of the feature maps
    are obtained via a layer of patch extraction convolutional filters. The
    resulting feature maps are normalized and fed through LISTA sub-network of
    `T` iterations. The LISTA output patches are de-normalized with `scale_norm`
    and stitched back into their respective positions to re-construct the final
    output image.

    Args:
      x: input layer
      conf: configuration dictionary
    Returns:
      y: output
    """
    mb_size = conf['mb_size']
    iw = conf['iw']
    fw = conf['fw']
    pw = conf['pw']
    ps = conf['ps']
    n_chans = conf['n_chans']
    n_c = conf['n_c']
    cw = conf['cw']
    thresh0 = conf['thresh0']
    e_rank = conf['e_rank']
    s_rank = conf['s_rank']
    d_rank = conf['d_rank']
    
    n_f = n_chans*pw*pw
    bs = mb_size
    
    # Initialize constant filters
    with tf.variable_scope('const_filt'):
        w_shift1 = tf.constant(_init_shift(n_chans, pw), name='w_shift1')
        w_shift2 = tf.constant(_init_shift(1, pw), name='w_shift2')
        w_stitch = tf.constant(_init_stitch(pw), name='w_stitch')
    
    # Initialize feature extraction filters
    with tf.device('/cpu:0' if FLAGS.dev_assign else None):
        w_conv = tf.get_variable('w_conv', [fw, fw, 1, n_chans], tf.float32,
            initializer=tf.truncated_normal_initializer(0., _relu_std(fw, 1)))
    
    # Feature Extraction
    # [bs, iw, iw, 1] -> [bs, iw, iw, n_chans]
    x_conv = tf.nn.conv2d(x, w_conv, [1, 1, 1, 1], 'SAME', name='x_conv')
    
    # Shift with pw*pw dirac delta filters to create overlapping patches.
    # Only obtain patches every `ps` strides.
    # A patch is resembled as the flattened array along the last dimension.
    # [bs, iw, iw, n_chans] --> [bs, iw//ps, iw//ps, n_chans*pw*pw]
    x_shift = tf.nn.depthwise_conv2d(x_conv, w_shift1, [1, ps, ps, 1], 'SAME',
        name='x_shift')
    
    # 4D tensor -> matrix
    # [bs, iw//ps, iw//ps, n_chans*pw*pw] -> [bs*(iw//ps)*(iw//ps), n_chans*pw*pw]
    x_in = tf.reshape(x_shift, [bs*(iw//ps)*(iw//ps), n_f], name='x_in')
    
    # Feed into sub-network
    with tf.variable_scope('lcod'):
        with tf.device('/cpu:0' if FLAGS.dev_assign else None):
            thresh = tf.get_variable('thresh',
                [n_c],
                dtype=tf.float32,
                initializer=tf.constant_initializer(thresh0))
            
            # Initial values
            e = np.random.randn(n_f, n_c).astype(np.float32) * _relu_std(1, n_f)
            s = (np.eye(n_c) - e.T.dot(e)).astype(np.float32)
            d = np.random.randn(n_c, pw*pw).astype(np.float32) * _relu_std(1, n_c)
            # Encoder
            w_e = _low_rank_helper(e, e_rank, 'e')
            # S matrix
            w_s = _low_rank_helper(s, s_rank, 's')
            # Decoder
            w_d = _low_rank_helper(d, d_rank, 'd')
        
        # Sub-network
        # [bs*(iw//ps)*(iw//ps), n_f] -> [bs*(iw//ps)*(iw//ps), n_c]
        z = _lcod(x_in, w_e, w_s, thresh, conf)
        
        # Decode
        # [bs*(iw//ps)*(iw//ps), n_c] -> [bs*(iw//ps)*(iw//ps), pw*pw]
        y_out = tf.matmul(z, w_d, name='y_out')
        
    # matrix --> 4D tensor
    # [bs*(iw//ps)*(iw//ps), pw*pw] -> [bs, iw//ps, iw//ps, pw*pw]
    y_shift = tf.reshape(y_out, [bs, iw // ps, iw // ps, pw*pw],
                         name='y_shift')
    
    # Average overlapping images together
    with tf.name_scope('overlap_avg'):
        mask_input = tf.ones([bs, iw // ps, iw // ps, pw*pw],
                             dtype=tf.float32)
        mask = tf.nn.deconv2d(mask_input, w_stitch,
            [bs, iw, iw, 1], [1, ps, ps, 1], 'SAME', name='mask')
        y_overlap = tf.nn.deconv2d(y_shift, w_stitch, [bs, iw, iw, 1],
            [1, ps, ps, 1], 'SAME', name='y_overlap')
        y_average = tf.div(y_overlap, mask + 1e-8, name='y_average')
    
    # Add x to y
    pred0 = tf.add(y_average, x, name='res_skip')
    
    # Crop to remove convolution boundary effects
    with tf.name_scope('crop'):
        crop_begin = [0, cw, cw, 0]
        crop_size = [bs, iw - 2*cw, iw - 2*cw, 1]
        pred_crop = tf.slice(pred0, crop_begin, crop_size, name='pred_crop')
    
    pred = tf.identity(pred_crop, name='pred')
    
    model_vars = [w_conv, thresh, w_e, w_s, w_d]
    
    for var in model_vars:
        name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', var.op.name)
        tf.histogram_summary(name, var)
    
    return pred, model_vars


def loss(y, model_vars, Y, l2_reg, scope=None):
    """
    L2-loss model on top of the network raw output.
    
    Args:
      y: network output tensor
      model_vars: [w_conv, thresh, w_e, w_s, w_d]
      Y: ground truth tensor
      l2_reg: l2 regularization strength
      scope: unique prefix string identifying the tower, e.g. 'tower_00'
    Returns:
      total_loss: total loss Tensor
    """
    sq_loss = tf.nn.l2_loss(y - Y, name='sq_loss')
    tf.add_to_collection('losses', sq_loss)
    if l2_reg > 0:
        with tf.name_scope('l2_decay'):
            w_conv, thresh, w_e, w_s, w_d = model_vars
            for decay_var in [w_conv, w_e, w_s, w_d]:
                weight_decay = tf.mul(tf.nn.l2_loss(decay_var), l2_reg)
                tf.add_to_collection('losses', weight_decay)
    total_loss = tf.add_n(tf.get_collection('losses', scope=scope), name='total_loss')
    
    # Add loss summaries
    for loss in tf.get_collection('losses', scope=scope) + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', loss.op.name)
        tf.scalar_summary(loss_name, loss)
    
    return total_loss