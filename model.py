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


def _lista(x, w_e, w_s, thresh, T):
    """
    Learned Iterative Shrinkage-Thresholding Algorithm (LISTA). LISTA is an
    approximately sparse encoder. It approximates (in an L2 sense) a sparse code
    of `x` according to dictionary `w_e`. Note that during backpropagation, `w_e`
    isn't strictly a dictionary (i.e. dictionary atoms are not strictly normalized).
    LISTA is a differentiable version of the iterative FISTA algorithm.
    Args:
      x: [n, n_f] tensor
      w_e: [n_f, n_c] encoder tensor
      w_s: [n_c, n_f] mutual inhibition tensor
      thresh: soft threshold
      T: number of iterations
    Returns:
      z: LISTA output
    """
    b = tf.matmul(x, w_e, name='b')
    with tf.name_scope('itr_00'):
        z = _st(b, thresh, name='z')
    for t in range(1, T+1):
        with tf.name_scope('itr_%02d' % t):
            c = b + tf.matmul(z, w_s, name='c')
            z = _st(c, thresh, name='z')
    return z


def _lcod(x, w_e, w_s, thresh, T):
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
      T: number of iterations
    Returns:
      z: LCoD output
    """
    with tf.name_scope('itr_00'):
        b = tf.matmul(x, w_e, name='b')
        z = tf.zeros_like(b, dtype=tf.float32, name='z')

    for t in range(1, T):
        with tf.name_scope('itr_%02d' % t):
            z_bar = _st(b, thresh, name='z_bar')
            with tf.name_scope('greedy_heuristic'):
                # no tf.tile b/c tf.select will brodcast?
                if t > 1:
                    z_diff = tf.sub(z_bar, z, name='z_diff')
                else:
                    z_diff = z_bar
                abs_z_diff = tf.abs(z_diff, name='abs_z_diff')

                tmp = tf.reduce_max(abs_z_diff, 1, True)
                tmp2 = tf.equal(abs_z_diff, tmp)
                e = tf.select(tmp2, z_diff, tf.zeros_like(z_bar, dtype=tf.float32),
                           name='e')
                ks = tf.argmax(abs_z_diff, 1, name='ks')
                
            with tf.name_scope('update_b'):
                s_slices = tf.gather(w_s, ks, name='s_slices')
                b = tf.add(b, tf.mul(e, s_slices), name='b')

            with tf.name_scope('update_z'):
                z = tf.select(tmp2, z_bar, z, name='z')

    with tf.name_scope('itr_%02d' % T):
        z = _st(b, thresh, name='z')

    return z


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
    fw = conf['fw']
    pw = conf['pw']
    ps = conf['ps']
    n_chans = conf['n_chans']
    n_c = conf['n_c']
    T = conf['T']
    cw = conf['cw']
    thresh0 = conf['thresh0']
    subnet_name = conf['subnet_name'].lower()
    e_rank = conf['e_rank']
    s_rank = conf['s_rank']
    d_rank = conf['d_rank']
    
    n_f = n_chans*pw*pw
    
    if subnet_name == 'lista':
        subnet = _lista
    elif subnet_name == 'lcod':
        subnet = _lcod
    else:
        raise ValueError('subnet_name must be "lista" or "lcod"')
    
    # Initialize constant filters
    with tf.variable_scope('const_filt'):
        w_shift = tf.constant(_init_shift(n_chans, pw), name='w_shift')
        w_stitch = tf.constant(_init_stitch(pw), name='w_stitch')
    
    # Initialize feature extraction filters
    with tf.device('/cpu:0' if FLAGS.dev_assign else None):
        w_conv = tf.get_variable('w_conv', [fw, fw, 1, n_chans], tf.float32,
            initializer=tf.truncated_normal_initializer(0., _relu_std(fw, 1)))
    
    # Get dimensions
    with tf.name_scope('dimensions'):
        bs, h, w = [tf.shape(x)[i] for i in range(3)]
        bs = tf.identity(bs, name='bs')
        h = tf.identity(h, name='h')
        w = tf.identity(w, name='w')
    
    # Feature Extraction
    # [bs, h, w, 1] -> [bs, h, w, n_chans]
    x_conv = tf.nn.conv2d(x, w_conv, [1, 1, 1, 1], 'SAME', name='x_conv')
    
    # Shift with pw*pw dirac delta filters to create overlapping patches.
    # Only obtain patches every `ps` strides.
    # A patch is resembled as the flattened array along the last dimension.
    # [bs, h, w, n_chans] --> [bs, h//ps, w//ps, n_chans*pw*pw]
    x_shift = tf.nn.depthwise_conv2d(x_conv, w_shift, [1, ps, ps, 1], 'SAME',
        name='x_shift')
    
    # 4D tensor -> matrix
    # [bs, h//ps, w//ps, n_chans*pw*pw] -> [bs*(h//ps)*(w//ps), n_chans*pw*pw]
    x_in = tf.reshape(x_shift, [-1, n_f], name='x_in')
    
    # Feed into sub-network
    with tf.variable_scope(subnet_name):
        with tf.device('/cpu:0' if FLAGS.dev_assign else None):
            thresh = tf.get_variable('thresh',
                [n_c],
                dtype=tf.float32,
                initializer=tf.constant_initializer(thresh0))
            
            # Initial values
            e = np.random.randn(n_f, n_c).astype(np.float32) * _relu_std(1, n_f)
            if subnet_name == 'lista':
                L = 5.
                e /= L
                s = (np.eye(n_c) - e.T.dot(e) / L).astype(np.float32)
            else:
                s = (np.eye(n_c) - e.T.dot(e)).astype(np.float32)
            d = np.random.randn(n_c, pw*pw).astype(np.float32) * _relu_std(1, n_c)
            # Encoder
            w_e = _low_rank_helper(e, e_rank, 'e')
            # S matrix
            w_s = _low_rank_helper(s, s_rank, 's')
            # Decoder
            w_d = _low_rank_helper(d, d_rank, 'd')
        
        # Sub-network
        # [bs*(h//ps)*(w//ps), n_f] -> [bs*(h//ps)*(w//ps), n_c]
        z = subnet(x_in, w_e, w_s, thresh, T)
        
        # Decode
        # [bs*(h//ps)*(w//ps), n_c] -> [bs*(h//ps)*(w//ps), pw*pw]
        y_out = tf.matmul(z, w_d, name='y_out')
        
    # matrix --> 4D tensor
    # [bs*(h//ps)*(w//ps), pw*pw] -> [bs, h//ps, w//ps, pw*pw]
    y_shift = tf.reshape(y_out, tf.pack([bs, h // ps, w // ps, pw*pw]),
                         name='y_shift')
    
    # Average overlapping images together
    with tf.name_scope('overlap_avg'):
        mask_input = tf.ones(tf.pack([bs, h // ps, w // ps, pw*pw]),
                             dtype=tf.float32)
        mask = tf.nn.deconv2d(mask_input, w_stitch,
            tf.pack([bs, h, w, 1]), [1, ps, ps, 1], 'SAME', name='mask')
        y_overlap = tf.nn.deconv2d(y_shift, w_stitch, tf.pack([bs, h, w, 1]),
            [1, ps, ps, 1], 'SAME', name='y_overlap')
        y_average = tf.div(y_overlap, mask + 1e-8, name='y_average')
    
    # Add x to y
    pred0 = tf.add(y_average, x, name='res_skip')
    
    # Crop to remove convolution boundary effects
    with tf.name_scope('crop'):
        crop_begin = [0, cw, cw, 0]
        crop_size = tf.pack([-1, h - 2*cw, w - 2*cw, 1])
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
