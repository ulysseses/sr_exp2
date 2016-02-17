from __future__ import division, absolute_import, print_function
from six.moves import range, zip

import os
import time
import numpy as np
import scipy.misc as sm
from scipy.ndimage.filters import gaussian_filter

import h5py
from fuel.datasets.hdf5 import H5PYDataset


def unit2byte(img_unit):
    """Convert `img_unit` from floating [0, 1] to integer [0, 255]"""
    return np.clip(np.round(255*img_unit), 0, 255).astype('uint8')


def byte2unit(img_byte):
    """Convert `img_byte`from integer [0, 255] to floating [0, 1]"""
    return img_byte.astype('float32') / 255.


def imresize(img, s):
    """
    Resize routine. If `s` < 1, then filter the image with a gaussian
    to prevent aliasing artifacts.

    Args:
      img: uint8 image
      s: down/up-sampling factor

    Returns:
      img: uint8 resized & bi-cubic interpolated image
    """
    img = sm.imresize(img, s, interp='bicubic')
    return img


def shave(img, border):
    """Shave off border from image.

    Args:
      img: image
      border: border

    Returns:
      img: img cropped of border
    """
    return img[border: -border, border: -border]


def padZeroBorder(img, border=3):
    """Pad image with a border of zeros

    Args:
      img: image
      border (3): border

    Returns:
      img_new: zero-padded `img`
    """
    h, w = img.shape
    img_new = np.zeros((h + 2*border, w + 2*border), dtype=img.dtype)
    img_new[border:-border, border:-border] = img
    return img_new


def rgb2ycc(img_rgb):
    """Convert image from rgb to ycbcr colorspace.

    Args:
      img_rgb: image in rgb colorspace

    Returns:
      img_ycc: image in ycbcr colorspace
    """
    img_rgb = img_rgb.astype('float32')
    img_ycc = np.empty_like(img_rgb, dtype='float32')
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    y, cb, cr = img_ycc[:,:,0], img_ycc[:,:,1], img_ycc[:,:,2]

    y[:] = .299*r + .587*g + .114*b
    cb[:] = 128 -.168736*r -.331364*g + .5*b
    cr[:] = 128 +.5*r - .418688*g - .081312*b

    img_ycc = np.clip(np.round(img_ycc), 0, 255).astype('uint8')
    return img_ycc


def ycc2rgb(img_ycc):
    """Convert image from ycbcr to rgb colorspace.

    Args:
      img_ycc: uint8 image in ycbcr colorspace

    Returns:
      img_rgb: uint8 image in rgb colorspace
    """
    img_ycc = img_ycc.astype('float32')
    img_rgb = np.empty_like(img_ycc, dtype='float32')
    y, cb, cr = img_ycc[:,:,0], img_ycc[:,:,1], img_ycc[:,:,2]
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

    r[:] = y + 1.402 * (cr-128)
    g[:] = y - .34414 * (cb-128) -  .71414 * (cr-128)
    b[:] = y + 1.772 * (cb-128)

    img_rgb = np.clip(np.round(img_rgb), 0, 255).astype('uint8')
    return img_rgb


def modcrop(img, modulo):
    """
    Crop `img` s.t. its dimensions are an integer multiple of `modulo`

    Args:
      img: image
      modulo: modulo factor

    For example:

    ```
    # 'img' is [[1, 2, 3], [4, 5, 6],
    #           [7, 8, 9], [1, 2, 3],
    #           [4, 5, 6], [7, 8, 9]]
    modcrop(img, 2) ==> [[1, 2, 3], [4, 5, 6],
                         [7, 8, 9], [1, 2, 3]]
    ```
    """
    h, w = img.shape[0], img.shape[1]
    h = (h // modulo) * modulo
    w = (w // modulo) * modulo
    img = img[:h, :w]
    return img


def padcrop(img, modulo):
    """
    Pad `img` s.t. its dimensions are an integer multiple of `modulo`

    Args:
      img: image
      modulo: modulo factor

    For example:

    ```
    # 'img' is [[1, 2, 3], [4, 5, 6],
    #           [7, 8, 9], [1, 2, 3],
    #           [4, 5, 6], [7, 8, 9]]
    modcrop(img, 2) ==> [[1, 2, 3], [4, 5, 6],
                         [7, 8, 9], [1, 2, 3],
                         [4, 5, 6], [7, 8, 9],
                         [0, 0, 0], [0, 0, 0]]
    ```
    """
    h, w = img.shape[0], img.shape[1]
    h2, w2 = int(np.ceil(h / modulo)) * modulo, int(np.ceil(w / modulo)) * modulo
    shp2 = list(img.shape)
    shp2[0] = h2
    shp2[1] = w2
    img2 = np.zeros(shp2, dtype=img.dtype)
    img2[:h, :w] = img
    return img2


def _crop_gen(img, iw, s):
    """
    Generate a strided series of cropped patches from an image.

    Args:
      img: image
      iw: crop width
      s: stride

    Yields:
      crop: 2d cropped region of `img`
    """
    for i in range(0, img.shape[0] - iw + 1, s):
        for j in range(0, img.shape[1] - iw + 1, s):
            crop = img[i : i + iw, j : j + iw]
            yield crop


def _num_crops(img, iw, s, tup=False):
    h, w = img.shape[0], img.shape[1]
    n_y, n_x = len(range(0, h - iw + 1, s)), len(range(0, w - iw + 1, s))
    if tup:
        return n_y, n_x
    else:
        return n_y * n_x


def _get_filenames(path):
    ext_set = set(['jpg', 'jpeg', 'png', 'bmp'])
    def _valid_ext(s):
        lst = s.split('.')
        if len(lst) != 2: return False
        sfx = lst[1].lower()
        return sfx in ext_set

    fns = [os.path.join(path, fn) for fn in os.listdir(path)
           if _valid_ext(fn)]
    return fns


def store_hdf5(conf, pp=None, **kwargs):
    """
    Generate crops from images in a directory and store into hdf5. Optionally
    preprocess them with and/or prune low-variance crops.

    Args:
      conf:
        path_h5: path to write hdf5 file
        path_tr: path to training images
        path_te: path to testing images
        path_va: path to validation images
        iw: crop width
        stride: stride
        sr: resize/super-resolution factor
        border: how much to shave off before resizing down `img_hr` --> `img_lr`
        augment: if True, augment dataset with rotations and flips
        prune: remove the `prune`%% lowest variance _num_crop
        chunk_size: number of rows to save in memory before flushing hdf5
      pp (None): if not None, function/lambda to preprocess image
      **kwargs: passed to h5py.File
    """
    path_h5             = conf['path_h5']
    path_tr             = conf['path_tr']
    path_te             = conf['path_te']
    path_va             = conf['path_va']
    iw                  = conf['iw']
    stride              = conf['stride']
    sr                  = conf['sr']
    augment             = conf['augment']
    prune               = conf['prune']
    border              = conf['border']
    chunk_size          = conf['chunk_size']
    data_cached         = conf['data_cached']

    if data_cached:
        print('data already cached...')
        return
    print('caching data...')
    start_time = time.time()
    
    # Count number of training/testing examples
    fns_tr = _get_filenames(path_tr)
    fns_te = _get_filenames(path_te)
    fns_va = _get_filenames(path_va)

    # Create a resizable h5py Dataset
    f = h5py.File(path_h5, mode='w')
    LRh5 = f.create_dataset('LR', (chunk_size, iw, iw, 1), dtype=np.float32,
                            maxshape=(None, iw, iw, 1))
    HRh5 = f.create_dataset('HR', (chunk_size, iw, iw, 1), dtype=np.float32,
                            maxshape=(None, iw, iw, 1))

    # Fill up with training data
    lrs = np.empty((chunk_size, iw, iw, 1), dtype=np.float32)
    hrs = np.empty((chunk_size, iw, iw, 1), dtype=np.float32)
    ind_tr = 0
    for fn in fns_tr:
        img = sm.imread(fn)
        if len(img.shape) == 3:
            img = rgb2ycc(sm.imread(fn))[:, :, 0]  # rgb --> y
        else:
            pass  # img is already y
        img_lr, img_hr = lr_hr(img, sr, border)
        img_lr, img_hr = byte2unit(img_lr), byte2unit(img_hr)

        ind_arr = 0
        for crop_lr, crop_hr in zip(_crop_gen(img_lr, iw, stride),
                                    _crop_gen(img_hr, iw, stride)):
            lrs[ind_arr] = crop_lr[..., np.newaxis]
            hrs[ind_arr] = crop_hr[..., np.newaxis]
            ind_arr += 1
            if ind_arr == chunk_size - 1:
                LRh5[ind_tr : ind_tr + ind_arr] = lrs[:ind_arr]
                HRh5[ind_tr : ind_tr + ind_arr] = hrs[:ind_arr]
                ind_tr += ind_arr
                ind_arr = 0
                LRh5.resize(ind_tr + chunk_size, axis=0)
                HRh5.resize(ind_tr + chunk_size, axis=0)
                f.flush()
        LRh5[ind_tr : ind_tr + ind_arr] = lrs[:ind_arr]
        HRh5[ind_tr : ind_tr + ind_arr] = hrs[:ind_arr]
        ind_tr += ind_arr
        ind_arr = 0
        LRh5.resize(ind_tr + chunk_size, axis=0)
        HRh5.resize(ind_tr + chunk_size, axis=0)
        f.flush()

    # Prune
    if prune > 0:
        HR_vars = np.empty(ind_tr, dtype=np.float32)
        for i in range(0, ind_tr, chunk_size):
            HR_vars[i : i + chunk_size] = np.var(HRh5[i : i + chunk_size],
                                                 axis=(1, 2, 3))
        bottom = int(round(prune * n))
        cutoff = HR_vars[np.argpartition(HR_vars, bottom)[:bottom]].max()

        ind_tr = 0
        for fn in fns_tr:
            img = sm.imread(fn)
            if len(img.shape) == 3:
                img = rgb2ycc(sm.imread(fn))[:, :, 0]  # rgb --> y
            else:
                pass  # img is already y
            img_lr, img_hr = lr_hr(img, sr, border)
            img_lr, img_hr = byte2unit(img_lr), byte2unit(img_hr)

            ind_arr = 0
            for crop_lr, crop_hr in zip(_crop_gen(img_lr, iw, stride),
                                        _crop_gen(img_hr, iw, stride)):
                if np.var(crop_lr) > cutoff:
                    lrs[ind_arr] = crop_lr[..., np.newaxis]
                    hrs[ind_arr] = crop_hr[..., np.newaxis]
                    ind_arr += 1
                if ind_arr == chunk_size - 1:
                    LRh5[ind_tr : ind_tr + ind_arr] = lrs[:ind_arr]
                    HRh5[ind_tr : ind_tr + ind_arr] = hrs[:ind_arr]
                    ind_tr += ind_arr
                    ind_arr = 0
                    LRh5.resize(ind_tr + chunk_size, axis=0)
                    HRh5.resize(ind_tr + chunk_size, axis=0)
                    f.flush()
            LRh5[ind_tr : ind_tr + ind_arr] = lrs[:ind_arr]
            HRh5[ind_tr : ind_tr + ind_arr] = hrs[:ind_arr]
            ind_tr += ind_arr
            ind_arr = 0
            LRh5.resize(ind_tr + chunk_size, axis=0)
            HRh5.resize(ind_tr + chunk_size, axis=0)
            f.flush()

    # Augment
    if augment:
        LRh5.resize(ind_tr * 4, axis=0)
        HRh5.resize(ind_tr * 4, axis=0)
        for i0, i1, i2, i3 in zip(range(0, ind_tr, chunk_size),
                                  range(ind_tr, 2*ind_tr, chunk_size),
                                  range(2*ind_tr, 3*ind_tr, chunk_size),
                                  range(3*ind_tr, 4*ind_tr, chunk_size)):
            shp0 = min(chunk_size, ind_tr)
            lrs[:shp0] = LRh5[i0 : min(i0 + chunk_size, ind_tr)]
            hrs[:shp0] = HRh5[i0 : min(i0 + chunk_size, ind_tr)]

            aug1_LR = np.transpose(lrs[:shp0, :, :, 0], (0, 2, 1))[..., np.newaxis]
            aug1_HR = np.transpose(hrs[:shp0, :, :, 0], (0, 2, 1))[..., np.newaxis]
            aug2_LR = np.fliplr(lrs[:shp0, :, :, 0])[..., np.newaxis]
            aug2_HR = np.fliplr(hrs[:shp0, :, :, 0])[..., np.newaxis]
            aug3_LR = np.transpose(np.fliplr(lrs[:shp0, :, :, 0]),
                                   (0, 2, 1))[..., np.newaxis]
            aug3_HR = np.transpose(np.fliplr(hrs[:shp0, :, :, 0]),
                                   (0, 2, 1))[..., np.newaxis]

            LRh5[i1 : min(i1 + chunk_size, 2*ind_tr)] = aug1_LR[:shp0]
            HRh5[i1 : min(i1 + chunk_size, 2*ind_tr)] = aug1_HR[:shp0]
            f.flush()
            LRh5[i2 : min(i2 + chunk_size, 3*ind_tr)] = aug2_LR[:shp0]
            HRh5[i2 : min(i2 + chunk_size, 3*ind_tr)] = aug2_HR[:shp0]
            f.flush()
            LRh5[i3 : min(i3 + chunk_size, 4*ind_tr)] = aug3_LR[:shp0]
            HRh5[i3 : min(i3 + chunk_size, 4*ind_tr)] = aug3_HR[:shp0]
            f.flush()
        ind_tr *= 4

    # Fill up with testing and validation data
    ind_te = ind_tr
    for fn in fns_te:
        img = sm.imread(fn)
        if len(img.shape) == 3:
            img = rgb2ycc(sm.imread(fn))[:, :, 0]  # rgb --> y
        else:
            pass  # img is already y
        img_lr, img_hr = lr_hr(img, sr, border)
        img_lr, img_hr = byte2unit(img_lr), byte2unit(img_hr)

        ind_arr = 0
        for crop_lr, crop_hr in zip(_crop_gen(img_lr, iw, stride),
                                    _crop_gen(img_hr, iw, stride)):
            lrs[ind_arr] = crop_lr[..., np.newaxis]
            hrs[ind_arr] = crop_hr[..., np.newaxis]
            ind_arr += 1
            if ind_arr == chunk_size - 1:
                LRh5[ind_te : ind_te + ind_arr] = lrs[:ind_arr]
                HRh5[ind_te : ind_te + ind_arr] = hrs[:ind_arr]
                ind_te += ind_arr
                ind_arr = 0
                LRh5.resize(ind_te + chunk_size, axis=0)
                HRh5.resize(ind_te + chunk_size, axis=0)
                f.flush()
        LRh5[ind_te : ind_te + ind_arr] = lrs[:ind_arr]
        HRh5[ind_te : ind_te + ind_arr] = hrs[:ind_arr]
        ind_te += ind_arr
        ind_arr = 0
        LRh5.resize(ind_te + chunk_size, axis=0)
        HRh5.resize(ind_te + chunk_size, axis=0)
        f.flush()

    ind_va = ind_te
    for fn in fns_va:
        img = sm.imread(fn)
        if len(img.shape) == 3:
            img = rgb2ycc(sm.imread(fn))[:, :, 0]  # rgb --> y
        else:
            pass  # img is already y
        img_lr, img_hr = lr_hr(img, sr, border)
        img_lr, img_hr = byte2unit(img_lr), byte2unit(img_hr)

        ind_arr = 0
        for crop_lr, crop_hr in zip(_crop_gen(img_lr, iw, stride),
                                    _crop_gen(img_hr, iw, stride)):
            lrs[ind_arr] = crop_lr[..., np.newaxis]
            hrs[ind_arr] = crop_hr[..., np.newaxis]
            ind_arr += 1
            if ind_arr == chunk_size - 1:
                LRh5[ind_va : ind_va + ind_arr] = lrs[:ind_arr]
                HRh5[ind_va : ind_va + ind_arr] = hrs[:ind_arr]
                ind_va += ind_arr
                ind_arr = 0
                LRh5.resize(ind_va + chunk_size, axis=0)
                HRh5.resize(ind_va + chunk_size, axis=0)
                f.flush()
        LRh5[ind_va : ind_va + ind_arr] = lrs[:ind_arr]
        HRh5[ind_va : ind_va + ind_arr] = hrs[:ind_arr]
        ind_va += ind_arr
        ind_arr = 0
        LRh5.resize(ind_va, axis=0)
        HRh5.resize(ind_va, axis=0)
        f.flush()

    split_dict = {
        'train': {'LR': (0, ind_tr), 'HR': (0, ind_tr)},
        'test': {'LR': (ind_tr, ind_te), 'HR': (ind_tr, ind_te)},
        'val': {'LR': (ind_te, ind_va), 'HR': (ind_te, ind_va)}
    }
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.close()
    duration = time.time() - start_time
    print('n_tr:', ind_tr)
    print('n_te:', ind_te - ind_tr)
    print('n_va:', ind_va - ind_te)
    print('total caching time: %.1f min' % (float(duration) / 60.))


def lr_hr(img, sr, border=3):
    """Generate LR & HR pair from image.

    Args:
      img: uint8 image
      sr: resize/super-resolution factor
      border (3): how much to shave off before resizing down `img_hr` --> `img_lr`
    """
    img_hr = modcrop(img, sr)
    down_shp = (img_hr.shape[0] // sr, img_hr.shape[1] // sr)
    img_lr = imresize(img_hr, down_shp)
    img_lr = imresize(img_lr, img_hr.shape)
    img_lr = shave(img_lr, border)
    img_hr = shave(img_hr, border)
    return img_lr, img_hr


def num_patches(img, iw, stride):
    """
    Calculate the number of patches in H/V directions.

    Args:
      img: image
      iw: crop width
      stride: stride
    Returns:
      n_y: number of patches per column
      n_x: number of patches per row
    """
    h, w = img.shape[:2]
    n_y = len(range(0, h - iw + 1, stride))
    n_x = len(range(0, w - iw + 1, stride))
    return n_y, n_x


def img2patches(img, iw, stride):
    """
    Convert an image to a tensor containing overlapping patches.

    Args:
      img: image
      iw: crop width
      stride: stride
    Returns:
      patches: 4d tensor of overlapping patches
    """
    h, w = img.shape[:2]
    n_y = len(range(0, h - iw + 1, stride))
    n_x = len(range(0, w - iw + 1, stride))
    patches = np.empty((n_y*n_x, iw, iw, 1), dtype=np.float32)
    
    ind = 0
    for i in range(0, h - iw + 1, stride):
        for j in range(0, w - iw + 1, stride):
            patches[ind] = img[i : i + iw, j : j + iw, np.newaxis]
            ind += 1
    assert ind == n_y*n_x

    return patches


def patches2img(patches, n_y, n_x, stride):
    """
    Convert a tensor containing overlapping patches to an image.

    Args:
      patches: 4d tensor of overlapping patches
      n_y: number of patches per column
      n_x: number of patches per row
      stride: stride
    Returns:
      img: output image
    """
    iw = patches.shape[1]
    h = iw + (n_y - 1) * stride
    w = iw + (n_x - 1) * stride
    img = np.zeros((h, w), dtype=np.float32)
    mask = 1e-8 * np.ones((h, w), dtype=np.float32)

    ind = 0
    for i in range(0, h - iw + 1, stride):
        for j in range(0, w - iw + 1, stride):
            img[i : i + iw, j : j + iw] += patches[ind, :, :, 0]
            ind += 1
            mask[i : i + iw, j : j + iw] += 1.0
    assert ind == n_y*n_x

    img /= mask
    return img
