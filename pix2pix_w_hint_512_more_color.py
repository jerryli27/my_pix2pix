#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file loads a checkpoint from trained 128x128 model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import urllib
from sklearn.neighbors import NearestNeighbors

from general_util import imread

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--user_hint_path", default=None, help="path to a hint image.")
parser.add_argument("--pretrained_sketch_net_path", default=None, help="path to the pretrained sketch network checkpoint")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=10, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
# to get tracing working on GPU, LD_LIBRARY_PATH may need to be modified:
# LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split A image into brightness (A) and color (B), ignore B image")
parser.add_argument("--gray_input_a", action="store_true", help="Treat A image as grayscale image.")
parser.add_argument("--gray_input_b", action="store_true", help="Treat B image as grayscale image.")
parser.add_argument("--use_hint", action="store_true", help="Supply hints to input. Training dimension 1 -> 4.")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default= 572,# 286,
                    help="scale images to this size before cropping to `CROP_SIZE`x`CROP_SIZE`")
parser.add_argument("--crop_size", type=int, default= 512,
                    help="scale images to this size before cropping to `CROP_SIZE`x`CROP_SIZE`")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--gpu_percentage", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--from_128", dest="from_128", action="store_true", help="Indicate whether model is from 128x128.")
parser.add_argument("--train_sketch", dest="train_sketch", action="store_true",
                    help="Indicate whether the model is for sketch generation. Variable scope will change accordingly.")
parser.add_argument("--use_sketch_loss", dest="use_sketch_loss", action="store_true",
                    help="Use the pretrained sketch generator network to compare the sketches of the generated image "
                         "versus that of the original image.")
parser.add_argument("--sketch_weight", type=float, default=1.0, help="weight on sketch loss term.")
parser.add_argument("--use_bin", dest="use_bin", action="store_true",
                    help="Output a probability distribution of color bins instead of one single color. It should make "
                         "the output more colorful.")
parser.add_argument("--num_bin_per_channel", type=int, default=6, help="number of bins per r, g, b channel")



a = parser.parse_args()

if a.use_sketch_loss and a.pretrained_sketch_net_path is None:
    parser.error("If you want to use sketch loss, please provide a valid pretrained_sketch_net_path.")

EPS = 1e-12
CROP_SIZE = a.crop_size  # 128 # 256

SKETCH_VAR_SCOPE_PREFIX = "sketch_"

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch, input_hints")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, gen_loss_GAN, gen_loss_L1, gen_loss_sketch, train")


def conv(batch_input, out_channels, stride, trainable=True):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02), trainable=trainable)
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input, trainable=True):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer, trainable=trainable)
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02), trainable=trainable)
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels, trainable=True):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02), trainable=trainable)
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))

class ImgToRgbBinEncoder():
    def __init__(self,bin_num=6):
        self.bin_num=bin_num
        index_matrix=  []
        pixel_max_val = 1.0  # Originally it was 255.0 but for the current program it is different.
        pixel_min_val = -1.0
        assert pixel_min_val < pixel_max_val
        step_size = (pixel_max_val - pixel_min_val) / (bin_num - 1)
        r, g, b = pixel_min_val, pixel_min_val, pixel_min_val
        while r <= pixel_max_val:
            # r_rounded = round(r)
            g = pixel_min_val
            while g <= pixel_max_val:
                # g_rounded = round(g)
                b = pixel_min_val
                while b <= pixel_max_val:
                    # b_rounded = round(b)
                    # index_matrix.append([r_rounded, g_rounded, b_rounded])
                    index_matrix.append([r, g, b])
                    b += step_size
                g += step_size
            r += step_size
        # self.index_matrix = np.array(index_matrix, dtype=np.uint8)
        self.index_matrix = np.array(index_matrix, dtype=np.float32)
        # self.nnencode = NNEncode(5,5,cc=self.index_matrix)
        self.nnencode = NNEncode(5.,0.05,cc=self.index_matrix)
    def img_to_bin(self, img, return_sparse = False):

        """

        :param img:  An image represented in numpy array with shape (batch, height, width, 3)
        :param bin_num: number of bins for each color dimension
        :return:  An image represented in numpy array with shape (batch, height, width, bin_num ** 3)
        """
        if len(img.shape) != 4 and len(img.shape) != 3:
            raise AssertionError("The image must have shape (batch, height, width, 3), or (height, width, 3), not %s" %str(img.shape))
        if len(img.shape) == 4:
            batch, height, width, num_channel = img.shape
        else:
            height, width, num_channel = img.shape
        if num_channel != 3:
            raise AssertionError("The image must have 3 channels representing rgb. not %d." %num_channel)

        # nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(index_matrix)
        #
        # img_resized = np.reshape(img, (batch * height * width, num_channel))
        #
        # # Shape batch * height * width, 5
        # distances, indices = nbrs.kneighbors(img_resized)
        #
        #
        #
        # # In the original paper they used a gaussian kernel with delta = 5.
        # distances = gaussian_kernel(distances, std=5.0)
        #
        # rgb_bin = np.zeros((batch * height * width, bin_num ** 3))
        #
        # for bhw in range(batch * height * width):
        #     for i in range(5):
        #         rgb_bin[bhw,indices[bhw,i]] = distances[bhw, i]
        #
        #
        # return rgb_bin

        # return self.nnencode.encode_points_mtx_nd(img,axis=3, return_sparse=return_sparse)
        return self.nnencode.encode_points_mtx_nd(img,axis=len(img.shape)-1, return_sparse=return_sparse)



    def bin_to_img(self, rgb_bin, t = 0.01, do_softmax = True):
        """
        This function uses annealed-mean technique in the paper.
        :param rgb_bin:
        :param t: t = 0.38 in the original paper
        """

        if len(rgb_bin.shape) != 4 and len(rgb_bin.shape) != 3:
            raise AssertionError("The rgb_bin must have shape (batch, height, width, 3), or (height, width, 3), not %s" %str(rgb_bin.shape))
        if len(rgb_bin.shape) == 4:
            batch, height, width, num_channel = rgb_bin.shape
        else:
            height, width, num_channel = rgb_bin.shape
        if num_channel != self.bin_num**3:
            raise AssertionError("The rgb_bin must have bin_num**3 channels, not %d." % num_channel)
        # This is not the correct way to normalize. The correct way is to just apply a softmax...
        # rgb_bin_normalized = rgb_bin/np.sum(rgb_bin,axis=3,keepdims=True)
        if do_softmax:
            rgb_bin_exp = np.exp(rgb_bin)
            # rgb_bin_softmax = rgb_bin_exp / np.sum(rgb_bin_exp, axis=3,keepdims=True)
            rgb_bin_softmax = rgb_bin_exp / np.sum(rgb_bin_exp, axis=len(rgb_bin.shape) - 1,keepdims=True)
        else:
            rgb_bin_softmax = rgb_bin

        exp_log_z_div_t = np.exp(np.divide(np.log(rgb_bin_softmax),t))
        # annealed_mean = exp_log_z_div_t / np.add(np.sum(exp_log_z_div_t, axis=3, keepdims=True),0.000001)
        # return self.nnencode.decode_points_mtx_nd(annealed_mean, axis=3)
        annealed_mean = exp_log_z_div_t / np.add(np.sum(exp_log_z_div_t, axis=len(rgb_bin.shape) - 1, keepdims=True),0.000001)
        return self.nnencode.decode_points_mtx_nd(annealed_mean, axis=len(rgb_bin.shape) - 1)


    def gaussian_kernel(self,arr,std):
        return np.exp(-arr**2 / (2 * std**2))



class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''
    def __init__(self,NN,sigma,km_filepath='',cc=-1):
        if(check_value(cc,-1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc)
        self.closest_neighbor = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self,pts_nd,axis=1,return_sparse=False,sameBlock=True):
        pts_flt = flatten_nd_array(pts_nd,axis=axis)
        P = pts_flt.shape[0]
        if(sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0 # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P,self.K))
            self.p_inds = np.arange(0,P,dtype='int')[:,na()]

        P = pts_flt.shape[0]

        if return_sparse:
            (dists, inds) = self.nbrs.closest_neighbor(pts_flt)
        else:
            (dists,inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists**2/(2*self.sigma**2))
        wts = wts/np.sum(wts,axis=1)[:,na()]

        self.pts_enc_flt[self.p_inds,inds] = wts
        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt,pts_nd,axis=axis)

        return pts_enc_nd

    def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd,axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt,self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)
        return pts_dec_nd

    # def decode_1hot_mtx_nd(self,pts_enc_nd,axis=1,returnEncode=False):
    #     pts_1hot_nd = nd_argmax_1hot(pts_enc_nd,axis=axis)
    #     pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
    #     if(returnEncode):
    #         return (pts_dec_nd,pts_1hot_nd)
    #     else:
    #         return pts_dec_nd

# *****************************
# ***** Utility functions *****
# *****************************
def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if(np.array(inds).size==1):
        if(inds==val):
            return True
    return False

def na(): # shorthand for new axis
    return np.newaxis

def flatten_nd_array(pts_nd,axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS,SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        # print NEW_SHP
        # print pts_flt.shape
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out

def load_examples(user_hint = None):
    if not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
            a_images = tf.expand_dims(L_chan, axis=2) / 50 - 1 # black and white with input range [0, 100]
            b_images = tf.stack([a_chan, b_chan], axis=2) / 110 # color channels with input range ~[-110, 110], not exact
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1] # [height, width, channels]

            # a_images = raw_input[:,:width//2,:] * 2 - 1
            # b_images = raw_input[:,width//2:,:] * 2 - 1

            # Modified code: change a_images and b_images to 0~1 before turning into grayscale and rescaling.
            a_images = raw_input[:,:width//2,:]
            b_images = raw_input[:,width//2:,:]
            if a.gray_input_a:
                a_images = tf.image.rgb_to_grayscale(a_images)
            if a.gray_input_b:
                b_images = tf.image.rgb_to_grayscale(b_images)

            a_images =  a_images * 2 - 1
            b_images = b_images * 2 - 1

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    def append_hint(inputs, targets, user_hint = None):
        num_hints = 40
        blank_hint = np.ones(targets.get_shape().as_list()[:-1] + [4], dtype=np.float32) * -1
        # blank_hint = np.ones(targets.get_shape().as_list()[:-1] + [3], dtype=np.float32)
        output = tf.get_variable('output', initializer=blank_hint, dtype=tf.float32, trainable=False)

        if user_hint is None:
            # Include the pixels around it ONLY if the current solution fails.
            rd_indices_h = tf.random_uniform([num_hints, 1], minval=0, maxval=targets.get_shape().as_list()[-3], dtype=tf.int32)
            rd_indices_w = tf.random_uniform([num_hints, 1], minval=0, maxval=targets.get_shape().as_list()[-2], dtype=tf.int32)
            rd_indices_2d = tf.concat(1,(rd_indices_h,rd_indices_w))

            targets_rgba = tf.concat(2,(targets,np.ones(targets.get_shape().as_list()[:-1] + [1])))
            hints = tf.gather_nd(targets_rgba, rd_indices_2d)

            # hints = tf.gather_nd(targets, rd_indices)
            clear_hint_op = tf.assign(output, blank_hint)
            random_condition = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32, name="random_hint_condition")
            with tf.control_dependencies([clear_hint_op,hints]):
                # I cannot assign this to any other variable, otherwise it will cause the program to be confused on
                # whether the clear hint op should be ran first or the scatter update first.
                half_constant = tf.constant(0.50)

                output = tf.cond(tf.less(random_condition, half_constant), lambda: output, lambda: tf.scatter_nd_update(output, rd_indices_2d, hints))
        else:
            output = tf.assign(output, user_hint)

        assert len(output.get_shape().as_list()) == 3
        return tf.concat(2, (inputs, output), name='input_concat'), output
        # return tf.concat(2, (inputs, output), name='input_concat'), output

    with tf.name_scope("target_images"):
        target_images = transform(targets)
    with tf.name_scope("input_images"):
        input_images = transform(inputs)
        if a.use_hint:
            input_images, input_hints = append_hint(input_images, target_images, user_hint=user_hint)


    if a.use_hint:
        paths, inputs, targets, input_hints = tf.train.batch([paths, input_images, target_images, input_hints], batch_size=a.batch_size)
    else:
        paths, inputs, targets = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths,
        inputs=inputs,
        targets=targets,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
        input_hints=input_hints if a.use_hint else None,
    )


def create_model(inputs, targets):
    def create_generator(generator_inputs, generator_outputs_channels, trainable = True):
        layers = []

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = conv(generator_inputs, a.ngf, stride=2, trainable=trainable)
            layers.append(output)
        layer_specs = [
            a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ]
        # else:
        #     layer_specs = [
        #         a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        #         a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        #         a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        #         a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        #         a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        #         a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        #         a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        #     ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv(rectified, out_channels, stride=2, trainable=trainable)
                output = batchnorm(convolved)
                layers.append(output)

        layer_specs = [
            (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]
        # else:
        #     layer_specs = [
        #         (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        #         (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        #         (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        #         (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        #         (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        #         (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        #         (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        #     ]
        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    # Can't find concat_v2 so commenting this out.
                    #input = tf.concat_v2([layers[-1], layers[skip_layer]], axis=3)
                    input = tf.concat(3, [layers[-1], layers[skip_layer]])

                rectified = tf.nn.relu(input)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = deconv(rectified, out_channels, trainable=trainable)
                output = batchnorm(output, trainable=trainable)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            #input = tf.concat_v2([layers[-1], layers[0]], axis=3)
            input = tf.concat(3,[layers[-1], layers[0]])
            rectified = tf.nn.relu(input)
            output = deconv(rectified, generator_outputs_channels, trainable=trainable)
            # output = tf.tanh(output) # TODO: commented this out because I'm going to change the loss function to softmax cross entropy instead.
            layers.append(output)

        return layers[-1]

    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        # input = tf.concat_v2([discrim_inputs, discrim_targets], axis=3)
        input = tf.concat(3, [discrim_inputs, discrim_targets])

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator" if not a.train_sketch else SKETCH_VAR_SCOPE_PREFIX + "generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    if a.use_sketch_loss:
        with tf.variable_scope(SKETCH_VAR_SCOPE_PREFIX + "generator") as scope:
            fake_sketches = create_generator(outputs, 1, trainable=False)
        with tf.variable_scope(SKETCH_VAR_SCOPE_PREFIX + "generator", reuse=True) as scope:
            real_sketches = create_generator(targets, 1, trainable=False)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator" if not a.train_sketch else SKETCH_VAR_SCOPE_PREFIX + "discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            if a.use_hint:
                print("creating discr without hint. FOR NOW")
                print(inputs[...,:1].get_shape().as_list())
                predict_real = create_discriminator(inputs[...,:1], targets)
            else:
                predict_real = create_discriminator(inputs, targets)
                # TODO: change back later
            # predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator" if not a.train_sketch else SKETCH_VAR_SCOPE_PREFIX + "discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            if a.use_hint:
                print("creating discr without hint. FOR NOW")
                print(inputs[...,:1].get_shape().as_list())
                predict_fake = create_discriminator(inputs[...,:1], outputs)
            else:
                predict_fake = create_discriminator(inputs, outputs)
                # TODO: change back later
            # predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        # gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss_L1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
        if a.use_sketch_loss:
            gen_loss_sketch = tf.reduce_mean(tf.abs(fake_sketches - real_sketches))
            gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight + gen_loss_sketch * a.sketch_weight
        else:
            gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator" if not a.train_sketch else SKETCH_VAR_SCOPE_PREFIX + "discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_train = discrim_optim.minimize(discrim_loss, var_list=discrim_tvars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator" if not a.train_sketch else SKETCH_VAR_SCOPE_PREFIX + "generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_train = gen_optim.minimize(gen_loss, var_list=gen_tvars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    if a.use_sketch_loss:
        update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1, gen_loss_sketch])
    else:
        update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_loss_sketch=ema.average(gen_loss_sketch) if a.use_sketch_loss else None,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def save_images(fetches, image_dir, step=None):
    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path))
        fileset = {"name": name, "step": step}
        kinds = ["inputs", "hints", "outputs", "targets"] if a.use_hint else ["inputs", "outputs", "targets"]
        for kind in kinds:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "w") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><meta content=\"text/html;charset=utf-8\" http-equiv=\"Content-Type\"><meta content=\"utf-8\" http-equiv=\"encoding\"><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>hint</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")
        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        kinds = ["inputs", "hints", "outputs", "targets"] if a.use_hint else ["inputs", "outputs", "targets"]
        for kind in kinds:
            index.write("<td><img src=\"images/%s\"></td>" % urllib.quote(fileset[kind]))

        index.write("</tr>")
    return index_path

def main():
    if a.from_128:
        assert a.checkpoint is not None

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    if a.use_bin:
        i2b_encoder = ImgToRgbBinEncoder(a.num_bin_per_channel)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization", "gray_input_a", "gray_input_b", "use_hint"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).iteritems():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False
        if a.user_hint_path is not None:
            if a.user_hint_path == "BLANK":
                user_hint = np.ones([CROP_SIZE,CROP_SIZE,4], dtype=np.float32) * -1
            else:
                user_hint = (np.array(imread(a.user_hint_path, shape=(CROP_SIZE,CROP_SIZE), rgba=True)) - 255.0/2) / (255.0/2)
                assert np.max(user_hint) <=1.0 and np.min(user_hint) >=-1.0
        else:
            user_hint = None

    else:
        assert a.user_hint_path == None # Can't train with one single specified hint image. Doesn't make sense.
        user_hint = None

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples(user_hint=user_hint)

    print("examples count = %d" % examples.count)

    # input_ph = tf.placeholder(tf.float32,shape=examples.inputs.get_shape())
    # target_ph = tf.placeholder(tf.float32,shape=examples.targets.get_shape())
    if a.use_bin:
        inputs = tf.placeholder(tf.float32,shape=examples.inputs.get_shape().as_list(), name="inputs")
        targets = tf.placeholder(tf.float32,shape=examples.targets.get_shape().as_list()[:-1] + [a.num_bin_per_channel ** 3], name="targets_bin")
        # i2b_encoder.img_to_bin(examples.targets)
    else:
        inputs = examples.inputs
        targets = examples.targets


    model = create_model(inputs, targets)
    # model = create_model(input_ph, target_ph)

    def deprocess(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        if a.lab_colorization:
            # colorization mode images can be 1 channel (L) or 2 channels (a,b)
            num_channels = int(image.get_shape()[-1])
            if num_channels == 1:
                return tf.image.convert_image_dtype((image + 1) / 2, dtype=tf.uint8, saturate=True)
            elif num_channels == 2:
                # (a, b) color channels, convert to rgb
                # a_chan and b_chan have range [-1, 1] => [-110, 110]
                a_chan, b_chan = tf.unstack(image * 110, axis=3)
                # get L_chan from inputs or targets
                if a.which_direction == "AtoB":
                    brightness = examples.inputs
                elif a.which_direction == "BtoA":
                    brightness = examples.targets
                else:
                    raise Exception("invalid direction")
                # L_chan has range [-1, 1] => [0, 100]
                L_chan = tf.squeeze((brightness + 1) / 2 * 100, axis=3)
                lab = tf.stack([L_chan, a_chan, b_chan], axis=3)
                rgb = lab_to_rgb(lab)
                return tf.image.convert_image_dtype(rgb, dtype=tf.uint8, saturate=True)
            else:
                raise Exception("unexpected number of channels")
        else:
            num_channels = int(image.get_shape()[-1])
            if num_channels != 4:
                return tf.image.convert_image_dtype((image + 1) / 2, dtype=tf.uint8, saturate=True)
            else:
                print('using hint! correct!')
                print('image shape: %s' %(str(image.get_shape().as_list())))
                # image_r,image_g,image_b,image_a = tf.unpack(image,axis=3)
                # image_r = (image_r + 1) / 2
                # image_g = (image_g + 1) / 2
                # image_b = (image_b + 1) / 2
                # image_rgba = tf.pack((image_r,image_g,image_b,image_a),axis=3)
                image_rgba = (image + 1) / 2
                return tf.image.convert_image_dtype(image_rgba, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("deprocess_inputs"):
        if a.use_hint:
            deprocessed_inputs = deprocess(examples.inputs[...,:1])
            deprocessed_hints = deprocess(examples.inputs[...,1:])
        else:
            deprocessed_inputs = deprocess(examples.inputs)


    with tf.name_scope("deprocess_targets"):
        deprocessed_targets = deprocess(examples.targets)

    with tf.name_scope("deprocess_outputs"):
        outputs = model.outputs
        if a.use_bin:
            # outputs = i2b_encoder.bin_to_img(outputs)
            deprocessed_outputs = outputs
        else:
            deprocessed_outputs = deprocess(outputs)

    with tf.name_scope("encode_images"):
        if a.use_bin:
            outputs_images = tf.placeholder(tf.float32, shape=[None] + examples.targets.get_shape().as_list()[1:], name='outputs_images')
            print(outputs_images.get_shape().as_list()) # TODO: delete.
            deprocessed_outputs_images = deprocess(outputs_images)
            encoded_outputs = tf.map_fn(tf.image.encode_png, deprocessed_outputs_images, dtype=tf.string, name="encoded_outputs")
        if a.use_hint:
            display_fetches = {
                "paths": examples.paths,
                "inputs": tf.map_fn(tf.image.encode_png, deprocessed_inputs, dtype=tf.string, name="input_pngs"),
                "hints": tf.map_fn(tf.image.encode_png, deprocessed_hints, dtype=tf.string, name="hint_pngs"),
                "targets": tf.map_fn(tf.image.encode_png, deprocessed_targets, dtype=tf.string, name="target_pngs"),
                "outputs": tf.map_fn(tf.image.encode_png, deprocessed_outputs, dtype=tf.string, name="output_pngs") if not a.use_bin else deprocessed_outputs, # TODO: change this part to be dependent on use_bin
            }
        else:
            display_fetches = {
                "paths": examples.paths,
                "inputs": tf.map_fn(tf.image.encode_png, deprocessed_inputs, dtype=tf.string, name="input_pngs"),
                "targets": tf.map_fn(tf.image.encode_png, deprocessed_targets, dtype=tf.string, name="target_pngs"),
                "outputs": tf.map_fn(tf.image.encode_png, deprocessed_outputs, dtype=tf.string, name="output_pngs") if not a.use_bin else deprocessed_outputs,

            }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", deprocessed_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", deprocessed_targets)

    if not a.use_bin:
        with tf.name_scope("outputs_summary"):
            tf.summary.image("outputs", deprocessed_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    if a.use_sketch_loss:
        tf.summary.scalar("gen_loss_sketch", model.gen_loss_sketch)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = a.gpu_percentage
    # Get all variables in the model.
    # TODO: Only need to do this once when I load from 128x128 model. I should disable this part if the model does not
    # come from 128x128.
    if a.from_128:
        generator_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator' if not a.train_sketch else SKETCH_VAR_SCOPE_PREFIX + "generator")
        discriminator_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator' if not a.train_sketch else SKETCH_VAR_SCOPE_PREFIX + "discriminator")
        other_var = [var for var in tf.global_variables() if (var not in generator_var and var not in discriminator_var)]
        print("number of generator var = %d, number of discriminator var = %d, number of other var = %d"
              %(len(generator_var), len(discriminator_var), len(other_var)))
        assert len(generator_var) != 0 and len(discriminator_var) != 0
        saver = tf.train.Saver(max_to_keep=1, var_list=generator_var + discriminator_var)
        with tf.Session(config=config) as sess:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)
            sess.run(tf.initialize_variables(other_var))
            saver = tf.train.Saver(max_to_keep=1)
            saver.save(sess,checkpoint)
    else:
        # If there is a checkpoint, then the sketch generator variables should already be stored in there.
        if a.use_sketch_loss and a.checkpoint is None:
            sketch_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=SKETCH_VAR_SCOPE_PREFIX + "generator")
            other_var = [var for var in tf.global_variables() if
                         (var not in sketch_var)]
            print("number of sketch var = %d, number of other var = %d"
                  % (len(sketch_var), len(other_var)))
            assert len(sketch_var) != 0
            saver = tf.train.Saver(max_to_keep=1, var_list=sketch_var)
            with tf.Session(config=config) as sess:
                print("loading sketch generator model from checkpoint")
                pretrained_sketch_checkpoint = tf.train.latest_checkpoint(a.pretrained_sketch_net_path)
                saver.restore(sess, pretrained_sketch_checkpoint)
                sess.run(tf.initialize_variables(other_var))
                saver = tf.train.Saver(max_to_keep=1)
        else:
            saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)


        if a.mode == "test":
            # testing
            # run a single epoch over all input data
            for step in range(examples.steps_per_epoch):
                results = sess.run(display_fetches)
                filesets = save_images(results, image_dir)
                for i, path in enumerate(results["paths"]):
                    print(step * a.batch_size + i + 1, "evaluated image", os.path.basename(path))
                index_path = append_index(filesets)

            print("wrote index at", index_path)
        else:
            # training
            max_steps = 2**32
            if a.max_epochs is not None:
                max_steps = examples.steps_per_epoch * a.max_epochs
            if a.max_steps is not None:
                max_steps = a.max_steps

            start_time = time.time()
            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                # # First get input and target
                # io_fetches = {
                #     "inputs": examples.inputs,
                #     "targets": examples.targets
                # }
                # # if should(a.display_freq):
                # #     io_fetches["display"] = display_fetches
                # io_results = sess.run(io_fetches, options=options, run_metadata=run_metadata)
                #
                # print(np.sum(io_results["inputs"][...,1:]))
                # print(np.sum(io_results["targets"]))

                if a.use_bin:

                    current_inputs, current_targets = sess.run([examples.inputs, examples.targets])
                    current_targets_bin = i2b_encoder.img_to_bin(current_targets)
                    feed_dict = {inputs: current_inputs, targets: current_targets_bin}
                else:
                    feed_dict = None

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1
                    if a.use_sketch_loss:
                        fetches["gen_loss_sketch"] = model.gen_loss_sketch

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                # Without this step the hints won't be updated.
                # if a.use_hint:
                #     fetches["inputs"] = examples.inputs
                #     fetches["update_hint"] = examples.input_hints
                #     fetches["deprocessed_hints"] = deprocessed_hints

                # results = sess.run(fetches, options=options, run_metadata=run_metadata,
                #                    feed_dict={input_ph:io_results["inputs"], target_ph:io_results["targets"]})
                results = sess.run(fetches, options=options, run_metadata=run_metadata, feed_dict=feed_dict)

                # print(np.sum(results["update_hint"]))
                # print(np.sum(results["inputs"][...,1:]))
                # print(np.sum(results["deprocessed_hints"]))

                if should(a.summary_freq):
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")

                    if a.use_bin:
                        contents = results["display"]["outputs"]
                        content_images = []
                        for content_i, content in enumerate(contents):
                            content_image = i2b_encoder.bin_to_img(content, t=1.00)
                            print('current_targets_bin shape: %s sum: %f' %(str(content.shape), np.sum(content)))
                            print('content shape: %s sum: %f' %(str(content_image.shape), np.sum(content_image)))
                            content_images.append(content_image)

                        content_images = np.array(content_images, dtype=np.float32)
                        outputs_images_feed_dict = {outputs_images: content_images}
                        encoded_content_images, = sess.run([encoded_outputs], feed_dict=outputs_images_feed_dict)
                        results["display"]["outputs"] = encoded_content_images
                        # print('encoded images shape: %s' %(str(encoded_images.shape)))
                        # results["display"]["outputs"] = encoded_images
                    filesets = save_images(results["display"], image_dir, step=results["global_step"])
                    # filesets = save_images(io_results["display"], image_dir, step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    global_step = results["global_step"]
                    print("progress  epoch %d  step %d  image/sec %0.1f" % (global_step // examples.steps_per_epoch, global_step % examples.steps_per_epoch, global_step * a.batch_size / (time.time() - start_time)))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])
                    if a.use_sketch_loss:
                        print("gen_loss_sketch", results["gen_loss_sketch"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()

"""

python pix2pix_w_hint_512_more_color.py --mode train --output_dir sanity_check_train_more_color --max_epochs 2000 --input_dir /mnt/tf_drive/home/ubuntu/pixiv_full_128_combined/tiny --which_direction AtoB --display_freq=1000 --gray_input_a --batch_size 1 --lr 0.008 --gpu_percentage 0.45 --scale_size=143 --crop_size=128 --use_sketch_loss --pretrained_sketch_net_path pixiv_full_128_to_sketch_train --use_hint --use_bin
--mode train --output_dir sanity_check_train --max_epochs 200 --input_dir /home/xor/pixiv_full_128_combined/tiny --which_direction AtoB --gray_input_a --display_freq=5 --use_hint
--mode test --output_dir sanity_check_test --input_dir /home/xor/pixiv_full_128_combined/tiny --which_direction AtoB --gray_input_a --use_hint --checkpoint sanity_check_train
"""
"""
TODO: don't forget to add  --use_bin
python pix2pix_w_hint_512.py --mode train --output_dir pixiv_full_512_w_hint_train --max_epochs 20 --input_dir /mnt/data_drive/home/ubuntu/pixiv_full_512_combined/train --which_direction AtoB --display_freq=1000 --gray_input_a --use_hint --batch_size 4 --lr 0.0008 --gpu_percentage 0.45 --checkpoint=pixiv_full_512_w_hint_train
python pix2pix_w_hint_512.py --mode train --output_dir pixiv_full_128_wgan_sketch_loss --max_epochs 20 --input_dir /mnt/tf_drive/home/ubuntu/pixiv_full_128_combined/train --which_direction AtoB --display_freq=1000 --gray_input_a --batch_size 4 --lr 0.0008 --gpu_percentage 0.45 --scale_size=143 --crop_size=128 --use_sketch_loss --pretrained_sketch_net_path pixiv_full_128_to_sketch_train
python pix2pix_w_hint_512.py --mode train --output_dir pixiv_full_128_wgan_w_hint_sketch_loss --max_epochs 20 --input_dir /mnt/tf_drive/home/ubuntu/pixiv_full_128_combined/train --which_direction AtoB --display_freq=1000 --gray_input_a --batch_size 4 --lr 0.0008 --gpu_percentage 0.45 --scale_size=143 --crop_size=128 --use_sketch_loss --pretrained_sketch_net_path pixiv_full_128_to_sketch_train --use_hint --use_bin
python pix2pix_w_hint_512.py --mode train --output_dir pixiv_full_512_wgan_w_hint_sketch_loss --max_epochs 20 --input_dir /mnt/data_drive/home/ubuntu/pixiv_full_512_combined/train --which_direction AtoB --display_freq=1000 --gray_input_a --batch_size 4 --lr 0.0002 --gpu_percentage 0.45 --scale_size=572 --crop_size=512 --use_sketch_loss --pretrained_sketch_net_path pixiv_full_128_to_sketch_train --use_hint --from_128 --checkpoint=pixiv_full_512_wgan_w_hint_sketch_loss
# TO train a network that turns colored images into sketches:
python pix2pix_w_hint_512.py --mode train --output_dir pixiv_full_128_to_sketch_train --max_epochs 20 --input_dir /mnt/tf_drive/home/ubuntu/pixiv_full_128_combined/train --which_direction BtoA --display_freq=1000 --gray_input_a --batch_size 4 --lr 0.002 --gpu_percentage 0.45 --scale_size=143 --crop_size=128

python pix2pix_w_hint_512.py --mode train --output_dir pixiv_full_128_to_sketch_train --max_epochs 20 --input_dir /mnt/tf_drive/home/ubuntu/pixiv_full_128_combined/train --which_direction BtoA --display_freq=1000 --gray_input_a --batch_size 4 --lr 0.002 --gpu_percentage 0.45 --scale_size=143 --crop_size=128 --train_sketch


"""

"""
python pix2pix_w_hint.py --mode test --output_dir pixiv_full_128_w_hint_test --input_dir /mnt/tf_drive/home/ubuntu/pixiv_full_128_combined/test --checkpoint pixiv_full_128_w_hint_train --gpu_percentage 0.45 --use_hint
python pix2pix_w_hint.py --mode test --output_dir pixiv_full_128_tiny_test --input_dir /mnt/tf_drive/home/ubuntu/pixiv_full_128_combined/test --checkpoint pixiv_full_128_tiny --gpu_percentage 0.45 --use_hint
python pix2pix_w_hint.py --mode test --output_dir pixiv_full_128_w_hint_test --input_dir /mnt/tf_drive/home/ubuntu/pixiv_full_128_combined/test --checkpoint pixiv_full_128_w_hint_train --gpu_percentage 0.45 --use_hint --user_hint_path=BLANK
"""