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

from general_util import imread, get_all_image_paths
from neural_util import decode_image, decode_image_with_file_name
from sketches_util import sketch_extractor

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--user_hint_path", default=None, help="path to a hint image.")
parser.add_argument("--pretrained_sketch_net_path", default=None, help="path to the pretrained sketch network checkpoint")
parser.add_argument("--single_input", dest="single_input", action="store_true",
                    help="Input image is a single image instead of a combination of the source and target.")
parser.set_defaults(single_input=False)
parser.add_argument("--output_ab", dest="output_ab", action="store_true",
                    help="The generator network outputs only ab channel instead of lab. "
                         "Must be used with lab_colorization.")
parser.set_defaults(output_ab=False)
parser.add_argument("--gen_sketch_input", dest="gen_sketch_input", action="store_true",
                    help="Input image is generated using the sketch generator network.")
parser.set_defaults(gen_sketch_input=False)

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
parser.add_argument("--ngf", type=int, default=32, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=32, help="number of discriminator filters in first conv layer")
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
parser.add_argument("--gpu_percentage", type=float, default=0.45, help="precent of gpu memory allocated.")
parser.add_argument("--from_128", dest="from_128", action="store_true", help="Indicate whether model is from 128x128.")
parser.add_argument("--train_sketch", dest="train_sketch", action="store_true",
                    help="Indicate whether the model is for sketch generation. Variable scope will change accordingly.")
parser.add_argument("--use_sketch_loss", dest="use_sketch_loss", action="store_true",
                    help="Use the pretrained sketch generator network to compare the sketches of the generated image "
                         "versus that of the original image.")
parser.add_argument("--sketch_weight", type=float, default=1.0, help="weight on sketch loss term.")
parser.add_argument("--hint_prob", type=float, default=0.5, help="The probability of having hint as extra input channels.")
parser.add_argument("--mix_prob", type=float, default=0.5, help="The probability of having old sketch as the input instead of the new one.")



a = parser.parse_args()

if a.use_sketch_loss and a.pretrained_sketch_net_path is None:
    parser.error("If you want to use sketch loss, please provide a valid pretrained_sketch_net_path.")
if a.gen_sketch_input and not a.use_sketch_loss:
    parser.error("If you want to use gen_sketch_input, please also turn on sketch loss.")
if a.gen_sketch_input and a.mix_prob > 0:
    parser.error("If you want to use gen_sketch_input, please set mix_prob to 0 or below.")
if a.mode != "test" and a.single_input and (a.mix_prob < 1 and not a.gen_sketch_input):
    parser.error("single input mode is intended for either test mode where only output is needed, or training where the "
                 "sketch is generated simply by dilation.")
if a.output_ab and not a.lab_colorization :
    parser.error("If you want the generator to output only a and b channels, please also add lab_colorization flag.")
if a.sketch_weight != 10.0:
    input("Are you sure you don't want sketch_weight to be 10.0?")

EPS = 1e-12
CROP_SIZE = a.crop_size  # 128 # 256
CLIP_VALUE = 0.04  # 0.01
APPROXIMATE_NUMBER_OF_TOTAL_PARAMETERS = 98218176

SKETCH_VAR_SCOPE_PREFIX = "sketch_"

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch, input_hints")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, real_sketches, fake_sketches, discrim_loss, gen_loss_GAN, gen_loss_L1, gen_loss_sketch, train")


def conv(batch_input, out_channels, stride, shift=4, pad = 1, trainable=True):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [shift, shift, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02), trainable=trainable)
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        if pad > 0:
            padded_input = tf.pad(batch_input, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="CONSTANT")
        else:
            assert pad == 0
            padded_input = batch_input
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


def deconv(batch_input, out_channels, stride = 2, shift = 4, trainable=True):
    with tf.variable_scope("deconv"):
        # batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        input_shape = tf.shape(batch_input)
        batch, in_height, in_width, = input_shape[0], input_shape[1], input_shape[2],
        in_channels = int(batch_input.get_shape()[3])
        filter = tf.get_variable("filter", [shift, shift, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02), trainable=trainable)
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * stride, in_width * stride, out_channels], [1, stride, stride, 1], padding="SAME")
        # return conv
        output_shape = tf.shape(conv)
        output_height, output_width, = output_shape[1], output_shape[2],
        conv = tf.reshape(conv, [batch, output_height, output_width, out_channels])
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
    # Input range [0, 255]
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
    # Input range for l is 0 ~ 100 and ab is -110 ~ 110
    # Output range is 0 ~ 1....???
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

def safe_concat(layer_1, layer_2, axis):
    concat_first_layer_shape = tf.shape(layer_1)
    concat_second_layer_shape = tf.shape(layer_2)
    return tf.cond(tf.logical_and(tf.equal(concat_first_layer_shape[1], concat_second_layer_shape[1]),tf.equal(concat_first_layer_shape[2], concat_second_layer_shape[2])), lambda: tf.concat(axis, [layer_1, layer_2]), lambda: tf.concat(axis, [tf.image.resize_nearest_neighbor(layer_1, (concat_second_layer_shape[1], concat_second_layer_shape[2])), layer_2]))

def create_generator(generator_inputs, generator_outputs_channels, trainable=True):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        # output = conv(generator_inputs, a.ngf, stride=2, trainable=trainable)
        # output = conv(generator_inputs, a.ngf, stride=1, shift=3, trainable=trainable)
        # layers.append(output)
        convolved = conv(generator_inputs, a.ngf, stride=1, shift=3, trainable=trainable)
        output = batchnorm(convolved, trainable=trainable)
        # rectified = lrelu(output, 0.2)
        rectified = tf.nn.relu(output)
        layers.append(rectified)

    layer_specs = [
        a.ngf * 2,  # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
        a.ngf * 2,  # encoder_3: [batch, 128, 128, ngf * 2] => [batch, 128, 128, ngf * 2]
        a.ngf * 4,  # encoder_4: [batch, 128, 128, ngf * 2] => [batch, 64, 64, ngf * 4]
        a.ngf * 4,  # encoder_5: [batch, 64, 64, ngf * 4] => [batch, 64, 64, ngf * 4]
        a.ngf * 8,  # encoder_6: [batch, 64, 64, ngf * 4] => [batch, 32, 32, ngf * 8]
        a.ngf * 8,  # encoder_7: [batch, 32, 32, ngf * 8] => [batch, 32, 32, ngf * 8]
        a.ngf * 16,  # encoder_8: [batch, 32, 32, ngf * 8] => [batch, 16, 16, ngf * 16]
        a.ngf * 16,  # encoder_9: [batch, 16, 16, ngf * 16] => [batch, 16, 16, ngf * 16]
    ]
    #
    #  layer_specs = [
    #     a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
    #     a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
    #     a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
    #     a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
    #     a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
    #     a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
    # ]
    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            # rectified = lrelu(layers[-1], 0.2)
            if (len(layers) + 1) % 2 == 0:
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv(layers[-1], out_channels, stride=2, shift=4, trainable=trainable)
            else:
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv(layers[-1], out_channels, stride=1, shift=3, trainable=trainable)
            output = batchnorm(convolved, trainable=trainable)
            rectified = tf.nn.relu(output)
            layers.append(rectified)
    # for out_channels in layer_specs:
    #     with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
    #         rectified = lrelu(layers[-1], 0.2)
    #         # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
    #         convolved = conv(rectified, out_channels, stride=2, trainable=trainable)
    #         output = batchnorm(convolved, trainable=trainable)
    #         layers.append(output)

    layer_specs = [
        (a.ngf * 16),  # decoder_8: [batch, 16, 16, ngf * 16 * 2]=> [batch, 32, 32, ngf * 16]
        (a.ngf * 8),  # decoder_7: [batch, 32, 32, ngf * 16] => [batch, 32, 32, ngf * 8]
        (a.ngf * 8),  # decoder_6: [batch, 32, 32, ngf * 8 * 2] => [batch, 64, 64, ngf * 8]
        (a.ngf * 4),  # decoder_5: [batch, 64, 64, ngf * 8] => [batch, 64, 64, ngf * 4]
        (a.ngf * 4),  # decoder_4: [batch, 64, 64, ngf * 4 * 2] => [batch, 128, 128, ngf * 4]
        (a.ngf * 2),  # decoder_3: [batch, 128, 128, ngf * 4] => [batch, 128, 128, ngf * 2]
        (a.ngf * 2),  # decoder_2: [batch, 128, 128, ngf * 2 * 2] => [batch, 256, 256, ngf * 2]
        (a.ngf * 1),  # decoder_1: [batch, 256, 256, ngf * 2] => [batch, 256, 256, ngf]
    ]
    # layer_specs = [
    #     (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
    #     (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
    #     (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
    #     (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
    #     (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
    #     (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    # ]
    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer % 2 != 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
                # [batch, in_height, in_width, in_channels] => [batch, in_height, in_width, out_channels]
                output = deconv(input, out_channels, 1, 3, trainable=trainable)
            else:
                # Can't find concat_v2 so commenting this out.
                # input = tf.concat_v2([layers[-1], layers[skip_layer]], axis=3)
                if decoder_layer == 0:
                    # input = tf.concat(3, [layers[-1], layers[-2]])
                    input = safe_concat(layers[-1], layers[-2], 3)
                else:
                    # input = tf.concat(3, [layers[-1], layers[skip_layer]])
                    input = safe_concat(layers[-1], layers[skip_layer], 3)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = deconv(input, out_channels, 2, 4, trainable=trainable)
            # if decoder_layer == 0:
            #     # first decoder layer doesn't have skip connections
            #     # since it is directly connected to the skip_layer
            #     input = layers[-1]
            # else:
            #     # Can't find concat_v2 so commenting this out.
            #     #input = tf.concat_v2([layers[-1], layers[skip_layer]], axis=3)
            #     input = tf.concat(3, [layers[-1], layers[skip_layer]])
            #
            # rectified = tf.nn.relu(input)
            # # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            # output = deconv(rectified, out_channels, trainable=trainable)
            output = batchnorm(output, trainable=trainable)
            #
            # if dropout > 0.0:
            #     output = tf.nn.dropout(output, keep_prob=1 - dropout)
            output = tf.nn.relu(output)
            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        # input = tf.concat_v2([layers[-1], layers[0]], axis=3)

        # input = tf.concat(3, [layers[-1], layers[0]])
        input = safe_concat(layers[-1], layers[0], 3)
        output = deconv(input, generator_outputs_channels, 1, 3, trainable=trainable)
        # output = tf.tanh(output)
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
        # convolved = conv(input, a.ndf, stride=2)
        convolved = conv(input, a.ndf, stride=2, shift=4)
        normed = batchnorm(convolved)
        rectified = lrelu(normed, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    layer_specs = [
        a.ndf,  # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
        a.ndf * 2,  # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
        a.ndf * 2,  # encoder_3: [batch, 128, 128, ngf * 2] => [batch, 128, 128, ngf * 2]
        a.ndf * 4,  # encoder_4: [batch, 128, 128, ngf * 2] => [batch, 64, 64, ngf * 4]
        a.ndf * 4,  # encoder_5: [batch, 64, 64, ngf * 4] => [batch, 64, 64, ngf * 4]
        a.ndf * 8,  # encoder_6: [batch, 64, 64, ngf * 4] => [batch, 32, 32, ngf * 8]
    ]
    for out_channels in layer_specs:
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            if (len(layers) + 1) % 2 == 0:
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv(layers[-1], out_channels, stride=1, shift=3)
            else:
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv(layers[-1], out_channels, stride=2, shift=4)

            normed = batchnorm(convolved)
            # rectified = lrelu(normed, 0.2)
            rectified = tf.nn.relu(normed)
            layers.append(rectified)
    # for i in range(n_layers):
    #     with tf.variable_scope("layer_%d" % (len(layers) + 1)):
    #         out_channels = a.ndf * min(2**(i+1), 8)
    #         stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
    #         convolved = conv(layers[-1], out_channels, stride=stride)
    #         normalized = batchnorm(convolved)
    #         rectified = lrelu(normalized, 0.2)
    #         layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        # convolved = conv(rectified, out_channels=1, stride=1)
        # output = tf.sigmoid(convolved)
        # layers.append(output)
        # With WGAN, sigmoid for the last layer is no longer needed
        convolved = conv(rectified, out_channels=1, stride=1, shift=3)
        layers.append(convolved)

    return layers[-1]


def load_examples(user_hint = None):
    if not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    # input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    # decode = tf.image.decode_jpeg
    # if len(input_paths) == 0:
    #     input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
    #     decode = tf.image.decode_png
    input_paths = get_all_image_paths(a.input_dir)
    decode = decode_image_with_file_name

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
        raw_input = decode(contents, paths, channels=3)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1]  # [height, width, channels]

        # a_images = raw_input[:,:width//2,:] * 2 - 1
        # b_images = raw_input[:,width//2:,:] * 2 - 1

        # Modified code: change a_images and b_images to 0~1 before turning into grayscale and rescaling.
        if a.single_input:
            a_images = raw_input
            b_images = raw_input
        else:
            a_images = raw_input[:, :width // 2, :]
            b_images = raw_input[:, width // 2:, :]
        if a.gray_input_a:
            a_images = tf.image.rgb_to_grayscale(a_images)
        if a.gray_input_b:
            b_images = tf.image.rgb_to_grayscale(b_images)

        if a.mix_prob >= 1:
            a_images = sketch_extractor(b_images, color_space="rgb", max_val=1.0, min_val=0.0)
        elif a.mix_prob <= 0:
            if a.gen_sketch_input:
                with tf.variable_scope(SKETCH_VAR_SCOPE_PREFIX + "generator") as scope:
                    a_images = create_generator(tf.expand_dims(b_images,axis=0), 1, trainable=False)[0]

            else:
                a_images = a_images
        else:
            random_mix_condition = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32,
                                                     name="random_mix_condition")
            mix_prob = tf.constant(a.mix_prob)
            a_images = tf.cond(tf.greater_equal(random_mix_condition, mix_prob), lambda: a_images,
                               lambda: sketch_extractor(b_images, color_space="rgb", max_val=1.0, min_val=0.0))

        if a.lab_colorization:
            # if a.which_direction=="AtoB":
            #     lab = rgb_to_lab(b_images)
            # else:
            #     lab = rgb_to_lab(a_images)

            # This doesn't work when I'm trying to train sketch gen...
            # if a.which_direction=="AtoB":
            #     lab = rgb_to_lab(b_images)
            # else:
            #     lab = rgb_to_lab(a_images)
            lab = rgb_to_lab(b_images)
            L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)

            L_chan = tf.expand_dims(L_chan, axis=2) / 50 - 1 # black and white with input range [0, 100]
            ab_chan = tf.stack([a_chan, b_chan], axis=2) / 110 # color channels with input range ~[-110, 110], not exact

            # if a.which_direction=="AtoB":
            #     b_images = tf.concat(2,[L_chan, ab_chan])
            #     a_images =  a_images * 2 - 1
            # else:
            #     a_images = tf.concat(2,[L_chan, ab_chan])
            #     b_images = b_images * 2 - 1
            b_images = tf.concat(2,[L_chan, ab_chan])
            a_images = a_images * 2 - 1
        else:
            a_images = a_images * 2 - 1
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
                half_constant = tf.constant(a.hint_prob)

                output = tf.cond(tf.greater_equal(random_condition, half_constant), lambda: output, lambda: tf.scatter_nd_update(output, rd_indices_2d, hints))
        else:
            output = tf.assign(output, user_hint)

        assert len(output.get_shape().as_list()) == 3
        return tf.concat(2, (inputs, output), name='input_concat'), output
        # return tf.concat(2, (inputs, output), name='input_concat'), output

    with tf.name_scope("target_images"):
        target_images = transform(targets)
    with tf.name_scope("input_images"):
        input_images = transform(inputs)


        # random_mix_condition = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32,
        #                                          name="random_mix_condition")
        # mix_prob = tf.constant(a.mix_prob)
        # input_images = tf.cond(tf.greater_equal(random_mix_condition, mix_prob), lambda: input_images,
        #                    lambda: sketch_extractor(target_images, max_val=1.0, min_val=-1.0, color_space="lab" if a.lab_colorization else "rgb"))


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

    if a.use_sketch_loss:
        with tf.variable_scope(SKETCH_VAR_SCOPE_PREFIX + "generator", reuse=True if a.gen_sketch_input else False) as scope:
            real_sketches = create_generator(targets, 1, trainable=False)
            # real_sketches = sketch_extractor(targets, color_space="lab" if a.lab_colorization else "rgb")

    with tf.variable_scope("generator" if not a.train_sketch else SKETCH_VAR_SCOPE_PREFIX + "generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        if a.output_ab:
            assert out_channels == 3
            ab_outputs = create_generator(inputs, 2)
            # Concatenate the l layer in the input with the ab output by the generator.
            outputs = tf.concat(3, (inputs[..., :1], ab_outputs))
            print(outputs.get_shape().as_list())
            assert int(outputs.get_shape()[-1]) == out_channels
        else:
            if a.gen_sketch_input:
                assert a.use_sketch_loss
                # Always use the sketch generated as the input if the gen_sketch_input is on.
                outputs = create_generator(real_sketches,
                                           out_channels)
            else:
                outputs = create_generator(inputs,
                                           out_channels)  # if not a.train_sketch else create_sketch_generator(inputs, out_channels)

    if a.use_sketch_loss:
        with tf.variable_scope(SKETCH_VAR_SCOPE_PREFIX + "generator", reuse=True) as scope:
            fake_sketches = create_generator(outputs, 1, trainable=False)
            # fake_sketches = sketch_extractor(outputs, color_space="lab" if a.lab_colorization else "rgb")


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
        # # minimizing -tf.log will try to get inputs to 1
        # # predict_real => 1
        # # predict_fake => 0
        # discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

        # Use wgan loss
        discrim_loss = -tf.reduce_mean(predict_real) + tf.reduce_mean(predict_fake)

    with tf.name_scope("generator_loss"):
        # # predict_fake => 1
        # # abs(targets - outputs) => 0
        # gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))

        # WGAN loss
        gen_loss_GAN = -tf.reduce_mean(predict_fake)
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        if a.use_sketch_loss:
            gen_loss_sketch = tf.reduce_mean(tf.abs(fake_sketches - real_sketches))
            gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight + gen_loss_sketch * a.sketch_weight
        else:
            gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator" if not a.train_sketch else SKETCH_VAR_SCOPE_PREFIX + "discriminator")]
        # discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        # WGAN does not use momentum based optimizer
        discrim_optim = tf.train.RMSPropOptimizer(a.lr)
        # discrim_train = discrim_optim.minimize(discrim_loss, var_list=discrim_tvars)

        # WGAN adds a clip and train discriminator 5 times
        discrim_min = discrim_optim.minimize(discrim_loss, var_list=discrim_tvars)
        discrim_clips = [var.assign(tf.clip_by_value(var, -CLIP_VALUE, CLIP_VALUE)) for var in discrim_tvars]
        # No difference between control dependencies and group.
        # with tf.control_dependencies([discrim_min] + discrim_clips):
        #     discrim_train = tf.no_op("discrim_train")
        with tf.control_dependencies([discrim_min]):
            discrim_train = tf.group(*discrim_clips)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator" if not a.train_sketch else SKETCH_VAR_SCOPE_PREFIX + "generator")]
            # gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_optim = tf.train.RMSPropOptimizer(a.lr)
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
        real_sketches=real_sketches if a.use_sketch_loss else None,
        fake_sketches=fake_sketches if a.use_sketch_loss else None,
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
        kinds = ["outputs", ]
        if not (a.single_input and a.mode == "test"):
            kinds = kinds + ["inputs", "targets"]
            if a.use_hint:
                kinds.append("hints")
            if a.use_sketch_loss:
                kinds = kinds + ["real_sketches", "fake_sketches"]
        # kinds = ["outputs",] if a.single_input else (["inputs", "hints", "outputs", "targets"] if a.use_hint else ["inputs", "outputs", "targets"])
        for kind in kinds:
            if (a.single_input and a.mode == "test"):
                # Do not modify file name when single input.
                filename = name + ".png"
            else:
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
        if not (a.single_input and a.mode == "test"):
            if a.use_hint:
                if a.use_sketch_loss:
                    index.write("<th>name</th><th>input</th><th>hint</th><th>output</th><th>target</th><th>real_sketch</th><th>fake_sketch</th></tr>")
                else:
                    index.write("<th>name</th><th>input</th><th>hint</th><th>output</th><th>target</th></tr>")
            else:
                if a.use_sketch_loss:
                    index.write("<th>name</th><th>input</th><th>output</th><th>target</th><th>real_sketch</th><th>fake_sketch</th></tr>")
                else:
                    index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")
        else:
            index.write("<th>name</th><th>output</th></tr>")

    for fileset in filesets:
        index.write("<tr>")
        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])
        kinds = ["outputs", ]
        if not (a.single_input and a.mode == "test"):
            kinds = ["inputs", "hints", "outputs", "targets"] if a.use_hint else ["inputs", "outputs", "targets"]
            if a.use_sketch_loss:
                kinds = kinds + ["real_sketches", "fake_sketches"]
        for kind in kinds:
            index.write("<td><img src=\"images/%s\"></td>" % urllib.quote(fileset[kind]))

        index.write("</tr>")
    return index_path

def main():
    if a.from_128:
        assert a.checkpoint is not None

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

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
        if not (a.mix_prob == 1.0 or a.mix_prob <= 0):
            input("Are you sure you don't want to set the mix_prob to 1 (always using the simple sketch generator) or "
                  "0 (always use the input image)?")
        if a.user_hint_path is not None:
            a.hint_prob = 1.0
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

    model = create_model(examples.inputs, examples.targets)
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
            elif num_channels == 3:
                # (a, b) color channels, convert to rgb
                # a_chan and b_chan have range [-1, 1] => [-110, 110]
                L_chan, a_chan, b_chan = tf.unstack(image, axis=3)
                L_chan = (L_chan + 1) * 50
                a_chan = a_chan * 110
                b_chan = b_chan * 110
                lab = tf.stack([L_chan, a_chan, b_chan], axis=3)
                rgb = lab_to_rgb(lab)
                return tf.image.convert_image_dtype(rgb, dtype=tf.uint8, saturate=True)
            elif num_channels == 4:
                print('using hint! correct!')
                # (a, b) color channels, convert to rgb
                # a_chan and b_chan have range [-1, 1] => [-110, 110]
                L_chan, a_chan, b_chan, opacity_chan = tf.unstack(image, axis=3)
                L_chan = (L_chan + 1) * 50
                a_chan = a_chan * 110
                b_chan = b_chan * 110
                lab = tf.stack([L_chan, a_chan, b_chan], axis=3)
                rgb = lab_to_rgb(lab)
                opacity_chan = tf.expand_dims((opacity_chan + 1) / 2,3)  # [-1, 1] => [0, 1]
                rgba = tf.concat(3, (rgb, opacity_chan))
                return tf.image.convert_image_dtype(rgba, dtype=tf.uint8, saturate=True)
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
        deprocessed_outputs = deprocess(model.outputs)

    if a.use_sketch_loss:
        with tf.name_scope("deprocess_real_sketches"):
            deprocessed_real_sketches = deprocess(model.real_sketches)
        with tf.name_scope("deprocess_fake_sketches"):
            deprocessed_fake_sketches = deprocess(model.fake_sketches)

    with tf.name_scope("encode_images"):
        if a.use_hint:
            if a.use_sketch_loss:
                display_fetches = {
                    "paths": examples.paths,
                    "inputs": tf.map_fn(tf.image.encode_png, deprocessed_inputs, dtype=tf.string, name="input_pngs"),
                    "hints": tf.map_fn(tf.image.encode_png, deprocessed_hints, dtype=tf.string, name="hint_pngs"),
                    "targets": tf.map_fn(tf.image.encode_png, deprocessed_targets, dtype=tf.string, name="target_pngs"),
                    "outputs": tf.map_fn(tf.image.encode_png, deprocessed_outputs, dtype=tf.string, name="output_pngs"),
                    "real_sketches": tf.map_fn(tf.image.encode_png, deprocessed_real_sketches, dtype=tf.string, name="real_sketches_pngs"),
                    "fake_sketches": tf.map_fn(tf.image.encode_png, deprocessed_fake_sketches, dtype=tf.string, name="fake_sketches_pngs"),
                }
            else:
                display_fetches = {
                    "paths": examples.paths,
                    "inputs": tf.map_fn(tf.image.encode_png, deprocessed_inputs, dtype=tf.string, name="input_pngs"),
                    "hints": tf.map_fn(tf.image.encode_png, deprocessed_hints, dtype=tf.string, name="hint_pngs"),
                    "targets": tf.map_fn(tf.image.encode_png, deprocessed_targets, dtype=tf.string, name="target_pngs"),
                    "outputs": tf.map_fn(tf.image.encode_png, deprocessed_outputs, dtype=tf.string, name="output_pngs"),
                }
        else:
            if a.use_sketch_loss:
                display_fetches = {
                    "paths": examples.paths,
                    "inputs": tf.map_fn(tf.image.encode_png, deprocessed_inputs, dtype=tf.string, name="input_pngs"),
                    "targets": tf.map_fn(tf.image.encode_png, deprocessed_targets, dtype=tf.string, name="target_pngs"),
                    "outputs": tf.map_fn(tf.image.encode_png, deprocessed_outputs, dtype=tf.string, name="output_pngs"),
                    "real_sketches": tf.map_fn(tf.image.encode_png, deprocessed_real_sketches, dtype=tf.string, name="real_sketches_pngs"),
                    "fake_sketches": tf.map_fn(tf.image.encode_png, deprocessed_fake_sketches, dtype=tf.string, name="fake_sketches_pngs"),
                }
            else:
                display_fetches = {
                    "paths": examples.paths,
                    "inputs": tf.map_fn(tf.image.encode_png, deprocessed_inputs, dtype=tf.string, name="input_pngs"),
                    "targets": tf.map_fn(tf.image.encode_png, deprocessed_targets, dtype=tf.string, name="target_pngs"),
                    "outputs": tf.map_fn(tf.image.encode_png, deprocessed_outputs, dtype=tf.string, name="output_pngs"),
                }


    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", deprocessed_inputs)
    if a.use_hint:
        with tf.name_scope("hints_summary"):
            tf.summary.image("inputs", deprocessed_hints)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", deprocessed_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", deprocessed_outputs)

    if a.use_sketch_loss:
        with tf.name_scope("real_sketches_summary"):
            tf.summary.image("real_sketches", deprocessed_real_sketches)
        with tf.name_scope("fake_sketches_summary"):
            tf.summary.image("fake_sketches", deprocessed_fake_sketches)

    with tf.name_scope("predict_real_summary"):
        # Changed this because I no longer have tanh in the discriminator. In order for the image to be within range,
        # I need to apply tanh here.
        # tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))
        tf.summary.image("predict_real", tf.image.convert_image_dtype((tf.nn.tanh(model.predict_real) + 1) / 2, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        # tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))
        tf.summary.image("predict_fake", tf.image.convert_image_dtype((tf.nn.tanh(model.predict_fake)) + 1 / 2, dtype=tf.uint8))

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
    if a.gpu_percentage > 0:
        config.gpu_options.per_process_gpu_memory_fraction = a.gpu_percentage
    else:
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
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
        if a.use_sketch_loss and a.mode != "test" and a.checkpoint is None:
            sketch_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=SKETCH_VAR_SCOPE_PREFIX + "generator")
            # This is a sanity check to make sure sketch variables are not trainable.
            assert len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=SKETCH_VAR_SCOPE_PREFIX + "generator")) == 0
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
                sketch_var_value_before = sketch_var[0].eval()[0, 0, 0, 0]
                print("Sketch var value before supervised session: %f" %(sketch_var_value_before))
                saver = tf.train.Saver(max_to_keep=1)
                saver.save(sess,save_path=os.path.join(a.output_dir, "model"))
        else:
            saver = tf.train.Saver(max_to_keep=1)
        # sketch_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=SKETCH_VAR_SCOPE_PREFIX + "generator")
        # if not a.train_sketch:
        #     assert len(sketch_var) == 0
        # saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)
        elif a.use_sketch_loss:
            print("loading model with saved sketch generator variables.")
            checkpoint = tf.train.latest_checkpoint(a.output_dir)
            print(checkpoint)
            assert checkpoint is not None
            saver.restore(sess, checkpoint)

            sketch_var_value_after = sketch_var[0].eval(session=sess)[0, 0, 0, 0]
            print("Sketch var value after supervised session: %f" %(sketch_var_value_after))
            assert sketch_var_value_after == sketch_var_value_before



        if a.mode == "test":
            # testing
            # run a single epoch over all input data
            for step in range(examples.steps_per_epoch):
                results = sess.run(display_fetches)
                filesets = save_images(results, image_dir)
                if not a.single_input:
                    for i, path in enumerate(results["paths"]):
                        print(step * a.batch_size + i + 1, "evaluated image", os.path.basename(path))
                else:
                    if step % 100 == 0:
                        print("Evaluated %d out of %d steps." %(step,examples.steps_per_epoch))
                index_path = append_index(filesets)
            print("wrote index at", index_path)
        else:
            # training
            max_steps = 2**32
            if a.max_epochs is not None:
                max_steps = examples.steps_per_epoch * a.max_epochs
            if a.max_steps is not None:
                max_steps = a.max_steps

            initial_global_step, = sess.run([sv.global_step])

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
                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                # print(np.sum(results["update_hint"]))
                # print(np.sum(results["inputs"][...,1:]))
                # print(np.sum(results["deprocessed_hints"]))

                if should(a.summary_freq):
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], image_dir, step=results["global_step"])
                    # filesets = save_images(io_results["display"], image_dir, step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    global_step = results["global_step"]
                    print("progress  epoch %d  step %d  image/sec %0.1f" % (global_step // examples.steps_per_epoch, global_step % examples.steps_per_epoch, (global_step - initial_global_step) * a.batch_size / (time.time() - start_time)))
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
--mode train --output_dir sanity_check_train --max_epochs 200 --input_dir /home/xor/pixiv_full_128_combined/tiny --which_direction AtoB --gray_input_a --display_freq=5 --use_hint
--mode test --output_dir sanity_check_test --input_dir /home/xor/pixiv_full_128_combined/tiny --which_direction AtoB --gray_input_a --use_hint --checkpoint sanity_check_train
"""
"""
# Sanity check
python pix2pix_w_hint_lab_wgan_larger_sketch_mix.py --mode train --output_dir pix2pix_w_hint_lab_wgan_larger_sketch_mix_sanity_check --max_epochs 2000 --input_dir /mnt/tf_drive/home/ubuntu/pixiv_full_128_combined/tiny --which_direction AtoB --display_freq=200 --gray_input_a --batch_size 1 --lr 0.0008 --gpu_percentage 0.1 --scale_size=143 --crop_size=128 --use_sketch_loss --pretrained_sketch_net_path sketch_colored_pair_cleaned_128_w_hint_lab_wgan_larger_train_sketch --use_hint --lab_colorization
# Train
python pix2pix_w_hint_lab_wgan_larger_sketch_mix.py --mode train --output_dir pixiv_downloaded_128_w_hint_lab_wgan_larger_sketch_mix --max_epochs 20 --input_dir /mnt/data_drive/home/ubuntu/pixiv_downloaded_sketches_lnet_128_combined/train --which_direction AtoB --display_freq=1000 --gray_input_a --batch_size 4 --lr 0.0008 --gpu_percentage 0.75 --scale_size=143 --crop_size=128 --use_sketch_loss --pretrained_sketch_net_path sketch_colored_pair_cleaned_128_w_hint_lab_wgan_larger_train_sketch --use_hint --lab_colorization
# Train 512
python pix2pix_w_hint_lab_wgan_larger_sketch_mix.py --mode train --output_dir pixiv_downloaded_512_w_hint_lab_wgan_larger_sketch_mix --max_epochs 20 --input_dir /mnt/data_drive/home/ubuntu/pixiv_downloaded_sketches_lnet_512_combined/train --which_direction AtoB --display_freq=1000 --gray_input_a --batch_size 4 --lr 0.0008 --gpu_percentage 0.9 --scale_size=572 --crop_size=512 --use_sketch_loss --pretrained_sketch_net_path pixiv_full_128_to_sketch_train --use_hint --lab_colorization --checkpoint=pixiv_downloaded_512_w_hint_lab_wgan_larger_sketch_mix --from_128
# TO train a network that turns colored images into sketches:
python pix2pix_w_hint_512.py --mode train --output_dir pixiv_full_128_to_sketch_train --max_epochs 20 --input_dir /mnt/tf_drive/home/ubuntu/pixiv_full_128_combined/train --which_direction BtoA --display_freq=1000 --gray_input_a --batch_size 4 --lr 0.002 --gpu_percentage 0.45 --scale_size=143 --crop_size=128

python pix2pix_w_hint_lab_wgan_larger_sketch_mix.py --mode train --output_dir sketch_colored_pair_cleaned_128_w_hint_lab_wgan_larger_sketch_mix_train_sketch --max_epochs 20 --input_dir /mnt/data_drive/home/ubuntu/sketch_colored_pair_128_combined_cleaned/sketch_colored_pair_128_combined/train/ --which_direction BtoA --display_freq=1000 --gray_input_a --batch_size 4 --lr 0.0008 --gpu_percentage 0.45 --scale_size=143 --crop_size=128 --lab_colorization --train_sketch
python pix2pix_w_hint_lab_wgan_larger_sketch_mix.py --mode test --output_dir sketch_colored_pair_cleaned_128_w_hint_lab_wgan_larger_sketch_mix_train_sketch_test --max_epochs 20 --input_dir /mnt/data_drive/home/ubuntu/sketch_colored_pair_128_combined_cleaned/sketch_colored_pair_128_combined/test/ --which_direction BtoA --display_freq=1000 --gray_input_a --batch_size 4 --lr 0.0008 --gpu_percentage 0.45 --scale_size=143 --crop_size=128 --lab_colorization --train_sketch --checkpoint=sketch_colored_pair_cleaned_128_w_hint_lab_wgan_larger_sketch_mix_train_sketch
# Test
python pix2pix_w_hint_lab_wgan_larger_sketch_mix.py --mode test --output_dir pixiv_downloaded_128_w_hint_lab_wgan_larger_sketch_mix_test_with_hint --max_epochs 20 --input_dir /mnt/data_drive/home/ubuntu/pixiv_downloaded_sketches_lnet_128_combined/test --which_direction AtoB --display_freq=1000 --gray_input_a --batch_size 4 --lr 0.0008 --gpu_percentage 0.45 --scale_size=143 --crop_size=128 --use_sketch_loss --pretrained_sketch_net_path pixiv_full_128_to_sketch_train --use_hint --lab_colorization --checkpoint=pixiv_downloaded_128_w_hint_lab_wgan_larger_sketch_mix
python pix2pix_w_hint_lab_wgan_larger_sketch_mix.py --mode test --output_dir pixiv_downloaded_128_w_hint_lab_wgan_larger_sketch_mix_test_no_hint --max_epochs 20 --input_dir /mnt/data_drive/home/ubuntu/pixiv_downloaded_sketches_lnet_128_combined/test --which_direction AtoB --display_freq=1000 --gray_input_a --batch_size 4 --lr 0.0008 --gpu_percentage 0.45 --scale_size=143 --crop_size=128 --use_sketch_loss --pretrained_sketch_net_path pixiv_full_128_to_sketch_train --use_hint --lab_colorization --checkpoint=pixiv_downloaded_128_w_hint_lab_wgan_larger_sketch_mix --user_hint_path=BLANK
python pix2pix_w_hint_lab_wgan_larger_sketch_mix.py --mode test --output_dir pixiv_downloaded_128_w_hint_lab_wgan_larger_sketch_mix_test_no_hint --max_epochs 20 --input_dir sketches_combined --which_direction AtoB --display_freq=1000 --gray_input_a --batch_size 4 --lr 0.0008 --gpu_percentage 0.45 --scale_size=143 --crop_size=128 --use_sketch_loss --pretrained_sketch_net_path pixiv_full_128_to_sketch_train --use_hint --lab_colorization --checkpoint=pixiv_downloaded_128_w_hint_lab_wgan_larger_sketch_mix --user_hint_path=BLANK

# Create the new sketch database.
python pix2pix_w_hint_lab_wgan_larger_sketch_mix.py --mode test --output_dir /mnt/data_drive/home/ubuntu/pixiv_downloaded_sketches_lnet_128/new_sketch --max_epochs 20 --input_dir /mnt/data_drive/home/ubuntu/pixiv_downloaded_sketches_lnet_128/color/ --which_direction BtoA --display_freq=1000 --gray_input_a --batch_size 4 --lr 0.0008 --gpu_percentage 0.25 --scale_size=143 --crop_size=128 --lab_colorization --train_sketch --checkpoint=sketch_colored_pair_cleaned_128_w_hint_lab_wgan_larger_sketch_mix_train_sketch --single_input
"""

"""
python pix2pix_w_hint.py --mode test --output_dir pixiv_full_128_w_hint_test --input_dir /mnt/tf_drive/home/ubuntu/pixiv_full_128_combined/test --checkpoint pixiv_full_128_w_hint_train --gpu_percentage 0.45 --use_hint
python pix2pix_w_hint.py --mode test --output_dir pixiv_full_128_tiny_test --input_dir /mnt/tf_drive/home/ubuntu/pixiv_full_128_combined/test --checkpoint pixiv_full_128_tiny --gpu_percentage 0.45 --use_hint
python pix2pix_w_hint.py --mode test --output_dir pixiv_full_128_w_hint_test --input_dir /mnt/tf_drive/home/ubuntu/pixiv_full_128_combined/test --checkpoint pixiv_full_128_w_hint_train --gpu_percentage 0.45 --use_hint --user_hint_path=BLANK
"""