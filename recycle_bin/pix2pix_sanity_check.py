#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os
import random
import time
import math

import numpy as np
import tensorflow as tf

from neural_util import conv, lrelu, batchnorm, deconv, fully_connected

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

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
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, gen_loss_GAN, train")


# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c


def _gaussian_mixture_circle(batchsize, num_cluster=8, scale=1, std=1):
    # Taken from https://github.com/musyoku/wasserstein-gan/blob/master/train_gaussian_mixture/sampler.py
    rand_indices = np.random.randint(0, num_cluster, size=batchsize)
    base_angle = math.pi * 2 / num_cluster
    angle = rand_indices * base_angle - math.pi / 2
    mean = np.zeros((batchsize, 2), dtype=np.float32)
    mean[:, 0] = np.cos(angle) * scale
    mean[:, 1] = np.sin(angle) * scale
    return np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32)

def _load_examples():
    size = 640
    target_images = _gaussian_mixture_circle(size,scale=0.5, std=0.1)
    input_images = np.random.randint(0, 8, size=size)
    target_producer = tf.train.input_producer(target_images)
    input_producer = tf.train.input_producer(input_images)
    inputs, targets = tf.train.batch([input_producer, target_producer], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(size / a.batch_size))

    return Examples(
        inputs=inputs,
        targets=targets,
        count=size,
        steps_per_epoch=steps_per_epoch,
    )

def create_model(inputs, targets):
    def create_generator(generator_inputs, generator_outputs_channels):
        layers = []

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = fully_connected(generator_inputs, a.ngf, activation_fn=tf.nn.tanh)
            layers.append(output)

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            #input = tf.concat_v2([layers[-1], layers[0]], axis=3)
            output = fully_connected(layers[-1], a.ndf, activation_fn=tf.nn.tanh)
            layers.append(output)

        return layers[-1]

    def create_discriminator(discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        # input = tf.concat_v2([discrim_inputs, discrim_targets], axis=3)
        input = tf.concat(3, [discrim_targets])

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            output = fully_connected(input, a.ndf, activation_fn=tf.nn.tanh)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        # discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

        # Use wgan loss
        discrim_loss = tf.reduce_mean(-predict_real + predict_fake)

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        # gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        # gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        # gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

        # WGAN loss
        gen_loss_GAN = tf.reduce_mean(-predict_fake)
        gen_loss = gen_loss_GAN * a.gan_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_train = discrim_optim.minimize(discrim_loss, var_list=discrim_tvars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_train = gen_optim.minimize(gen_loss, var_list=gen_tvars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        gen_loss_GAN=ema.average(gen_loss_GAN),
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def main():
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
        options = {"which_direction", "ngf", "ndf", "lab_colorization", "gray_input_a", "gray_input_b"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).iteritems():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = _load_examples()

    print("examples count = %d" % examples.count)

    model = create_model(examples.inputs, examples.targets)

    def deprocess(image):
        return image
        # if a.aspect_ratio != 1.0:
        #     # upscale to correct aspect ratio
        #     size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
        #     image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)
        #
        # if a.lab_colorization:
        #     # colorization mode images can be 1 channel (L) or 2 channels (a,b)
        #     num_channels = int(image.get_shape()[-1])
        #     if num_channels == 1:
        #         return tf.image.convert_image_dtype((image + 1) / 2, dtype=tf.uint8, saturate=True)
        #     elif num_channels == 2:
        #         # (a, b) color channels, convert to rgb
        #         # a_chan and b_chan have range [-1, 1] => [-110, 110]
        #         a_chan, b_chan = tf.unstack(image * 110, axis=3)
        #         # get L_chan from inputs or targets
        #         if a.which_direction == "AtoB":
        #             brightness = examples.inputs
        #         elif a.which_direction == "BtoA":
        #             brightness = examples.targets
        #         else:
        #             raise Exception("invalid direction")
        #         # L_chan has range [-1, 1] => [0, 100]
        #         L_chan = tf.squeeze((brightness + 1) / 2 * 100, axis=3)
        #         lab = tf.stack([L_chan, a_chan, b_chan], axis=3)
        #         rgb = lab_to_rgb(lab)
        #         return tf.image.convert_image_dtype(rgb, dtype=tf.uint8, saturate=True)
        #     else:
        #         raise Exception("unexpected number of channels")
        # else:
        #     return tf.image.convert_image_dtype((image + 1) / 2, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("deprocess_inputs"):
        deprocessed_inputs = deprocess(examples.inputs)

    with tf.name_scope("deprocess_targets"):
        deprocessed_targets = deprocess(examples.targets)

    with tf.name_scope("deprocess_outputs"):
        deprocessed_outputs = deprocess(model.outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "inputs": deprocessed_inputs,
            "targets": deprocessed_targets,
            "outputs": deprocessed_outputs,
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", deprocessed_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", deprocessed_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", deprocessed_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        if a.mode == "test":
            # testing
            raise NotImplementedError
            # # run a single epoch over all input data
            # for step in range(examples.steps_per_epoch):
            #     results = sess.run(display_fetches)
            #     filesets = save_images(results, image_dir)
            #     for i, path in enumerate(results["paths"]):
            #         print(step * a.batch_size + i + 1, "evaluated image", os.path.basename(path))
            #     index_path = append_index(filesets)
            #
            # print("wrote index at", index_path)
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

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    # print("saving display images")
                    # filesets = save_images(results["display"], image_dir, step=results["global_step"])
                    # append_index(filesets, step=True)
                    print("saving display images")
                    # filesets = save_images(results["display"], image_dir, step=results["global_step"])
                    # append_index(filesets, step=True)

                    index_path = os.path.join(a.output_dir, "index.csv")
                    index = open(index_path, "a")
                    for i, in_path in enumerate(fetches["paths"]):
                        for kind in ["inputs", "outputs", "targets"]:
                            contents = fetches[kind][i]
                            index.write('%s\t' %(str(contents)))
                        index.write("\n")


                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    global_step = results["global_step"]
                    print("progress  epoch %d  step %d  image/sec %0.1f" % (global_step // examples.steps_per_epoch, global_step % examples.steps_per_epoch, global_step * a.batch_size / (time.time() - start_time)))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()
"""--input_dir=/home/xor/pixiv_images/test_images_sketches/line/ --b_dir=/home/xor/pixiv_images/test_images_sketches/color/ --operation=combine --output_dir=/home/xor/pixiv_images/test_images_sketches_combined/"""
"""
# train the model (this may take 1-8 hours depending on GPU, on CPU you will be waiting for a bit)
python pix2pix.py --mode train --output_dir facades_train --max_epochs 200 --input_dir facades/train --which_direction BtoA --display_freq=5000
# test the model
python pix2pix.py --mode test --output_dir facades_test --input_dir facades/val --checkpoint facades_train
"""


"""
--mode
train
--output_dir
sanity_check_train
--max_epochs
200
--input_dir
/home/xor/pixiv_images/test_images_sketches_combined
--which_direction
AtoB
"""
"""
python pix2pix.py --mode train --output_dir pixiv_full_128_train --max_epochs 20 --input_dir /home/ubuntu/pixiv_full_128_combined/train --which_direction AtoB --display_freq=5000 --gray_input_a --batch_size 1
"""

"""
python pix2pix.py --mode test --output_dir pixiv_full_128_test --input_dir /mnt/tf_drive/home/ubuntu/pixiv_full_128_combined/test --checkpoint pixiv_full_128_train
"""