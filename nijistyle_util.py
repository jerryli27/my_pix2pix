"""
This file contains one function that implemented the papers:
"A Neural Algorithm of Artistic Style" (https://arxiv.org/abs/1508.06576),
"Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis" (arxiv.org/abs/1601.04589),
"Instance Normalization - The Missing Ingredient for Fast Stylization" (https://arxiv.org/abs/1607.08022),
"Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks" (https://arxiv.org/abs/1603.01768).

In addition, it contains one more functionality to control the degree of stylization of the content image by using a
weighted mask for the content image ("content_img_style_weight_mask" in the code)
The code skeleton was borrowed from https://github.com/anishathalye/neural-style.
"""

from sys import stderr

import numpy as np
import tensorflow as tf
from typing import Union, Tuple, List, Iterable

import neural_doodle_util
import neural_util
import vgg
from general_util import get_np_array_num_elements
from mrf_util import mrf_loss

try:
    reduce
except NameError:
    from functools import reduce

CONTENT_LAYER = 'layer_2'
STYLE_LAYERS = ('layer_1', 'layer_2', 'layer_3', 'layer_4')  # This is used for texture generation (without content)
STYLE_LAYERS_WITH_CONTENT = ('layer_1','layer_2', 'layer_3', 'layer_4')# ('layer_1', 'layer_2', 'layer_3', 'layer_4')
STYLE_LAYERS_MRF = ('relu3_1', 'relu4_1')  # According to https://arxiv.org/abs/1601.04589.


def stylize(network, content, styles, shape, iterations, save_dir, content_weight=5.0, style_weight=100.0, tv_weight=100.0,
            style_blend_weights=None, learning_rate=10.0, initial=None, use_mrf=False, use_semantic_masks=False,
            mask_resize_as_feature=True, output_semantic_mask=None, style_semantic_masks=None,
            semantic_masks_weight=1.0, print_iterations=None, checkpoint_iterations=None,
            semantic_masks_num_layers=4, content_img_style_weight_mask=None):
    """
    Stylize images.
    :param network: Path to pretrained vgg19 network. It can be downloaded at
    http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
    :param content: The content image. If left blank, it will enter texture generation mode (style synthesis without
    context loss).
    :param styles: A list of style images as numpy arrays.
    :param shape: The shape of the output image. It should be with format (1, height, width, 3)
    :param iterations: The number of iterations to run.
    :param content_weight: The weight for content loss. The larger the weight, the more the output will look like
    the content image.
    :param style_weight: The weight for style loss. The larger the weight, the more the output will have a style that
    looks like the style images.
    :param tv_weight: The weight for total-variation loss. The larger the weight, the smoother the output will be.
    :param style_blend_weights: If inputting multiple style images, this controls the balance between their styles.
    If left as None, it will treat all style images as equal.
    :param learning_rate: As name suggests.
    :param initial: The initial starting point for the output. If left blank, the initial would just be noise.
    :param use_mrf: Whether we use markov-random-field loss instead of gramian loss. mrf_util.py contains more info.
    :param use_semantic_masks: Whether we use semantic masks as additional semantic information. Please check the paper
    "Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks" for more information.
    :param mask_resize_as_feature: If true, resize the mask and use the resized mask as additional feature besides the
    vgg network layers. If false, pass the masks (must have exactly 3 masks) into the vgg network and use the outputted
    layers as additional features.
    :param output_semantic_mask: The semantic masks you would like to apply to the outputted image.The mask should have
    shape (batch_size, height, width, semantic_masks_num_layers) Unlike the neural doodle paper, here I use one
    black-and-white image for each semantic mask (the paper had semantic masks represented as rgb images, limiting the
    semantic channels to 3).
    :param style_semantic_masks: A list of semantic masks you would like to apply to each style image. The mask should
    have shape (batch_size, height, width, semantic_masks_num_layers)
    :param semantic_masks_weight: How heavily you'd like to weight the semantic masks as compared to other sources of
    semantic information obtained through passing the image through vgg network. Default is 1.0.
    :param print_iterations: Print loss information every n iterations.
    :param checkpoint_iterations: Save a checkpoint as well as the best image so far every n iterations.
    :param semantic_masks_num_layers: The number of semantic masks each image have.
    :param content_img_style_weight_mask: One black-and-white mask specifying how much we should "stylize" each pixel
    in the outputted image. The areas where the mask has higher value would be stylized more than other areas. A
    completely white mask would mean that we stylize the output image just as before, while a completely dark mask
    would mean that we do not stylize the output image at all, so it should look pretty much the same as content image.
    If you do not wish to use this feature, just leave it as None.
    :return: a tuple where the first item is either the current iteration or None, indicating it has finished training.
    The second item is the image that has the lowest loss so far. The tuples are yielded every 'checkpoint_iterations'
    iterations as well as the last iteration.
    :rtype: iterator[tuple[int|None,image]]
    """
    global STYLE_LAYERS
    if content is not None:
        STYLE_LAYERS = STYLE_LAYERS_WITH_CONTENT
    if use_mrf:
        raise NotImplementedError
        STYLE_LAYERS = STYLE_LAYERS_MRF  # Easiest way to be compatible with no-mrf versions.
    if use_semantic_masks:
        raise NotImplementedError
        assert semantic_masks_weight is not None
        assert output_semantic_mask is not None
        assert style_semantic_masks is not None
    if content_img_style_weight_mask is not None:
        if shape[1] != content_img_style_weight_mask.shape[1] or shape[2] != content_img_style_weight_mask.shape[2]:
            raise AssertionError("The shape of style_weight_mask is incorrect. It must have the same height and width "
                                 "as the output image. The output image has shape: %s and the style weight mask has "
                                 "shape: %s" % (str(shape), str(content_img_style_weight_mask.shape)))
        if content_img_style_weight_mask.dtype != np.float32:
            raise AssertionError('The dtype of style_weight_mask must be float32. it is now %s' % str(
                content_img_style_weight_mask.dtype))

    # Append a (1,) in front of the shapes of the style images. So the style_shapes contains (1, height, width, 3).
    # 3 corresponds to rgb.
    style_shapes = [(1,) + style.shape for style in styles]
    if style_blend_weights is None:
        style_blend_weights = [1.0 / len(styles) for _ in styles]
    content_features = {}
    style_features = [{} for _ in styles]
    output_semantic_mask_features = {}

    # The default behavior of tensorflow was to allocate all gpu memory. Here it is set to only use as much gpu memory
    # as it needs.
    with tf.Graph().as_default(), tf.Session(
            config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        # Compute content features in feed-forward mode
        content_image = tf.placeholder(tf.uint8, shape=shape, name='content_image')
        content_image_float = tf.image.convert_image_dtype(content_image, dtype=tf.float32) * 2 - 1

        with tf.variable_scope("discriminator", reuse=False):
            net = vgg.net(content_image_float, trainable=False)
        content_features[CONTENT_LAYER] = net[CONTENT_LAYER]
        net_layer_sizes = vgg.get_net_layer_sizes(net)

        if content is not None:
            # content_pre = np.array([vgg.preprocess(content, mean_pixel)])
            content_pre = np.array([content])
            content_pre = content_pre.astype(dtype=np.uint8)

        # Compute style features in feed-forward mode.
        if content_img_style_weight_mask is not None:
            style_weight_mask_layer_dict = neural_doodle_util.masks_average_pool(content_img_style_weight_mask)

        for i in range(len(styles)):
            # Using precompute_image_features, which calculates on cpu and thus allow larger images.
            style_features[i] = _precompute_image_features(styles[i], STYLE_LAYERS, style_shapes[i], save_dir)

        if initial is None:
            initial = tf.random_normal(shape) * 0.256
        else:
            # initial = np.array([vgg.preprocess(initial, mean_pixel)])
            initial = np.array([initial])
            initial = initial.astype('float32')
        # image = tf.Variable(initial)
        # image_uint8 = tf.cast(image, tf.uint8)
        # image_float = tf.image.convert_image_dtype(image_uint8,dtype=tf.float32) * 2 - 1

        image_float = tf.Variable(initial)
        image = tf.image.convert_image_dtype((image_float + 1) / 2,dtype=tf.uint8)

        with tf.variable_scope("discriminator", reuse=True):
            net= vgg.net(image_float, trainable=False)

        # content loss
        _, height, width, number = map(lambda i: i.value, content_features[CONTENT_LAYER].get_shape())
        content_features_size = height * width * number
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) /
                                         content_features_size)
        # style loss
        style_loss = 0
        for i in range(len(styles)):
            style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                if content_img_style_weight_mask is not None:
                    # Apply_style_weight_mask_to_feature_layer, then normalize with average of that style weight mask.
                    layer = neural_doodle_util.vgg_layer_dot_mask(style_weight_mask_layer_dict[style_layer], layer) \
                            / (tf.reduce_mean(style_weight_mask_layer_dict[style_layer]) + 0.000001)

                if use_mrf:
                    if use_semantic_masks:
                        # TODO: Compare the effect of concatenate masks to vgg layers versus dotting them with vgg
                        # layers. If you change this to dot, don't forget to also change that in neural_doodle_util.
                        layer = neural_doodle_util.concatenate_mask_layer_tf(output_semantic_mask_features[style_layer],
                                                                             layer)
                        # layer = neural_doodle_util.vgg_layer_dot_mask(output_semantic_mask_features[style_layer], layer)
                    style_losses.append(mrf_loss(style_features[i][style_layer], layer, name='%d%s' % (i, style_layer)))
                else:
                    if use_semantic_masks:
                        gram = neural_doodle_util.gramian_with_mask(layer, output_semantic_mask_features[style_layer])
                    else:
                        gram = neural_util.gramian(layer)
                    style_gram = style_features[i][style_layer]
                    style_gram_size = get_np_array_num_elements(style_gram)
                    style_losses.append(tf.nn.l2_loss(
                        gram - style_gram) / style_gram_size)  # TODO: Check normalization constants. the style loss is way too big compared to the other two.
            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)
        # total variation denoising
        tv_loss = tf.mul(neural_util.total_variation(image_float), tv_weight)

        # overall loss
        if content is None:  # If we are doing style/texture regeration only.
            loss = style_loss + tv_loss
        else:
            loss = content_loss + style_loss + tv_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        def print_progress(i, feed_dict, last=False):
            stderr.write('Iteration %d/%d\n' % (i + 1, iterations))
            if last or (print_iterations is not None and print_iterations != 0 and i % print_iterations == 0):
                if content is not None:
                    stderr.write('  content loss: %g\n' % content_loss.eval(feed_dict=feed_dict))
                stderr.write('    style loss: %g\n' % style_loss.eval(feed_dict=feed_dict))
                stderr.write('       tv loss: %g\n' % tv_loss.eval(feed_dict=feed_dict))
                stderr.write('    total loss: %g\n' % loss.eval(feed_dict=feed_dict))

        # Load discriminator weight.

        if '0.12.0' in tf.__version__:
            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        else:
            all_vars = tf.get_collection(tf.GraphKeys.VARIABLES)

        # discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_tvars = [var for var in all_vars if var.name.startswith("discriminator")]
        saver = tf.train.Saver(discrim_tvars)

        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise AssertionError("Cannot load from save directory.")

        var_not_saved = [item for item in all_vars if item not in discrim_tvars]
        sess.run(tf.initialize_variables(var_not_saved))









        # optimization
        best_loss = float('inf')
        best = np.zeros(shape=shape)
        feed_dict = {}
        if content is not None:
            feed_dict[content_image] = content_pre
        sess.run(tf.initialize_all_variables(), feed_dict=feed_dict)
        for i in range(iterations):
            last_step = (i == iterations - 1)
            print_progress(i, feed_dict, last=last_step)
            train_step.run(feed_dict=feed_dict)

            if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                this_loss = loss.eval(feed_dict=feed_dict)
                if this_loss < best_loss:
                    best_loss = this_loss
                    best = image.eval()
                # yield (
                #     (None if last_step else i),
                #     vgg.unprocess(best.reshape(shape[1:]), mean_pixel)
                # )
                print(best)
                best_float32 = image_float.eval()
                print(best_float32)
                yield (
                    (None if last_step else i),
                    best.reshape(shape[1:])
                )


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)



def _precompute_image_features(img, layers, shape, save_dir):
    # type: (np.ndarray, Union[Tuple[str], List[str]], Union[Tuple[int], List[int]]) -> Dict[str, np.ndarray]
    """
    Precompute the features of the image by passing it through the vgg network and storing the computed layers.
    :param img: the image of which the features would be precomputed. It must have shape (height, width, 3)
    :param layers: A list of string specifying which layers would we be returning. Check vgg.py for layer names.
    :param shape: shape of the image placeholder.
    :param vgg_data: The vgg network represented as a dictionary. It can be obtained by vgg.pre_read_net.
    :param mean_pixel: The mean pixel value for the vgg network. It can be obtained by vgg.read_net or just hardcoded.
    :param use_mrf: Whether we're using mrf loss. If true, it does not calculate and store the gram matrix.
    :param use_semantic_masks: Whether we're using semantic masks. If true, it does not calculate and store the gram
    matrix.
    :return: A dictionary containing the precomputed feature for each layer.
    """
    features_dict = {}
    g = tf.Graph()
    # Choose to use cpu here because we only need to compute this once and using cpu would provide us more memory
    # than the gpu and therefore allow us to process larger style images using the extra memory. This will not have
    # an effect on the training speed later since the gram matrix size is not related to the size of the image.
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        with tf.variable_scope("discriminator", reuse=False):
            image = tf.placeholder(tf.uint8, shape=shape)
            image_float = tf.image.convert_image_dtype(image,dtype=tf.float32) * 2 - 1
            net = vgg.net(image_float, trainable=False)
            style_pre = np.array([img])
            style_pre = style_pre.astype(np.uint8)

            if '0.12.0' in tf.__version__:
                all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            else:
                all_vars = tf.get_collection(tf.GraphKeys.VARIABLES)

            discrim_tvars = [var for var in all_vars if var.name.startswith("discriminator")]
            saver = tf.train.Saver(discrim_tvars)

            ckpt = tf.train.get_checkpoint_state(save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise AssertionError("Cannot load from save directory.")

            var_not_saved = [item for item in all_vars if item not in discrim_tvars]
            sess.run(tf.initialize_variables(var_not_saved))


            for layer in layers:
                # Calculate and store gramian.
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                features_dict[layer] = gram
    return features_dict