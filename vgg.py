# The code skeleton mainly comes from https://github.com/anishathalye/neural-style.
# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.
import numpy as np
import scipy.io
import scipy.ndimage
import tensorflow as tf

#
#
# def net(data_path, input_image):
#     # The stride multiplier feature is an attempt to make the style/texture features look larger. It is not fully
#     # developed yet so please keep it at 1.
#     layers = (
#         'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
#
#         'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
#
#         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
#         'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
#
#         'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
#         'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
#
#         'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
#         'relu5_3', 'conv5_4', 'relu5_4'
#     )
#
#     data = scipy.io.loadmat(data_path)
#     mean = data['normalization'][0][0][0]
#     mean_pixel = np.mean(mean, axis=(0, 1))
#     weights = data['layers'][0]
#
#     net = {}
#     current = input_image
#     for i, name in enumerate(layers):
#         kind = name[:4]
#         if kind == 'conv':
#             kernels, bias = weights[i][0][0][0][0]
#             # matconvnet: weights are [width, height, in_channels, out_channels]
#             # tensorflow: weights are [height, width, in_channels, out_channels]
#             kernels = np.transpose(kernels, (1, 0, 2, 3))
#             bias = bias.reshape(-1)
#             current = _conv_layer(current, kernels, bias)
#         elif kind == 'relu':
#             current = tf.nn.relu(current)
#         elif kind == 'pool':
#             current = _pool_layer(current)
#         net[name] = current
#
#     assert len(net) == len(layers)
#     return net, mean_pixel

# Given the path to vgg net, read the data and compute the mean pixel.
def read_net(data_path):
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    return data, mean_pixel

# Given the data from scipy.io.loadmat(data_path), generate the net directly
def pre_read_net(data, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current

    assert len(net) == len(layers)
    return net


def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')

def preprocess(image, mean_pixel):
    return image - mean_pixel

def unprocess(image, mean_pixel):
    return image + mean_pixel

def get_net_layer_sizes(net):
    net_layer_sizes = {}
    for key, val in net.iteritems():
        net_layer_sizes[key] = map(lambda i: i.value, val.get_shape())
    return net_layer_sizes



















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

def net(discrim_targets, trainable=True):
    n_layers = 3
    ndf = 64
    layers = []
    ret = {}

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    # input = tf.concat_v2([discrim_inputs, discrim_targets], axis=3)
    # input = tf.concat(3, [discrim_inputs, discrim_targets])
    input = discrim_targets

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(input, ndf, stride=2, trainable=trainable)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)
        ret["layer_1"] = rectified

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = conv(layers[-1], out_channels, stride=stride, trainable=trainable)
            normalized = batchnorm(convolved, trainable=trainable)
            rectified = lrelu(normalized, 0.2)
            ret["layer_%d" % (len(layers) + 1)] = rectified
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1, trainable=trainable)
        output = tf.sigmoid(convolved)
        ret["layer_%d" % (len(layers) + 1)] = output
        layers.append(output)

    return ret