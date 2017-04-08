from operator import mul
import math
import tensorflow as tf
from typing import Union, Callable, Tuple

def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
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


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer)
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv

def fully_connected(input, out_channels, activation_fn=None, name='', trainable=True):
    # type: (tf.Tensor, int, Union[None,Callable[[tf.Tensor], tf.Tensor]], str, bool) -> tf.Tensor
    with tf.variable_scope('fully_connected_' + name):
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input

        input_shape = [i.value for i in input.get_shape()]
        batch_size = input_shape[0]
        num_channels = reduce(mul, input_shape[1:])

        weights_shape = [num_channels, out_channels]
        weights_init_stdv = math.sqrt(1.0 / (num_channels))
        weights_initializer = tf.truncated_normal_initializer(stddev=weights_init_stdv)
        weights_init = tf.get_variable('weights_init', shape=weights_shape, dtype=tf.float32,
                                       initializer=weights_initializer, trainable=trainable)

        bias_shape = [out_channels]
        bias_init = tf.get_variable('bias_init', shape=bias_shape, dtype=tf.float32,
                                    initializer=tf.constant_initializer(), trainable=trainable)

        fc1 = tf.reshape(input, [-1, num_channels])
        fc1 = tf.nn.bias_add(tf.matmul(fc1, weights_init), bias_init)

        if activation_fn is not None:
            fc1 = activation_fn(fc1)
        return fc1