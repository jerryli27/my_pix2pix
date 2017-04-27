
"""
This file provides functions for turning any image into a 'sketch'.
"""

import random
import shutil
import time

import cv2
import tensorflow as tf
from PIL import ImageStat
from scipy.stats import threshold

from general_util import *
#
# neiborhood8 = np.array([[1, 1, 1],
#                         [1, 1, 1],
#                         [1, 1, 1]],
#                        np.uint8)
neiborhood8 = np.array([[1 for i in range(3)] for j in range(3)],
                       np.uint8)

def image_to_sketch(img):
    """
    Ideas are from http://qiita.com/khsk/items/6cf4bae0166e4b12b942 and
    http://qiita.com/taizan/items/cf77fd37ec3a0bef5d9d
    :param image: An image represented in numpy array with shape (height, width, 3) or (batch, height, width, 3)
    :return: A sketch of the image with shape (height, width) or (batch, height, width)
    """

    # We must apply a lower threshold. Otherwise the sketch image will be filled with non-zero values that may provide
    # hints to the cnn trained. (It is unlikely to occur in human provided sketches that we have many pixels with
    # brightness lower than 32. )
    SKETCH_LOWEST_BRIGHTNESS = 32


    if len(img.shape) == 4:
        img_diff_dilation_gray =  np.array([image_to_sketch(img[i,...]) for i in range(img.shape[0])])
        return img_diff_dilation_gray
    elif len(img.shape) == 3:
        assert img.dtype == np.float32  # Otherwise the conversion does not work properly.

        # kernel = np.ones((5, 5), np.uint8)
        # img_dilation = cv2.dilate(img, kernel, iterations=1)
        # img_diff_dilation = np.abs(np.subtract(img, img_dilation))
        # img_diff_dilation_gray = cv2.cvtColor(img_diff_dilation, cv2.COLOR_RGB2GRAY)

        img_dilate = cv2.dilate(img, neiborhood8, iterations=1)
        img_diff = cv2.absdiff(img, img_dilate)
        img_diff_dilation_gray = cv2.cvtColor(img_diff, cv2.COLOR_RGB2GRAY)

        img_diff_dilation_gray_thresholded = threshold(img_diff_dilation_gray,threshmin=SKETCH_LOWEST_BRIGHTNESS)
        # img_diff_dilation_gray_thresholded = img_diff_dilation_gray_thresholded.astype(np.bool).astype(np.float32) * 255
        # Usually Sketches are represented by dark lines on white background, so I need to invert.
        img_diff_dilation_gray_inverted = 255-img_diff_dilation_gray_thresholded


        return img_diff_dilation_gray_inverted
    else:
        print('Image has to be either of shape (height, width, num_features) or (batch_size, height, width, num_features)')
        raise AssertionError


def image_to_sketch_experiment(img):
    """
    Ideas are from http://qiita.com/khsk/items/6cf4bae0166e4b12b942 and
    http://qiita.com/taizan/items/cf77fd37ec3a0bef5d9d
    :param image: An image represented in numpy array with shape (height, width, 3) or (batch, height, width, 3)
    :return: A sketch of the image with shape (height, width) or (batch, height, width)
    """

    # We must apply a lower threshold. Otherwise the sketch image will be filled with non-zero values that may provide
    # hints to the cnn trained. (It is unlikely to occur in human provided sketches that we have many pixels with
    # brightness lower than 32. )
    SKETCH_LOWEST_BRIGHTNESS = 32


    if len(img.shape) == 4:
        img_diff_dilation_gray =  np.array([image_to_sketch(img[i,...]) for i in range(img.shape[0])])
        return img_diff_dilation_gray
    elif len(img.shape) == 3:
        assert img.dtype == np.float32  # Otherwise the conversion does not work properly.

        # kernel = np.ones((5, 5), np.uint8)
        # img_dilation = cv2.dilate(img, kernel, iterations=1)
        # img_diff_dilation = np.abs(np.subtract(img, img_dilation))
        # img_diff_dilation_gray = cv2.cvtColor(img_diff_dilation, cv2.COLOR_RGB2GRAY)

        img_dilate = cv2.dilate(img, neiborhood8, iterations=1)
        img_diff = cv2.absdiff(img, img_dilate)
        img_diff_dilation_gray = cv2.cvtColor(img_diff, cv2.COLOR_RGB2GRAY)

        img_diff_dilation_gray_thresholded = threshold(img_diff_dilation_gray,threshmin=SKETCH_LOWEST_BRIGHTNESS)
        img_diff_dilation_gray_thresholded = img_diff_dilation_gray_thresholded.astype(np.bool).astype(np.float32) * 255


        img_dilate = cv2.dilate(img_diff_dilation_gray_thresholded, neiborhood8, iterations=1)
        img_diff_dilation_gray_thresholded = cv2.absdiff(img_diff_dilation_gray_thresholded, img_dilate)

        # Usually Sketches are represented by dark lines on white background, so I need to invert.
        img_diff_dilation_gray_inverted = 255-img_diff_dilation_gray_thresholded


        return img_diff_dilation_gray_inverted
    else:
        print('Image has to be either of shape (height, width, num_features) or (batch_size, height, width, num_features)')
        raise AssertionError

def sketch_extractor(images, color_space, sketch_width=3, max_val=1.0, min_val=-1.0):
    images = (images - min_val) / (max_val - min_val) * 255.0
    image_shape = images.get_shape().as_list()
    is_single_image = False
    if len(image_shape) == 3:
        images = tf.expand_dims(images, axis=0)
        image_shape = images.get_shape().as_list()
        is_single_image = True
    if len(image_shape) != 4:
        raise AssertionError("Input image must have shape [batch_size, height, width, 1 or 3]")
    if image_shape[3] == 3:
        if color_space == "rgb":
            # gray_images = tf.image.rgb_to_grayscale(images)
            rgb_weights = [0.2989, 0.5870, 0.1140]
            gray_images = tf.reduce_sum(images * rgb_weights, axis=3, keep_dims=True)
        elif color_space == "lab":
            gray_images = images[..., :1]
        else:
            raise AttributeError("Color space must be either lab or rgb.")

    elif image_shape[3] == 1:
        gray_images = images
    else:
        raise AssertionError("Input image must have shape [batch_size, height, width, 1 or 3]")
    # filt = np.expand_dims(np.array([[1, 1, 1],
    #                                 [1, 1, 1],
    #                                 [1, 1, 1]],
    #                                 np.uint8), axis=2)
    assertion = tf.assert_greater_equal(sketch_width, 3,
                                 message="sketch_width has to be greater than 3 for dilation to work properly.")
    with tf.control_dependencies([assertion]):
        filt = tf.ones((sketch_width,sketch_width,1),)
    # filt = np.expand_dims(np.array([[1 for i in range(18)] for j in range(18)],
    #                        np.uint8), axis=2)
    # filt = np.expand_dims(np.array([[1, 1, 1, 1, 1],
    #                                 [1, 1, 1, 1, 1],
    #                                 [1, 1, 1, 1, 1],
    #                                 [1, 1, 1, 1, 1],
    #                                 [1, 1, 1, 1, 1],],
    #                                np.uint8), axis=2)
    stride = 1
    rate = 1
    padding = 'SAME'
    dil = tf.nn.dilation2d(gray_images, filt, (1, stride, stride, 1), (1, rate, rate, 1), padding,
                           name='image_dilated')
    sketch = 255 - tf.abs(gray_images - dil)
    # Did NOT apply a threshold here to clear out the low values because i think it may not be necessary.
    # sketch = tf.clip_by_value(sketch, 32, 255)
    sketch = sketch / 255.0 * (max_val - min_val) + min_val
    assert sketch.get_shape().as_list() == gray_images.get_shape().as_list()
    if is_single_image:
        sketch = sketch[0]
    return sketch