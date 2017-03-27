
import cv2
import os
import tensorflow as tf
import numpy as np

from PIL import ImageStat, Image
from neural_util import decode_image_with_file_name

def calc_mse(img, adjust_color_bias=True):
    # height, width, num_channels = img_ph.get_shape().as_list()
    SSE, bias = 0, [0, 0, 0]
    if adjust_color_bias:
        bias = tf.reduce_mean(img, axis=(0, 1), keep_dims=True)
        bias_mean = tf.reduce_mean(bias)
        bias = bias - bias_mean
    mu = tf.reduce_mean(img, axis=2, keep_dims=True)
    # sse = tf.reduce_sum(tf.square(img_ph - mu - bias))
    # mse = sse / (height * width)
    mse = tf.reduce_mean(tf.square(img - mu - bias)) * 3  # 3 for 3 channels.
    return mse

def detect_bw_tf(img, MSE_cutoff=0.001, adjust_color_bias=True):
    # type: (np.ndarray, tf.Tensor, tf.Tensor, float, bool) -> bool
    # Returns true if the image is NOT black and white or grayscale (that is to say, it passed the test).
    # Mainly copied from
    # http://stackoverflow.com/questions/14041562/python-pil-detect-if-an-image-is-completely-black-or-white
    # The cutoff default comes from 100.0 / 256 ** 2. The original default was 22 but I thought that was too low.
    # height, width, num_channels = img_ph.get_shape().as_list()
    num_channel = tf.shape(img, name="resized_image_shape")[-1]
    def _is_one_ch(num_ch):
        return tf.equal(num_ch, 1)
    def _is_three_ch(num_ch):
        return tf.equal(num_ch, 3)
    def _is_four_ch(num_ch):
        return tf.equal(num_ch, 4)
    def check_three_ch_above(image, num_ch):
        return tf.cond(_is_three_ch(num_ch), lambda: tf.greater(calc_mse(image, adjust_color_bias), MSE_cutoff), lambda: check_four_ch(image, num_ch))
    def check_four_ch(image, num_ch):
        assert_four_ch = tf.Assert(_is_four_ch(num_ch), [num_ch])
        with tf.control_dependencies([assert_four_ch, ]):
            return tf.greater(calc_mse(image[...,:3], adjust_color_bias), MSE_cutoff)
    return tf.cond(_is_one_ch(num_channel), lambda: tf.constant(False), lambda: check_three_ch_above(img, num_channel))

def detect_face(img, cascade_classifier, face_min_size=32):
    _, _, num_channels = img.shape
    if img.dtype == np.float32:
        img = np.array(img * 255.0, np.uint8)
    if num_channels == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = cv2.equalizeHist(gray)
    faces = cascade_classifier.detectMultiScale(gray,
                                                # detector options
                                                scaleFactor= 1.1,
                                                minNeighbors=5,
                                                minSize=(face_min_size, face_min_size))
    if len(faces) > 0:
        return True
    else:
        return False

# def sketch_extractor(images, max_val=1.0, min_val=-1.0):
#     images = (images - min_val) / (max_val - min_val) * 255.0
#     image_shape = images.get_shape().as_list()
#     is_single_image = False
#     if len(image_shape) == 3:
#         images = tf.expand_dims(images, axis=0)
#         image_shape = images.get_shape().as_list()
#         is_single_image = True
#     if len(image_shape) != 4:
#         raise AssertionError("Input image must have shape [batch_size, height, width, 1 or 3]")
#     if image_shape[3] == 3:
#         # gray_images = tf.image.rgb_to_grayscale(images)
#         rgb_weights = [0.2989, 0.5870, 0.1140]
#         gray_images = tf.reduce_sum(images * rgb_weights, axis=3, keep_dims=True)
#
#     elif image_shape[3] == 1:
#         gray_images = images
#     else:
#         raise AssertionError("Input image must have shape [batch_size, height, width, 1 or 3]")
#     filt = np.expand_dims(np.array([[1, 1, 1],
#                                     [1, 1, 1],
#                                     [1, 1, 1]],
#                                    np.uint8), axis=2)
#     stride = 1
#     rate = 1
#     padding = 'SAME'
#     dil = tf.nn.dilation2d(gray_images, filt, (1, stride, stride, 1), (1, rate, rate, 1), padding,
#                            name='image_dilated')
#     sketch = 255 - tf.abs(gray_images - dil)
#     # Did NOT apply a threshold here to clear out the low values because i think it may not be necessary.
#     # sketch = tf.clip_by_value(sketch, 32, 255)
#     sketch = sketch / 255.0 * (max_val - min_val) + min_val
#     assert sketch.get_shape().as_list() == gray_images.get_shape().as_list()
#     if is_single_image:
#         sketch = sketch[0]
#     return sketch

def load(paths, contents, channels = None):
    return tf.image.convert_image_dtype(decode_image_with_file_name(contents, paths, channels=channels), dtype=tf.float32)

#
# def find(path, do_sort = True):
#     # The part commented out is the original code. It only search for files within the immediate subdirectory.
#     # result = []
#     # for filename in os.listdir(d):
#     #     _, ext = os.path.splitext(filename.lower())
#     #     if ext == ".jpg" or ext == ".png":
#     #         result.append(os.path.join(d, filename))
#     # result.sort()
#     # return result
#     # This code search for all subdirectories and their subdirectories.
#     result = []
#     if os.path.isfile(path):
#         print("Using provided list of images.")
#         with open(path, 'r') as f:
#             for line in f.readlines():
#                 result.append(os.path.join(path, line.rstrip("\n")))
#     else:
#         for path, subdirs, files in os.walk(path):
#             for name in files:
#                 full_file_path = os.path.join(path, name)
#                 _, ext = os.path.splitext(full_file_path.lower())
#                 if ext == ".jpg" or ext == ".png":
#                     result.append(full_file_path)
#     if do_sort:
#         result.sort()
#     return result


def save(encoded, path):
    # _, ext = os.path.splitext(path.lower())
    # image = to_uint8(image=image)
    # if ext == ".jpg":
    #     encoded = encode_jpeg(image=image)
    # elif ext == ".png":
    #     encoded = encode_png(image=image)
    # else:
    #     raise Exception("invalid image suffix")

    if os.path.exists(path):
        raise Exception("file already exists at " + path)

    with open(path, "w") as f:
        f.write(encoded)

def png_path(path):
    basename, _ = os.path.splitext(os.path.basename(path))
    return os.path.join(os.path.dirname(path), basename + ".png")

def get_image_hw(image, dtype = tf.float32):
    image_shape = tf.shape(image, name="image_shape")

    height = tf.cast(image_shape[0], dtype)
    width = tf.cast(image_shape[1], dtype)
    return height, width

def pass_hw_test(height, width, hw_ratio_threshold):
    hw_ratio = tf.minimum(tf.div(height, width), tf.div(width, height))
    return tf.greater_equal(hw_ratio, hw_ratio_threshold, name="pass_hw_test")

def resize_image(image, height, width, resize_mode, new_size):

    if resize_mode == "pad":

        size_float = tf.maximum(height, width)
        size = tf.cast(size_float, tf.int32)
        # pad to correct ratio
        # oh = tf.math_ops.round((size - height) / 2)
        # ow = tf.math_ops.round((size - width) / 2)
        oh = tf.cast((size_float - height) // 2, tf.int32)
        ow = tf.cast((size_float - width) // 2, tf.int32)
        dst = tf.cond(tf.not_equal(height, width), lambda: tf.image.pad_to_bounding_box(image, oh, ow, size, size), lambda: image)
    elif resize_mode == "crop":
        # crop to correct ratio
        size_float = tf.minimum(height, width)
        size = tf.cast(size_float, tf.int32)
        oh = tf.cast((height - size_float) // 2, tf.int32)
        ow = tf.cast((width - size_float) // 2, tf.int32)
        dst = tf.cond(tf.not_equal(height, width), lambda: tf.image.crop_to_bounding_box(image, oh, ow, size, size), lambda: image)
    elif resize_mode == "reshape":
        dst = image
        # size = tf.shape(dst, name="resized_image_shape")[0]
        # dst = tf.control_flow_ops.cond(tf.math_ops.greater_equal(size,a.size),
        #                      tf.image.resize_images(dst, size=[a.size, a.size],method=tf.image.ResizeMethod.AREA,),
        #                      tf.image.resize_images(dst, size=[a.size, a.size],method=tf.image.ResizeMethod.BICUBIC,))
    elif resize_mode == "none":
        dst = image
        return dst
    else:
        raise AttributeError("Resize mode %s not supported." %(resize_mode))


    new_size = tf.shape(dst, name="resized_image_shape")[0]
    dst = tf.cond(tf.greater_equal(new_size, new_size),
                  lambda: tf.image.resize_images(dst, size=[new_size, new_size],
                                                          method=tf.image.ResizeMethod.AREA, ),
                  lambda: tf.image.resize_images(dst, size=[new_size, new_size],
                                                          method=tf.image.ResizeMethod.BICUBIC, ))
    return dst
