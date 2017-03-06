from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import os
import random
import tensorflow as tf
import numpy as np

from PIL import ImageStat, Image


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="output path")
parser.add_argument("--image_list_path", help="path to a file containing the path to images under `input-dir`."
                                              " If this is set, the list of images will be preprocessed instead of "
                                              "all images under `input-dir`")
parser.add_argument("--pad", action="store_true", help="pad instead of crop for resize operation")
parser.add_argument("--allow_bw", action="store_true", help="If set to be true, black and white or grayscale images "
                                                            "can be included. Default is to exclude them.")
parser.add_argument("--no_face_detection", action="store_true", help="If set to be true, images without anime "
                                                                     "character faces can be included. "
                                                                     "Default is to exclude them.")
parser.add_argument("--size", type=int, default=256, help="size to use for resize operation")
parser.add_argument("--verbose", action="store_true", help="If not set, then only print how many images it has processed. Otherwise print each file processed.")
parser.add_argument("--gpu_limit", type=float, default=0.5, help="The percentage of gpu this program can use.")
a = parser.parse_args()

IMG_TYPE_GRAYSCALE = 1
IMG_TYPE_COLOR = 2
IMG_TYPE_BW = 3
IMG_TYPE_UK = 4
cascade_file="./lbpcascade_animeface.xml"
if not os.path.isfile(cascade_file):
    try:
        os.system(
            "wget https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml")
        assert os.path.isfile(cascade_file)
    except:
        raise RuntimeError("%s: not found" % cascade_file)
cascade_classifier = cv2.CascadeClassifier(cascade_file)

def detect_bw(file, thumb_size=40, MSE_cutoff=22, adjust_color_bias=True):
    # type: (str, int, int, bool) -> int
    # Mainly copied from
    # http://stackoverflow.com/questions/14041562/python-pil-detect-if-an-image-is-completely-black-or-white
    pil_img = Image.open(file)
    bands = pil_img.getbands()
    if bands == ('R', 'G', 'B') or bands == ('R', 'G', 'B', 'A'):
        thumb = pil_img.resize((thumb_size, thumb_size))
        SSE, bias = 0, [0, 0, 0]
        if adjust_color_bias:
            bias = ImageStat.Stat(thumb).mean[:3]
            bias = [b - sum(bias) / 3 for b in bias]
        for pixel in thumb.getdata():
            mu = sum(pixel) / 3
            SSE += sum((pixel[i] - mu - bias[i]) * (pixel[i] - mu - bias[i]) for i in [0, 1, 2])
        MSE = float(SSE) / (thumb_size * thumb_size)
        if MSE <= MSE_cutoff:
            # print "grayscale\t",
            return IMG_TYPE_GRAYSCALE
        else:
            return IMG_TYPE_COLOR
            # print "Color\t\t\t",
        # print "( MSE=", MSE, ")"
    elif len(bands) == 1:
        # print "Black and white", bands
        return IMG_TYPE_BW
    else:
        # print "Don't know...", bands
        return IMG_TYPE_UK

def detect_bw_tf_op(img_ph, adjust_color_bias=True):
    height, width, num_channels = img_ph.get_shape().as_list()
    SSE, bias = 0, [0, 0, 0]
    if adjust_color_bias:
        bias = tf.reduce_mean(img_ph,axis=(0,1), keep_dims=True)
        bias_mean = tf.reduce_mean(bias)
        bias = bias - bias_mean
    mu = tf.reduce_mean(img_ph, axis=2, keep_dims=True)
    sse = tf.reduce_sum(tf.square(img_ph - mu - bias))
    mse = sse / (height * width)
    return mse


def detect_bw_tf(img, img_ph, detect_bw_op, MSE_cutoff=0.001, adjust_color_bias=True):
    # type: (np.ndarray, tf.Tensor, tf.Tensor, float, bool) -> bool
    # Mainly copied from
    # http://stackoverflow.com/questions/14041562/python-pil-detect-if-an-image-is-completely-black-or-white
    # The cutoff default comes from 100.0 / 256 ** 2. The original default was 22 but I thought that was too low.
    height, width, num_channels = img_ph.get_shape().as_list()
    if num_channels != 3 and num_channels != 4:
        return True
    mse = detect_bw_op.eval(feed_dict={img_ph:img})
    if mse <= MSE_cutoff:
        return True
    else:
        return False

def grayscale(img):
    img = img / 255
    img = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    return (np.expand_dims(img, axis=2) * 255).astype(np.uint8)


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

def sketch_extractor(images, max_val=1.0, min_val=-1.0):
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
        # gray_images = tf.image.rgb_to_grayscale(images)
        rgb_weights = [0.2989, 0.5870, 0.1140]
        gray_images = tf.reduce_sum(images * rgb_weights, axis=3, keep_dims=True)

    elif image_shape[3] == 1:
        gray_images = images
    else:
        raise AssertionError("Input image must have shape [batch_size, height, width, 1 or 3]")
    filt = np.expand_dims(np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]],
                                   np.uint8), axis=2)
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

def normalize(img):
    img -= img.min()
    img /= img.max()
    return img


def create_op(func, **placeholders):
    op = func(**placeholders)

    def f(**kwargs):
        feed_dict = {}
        for argname, argvalue in kwargs.iteritems():
            placeholder = placeholders[argname]
            feed_dict[placeholder] = argvalue
        return op.eval(feed_dict=feed_dict)

    return f

downscale = create_op(
    func=tf.image.resize_images,
    images=tf.placeholder(tf.float32, [None, None, None]),
    size=tf.placeholder(tf.int32, [2]),
    method=tf.image.ResizeMethod.AREA,
)

upscale = create_op(
    func=tf.image.resize_images,
    images=tf.placeholder(tf.float32, [None, None, None]),
    size=tf.placeholder(tf.int32, [2]),
    method=tf.image.ResizeMethod.BICUBIC,
)

decode_jpeg = create_op(
    func=tf.image.decode_jpeg,
    contents=tf.placeholder(tf.string),
)

decode_png = create_op(
    func=tf.image.decode_png,
    contents=tf.placeholder(tf.string),
)

rgb_to_grayscale = create_op(
    func=tf.image.rgb_to_grayscale,
    images=tf.placeholder(tf.float32),
)

grayscale_to_rgb = create_op(
    func=tf.image.grayscale_to_rgb,
    images=tf.placeholder(tf.float32),
)

encode_jpeg = create_op(
    func=tf.image.encode_jpeg,
    image=tf.placeholder(tf.uint8),
)

encode_png = create_op(
    func=tf.image.encode_png,
    image=tf.placeholder(tf.uint8),
)

crop = create_op(
    func=tf.image.crop_to_bounding_box,
    image=tf.placeholder(tf.float32),
    offset_height=tf.placeholder(tf.int32, []),
    offset_width=tf.placeholder(tf.int32, []),
    target_height=tf.placeholder(tf.int32, []),
    target_width=tf.placeholder(tf.int32, []),
)

pad = create_op(
    func=tf.image.pad_to_bounding_box,
    image=tf.placeholder(tf.float32),
    offset_height=tf.placeholder(tf.int32, []),
    offset_width=tf.placeholder(tf.int32, []),
    target_height=tf.placeholder(tf.int32, []),
    target_width=tf.placeholder(tf.int32, []),
)

to_uint8 = create_op(
    func=tf.image.convert_image_dtype,
    image=tf.placeholder(tf.float32),
    dtype=tf.uint8,
    saturate=True,
)

to_float32 = create_op(
    func=tf.image.convert_image_dtype,
    image=tf.placeholder(tf.uint8),
    dtype=tf.float32,
)

to_sketch_op = create_op(
    func=sketch_extractor,
    images=tf.placeholder(tf.float32, [a.size,a.size,3]),
    max_val=tf.placeholder(tf.float32, []),
    min_val=tf.placeholder(tf.float32, [])
)


def load(path):
    contents = open(path).read()
    _, ext = os.path.splitext(path.lower())

    if ext == ".jpg":
        image = decode_jpeg(contents=contents)
    elif ext == ".png":
        image = decode_png(contents=contents)
    else:
        raise Exception("invalid image suffix")

    return to_float32(image=image)


def find(d):
    # The part commented out is the original code. It only search for files within the immediate subdirectory.
    # result = []
    # for filename in os.listdir(d):
    #     _, ext = os.path.splitext(filename.lower())
    #     if ext == ".jpg" or ext == ".png":
    #         result.append(os.path.join(d, filename))
    # result.sort()
    # return result
    # This code search for all subdirectories and their subdirectories.
    result = []
    if a.image_list_path is not None and os.path.isfile(a.image_list_path):
        print("Using provided list of images.")
        with open(a.image_list_path, 'r') as f:
            for line in f.readlines():
                result.append(os.path.join(d, line.rstrip("\n")))
    else:
        for path, subdirs, files in os.walk(d):
            for name in files:
                full_file_path = os.path.join(path, name)
                _, ext = os.path.splitext(full_file_path.lower())
                if ext == ".jpg" or ext == ".png":
                    result.append(full_file_path)
    result.sort()
    return result


def save(image, path):
    _, ext = os.path.splitext(path.lower())
    image = to_uint8(image=image)
    if ext == ".jpg":
        encoded = encode_jpeg(image=image)
    elif ext == ".png":
        encoded = encode_png(image=image)
    else:
        raise Exception("invalid image suffix")

    if os.path.exists(path):
        raise Exception("file already exists at " + path)

    with open(path, "w") as f:
        f.write(encoded)


def png_path(path):
    basename, _ = os.path.splitext(os.path.basename(path))
    return os.path.join(os.path.dirname(path), basename + ".png")


def main():
    random.seed(0)

    color_dir = os.path.join(a.output_dir,"color")
    sketch_dir = os.path.join(a.output_dir,"sketch")
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
    if not os.path.exists(color_dir):
        os.makedirs(color_dir)
    if not os.path.exists(sketch_dir):
        os.makedirs(sketch_dir)

    if a.gpu_limit <= 0.0:
        print('Using cpu mode.')
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
    elif a.gpu_limit <= 1.0:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = a.gpu_limit
    else:
        raise AssertionError('The gpu limit is invalid. It must be between 0 and 1. It is now %f' %a.gpu_limit)

    with tf.Session(config=config) as sess:
        detect_bw_img_ph = tf.placeholder(tf.float32, [a.size,a.size,3], name="detect_bw_img_ph")
        detect_bw_op = detect_bw_tf_op(detect_bw_img_ph)
        num_image_passing_bw = 0
        num_image_passing_face = 0
        for num_image, src_path in enumerate(find(a.input_dir)):
            dst_path = png_path(os.path.join(color_dir, os.path.basename(src_path)))
            dst_sketch_path = png_path(os.path.join(sketch_dir, os.path.basename(src_path)))
            if not a.verbose:
                if num_image % 100 == 0:
                    print("Processed %d images." %num_image)
            else:
                print(src_path, "->", dst_path)
            try:
                src = load(src_path)
                # First resize and crop the images to the correct shape.
                height, width, num_channels = src.shape
                if num_channels != 3 and num_channels != 4:
                    continue
                dst = src
                if height != width:
                    if a.pad:
                        size = max(height, width)
                        # pad to correct ratio
                        oh = (size - height) // 2
                        ow = (size - width) // 2
                        dst = pad(image=dst, offset_height=oh, offset_width=ow, target_height=size, target_width=size)
                    else:
                        # crop to correct ratio
                        size = min(height, width)
                        oh = (height - size) // 2
                        ow = (width - size) // 2
                        dst = crop(image=dst, offset_height=oh, offset_width=ow, target_height=size, target_width=size)

                assert(dst.shape[0] == dst.shape[1])

                size, _, _ = dst.shape
                if size > a.size:
                    dst = downscale(images=dst, size=[a.size, a.size])
                elif size < a.size:
                    dst = upscale(images=dst, size=[a.size, a.size])

                if a.allow_bw or not detect_bw_tf(dst,detect_bw_img_ph,detect_bw_op):
                    num_image_passing_bw += 1
                    if a.no_face_detection or detect_face(dst, cascade_classifier, face_min_size=min(a.size / 8, 48)):
                        num_image_passing_face += 1
                        dst_sketch = to_sketch_op(images=dst,max_val=1.0, min_val=0.0)
                        save(dst, dst_path)
                        save(dst_sketch, dst_sketch_path)
            except Exception as exception:
                print("exception ", exception, " happened when processing ", src_path, "->", dst_path, ". Skipping this one. ")
        print("Number of images preprocessed in total: %d. In which %d passed black-and-white test and %d passed face test and was saved." %(num_image +1, num_image_passing_bw ,num_image_passing_face))

main()

"""python tools/process.py --input_dir=/mnt/data_drive/home/ubuntu/pixiv_downloaded_sketches_lnet_128/line/ --output_dir=/mnt/data_drive/home/ubuntu/pixiv_downloaded_sketches_lnet_128_combined/train/ --b_dir=/mnt/data_drive/home/ubuntu/pixiv_downloaded_sketches_lnet_128/color/ --operation=combine --size=128 --image_list_path=/mnt/data_drive/home/ubuntu/pixiv_downloaded_sketches_lnet_128/images_containing_face.txt --silent --gpu_limit=0.05"""
