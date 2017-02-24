from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import tensorflow as tf
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="output path")
parser.add_argument("--operation", required=True, choices=["grayscale", "resize", "blank", "combine"])
parser.add_argument("--pad", action="store_true", help="pad instead of crop for resize operation")
parser.add_argument("--size", type=int, default=256, help="size to use for resize operation")
parser.add_argument("--b_dir", type=str, help="path to folder containing B images for combine operation")
parser.add_argument("--silent", action="store_true", help="Only print how many images it has processed")
parser.add_argument("--gpu_limit", type=float, default=0.5, help="The percentage of gpu this program can use.")
a = parser.parse_args()


def grayscale(img):
    img = img / 255
    img = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    return (np.expand_dims(img, axis=2) * 255).astype(np.uint8)


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

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

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
        for num_image, src_path in enumerate(find(a.input_dir)):
            dst_path = png_path(os.path.join(a.output_dir, os.path.basename(src_path)))
            if a.silent:
                if num_image % 100 == 0:
                    print("Processed %d images." %num_image)
            else:
                print(src_path, "->", dst_path)
            src = load(src_path)

            if a.operation == "grayscale":
                dst = grayscale_to_rgb(images=rgb_to_grayscale(images=src))
            elif a.operation == "resize":
                height, width, _ = src.shape
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
            elif a.operation == "blank":
                height, width, _ = src.shape
                if height != width:
                    raise Exception("non-square image")

                image_size = width
                size = int(image_size * 0.3)
                offset = int(image_size / 2 - size / 2)

                dst = src
                dst[offset:offset + size,offset:offset + size,:] = np.ones([size, size, 3])
            elif a.operation == "combine":
                if a.b_dir is None:
                    raise Exception("missing b_dir")

                # find corresponding file in b_dir, could have a different extension
                # The code commented out was the original implementation that assumes all images are in the immediate
                # subdirectory
                # basename, _ = os.path.splitext(os.path.basename(src_path))
                # for ext in [".png", ".jpg"]:
                #     sibling_path = os.path.join(a.b_dir, basename + ext)
                #     if os.path.exists(sibling_path):
                #         sibling = load(sibling_path)
                #         break

                # The following code assumes that the structure of both folders are the same. So if a has
                # a/sub/dir/1.jpg then b should have b/sub/dir/1.jpg
                basename, _ = os.path.splitext(os.path.basename(src_path))
                subdir_name = src_path[len(a.input_dir):]
                # sibling_path = os.path.join(a.b_dir, subdir_name)
                sibling_path = a.b_dir + subdir_name
                if os.path.exists(sibling_path):
                    sibling = load(sibling_path)
                else:
                    raise Exception("could not find sibling image for " + src_path +
                                    ". Sibling path: " + sibling_path +" does not exist under subdir_name " + subdir_name + " bdir=" + a.b_dir)

                # make sure that dimensions are correct
                height, width, _ = src.shape
                if height != sibling.shape[0] or width != sibling.shape[1]:
                    raise Exception("differing sizes")

                # remove alpha channel
                src = src[:,:,:3]
                sibling = sibling[:,:,:3]
                if src.shape[2] == 1:
                    src = np.repeat(src, repeats=3, axis=2)
                if sibling.shape[2] == 1:
                    sibling = np.repeat(sibling, repeats=3, axis=2)

                dst = np.concatenate([src, sibling], axis=1)
            else:
                raise Exception("invalid operation")

            save(dst, dst_path)


main()
