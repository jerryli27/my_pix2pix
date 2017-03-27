"""
This file is for preprocessing image files that you have and convert them to formats that can be processed by the
Paints Sketch program. It does a few preprocessing steps by default, like filtering out images with a wierd height to
width ratio, filter out black and white images, and many more.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import time
import sys
import traceback

from shutil import copy

from preprocess_util import *
from sketches_util import sketch_extractor
from general_util import get_all_image_paths

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="Path to folder containing images to be preprocessed, "
                                                       "or a file containing paths to the images.")
parser.add_argument("--output_dir", required=True, help="Output path.")
parser.add_argument("--resize_mode", required=True, choices=["crop", "pad", "reshape", "none"],
                    help="Crop means cropping the iamges to a square at the center before resizing."
                         "Pad means padding the images to a square on the sides before resizing."
                         "Reshape means reshaping the images to a square and ignoring its height width ratio."
                         "None means output the image as it is, except the extension will be changed to png.")
parser.add_argument("--hw_ratio_threshold", type=float, default=0.5,
                    help="The minimum threshold for height width ratio or width height ratio, whichever is smaller."
                         "Set it to a number <= 0 to disable height width ratio check.")
parser.add_argument("--allow_bw", action="store_true", help="If set to be true, black and white or grayscale images "
                                                            "can be included. Default is to exclude them.")
# TODO: double check the effect of including face detection or not. In theory including face detection will result
# in the network learning skin color better.
parser.add_argument("--no_face_detection", action="store_true", help="If set to be true, images without anime "
                                                                     "character faces can be included. "
                                                                     "Default is to exclude them.")
parser.add_argument("--gen_sketch", action="store_true", help="If set to be true, sketches are generated along "
                                                                     "with the resized colored image."
                                                                     "Default is not save sketch")
parser.add_argument("--size", type=int, default=256, help="size to use for resize operation")
parser.add_argument("--verbose", action="store_true", help="If not set, then only print how many images it has processed. Otherwise print each file processed.")
parser.add_argument("--gpu_limit", type=float, default=0.5, help="The percentage of gpu this program can use.")
parser.add_argument("--output_list_only", action="store_true",
                    help="If set to be true, then the only output is a list of images that passed the tests.")
a = parser.parse_args()

if a.output_list_only and a.gen_sketch:
    parser.error("output_list_only cannot be used together with gen_sketch. If you would like to generate a list of "
                 "images passing the tests, please take off the gen_sketch flag.")
# if a.output_list_only and not os.path.isfile(a.output_dir):
#     parser.error("output_list_only flag is turned on. Do not provide a directory. "
#                  "Please provide a path to the file name to save the list of image paths instead. ")



def main():
    random.seed(0)

    color_dir = os.path.join(a.output_dir,"color")
    sketch_dir = os.path.join(a.output_dir,"sketch")
    if a.output_list_only:
        if not os.path.exists(os.path.dirname(a.output_dir)):
            os.makedirs(os.path.dirname(a.output_dir))
    else:
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

    if not a.no_face_detection:
        cascade_file = "./lbpcascade_animeface.xml"
        if not os.path.isfile(cascade_file):
            try:
                os.system(
                    "wget https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml")
                assert os.path.isfile(cascade_file)
            except:
                raise RuntimeError("%s: not found" % cascade_file)
        cascade_classifier = cv2.CascadeClassifier(cascade_file)

    with tf.Session(config=config) as sess:
        image_i = 0
        num_image_passing_hw_ratio = 0
        num_image_passing_bw = 0
        num_image_passing_face = 0

        all_image_paths = get_all_image_paths(a.input_dir)
        num_images = len(all_image_paths)

        output_image_path_list = a.output_dir if a.output_list_only else os.path.join(a.output_dir, "image_path_list.txt")
        output_image_path_list_file = open(output_image_path_list, "w")

        reader = tf.WholeFileReader()
        filename_queue = tf.train.string_input_producer(all_image_paths, shuffle=False)
        paths, contents = reader.read(filename_queue)
        src = load(paths, contents, channels=3)
        height, width = get_image_hw(src)
        # paths, src = tf.train.batch([paths, src], batch_size=a.batch_size)
        hw_ratio_test_result = pass_hw_test(height, width, a.hw_ratio_threshold)
        resized_image = tf.cond(hw_ratio_test_result, lambda: resize_image(src,height,width, a.resize_mode, a.size), lambda: tf.constant(0.0))
        sketch = sketch_extractor(src, color_space="rgb", max_val=1.0, min_val=0.0)
        sketch_resized = tf.cond(hw_ratio_test_result, lambda: resize_image(sketch,height,width, a.resize_mode, a.size), lambda: tf.constant(0.0))
        resized_image_encoded = tf.cond(hw_ratio_test_result, lambda: tf.image.encode_png(tf.image.convert_image_dtype(resized_image, dtype=tf.uint8),name="output_pngs"), lambda: tf.constant("", dtype=tf.string))
        sketch_resized_encoded = tf.cond(hw_ratio_test_result, lambda: tf.image.encode_png(tf.image.convert_image_dtype(sketch_resized, dtype=tf.uint8),name="sketch_pngs"), lambda: tf.constant("", dtype=tf.string))

        if a.allow_bw:
            bw_test_result = tf.constant(True)
        else:
            bw_test_result = tf.cond(hw_ratio_test_result, lambda: detect_bw_tf(resized_image),lambda: tf.constant(False))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        start_time = time.time()
        for image_i, src_path in enumerate(all_image_paths):
            if not a.output_list_only:
                dst_path = png_path(os.path.join(color_dir, os.path.basename(src_path)))
                dst_sketch_path = png_path(os.path.join(sketch_dir, os.path.basename(src_path)))
            if not a.verbose:
                if image_i % 100 == 0:
                    current_time = time.time()
                    remaining_time = 0.0 if image_i == 0 else (num_images - image_i) * (float(current_time - start_time) / image_i)
                    print('%.3f%% done. Remaining time: %.1fs' % (float(image_i) / num_images * 100, remaining_time))
                    # print("Processed %d images." %image_i)
            else:
                if a.output_list_only:
                    print(src_path)
                else:
                    print(src_path, "->", dst_path)
            try:
                fetches = {
                    "paths": paths,
                    "hw_ratio_test_result": hw_ratio_test_result,
                    "bw_test_result":bw_test_result
                }
                if not a.no_face_detection:
                    # Don't need the actual resized image if we're not doing face detection.
                    fetches["dst"] = resized_image
                if not a.output_list_only:
                    fetches["dst_encoded"] = resized_image_encoded
                    if a.gen_sketch:
                        # fetches["dst_sketch"] = sketch_resized
                        fetches["dst_sketch_encoded"] = sketch_resized_encoded
                results = sess.run(fetches)
                assert results["paths"] == src_path
                if results["hw_ratio_test_result"]:
                    num_image_passing_hw_ratio += 1
                    if results["bw_test_result"]:
                        num_image_passing_bw += 1
                        # a wierd bug: int(min(a.size / 8, 32)) works but min(a.size / 8, 32) does not.
                        # min(a.size / 8, 32) gives "Required argument 'rejectLevels' (pos 2) not found"
                        if a.no_face_detection or detect_face(results["dst"], cascade_classifier, face_min_size=int(min(a.size / 8, 16))):
                            num_image_passing_face += 1
                            output_image_path_list_file.write(results["paths"] + "\n")
                            if not a.output_list_only:
                                save(results["dst_encoded"], dst_path)
                                if a.gen_sketch:
                                    save(results["dst_sketch_encoded"], dst_sketch_path)
                                src_txt_path = src_path+"txt"
                                if os.path.isfile(src_txt_path):
                                    copy(src_txt_path, dst_path+".txt")
            except Exception as exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback)
                print("exception ", exception, " happened when processing ", src_path, ". Skipping this one. ")
        print("Number of images preprocessed in total: %d. In which %d passed hw ratio check, %d passed black-and-white test and %d passed face test and was saved." %(image_i +1, num_image_passing_hw_ratio, num_image_passing_bw ,num_image_passing_face))
        output_image_path_list_file.close()
        coord.request_stop()
        coord.join(threads)
        sess.close()
if __name__ == "__main__":
    main()
    # In the previous version it takes around 4 hours to process 50000 images with no face detection...

"""
python preprocess.py --input_dir=/mnt/pixiv_drive/home/ubuntu/PycharmProjects/PixivUtil2/pixiv_downloaded/ --output_dir=/mnt/data_drive/home/ubuntu/pixiv_new_128/ --size=128 --resize_mode=reshape --gpu_limit=0.25
"""