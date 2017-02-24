
"""
This file provides functions for turning any image into a 'sketch'.
"""

import random
import shutil
import time

import cv2
from PIL import ImageStat
from scipy.stats import threshold

import colorful_img_network_util
from general_util import *

neiborhood8 = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
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
        # Usually Sketches are represented by dark lines on white background, so I need to invert.
        img_diff_dilation_gray_inverted = 255-img_diff_dilation_gray_thresholded


        return img_diff_dilation_gray_inverted
    else:
        print('Image has to be either of shape (height, width, num_features) or (batch_size, height, width, num_features)')
        raise AssertionError


def generate_hint_from_image(img, max_num_hint = 15, min_num_hint = 5):
    """
    :param image: An image represented in numpy array with shape (height, width, 3) or (batch, height, width, 3)
    :return: A hint of the color usage of the original image image with shape (height, width, 4) or
    (batch, height, width, 4) where the last additional dimension stands for a (in rgba).
    """
    assert max_num_hint >= min_num_hint
    _hint_max_width = 15
    _hint_min_width = 5
    _hint_max_area = 100
    _hint_min_area = 25

    if len(img.shape) == 4:
        img_diff_dilation_gray =  np.array([generate_hint_from_image(img[i, ...]) for i in range(img.shape[0])])
        return img_diff_dilation_gray
    elif len(img.shape) == 3:
        if min_num_hint == max_num_hint:
            num_hints = max_num_hint
        else:
            num_hints = random.randint(min_num_hint, max_num_hint)


        height, width, rgb = img.shape
        assert rgb==3

        # All unmarked pixels are filled by 0,0,0,0 by default.
        ret = np.zeros(shape=(img.shape[0],img.shape[1],4))

        # Select random sites to give hints about the color used at that point.
        for hint_i in range(num_hints):
            curr_hint_width = random.randint(_hint_min_width, _hint_max_width)
            curr_hint_area = random.randint(_hint_min_area, _hint_max_area)
            curr_hint_height = int(curr_hint_area / curr_hint_width)

            rand_x = random.randint(0,width-1)
            rand_y = random.randint(0,height-1)
            ret[max(0, rand_y - curr_hint_height / 2):min(height, rand_y + curr_hint_height),max(0, rand_x - curr_hint_width / 2):min(width, rand_x + curr_hint_width),0:3] = img[max(0, rand_y - curr_hint_height / 2):min(height, rand_y + curr_hint_height),max(0, rand_x - curr_hint_width / 2):min(width, rand_x + curr_hint_width),:]
            ret[max(0, rand_y - curr_hint_height / 2):min(height, rand_y + curr_hint_height),max(0, rand_x - curr_hint_width / 2):min(width, rand_x + curr_hint_width),3] = np.ones((min(height, rand_y + curr_hint_height) - max(0, rand_y - curr_hint_height / 2),min(width, rand_x + curr_hint_width) - max(0, rand_x - curr_hint_width / 2))) * 255.0
        return ret

    else:
        print('Image has to be either of shape (height, width, num_features) or (batch_size, height, width, num_features)')
        raise AssertionError


IMG_TYPE_GRAYSCALE = 1
IMG_TYPE_COLOR = 2
IMG_TYPE_BW = 3
IMG_TYPE_UK = 4

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


def detect_complicated_img(file, threshold = 10.0):
    img = imread(file)
    sketch = image_to_sketch(img)
    sketch = np.expand_dims(np.expand_dims(sketch,axis=0),axis=3)
    total_var = np_total_variation(sketch)
    print(total_var)
    if total_var < threshold:
        return False
    else:
        return True

def training_data_clean(data_folder, start_percentage = 0.0):
    all_img_paths = get_all_image_paths_in_dir(data_folder)
    print('Read %d images.' %len(all_img_paths))
    if start_percentage != 0.0:
        all_img_paths = all_img_paths[int(len(all_img_paths) * start_percentage / 100.0):]
        print("Starting at %f%% with %d images left" %(start_percentage, len(all_img_paths)))
    for i, img_path in enumerate(all_img_paths):
        try:
            if not detect_bw(img_path) == IMG_TYPE_COLOR:
                shutil.copy(img_path, img_path + '.bak')
                os.remove(img_path)
            else:
                content_pre_list = read_and_resize_batch_images([img_path], None, None)
                height = float(content_pre_list.shape[1])
                width = float(content_pre_list.shape[2])
                if height / width > 2.0 or width / height > 2.0:
                    shutil.copy(img_path, img_path + '.bak')
                    os.remove(img_path)
            if i % 100 == 0:
                print("%.3f%% done." % (100.0 * float(i) / len((all_img_paths))))
        except:
            os.remove(img_path)


def read_resize_and_save_batch_images_with_sketches(dirs, height, width, save_path, sketch_save_path, max_size_g=32):
    # type: (List[str], int, int, str, str, int) -> None
    """
    Preprocessing the images and save them as npy will greatly increase the training speed since it decreased the
    time spent on cpu trying to read images and convert them to sketches.
    :param dirs: a list of strings of paths to images.
    :param height: height of outputted images. If height and width are both None, then the images are not resized.
    :param width: width of outputted images. If height and width are both None, then the images are not resized.
    :param save_path: The path to save the preprocessed images (as numpy array).
    :param max_size_g: the maximum size of the numpy array. If it exceeds this size, a warning will be displayed and
    nothing will be saved.
    :return: an numpy array representing the resized images. The shape is (num_image, height, width, 3). The numpy
    array is also saved at "save_dir".
    """
    # TODO: fix the uint8 issue...
    raise NotImplementedError('I did not fix the uint8 issue yet.')
    if height is None or width is None:
        raise AssertionError('The height and width has to be both non None or both None.')
    shape = (height, width)
    estimated_size = height * width * (1 + 3) * len(dirs) * 1 # 1 for the size of np.uint8
    print('Estimated numpy array size: %d' %estimated_size)
    max_bytes = max_size_g * (1024 ** 3)
    if estimated_size > max_bytes:
        raise AssertionError('The estimated size of the images (%fG) to be saved exceeds the max allowed size (%fG) '
                             'specified. ' %(float(estimated_size) / (1024**3), float(max_size_g)))

    images = np.array([imread(d, shape=shape, dtype=np.uint8) for d in dirs], np.uint8)
    print('Saving numpy array with size %.3f G' %(images.nbytes / float(1024 ** 3)))
    np.save(save_path, images)
    print('Finished saving. Now converting sketches')
    image_sketches = image_to_sketch(images)
    print('Finished converting sketches. Now saving it as np array with size %.3f G'
          %(image_sketches.nbytes / float(1024 ** 3)))
    np.save(sketch_save_path, image_sketches)
    return

def preprocess_img_in_dir_and_save(directory, height, width, save_dir, batch_size, max_size_g=32):
    assert save_dir[-1] == '/'
    all_img_dirs = get_all_image_paths_in_dir(directory)
    num_images = len(all_img_dirs)
    image_per_file = max_size_g * (1024 ** 3) / (height * width * (1 + 3) * 1)
    # Make sure that each file contains number of images that is divisible by batch size.
    num_images = num_images - num_images % batch_size
    image_per_file = image_per_file - image_per_file % batch_size

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir+'images/')
        os.makedirs(save_dir+'sketches/')

    with open(save_dir + 'record.txt', 'w') as record_f:
        i = 0
        while i < num_images:
            if (i + image_per_file > num_images):
                end_i = num_images
                current_file_image_dirs = get_batch_paths(all_img_dirs, i, num_images - i)
            else:
                end_i = i + image_per_file
                current_file_image_dirs = get_batch_paths(all_img_dirs, i, image_per_file)

            current_images_save_path = save_dir + 'images/%dx%d_%d_to_%d.npy' % (height,width,i,end_i)
            current_sketches_save_path = save_dir + 'sketches/%dx%d_%d_to_%d.npy' % (height,width,i,end_i)
            read_resize_and_save_batch_images_with_sketches(current_file_image_dirs, height, width,
                                                            current_images_save_path,current_sketches_save_path,
                                                            max_size_g=max_size_g)

            record_f.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\n' %(current_images_save_path, current_sketches_save_path,
                                                        batch_size,height,width,i,end_i))
            i = end_i
            print('%.3f%% Done.' %(float(end_i) / num_images * 100.0))
        assert i == num_images

def read_preprocessed_sketches_npy_record(save_dir):
    ret = []
    with open(save_dir + 'record.txt', 'r') as record_f:
        for line in record_f:
            line_split = line.split('\t')
            if len(line_split) == 7:
                for item in range(2,7):
                    line_split[item] = int(line_split[item])
                ret.append(line_split)
            elif len(line_split) == 0:
                pass
            else:
                raise AssertionError('Error in read_preprocessed_npy_record. Format of record.txt is wrong.')
    return ret

def find_corresponding_sketches_npy_from_record(record_list, start_index, batch_size):
    num_images = record_list[-1][-1]
    start_index = start_index % num_images
    for record_i, record in enumerate(record_list):
        if start_index + batch_size < record[6]:
            return record_i, max(0,start_index - record[5])

    # If the end index had exceeded the end of the record, start from the beginning instead of raising an error.
    return 0, 0
    raise AssertionError('Error in find_corresponding_npy_from_record.')

def calc_rgb_bin_distr_and_weights(directory, save_dir, bin_num = 6, lambd = 0.5):
    assert save_dir[-1] == '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_img_dirs = get_all_image_paths_in_dir(directory)
    random.shuffle(all_img_dirs)
    all_img_dirs = all_img_dirs[:len(all_img_dirs) / 10] # Only sample 10% of all the images to save time.
    num_imgs = len(all_img_dirs)
    img2rgbbin_encoder = colorful_img_network_util.ImgToRgbBinEncoder(bin_num=bin_num)

    rgbbin_sum = np.zeros((bin_num ** 3))
    start_time = time.time()

    for i, img_dir in enumerate(all_img_dirs):
        image = imread(img_dir,(64,64))
        image_rgbbin = img2rgbbin_encoder.nnencode.encode_points_mtx_nd(image,axis=2, return_sparse=False)
        image_rgbbin_mean = np.mean(image_rgbbin,axis=(0,1))
        rgbbin_sum = rgbbin_sum + image_rgbbin_mean
        if i % 100 == 0:
            end_time = time.time()
            remaining_time = 0.0 if i == 0 else (num_imgs - i) * (float(end_time-start_time) / i)
            print('%.3f%% done. Remaining time: %.1fs' %(float(i) / num_imgs * 100, remaining_time))

    rgbbin_avg = rgbbin_sum / num_imgs
    print(rgbbin_avg)
    np.testing.assert_almost_equal(np.sum(rgbbin_avg),1.0)

    rgbbin_avg_dir = save_dir + 'rgbbin_avg.npy'
    np.save(rgbbin_avg_dir,rgbbin_avg)

    # Didn't understand what the smoothing part is doing in the paper. It doesn't make sense to smooth a probability
    # distribution with a gaussian kernel... So I left it as it is
    # rgbbin_avg_smoothed = img2rgbbin_encoder.gaussian_kernel(rgbbin_avg, std=0.005)
    # rgbbin_avg_smoothed = rgbbin_avg_smoothed / np.sum(rgbbin_avg_smoothed)
    # print(rgbbin_avg_smoothed)
    rgbbin_avg_smoothed = rgbbin_avg
    weights = np.divide(1.0, (rgbbin_avg_smoothed * (1-lambd) + lambd / bin_num ** 3))
    weights = weights / np.sum(np.multiply(weights,rgbbin_avg_smoothed))
    print(weights)

    np.testing.assert_almost_equal(np.sum(np.multiply(weights, rgbbin_avg_smoothed)),1.0)

    weights_dir = save_dir + 'weights.npy'
    np.save(weights_dir,weights)


def calc_ab_bin_distr_and_weights(directory, save_dir, lambd = 0.5):
    assert save_dir[-1] == '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_img_dirs = get_all_image_paths_in_dir(directory)
    random.shuffle(all_img_dirs)
    all_img_dirs = all_img_dirs[:len(all_img_dirs) / 10] # Only sample 10% of all the images to save time.
    num_imgs = len(all_img_dirs)
    img2bin_encoder = colorful_img_network_util.ImgToABBinEncoder()

    bin_sum = np.zeros((colorful_img_network_util.LAB_NUM_BINS))
    start_time = time.time()

    for i, img_dir in enumerate(all_img_dirs):
        image = imread(img_dir,(64,64))
        image_lab = colorful_img_network_util.rgb_to_lab(image)
        image_bin = img2bin_encoder.nnencode.encode_points_mtx_nd(image_lab[...,1:],axis=2, return_sparse=False)
        image_bin_mean = np.mean(image_bin,axis=(0,1))
        bin_sum = bin_sum + image_bin_mean
        if i % 100 == 0:
            end_time = time.time()
            remaining_time = 0.0 if i == 0 else (num_imgs - i) * (float(end_time-start_time) / i)
            print('%.3f%% done. Remaining time: %.1fs' %(float(i) / num_imgs * 100, remaining_time))

    bin_avg = bin_sum / num_imgs
    print(bin_avg)
    np.testing.assert_almost_equal(np.sum(bin_avg),1.0)

    rgbbin_avg_dir = save_dir + 'rgbbin_avg.npy'
    np.save(rgbbin_avg_dir,bin_avg)

    # Didn't understand what the smoothing part is doing in the paper. It doesn't make sense to smooth a probability
    # distribution with a gaussian kernel... So I left it as it is
    # rgbbin_avg_smoothed = img2rgbbin_encoder.gaussian_kernel(rgbbin_avg, std=0.005)
    # rgbbin_avg_smoothed = rgbbin_avg_smoothed / np.sum(rgbbin_avg_smoothed)
    # print(rgbbin_avg_smoothed)
    bin_avg_smoothed = bin_avg
    weights = np.divide(1.0, (bin_avg_smoothed * (1-lambd) + lambd / colorful_img_network_util.LAB_NUM_BINS ** 3))
    weights = weights / np.sum(np.multiply(weights,bin_avg_smoothed))
    print(weights)

    np.testing.assert_almost_equal(np.sum(np.multiply(weights, bin_avg_smoothed)),1.0)

    weights_dir = save_dir + 'weights.npy'
    np.save(weights_dir,weights)


if __name__ == '__main__':
    # height = 256
    # width = 256
    # batch_size = 12
    #
    # print('Directly calling this file will start the process to convert all training images to preprocessed numpy '
    #       'files. Current setting is height = %d, width = %d, batch_size = %d' %(height, width, batch_size))
    #
    # preprocess_img_in_dir_and_save('/mnt/pixiv_drive/home/ubuntu/PycharmProjects/PixivUtil2/pixiv_downloaded/',
    #                                height, width, 'pixiv_img_preprocessed_npy/256/',batch_size, max_size_g=16)

    # print('testing calc_rgb_bin_distr_and_weights')
    # calc_rgb_bin_distr_and_weights('/home/xor/pixiv_testing/', 'rgb_bin_distr_and_weights/')

    print('testing calc_ab_bin_distr_and_weights')
    calc_ab_bin_distr_and_weights('/mnt/pixiv_drive/home/ubuntu/PycharmProjects/PixivUtil2/pixiv_downloaded/', 'resources/ab_bin_distr_and_weights/')
