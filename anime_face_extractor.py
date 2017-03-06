"""
The anime face extractor is taken from https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/examples/detect.py
This file is mainly for pre-processing the dataset. It will take a dataset that has already been separated into color
images and sketch images, go through all the color images, and compose a list of those that contains at least one face.
"""

import cv2
import sys
import time
import os.path
from argparse import ArgumentParser

from general_util import get_all_image_paths_in_dir, get_file_name

def detect(colored_save_dir, save_path="images_containing_face.txt", face_min_size=32, cascade_file="./lbpcascade_animeface.xml"):

    colored_image_dir_len = len(colored_save_dir)
    all_img_dirs = get_all_image_paths_in_dir(colored_save_dir)
    num_images = len(all_img_dirs)
    num_images_with_face = 0

    if not os.path.isfile(cascade_file):
        try:
            os.system("wget https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml")
            assert os.path.isfile(cascade_file)
        except:
            raise RuntimeError("%s: not found" % cascade_file)
    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))

    start_time = time.time()
    cascade = cv2.CascadeClassifier(cascade_file)
    with open(save_path, 'w') as fout:
        for i, filename in enumerate(all_img_dirs):
            image = cv2.imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = cascade.detectMultiScale(gray,
                                             # detector options
                                             scaleFactor= 1.1,
                                             minNeighbors=5,
                                             # minSize=(24, 24))
                                             minSize=(face_min_size, face_min_size))
            if len(faces) > 0:
                img_file_name = get_file_name(filename)
                img_subdir_name = os.path.dirname(filename)[colored_image_dir_len:]
                img_subdir_path = img_subdir_name + '/' + img_file_name + '.png'
                fout.write(img_subdir_path + '\n')
                num_images_with_face += 1

            if i % 100 == 0:
                end_time = time.time()
                remaining_time = 0.0 if i == 0 else (num_images - i) * (float(end_time - start_time) / i)
                percentage_containing_faces = float(num_images_with_face) / (i + 1) * 100
                print('%.3f%% done. %.3f%% images contains faces. Remaining time: %.1fs'
                      % (float(i) / num_images * 100, percentage_containing_faces,  remaining_time))

        # This will actually save the faces, but we don't need that for now.
        # for (x, y, w, h) in faces:
        #     # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        #     faces_cropped_image= image[y:y+h, x:x+w]
        #     faces_cropped_image = cv2.resize(faces_cropped_image, (hw, hw))
        #     cv2.imwrite(os.path.join(save_dir,"%d.png" %num_faces_detected), faces_cropped_image)
        #     num_faces_detected += 1
    print("DONE. Number of images containing faces: %d out of %d" %(num_images_with_face, num_images))

if __name__ == "__main__":
    parser = ArgumentParser()
    # '/home/ubuntu/pixiv_full/pixiv/' or /home/ubuntu/pixiv/pixiv_training_filtered/' or
    # '/mnt/pixiv_drive/home/ubuntu/PycharmProjects/PixivUtil2/pixiv_downloaded/' -> Number of images  442701.
    parser.add_argument('--colored_save_dir', dest='colored_save_dir',
                        help='The path to images containing anime characters. ',
                        default='/mnt/data_drive/home/ubuntu/pixiv_downloaded_sketches_lnet_128/color/')
    parser.add_argument('--save_path', dest='save_path',
                        help='The path to save the list of images containing faces.',
                        default="/mnt/data_drive/home/ubuntu/pixiv_downloaded_sketches_lnet_128/images_containing_face.txt",)
    parser.add_argument('--face_min_size', dest='face_min_size', type=int,
                        help='face_min_size.',
                        default=16)
    parser.add_argument('--cascade_file', dest='cascade_file',
                        help='The path to the cascade xml file. ', metavar='CASCADE_FILE',
                        default="./lbpcascade_animeface.xml")
    args = parser.parse_args()
    detect(args.colored_save_dir, args.save_path, args.face_min_size, args.cascade_file)