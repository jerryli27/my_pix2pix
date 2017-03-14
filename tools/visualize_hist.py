import cv2
import numpy as np
from matplotlib import pyplot as plt

from general_util import get_all_image_paths_in_dir

if __name__ == "__main__":
    all_img_dirs = get_all_image_paths_in_dir("../sketches/") # "../sketches/"

    for img_dir in all_img_dirs:
        img = cv2.imread(img_dir,0)
        plt.hist(img.ravel(),256,[0,256]); plt.show(block=True)