import unittest

import general_util
from sketches_util import *

class TestImageToSketchesUtil(unittest.TestCase):
    def test_image_to_sketch_experiment(self):
        img = general_util.imread('2821993.jpg') # '12746957.jpg')
        sketch = image_to_sketch_experiment(img)
        cv2.imshow('Input', cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8))
        cv2.imshow('Sketch', sketch.astype(np.uint8))
        cv2.waitKey(0)


    # def test_image_to_sketch(self):
    #     img = general_util.imread('2821993.jpg') # '12746957.jpg')
    #     sketch = image_to_sketch(img)
    #     cv2.imshow('Input', cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8))
    #     cv2.imshow('Sketch', sketch.astype(np.uint8))
    #     cv2.waitKey(0)
    #
    #     # actual_output = raw_input('Please enter "y" if you think the result is satisfactory')
    #     # expected_output = 'y'
    #     # self.assertEqual(actual_output, expected_output)
    #
    # def test_generate_hint_from_image(self):
    #     img = general_util.imread('face_36_566_115.png')
    #     sketch = generate_hint_from_image(img)
    #     cv2.imshow('Input', cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8))
    #     cv2.imshow('Hint', cv2.cvtColor(sketch.astype(np.uint8), cv2.COLOR_RGBA2BGR))
    #     cv2.waitKey(0)
    #
    #     # actual_output = raw_input('Please enter "y" if you think the result is satisfactory')
    #     # expected_output = 'y'
    #     # self.assertEqual(actual_output, expected_output)
    #
    #
    # def test_detect_bw(self):
    #     all_img_paths = general_util.get_all_image_paths_in_dir('/home/xor/pixiv_testing_renamed/')
    #
    #     for img_path in all_img_paths[:10]:
    #         print(img_path)
    #         detect_bw(img_path)
    #
    #
    #     # img = general_util.imread('face_36_566_115.png')
    #     # sketch = generate_hint_from_image(img)
    #     # cv2.imshow('Input', cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8))
    #     # cv2.imshow('Hint', cv2.cvtColor(sketch.astype(np.uint8), cv2.COLOR_RGBA2BGR))
    #     # cv2.waitKey(0)
    #
    #     # actual_output = raw_input('Please enter "y" if you think the result is satisfactory')
    #     # expected_output = 'y'
    #     # self.assertEqual(actual_output, expected_output)
    #
    # def test_detect_complicated_img(self):
    #
    #     all_img_paths = general_util.get_all_image_paths_in_dir('/home/xor/pixiv_images/train_images/')
    #
    #     for img_path in all_img_paths[10:]:
    #         print(img_path)
    #         print(detect_complicated_img(img_path))
    #
    #
    #         img = general_util.imread(img_path)
    #         sketch = image_to_sketch(img)
    #         cv2.imshow('Input', cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8))
    #         cv2.imshow('Sketch', sketch.astype(np.uint8))
    #         cv2.waitKey(0)


if __name__ == '__main__':
    unittest.main()
