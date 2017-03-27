import shutil
import tempfile

from preprocess_util import *
from general_util import get_all_image_paths

class PreprocessUtilTest(tf.test.TestCase):

    def test_detect_bw_tf(self):
        bw_file_names = get_all_image_paths("bw_test_images/bw/")
        colored_file_names = get_all_image_paths("bw_test_images/colored/")
        input_file_names = bw_file_names + colored_file_names

        with self.test_session(config = tf.ConfigProto(device_count = {'GPU': 0})) as sess:
            # Save test image

            reader = tf.WholeFileReader()
            filename_queue = tf.train.string_input_producer(input_file_names)
            paths, contents = reader.read(filename_queue)
            image_decoded = decode_image_with_file_name(contents, paths, channels=3)
            image_float = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
            is_bw = detect_bw_tf(image_float)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(len(input_file_names)):
                current_path,actual_output, = sess.run([paths, is_bw])
                expected_output = current_path in colored_file_names
                tf.assert_equal(actual_output, expected_output)
                # print("%s is %s" %(current_path, "colored" if expected_output else "grayscale"))

            coord.request_stop()
            coord.join(threads)
            sess.close()

if __name__ == '__main__':
    tf.test.main()
