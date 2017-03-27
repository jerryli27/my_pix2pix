import shutil
import tempfile

from neural_util import *

class NeuralUtilTest(tf.test.TestCase):
    def test_get_tf_string_extension(self):
        with self.test_session(config = tf.ConfigProto(device_count = {'GPU': 0})) as sess:
            file_name = "test.jpg"
            actual_output, = sess.run([get_tf_string_extension(file_name)])
            expected_output = "jpg"
            tf.assert_equal(actual_output,expected_output)

    def test_decode_image_with_file_name(self):
        dirpath = tempfile.mkdtemp()
        try:
            for rd, ext in enumerate(["jpg", "JPG", "png", 'PNG']):
                file_name = dirpath + '/image.' + ext
                with self.test_session(config = tf.ConfigProto(device_count = {'GPU': 0})) as sess:
                    # Save test image
                    img = np.zeros((128, 128, 3), dtype=np.uint8)
                    img[rd, 0, 0] = 1
                    encoder = tf.image.encode_jpeg if ext.lower() == "jpg" else tf.image.encode_png
                    encoded_img, = sess.run([encoder(img, name="output_imgs")])

                    with open(file_name, "w") as f:
                        f.write(encoded_img)
                    reader = tf.WholeFileReader()
                    file_name_tensor = tf.constant([file_name])
                    filename_queue = tf.train.string_input_producer([file_name])
                    paths, contents = reader.read(filename_queue)
                    raw_input = decode_image_with_file_name(contents, paths, channels=3)

                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                    actual_output, = sess.run([raw_input])

                    expected_output = img
                    tf.assert_equal(actual_output, expected_output)
        finally:

            shutil.rmtree(dirpath)

if __name__ == '__main__':
    tf.test.main()
