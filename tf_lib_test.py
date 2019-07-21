import numpy as np
import tensorflow as tf

from tf_lib import *


class TfLibTest(tf.test.TestCase):

    def test_partial_matrix_reduce_sum(self):
        x = tf.convert_to_tensor(
            [[1, 4, 2],   # -> Max 4, so select the 3 on next row. For this row uses default=0.0
             [7, 3, 1],   # -> Max 7, so select the 8 on the next row.
             [8, 1, 4],   # -> Max 8, so select 1 on the next row.
             [1, 3, 5],   # -> Only 3 is not masked out, chosen as max.
             [5, 2, 7],
             ], tf.float32)
        x = tf.transpose(x)
        want = np.array([[8, 9], [7, 4], [3, 9]], dtype=np.float32)
        y = partial_matrix_reduce_sum(x, 2)
        self.assertNDArrayNear(y, want, 1e-4,
                               f'x=\n{x} \ny=\n{y} \nwant=\n{want}\n')

        want = np.array([[16], [8], [7]], dtype=np.float32)
        y = partial_matrix_reduce_sum(x, 3)
        self.assertNDArrayNear(y, want, 1e-4,
                               f'x=\n{x} \ny=\n{y} \nwant=\n{want}\n')

    def test_partial_matrix_reduce_any(self):
        mask = tf.convert_to_tensor([
            [True, True, False],
            [True, False, False],
            [True, True, False],
            [False, True, True],
            [True, True, False]
        ], dtype=tf.bool)
        mask = tf.transpose(mask)
        y = partial_matrix_reduce_any(mask, 2)
        want = np.array([[True, True], [True, True], [
                        False, True]], dtype=np.bool)
        self.assertTrue(np.array_equal(y.numpy(), want),
                        f'mask=\n{mask} \ny=\n{y} \nwant=\n{want}\n')

        y = partial_matrix_reduce_any(mask, 3)
        want = np.array([[True], [True], [False]], dtype=np.bool)
        self.assertTrue(np.array_equal(y.numpy(), want),
                        f'mask=\n{mask} \ny=\n{y} \nwant=\n{want}\n')


if __name__ == '__main__':
    tf.test.main()
