import numpy as np
import tensorflow as tf

import optimizations


class OptimizationsTest(tf.test.TestCase):

    def test_adjust_gain(self):
        x = tf.convert_to_tensor([[1.0, -1.0, -0.5], [0.1, -0.1, 0.5]],
                                 dtype=tf.float32)
        y = optimizations.adjust_pct_gain(x, 1.1)
        want = [[1.0, -1.1, -0.55],
                [0.1, -0.11, 0.5]]
        self.assertNDArrayNear(y, want, 1e-4)

        log_x = tf.math.log(x / 100.0 + 1.0)
        log_y = optimizations.adjust_log_gain(log_x, 1.1)
        y = 100.0 * (tf.math.exp(log_y) - 1.0)
        self.assertNDArrayNear(y, want, 1e-4,
                               f'x=\n{x} \nlog_x=\n{log_x} \nlog_y=\n{log_y} \ny=\n{y} \nwant=\n{want}\n')

    def test_value_of_argmax_prev_value(self):
        x = tf.convert_to_tensor(
            [[1, 4, 2],   # -> Max 4, so select the 3 on next row. For this row uses default=0.0
             [7, 3, 1],   # -> Max 7, so select the 8 on the next row.
             [8, 1, 4],   # -> Max 8, so select 1 on the next row.
             [1, 3, 5],   # -> Only 3 is not masked out, chosen as max.
             [5, 2, 7],
             ], tf.float32)
        x2 = x * 10
        mask = tf.convert_to_tensor([
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [False, True, False],
            [True, True, False]
        ], dtype=tf.bool)
        y, y2 = optimizations.value_of_argmax_prev_value(
            x, [x, x2], mask, transposed=False, default=0.0)
        want = np.array([3, 8, 0, 2], dtype=np.float32)
        want2 = want * 10
        self.assertArrayNear(y, want, 1e-4,
                             f'x=\n{x} \ny=\n{y} \nwant=\n{want}\n')
        self.assertArrayNear(y2, want2, 1e-4,
                             f'x2=\n{x2} \ny2=\n{y2} \nwant2=\n{want2}\n')

        x = tf.transpose(x)
        mask = tf.transpose(mask)
        y = optimizations.value_of_argmax_prev_value(
            x, [x], mask, transposed=True, default=0.0)[0]
        self.assertArrayNear(y, want, 1e-4,
                             f'x=\n{x} \ny=\n{y} \nwant=\n{want}\n')


if __name__ == '__main__':
    tf.test.main()
