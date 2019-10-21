import numpy as np
import tensorflow as tf

import optimizations


class OptimizationsTest(tf.test.TestCase):

    def test_mix_gain(self):
        logits = tf.convert_to_tensor([1.0, 1.0])
        logits_mask = tf.convert_to_tensor([True, True], dtype=tf.bool)
        gains = tf.convert_to_tensor([
            [1.5, 2.0, 1, 1],
            [0.25, 1, 2, 1]])
        log_gains = tf.math.log(gains)

        # Try first only iwth penalties for loss.
        mix_gain, adjusted_mix_gain = optimizations.mix_gain(
            logits, logits_mask, tf.transpose(log_gains),
            1.1, 1)
        print(f'mix_gain={mix_gain.numpy()}, adjusted_mix_gain={adjusted_mix_gain.numpy()}')
        self.assertNear(mix_gain, 1.75, 1e-4)
        self.assertNear(adjusted_mix_gain, 1.728125, 1e-4)

        print('\n\n')

        # Test that a large gain suffers more from power function than many
        # smaller ones.
        gain_power = 0.5
        logits = tf.convert_to_tensor([1.0, -1000])
        gains = tf.convert_to_tensor([
            [2.0, 2.0, 1],
            [1.0, 4.0, 1]])
        log_gains = tf.math.log(gains)
        mix_gain, adjusted_mix_gain_a = optimizations.mix_gain(
            logits, logits_mask, tf.transpose(log_gains),
            1.1, gain_power)
        print(f'mix_gain={mix_gain.numpy()}, adjusted_mix_gain={adjusted_mix_gain_a.numpy()}')
        self.assertNear(mix_gain, 4.0, 1e-4)
        print('\n')

        logits = tf.convert_to_tensor([-1000.0, 1.0])
        mix_gain, adjusted_mix_gain_b = optimizations.mix_gain(
            logits, logits_mask, tf.transpose(log_gains),
            1.1, gain_power)
        print(f'mix_gain={mix_gain.numpy()}, adjusted_mix_gain={adjusted_mix_gain_b.numpy()}')
        self.assertNear(mix_gain, 4.0, 1e-4)
        self.assertGreater(adjusted_mix_gain_a, adjusted_mix_gain_b)

    def test_adjusted_log_gains(self):
        x = tf.convert_to_tensor([[1.0, -1.0, -0.5], [0.1, -0.1, 0.5]],
                                 dtype=tf.float32)
        want = [[1.0, -1.1, -0.55],
                [0.1, -0.11, 0.5]]

        log_x = tf.math.log(x / 100.0 + 1.0)
        log_y = optimizations.adjusted_log_gains(log_x, 1.1, 1)
        y = 100.0 * (tf.math.exp(log_y) - 1.0)
        self.assertNDArrayNear(y, want, 1e-4,
                               f'x=\n{x} \nlog_x=\n{log_x} \nlog_y=\n{log_y} \ny=\n{y} \nwant=\n{want}\n')

        # Now try the version that works on percentage points.
        y = optimizations.adjusted_pct_gains(x, 1.1)
        self.assertNDArrayNear(y, want, 1e-4)

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
