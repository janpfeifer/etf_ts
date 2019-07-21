# optimizations contains many optimization functions, written for Tensorflow
import math
import tensorflow as tf


def adjust_pct_daily_gain(values: tf.Tensor, loss_cost: float) -> tf.Tensor:
    """Adjusts negative pct gains by multiplying by loss_cost."""
    ones = tf.ones_like(values)
    return values * tf.where(values < 0.0, loss_cost * ones, ones)


def adjust_log_daily_gain(values: tf.Tensor, loss_cost: float) -> tf.Tensor:
    """Adjusts negative log gains by summing by log(loss_cost)."""
    ones = tf.ones_like(values)
    zeros = tf.zeros_like(values)

    # Calculate log gain assuming all are losses.
    log_adjusted_gain = tf.math.log(
        loss_cost * tf.math.exp(values) + (1.0 - loss_cost))
    return tf.where(values < 0.0, log_adjusted_gain, values)


def total_gain_from_log_gains(values: tf.Tensor) -> tf.Tensor:
    """Sum log gains and take the exponetial."""
    sum = tf.reduce_sum(values, axis=-1)
    return tf.math.exp(sum)
